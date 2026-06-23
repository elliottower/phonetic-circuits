"""Compose DAS directions across phonetic tasks.

Given learned DAS subspaces (rank-1 directions in R^768) per task per layer,
test whether phonological operations can be composed via linear algebra
on their causal subspaces.

Operations:
  AND  — shared subspace: project onto intersection of two task directions.
         If IIA stays high, the two tasks share a causal variable.
  OR   — union subspace: span of both directions (rank-2).
         If IIA is higher than either alone, the tasks encode complementary info.
  NOT  — ablate one task's direction, evaluate the other.
         If task B's IIA drops when we remove task A's direction, B depends on A's variable.
  TRANSFER — use task A's learned direction to intervene on task B.
         If IIA > chance, the tasks share causal structure.

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_das_composition --das-dir results_das
    python -m lib.run_das_composition --das-dir results_das --layer 9
"""
import argparse
import json
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

from lib.dataset import PhoneticDataset, PHONETIC_TASKS
from lib.run_das import collect_resids, eval_iia, train_das, log


MODEL_NAMES = {
    "gpt2": "gpt2",
}


def load_das_directions(das_dir, model_name="gpt2", k=1):
    """Load saved DAS directions (U matrices) per task per layer.

    Expects files like das_k1_op4_oronym_gpt2.json with a saved U matrix,
    or retrains if only IIA scores are available.
    """
    das_dir = Path(das_dir)
    directions = {}

    for path in sorted(das_dir.glob(f"das_k{k}_*_{model_name}.json")):
        with open(path) as f:
            data = json.load(f)
        for task, res in data.items():
            if task not in directions:
                directions[task] = {}
            for layer_str, layer_data in res.get("layers", {}).items():
                if "direction" in layer_data:
                    U = torch.tensor(layer_data["direction"])
                    directions[task][int(layer_str)] = U

    return directions


def intersect_subspaces(U_a, U_b):
    """AND: find the shared direction between two rank-1 subspaces."""
    cos_sim = abs(float((U_a.T @ U_b).squeeze()))
    shared = (U_a + U_b * torch.sign((U_a.T @ U_b).squeeze()))
    shared = shared / shared.norm()
    return shared.unsqueeze(1) if shared.dim() == 1 else shared, cos_sim


def union_subspaces(U_a, U_b):
    """OR: span of both directions (rank-2 subspace)."""
    combined = torch.cat([U_a, U_b], dim=1)
    Q, _ = torch.linalg.qr(combined)
    return Q


def ablate_direction(U_full, U_remove):
    """NOT: project out U_remove from U_full."""
    proj = U_remove @ U_remove.T
    residual = U_full - proj @ U_full
    if residual.norm() < 1e-8:
        return torch.zeros_like(U_full)
    return residual / residual.norm()


def run_composition_experiments(model, directions, args, device):
    tasks = sorted(directions.keys())
    layer = args.layer
    results = {"layer": layer, "experiments": {}}

    log(f"Layer {layer}: {len(tasks)} tasks with directions")

    task_data = {}
    for task in tasks:
        if layer not in directions[task]:
            log(f"  Skipping {task} — no direction at layer {layer}")
            continue
        dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=50)
        eval_data = collect_resids(model, dataset, layer, device, max_pairs=50)
        task_data[task] = eval_data

    available_tasks = [t for t in tasks if t in task_data and layer in directions[t]]

    # TRANSFER: use task A's direction on task B
    log("\n--- TRANSFER ---")
    transfer_results = {}
    for task_a in available_tasks:
        U_a = directions[task_a][layer].to(device)
        for task_b in available_tasks:
            iia = eval_iia(model, task_data[task_b], U_a, layer, device)
            key = f"{task_a}->{task_b}"
            transfer_results[key] = iia
            marker = " *SELF*" if task_a == task_b else ""
            log(f"  {key}: IIA = {iia:.3f}{marker}")
    results["experiments"]["transfer"] = transfer_results

    # AND: shared subspace between pairs
    log("\n--- AND (intersection) ---")
    and_results = {}
    for task_a, task_b in combinations(available_tasks, 2):
        U_a = directions[task_a][layer].to(device)
        U_b = directions[task_b][layer].to(device)
        U_shared, cos_sim = intersect_subspaces(U_a, U_b)
        U_shared = U_shared.to(device)

        iia_a = eval_iia(model, task_data[task_a], U_shared, layer, device)
        iia_b = eval_iia(model, task_data[task_b], U_shared, layer, device)

        key = f"{task_a}&{task_b}"
        and_results[key] = {
            "cos_sim": cos_sim,
            "iia_on_a": iia_a,
            "iia_on_b": iia_b,
        }
        log(f"  {key}: cos={cos_sim:.3f}, IIA_a={iia_a:.3f}, IIA_b={iia_b:.3f}")
    results["experiments"]["and"] = and_results

    # OR: union of two directions (rank-2)
    log("\n--- OR (union) ---")
    or_results = {}
    for task_a, task_b in combinations(available_tasks, 2):
        U_a = directions[task_a][layer].to(device)
        U_b = directions[task_b][layer].to(device)
        U_union = union_subspaces(U_a, U_b).to(device)

        iia_a = eval_iia(model, task_data[task_a], U_union, layer, device)
        iia_b = eval_iia(model, task_data[task_b], U_union, layer, device)

        key = f"{task_a}|{task_b}"
        or_results[key] = {
            "iia_on_a": iia_a,
            "iia_on_b": iia_b,
        }
        log(f"  {key}: IIA_a={iia_a:.3f}, IIA_b={iia_b:.3f}")
    results["experiments"]["or"] = or_results

    # NOT: ablate task A's direction, evaluate task B
    log("\n--- NOT (ablation) ---")
    not_results = {}
    for task_a in available_tasks:
        U_a = directions[task_a][layer].to(device)
        for task_b in available_tasks:
            if task_a == task_b:
                continue
            U_b = directions[task_b][layer].to(device)
            U_b_minus_a = ablate_direction(U_b, U_a)
            if U_b_minus_a.norm() < 1e-8:
                iia = 0.0
            else:
                U_b_minus_a = U_b_minus_a.to(device)
                iia = eval_iia(model, task_data[task_b], U_b_minus_a, layer, device)

            iia_orig = eval_iia(model, task_data[task_b], U_b, layer, device)
            key = f"{task_b}\\{task_a}"
            not_results[key] = {
                "iia_original": iia_orig,
                "iia_after_ablation": iia,
                "iia_drop": iia_orig - iia,
            }
            log(f"  {key}: orig={iia_orig:.3f}, after={iia:.3f}, drop={iia_orig - iia:+.3f}")
    results["experiments"]["not"] = not_results

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose DAS directions across tasks")
    parser.add_argument("--das-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain DAS directions instead of loading saved ones")
    parser.add_argument("--output-dir", type=str, default="results_das_composition")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = MODEL_NAMES.get(args.model, args.model)

    log(f"Loading model: {model_fullname} on {device}")
    model = HookedTransformer.from_pretrained(model_fullname, device=device)
    log(f"Model loaded")

    if args.retrain:
        log("Retraining DAS directions...")
        directions = {}
        for task in sorted(PHONETIC_TASKS):
            log(f"  Training {task} at layer {args.layer}...")
            dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=50)
            train_data = collect_resids(model, dataset, args.layer, device, max_pairs=50)
            U = train_das(model, train_data, args.layer, device, k=args.k, n_steps=100)
            directions.setdefault(task, {})[args.layer] = U.cpu()
    else:
        if not args.das_dir:
            parser.error("--das-dir is required when not using --retrain")
        log(f"Loading DAS directions from {args.das_dir}")
        directions = load_das_directions(args.das_dir, args.model, args.k)

    if not directions:
        log("No directions found. Use --retrain to train them fresh.")
    else:
        results = run_composition_experiments(model, directions, args, device)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"composition_L{args.layer}_k{args.k}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\nSaved: {out_path}")
