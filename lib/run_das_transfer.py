"""Cross-task DAS transfer: do phonological tasks share causal subspaces?

Trains DAS direction on task A, evaluates IIA on task B. If tasks share
causal structure, cross-task IIA should exceed random baseline.

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_das_transfer --layer 10
    python -m lib.run_das_transfer --tasks op1_hypocorism op4_oronym --layer 10
"""
import argparse
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

from lib.dataset import PhoneticDataset, PHONETIC_TASKS
from lib.run_das import train_das, eval_iia, collect_resids, log


def subspace_alignment(U_a: torch.Tensor, U_b: torch.Tensor) -> float:
    """Squared cosine of principal angle between two rank-k subspaces.

    For k=1 this is simply (U_a^T U_b)^2. For k>1 it returns the mean
    of the squared singular values of U_a^T U_b (average squared cosine
    of principal angles).
    """
    inner = U_a.T @ U_b
    if inner.numel() == 1:
        return inner.item() ** 2
    svals = torch.linalg.svdvals(inner)
    return (svals ** 2).mean().item()


def train_direction_for_task(model, task, layer, device, k, n_steps, num_examples):
    """Train DAS direction on 50% of task data, return (U, eval_data)."""
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=num_examples)
    n_total = len(dataset)
    n_train = min(int(n_total * 0.5), 50)
    n_eval = min(n_total - n_train, 50)

    log(f"  {task}: {n_total} examples, train={n_train}, eval={n_eval}")

    train_data = collect_resids(model, dataset, layer, device, max_pairs=n_train)

    eval_dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=num_examples)
    eval_dataset.df = eval_dataset.df.iloc[n_train:n_train + n_eval]
    eval_data = collect_resids(model, eval_dataset, layer, device, max_pairs=n_eval)

    U = train_das(model, train_data, layer, device, k=k, n_steps=n_steps)
    return U, eval_data


def run_transfer(model, tasks, layer, device, k, n_steps, num_examples):
    """Build the full transfer matrix.

    Returns:
        transfer_matrix: dict[source_task][target_task] = IIA
        alignment_matrix: dict[task_a][task_b] = cos^2 angle
        directions: dict[task] = U tensor
        self_iia: dict[task] = self-task IIA
        random_iia: dict[task] = random baseline IIA
        details: list of per-pair result dicts
    """
    directions = {}
    eval_data = {}
    self_iia = {}
    random_iia = {}

    d_model = model.cfg.d_model

    # Phase 1: train DAS direction per task, compute self-task IIA + random baseline
    log(f"\n--- Phase 1: Train DAS directions (layer {layer}, k={k}) ---")
    for task in tqdm(tasks, desc="Training"):
        log(f"\nTraining DAS for {task}")
        U, edata = train_direction_for_task(model, task, layer, device, k, n_steps, num_examples)
        directions[task] = U
        eval_data[task] = edata

        self_iia[task] = eval_iia(model, edata, U, layer, device)

        random_U = torch.randn(d_model, k, device=device)
        random_U, _ = torch.linalg.qr(random_U)
        random_iia[task] = eval_iia(model, edata, random_U, layer, device)

        log(f"  Self IIA: {self_iia[task]:.3f}  Random IIA: {random_iia[task]:.3f}")

    # Phase 2: cross-task transfer evaluation
    log(f"\n--- Phase 2: Cross-task transfer ---")
    transfer_matrix = {src: {} for src in tasks}
    details = []

    pairs = [(src, tgt) for src, tgt in product(tasks, tasks)]
    for src, tgt in tqdm(pairs, desc="Transfer eval"):
        if src == tgt:
            transfer_matrix[src][tgt] = self_iia[src]
            continue

        transfer_iia = eval_iia(model, eval_data[tgt], directions[src], layer, device)
        transfer_matrix[src][tgt] = transfer_iia

        details.append({
            "source_task": src,
            "target_task": tgt,
            "transfer_iia": transfer_iia,
            "self_iia_source": self_iia[src],
            "self_iia_target": self_iia[tgt],
            "random_iia_target": random_iia[tgt],
        })

    # Phase 3: subspace alignment between all direction pairs
    log(f"\n--- Phase 3: Subspace alignment ---")
    alignment_matrix = {a: {} for a in tasks}
    for a, b in product(tasks, tasks):
        alignment_matrix[a][b] = subspace_alignment(directions[a], directions[b])

    return transfer_matrix, alignment_matrix, directions, self_iia, random_iia, details


def print_summary(tasks, transfer_matrix, alignment_matrix, self_iia, random_iia):
    """Print transfer matrix and alignment as formatted tables."""
    task_labels = [t.replace("_", " ")[:12] for t in tasks]

    log("\n=== TRANSFER MATRIX (rows=source, cols=target) ===")
    header = f"{'source':<14}" + "".join(f"{l:>13}" for l in task_labels)
    log(header)
    log("-" * len(header))
    for i, src in enumerate(tasks):
        row = f"{task_labels[i]:<14}"
        for tgt in tasks:
            val = transfer_matrix[src][tgt]
            marker = " *" if src == tgt else "  "
            row += f"{val:>11.3f}{marker}"
        log(row)

    log(f"\n{'Self IIA':<14}" + "".join(f"{self_iia[t]:>13.3f}" for t in tasks))
    log(f"{'Random IIA':<14}" + "".join(f"{random_iia[t]:>13.3f}" for t in tasks))

    log("\n=== SUBSPACE ALIGNMENT (cos^2 principal angle) ===")
    header = f"{'':>14}" + "".join(f"{l:>13}" for l in task_labels)
    log(header)
    log("-" * len(header))
    for i, a in enumerate(tasks):
        row = f"{task_labels[i]:<14}"
        for b in tasks:
            val = alignment_matrix[a][b]
            row += f"{val:>13.3f}"
        log(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-task DAS transfer")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results_das_transfer")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    log(f"Loading model: {model_name} on {device}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    log(f"Model loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d={model.cfg.d_model}")

    tasks = args.tasks
    log(f"Tasks: {tasks}")
    log(f"Layer: {args.layer}, k={args.k}")

    t0 = time.time()
    transfer_matrix, alignment_matrix, directions, self_iia, random_iia, details = (
        run_transfer(model, tasks, args.layer, device, args.k, args.n_steps, args.num_examples)
    )
    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.1f}s")

    print_summary(tasks, transfer_matrix, alignment_matrix, self_iia, random_iia)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "model": args.model,
            "layer": args.layer,
            "k": args.k,
            "n_steps": args.n_steps,
            "tasks": tasks,
            "timestamp": datetime.now().isoformat(),
        },
        "transfer_matrix": transfer_matrix,
        "alignment_matrix": alignment_matrix,
        "self_iia": self_iia,
        "random_iia": random_iia,
        "details": details,
    }

    out_path = out_dir / f"transfer_L{args.layer}_k{args.k}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {out_path}")
