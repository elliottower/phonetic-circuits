"""End-to-end EAP-IG pipeline: attribution + faithfulness evaluation + plots.

Pure Python, no Modal dependency. Can run locally or on any GPU machine.

Usage:
    python -m lib.run_pipeline
    python -m lib.run_pipeline --tasks op4_oronym --model gpt2
    python -m lib.run_pipeline --skip-eval  # attribution only
"""
import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference" / "mib"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference" / "mib" / "EAP-IG" / "src"))

from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.attribute import attribute
from lib.dataset import PhoneticDataset, PHONETIC_TASKS
from lib.run_attribution import phonetic_logit_diff, MODEL_NAMES


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_attribution(model, task: str, args) -> Path:
    log(f"Attribution: {task}")
    dataset = PhoneticDataset(
        task=task,
        tokenizer=model.tokenizer,
        num_examples=args.num_examples,
    )
    log(f"  {len(dataset)} examples")
    dataloader = dataset.to_dataloader(batch_size=args.batch_size)

    graph = Graph.from_model(model, neuron_level=False, node_scores=False)
    metric_fn = partial(phonetic_logit_diff, mean=True, loss=True)
    method_map = {"EAP-IG": "EAP-IG-inputs", "EAP": "EAP"}
    eap_method = method_map.get(args.method, args.method)
    ig_steps = args.ig_steps if args.method == "EAP-IG" else None

    attribute(
        model, graph, dataloader, metric_fn,
        method=eap_method,
        intervention=args.ablation,
        ig_steps=ig_steps,
    )

    save_dir = Path(args.output_dir) / "circuits" / f"{args.method}_{args.ablation}_edge"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{task}_{args.model}.json"
    graph.to_json(str(save_path))
    log(f"  Saved circuit: {save_path}")
    return save_path


def run_evaluation(model, task: str, circuit_path: Path, args) -> dict:
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    log(f"Evaluation: {task}")
    graph = Graph.from_json(str(circuit_path))

    dataset = PhoneticDataset(
        task=task,
        tokenizer=model.tokenizer,
        num_examples=args.num_examples,
    )
    log(f"  {len(dataset)} examples")
    dataloader = dataset.to_dataloader(batch_size=args.batch_size)

    metric_fn = partial(phonetic_logit_diff, mean=False, loss=False)

    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = evaluate_area_under_curve(
        model, graph, dataloader,
        metrics=metric_fn,
        intervention=args.ablation,
    )

    result = {
        "weighted_edge_counts": weighted_edge_counts,
        "area_under": area_under,
        "area_from_1": area_from_1,
        "average": average,
        "faithfulnesses": faithfulnesses,
    }

    save_dir = Path(args.output_dir) / "results" / f"{args.method}_{args.ablation}_edge"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{task}_{args.model}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    log(f"  CPR (area under): {area_under:.4f}")
    log(f"  CMD (area from 1): {area_from_1:.4f}")
    log(f"  Saved: {out_path}")
    return result


def run_behavioral(model, task: str, args) -> dict:
    log(f"Behavioral: {task}")
    dataset = PhoneticDataset(
        task=task,
        tokenizer=model.tokenizer,
        num_examples=args.num_examples,
    )
    n_correct = 0
    total_ld = 0.0
    for i in tqdm(range(len(dataset)), desc=task, leave=False):
        clean, corrupted, (tid, fid) = dataset[i]
        tokens = model.to_tokens(clean)
        with torch.no_grad():
            logits = model(tokens)
        tl = logits[0, -1, tid].item()
        fl = logits[0, -1, fid].item()
        total_ld += tl - fl
        if tl > fl:
            n_correct += 1
    acc = n_correct / len(dataset)
    mean_ld = total_ld / len(dataset)
    log(f"  {task}: {acc:.1%} ({n_correct}/{len(dataset)}) mean_ld={mean_ld:.2f}")
    return {"task": task, "accuracy": acc, "mean_logit_diff": mean_ld, "n": len(dataset)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phonetic circuits EAP-IG pipeline")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--method", type=str, default="EAP-IG", choices=["EAP", "EAP-IG"])
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, default="patching")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-eval", action="store_true", help="Skip faithfulness evaluation")
    parser.add_argument("--skip-behavioral", action="store_true", help="Skip behavioral validation")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = MODEL_NAMES.get(args.model, args.model)

    log(f"Loading model: {model_fullname} on {device}")
    model = HookedTransformer.from_pretrained(model_fullname, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    log(f"Model loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d={model.cfg.d_model}")

    summary = {"model": args.model, "method": args.method, "tasks": {}}

    for task in args.tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task}")
        log(f"{'='*60}")

        task_result = {}

        if not args.skip_behavioral:
            beh = run_behavioral(model, task, args)
            task_result["behavioral"] = beh
            if beh["accuracy"] < 0.70:
                log(f"  WARN: {task} accuracy {beh['accuracy']:.1%} < 70% gate")

        circuit_path = run_attribution(model, task, args)
        task_result["circuit_path"] = str(circuit_path)

        if not args.skip_eval:
            eval_result = run_evaluation(model, task, circuit_path, args)
            task_result["cpr"] = eval_result["area_under"]
            task_result["cmd"] = eval_result["area_from_1"]

        summary["tasks"][task] = task_result

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved: {summary_path}")

    if not args.skip_eval:
        from lib.plot import load_curve, area_chart, overlay

        pkl_dir = Path(args.output_dir) / "results" / f"{args.method}_{args.ablation}_edge"
        plot_dir = Path(args.output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        curves = []
        for task in args.tasks:
            pkl_path = pkl_dir / f"{task}_{args.model}.pkl"
            if pkl_path.exists():
                c = load_curve(pkl_path, label=task)
                curves.append(c)
                fig = area_chart(c, title=f"{task} ({args.model})")
                out = plot_dir / f"{task}_{args.model}_area.png"
                fig.savefig(str(out), dpi=150, bbox_inches="tight")
                plt.close(fig)
                log(f"  Plot: {out}")

        if len(curves) > 1:
            fig = overlay(curves, title=f"Phonetic circuits faithfulness ({args.model})")
            out = plot_dir / f"overlay_{args.model}.png"
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log(f"  Overlay: {out}")

    log("\n=== RESULTS ===")
    for task, res in summary["tasks"].items():
        parts = [task]
        if "behavioral" in res:
            parts.append(f"acc={res['behavioral']['accuracy']:.1%}")
        if "cpr" in res:
            parts.append(f"CPR={res['cpr']:.3f}")
        if "cmd" in res:
            parts.append(f"CMD={res['cmd']:.3f}")
        log("  ".join(parts))
