"""Compositional ablation: test sub-circuit independence for chained operations.

For compound phonological operations (e.g., William Ding -> Building requires
Op 1 shortening + Op 4 fusion), ablating the Op 1 circuit should break compound
predictions but leave standalone Op 4 predictions intact, and vice versa.

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_compositional_ablation --circuit-a op1_hypocorism --circuit-b op4_oronym
    python -m lib.run_compositional_ablation  # default: op1 + op4
"""
import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

from lib.dataset import PhoneticDataset, PHONETIC_TASKS


MODEL_NAMES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
}

CIRCUITS_DIR = Path(__file__).resolve().parent.parent / "results" / "circuits"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_circuit_heads(task: str, top_k: int, model_name: str = "gpt2") -> list[tuple[int, int]]:
    """Load a circuit JSON and return the top-K heads by total absolute edge score.

    Aggregates absolute edge scores per destination head. Edge keys look like
    "a0.h1->a5.h3<q>" or "a10.h0->logits". We aggregate by the head that
    appears as *either* source or destination in any edge.
    """
    circuit_path = CIRCUITS_DIR / f"{task}_{model_name}.json"
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

    with open(circuit_path) as f:
        data = json.load(f)

    edges = data["edges"]
    head_scores = defaultdict(float)
    head_pattern = re.compile(r"a(\d+)\.h(\d+)")

    for edge_key, edge_data in edges.items():
        score = abs(edge_data["score"])
        # Find all heads mentioned in this edge key
        for match in head_pattern.finditer(edge_key):
            layer = int(match.group(1))
            head = int(match.group(2))
            head_scores[(layer, head)] += score

    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    top_heads = [lh for lh, _ in sorted_heads[:top_k]]

    return top_heads


def make_ablation_hooks(heads_to_ablate: set[tuple[int, int]]) -> list[tuple[str, callable]]:
    """Create TransformerLens hooks that zero-ablate the specified heads.

    Each hook targets blocks.{layer}.attn.hook_result and zeros out
    [:, :, head_idx, :] for ablated heads at that layer.
    """
    # Group heads by layer
    layer_heads = defaultdict(list)
    for layer, head in heads_to_ablate:
        layer_heads[layer].append(head)

    hooks = []
    for layer, head_list in layer_heads.items():
        hook_name = f"blocks.{layer}.attn.hook_result"

        def make_hook(_heads):
            def hook_fn(value, hook):
                for h in _heads:
                    value[:, :, h, :] = 0.0
                return value
            return hook_fn

        hooks.append((hook_name, make_hook(head_list)))

    return hooks


def evaluate_with_ablation(
    model,
    dataset: PhoneticDataset,
    heads_to_ablate: set[tuple[int, int]],
    device: str,
    num_examples: int,
) -> dict:
    """Run model with ablation hooks, compute logit_diff and accuracy.

    Returns dict with mean_logit_diff, accuracy, n_examples, and per-example diffs.
    """
    hooks = make_ablation_hooks(heads_to_ablate) if heads_to_ablate else []
    n = min(len(dataset), num_examples)

    logit_diffs = []
    correct = 0

    for i in tqdm(range(n), desc="Evaluating", leave=False):
        clean, _corrupted, (target_idx, foil_idx) = dataset[i]
        tokens = model.to_tokens(clean).to(device)

        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)

        target_logit = logits[0, -1, target_idx].item()
        foil_logit = logits[0, -1, foil_idx].item()
        diff = target_logit - foil_logit
        logit_diffs.append(diff)

        if target_logit > foil_logit:
            correct += 1

    mean_diff = sum(logit_diffs) / len(logit_diffs) if logit_diffs else 0.0
    accuracy = correct / n if n > 0 else 0.0

    return {
        "mean_logit_diff": mean_diff,
        "accuracy": accuracy,
        "n_examples": n,
        "logit_diffs": logit_diffs,
    }


def run_ablation_experiment(
    model,
    circuit_a_task: str,
    circuit_b_task: str,
    top_k: int,
    device: str,
    num_examples: int,
    model_name: str = "gpt2",
) -> dict:
    """Run the full compositional ablation experiment.

    Tests all four ablation conditions (baseline, ablate_a, ablate_b, ablate_both)
    on standalone examples for each task.
    """
    log(f"Loading circuit heads: {circuit_a_task} (top-{top_k})")
    heads_a = load_circuit_heads(circuit_a_task, top_k, model_name)
    log(f"  Circuit A heads: {heads_a}")

    log(f"Loading circuit heads: {circuit_b_task} (top-{top_k})")
    heads_b = load_circuit_heads(circuit_b_task, top_k, model_name)
    log(f"  Circuit B heads: {heads_b}")

    heads_a_set = set(heads_a)
    heads_b_set = set(heads_b)
    overlap = heads_a_set & heads_b_set
    log(f"Overlap: {len(overlap)} heads — {sorted(overlap)}")

    # Load datasets
    log(f"Loading dataset: {circuit_a_task}")
    dataset_a = PhoneticDataset(task=circuit_a_task, tokenizer=model.tokenizer, num_examples=num_examples)
    log(f"  {len(dataset_a)} examples")

    log(f"Loading dataset: {circuit_b_task}")
    dataset_b = PhoneticDataset(task=circuit_b_task, tokenizer=model.tokenizer, num_examples=num_examples)
    log(f"  {len(dataset_b)} examples")

    conditions = {
        "baseline": set(),
        f"ablate_{circuit_a_task}": heads_a_set,
        f"ablate_{circuit_b_task}": heads_b_set,
        "ablate_both": heads_a_set | heads_b_set,
    }

    results = {
        "config": {
            "circuit_a": circuit_a_task,
            "circuit_b": circuit_b_task,
            "top_k": top_k,
            "model": model_name,
            "num_examples": num_examples,
        },
        "heads": {
            circuit_a_task: [list(h) for h in heads_a],
            circuit_b_task: [list(h) for h in heads_b],
            "overlap": [list(h) for h in sorted(overlap)],
        },
        "conditions": {},
    }

    for condition_name, ablated_heads in conditions.items():
        log(f"\n--- Condition: {condition_name} ({len(ablated_heads)} heads ablated) ---")
        condition_results = {}

        # Evaluate on task A examples
        log(f"  Evaluating on {circuit_a_task} examples...")
        t0 = time.time()
        res_a = evaluate_with_ablation(model, dataset_a, ablated_heads, device, num_examples)
        elapsed = time.time() - t0
        condition_results[circuit_a_task] = {
            "mean_logit_diff": res_a["mean_logit_diff"],
            "accuracy": res_a["accuracy"],
            "n_examples": res_a["n_examples"],
        }
        log(f"    {circuit_a_task}: logit_diff={res_a['mean_logit_diff']:.4f}, acc={res_a['accuracy']:.2%} ({elapsed:.1f}s)")

        # Evaluate on task B examples
        log(f"  Evaluating on {circuit_b_task} examples...")
        t0 = time.time()
        res_b = evaluate_with_ablation(model, dataset_b, ablated_heads, device, num_examples)
        elapsed = time.time() - t0
        condition_results[circuit_b_task] = {
            "mean_logit_diff": res_b["mean_logit_diff"],
            "accuracy": res_b["accuracy"],
            "n_examples": res_b["n_examples"],
        }
        log(f"    {circuit_b_task}: logit_diff={res_b['mean_logit_diff']:.4f}, acc={res_b['accuracy']:.2%} ({elapsed:.1f}s)")

        results["conditions"][condition_name] = condition_results

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compositional ablation for phonetic circuit tasks")
    parser.add_argument("--circuit-a", type=str, default="op1_hypocorism",
                        help="First task circuit to test")
    parser.add_argument("--circuit-b", type=str, default="op4_oronym",
                        help="Second task circuit to test")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Number of top heads per circuit to ablate")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results_compositional_ablation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = MODEL_NAMES.get(args.model, args.model)

    log(f"Loading model: {model_fullname} on {device}")
    model = HookedTransformer.from_pretrained(model_fullname, device=device)
    model.cfg.use_attn_result = True
    log(f"Model loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d={model.cfg.d_model}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*60}")
    log(f"Compositional ablation: {args.circuit_a} x {args.circuit_b}")
    log(f"{'='*60}")

    t0 = time.time()
    results = run_ablation_experiment(
        model=model,
        circuit_a_task=args.circuit_a,
        circuit_b_task=args.circuit_b,
        top_k=args.top_k,
        device=device,
        num_examples=args.num_examples,
        model_name=args.model,
    )
    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.1f}s")

    out_path = out_dir / f"compositional_{args.circuit_a}_x_{args.circuit_b}_k{args.top_k}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved: {out_path}")

    # Summary table
    log("\n=== SUMMARY ===")
    log(f"{'Condition':<30} {'Task A logit_diff':>18} {'Task A acc':>10} {'Task B logit_diff':>18} {'Task B acc':>10}")
    log("-" * 90)
    for cond_name, cond_data in results["conditions"].items():
        a_data = cond_data[args.circuit_a]
        b_data = cond_data[args.circuit_b]
        log(f"{cond_name:<30} {a_data['mean_logit_diff']:>18.4f} {a_data['accuracy']:>10.2%} {b_data['mean_logit_diff']:>18.4f} {b_data['accuracy']:>10.2%}")
