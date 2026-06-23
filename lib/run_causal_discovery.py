"""Causal discovery on attention head activations for phonetic tasks.

Collects per-head outputs from GPT-2, runs causal discovery algorithms
to infer the causal DAG between heads — rather than assuming a graph.

Methods:
  - PC algorithm (constraint-based, conditional independence tests)
  - CD-NOD (exploits clean/corrupted distribution shift for edge orientation)
  - NOTEARS (continuous optimization, acyclicity constraint)
  - LiNGAM (non-Gaussianity for identifiability)

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_causal_discovery --tasks op4_oronym
    python -m lib.run_causal_discovery --tasks op4_oronym --methods pc cdnod
    python -m lib.run_causal_discovery  # all tasks, all methods
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

from lib.dataset import PhoneticDataset, PHONETIC_TASKS


MODEL_NAMES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
}

AVAILABLE_METHODS = ["pc", "cdnod", "notears", "lingam"]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_head_activations(model, dataset, device, max_examples=200):
    """Collect per-head outputs at last token position.

    Returns:
        acts_clean: (n_examples, n_heads_total) — each head's contribution to residual stream
        acts_corrupt: (n_examples, n_heads_total) — same for corrupted inputs
        head_names: list of "a{layer}.h{head}" strings
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    hook_names = [f"blocks.{l}.attn.hook_result" for l in range(n_layers)]

    head_names = []
    for l in range(n_layers):
        for h in range(n_heads):
            head_names.append(f"a{l}.h{h}")

    n = min(len(dataset), max_examples)
    acts_clean_list = []
    acts_corrupt_list = []

    for i in tqdm(range(n), desc="Collecting head activations"):
        clean, corrupted, _ = dataset[i]
        clean_toks = model.to_tokens(clean).to(device)
        corrupt_toks = model.to_tokens(corrupted).to(device)

        with torch.no_grad():
            _, cache_c = model.run_with_cache(clean_toks, names_filter=hook_names)
            _, cache_s = model.run_with_cache(corrupt_toks, names_filter=hook_names)

        clean_vec = []
        corrupt_vec = []
        for l in range(n_layers):
            hook = f"blocks.{l}.attn.hook_result"
            for h in range(n_heads):
                clean_vec.append(cache_c[hook][0, -1, h, :].norm().item())
                corrupt_vec.append(cache_s[hook][0, -1, h, :].norm().item())

        acts_clean_list.append(clean_vec)
        acts_corrupt_list.append(corrupt_vec)

    return np.array(acts_clean_list), np.array(acts_corrupt_list), head_names


def run_pc(data, var_names, alpha=0.05):
    """PC algorithm via causal-learn."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    log(f"  Running PC (alpha={alpha}, {data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()
    cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True,
            uc_rule=0, uc_priority=2, verbose=False, show_progress=False)
    elapsed = time.time() - t0

    adj = cg.G.graph
    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((var_names[i], var_names[j], "directed"))
            elif adj[i, j] == -1 and adj[j, i] == -1 and i < j:
                edges.append((var_names[i], var_names[j], "undirected"))

    return {
        "method": "pc",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency": adj.tolist(),
    }


def run_cdnod(data, c_indx, var_names, alpha=0.05):
    """CD-NOD: exploits distribution shift for edge orientation."""
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.cit import fisherz

    log(f"  Running CD-NOD (alpha={alpha}, {data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()
    cg = cdnod(data, c_indx, alpha=alpha, indep_test=fisherz, stable=True,
               uc_rule=0, uc_priority=2, verbose=False, show_progress=False)
    elapsed = time.time() - t0

    adj = cg.G.graph
    n_real = len(var_names)
    edges = []
    for i in range(n_real):
        for j in range(n_real):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((var_names[i], var_names[j], "directed"))
            elif adj[i, j] == -1 and adj[j, i] == -1 and i < j:
                edges.append((var_names[i], var_names[j], "undirected"))

    return {
        "method": "cdnod",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency": adj[:n_real, :n_real].tolist(),
    }


def run_notears(data, var_names):
    """NOTEARS: continuous optimization with acyclicity constraint."""
    from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

    log(f"  Running NOTEARS ({data.shape[0]} samples, {data.shape[1]} vars)")

    try:
        from notears.linear import notears_linear
        t0 = time.time()
        W = notears_linear(data, lambda1=0.1, loss_type="l2")
        elapsed = time.time() - t0
    except ImportError:
        from causallearn.search.FCMBased import lingam
        log("    notears not installed, falling back to DirectLiNGAM")
        return run_lingam(data, var_names)

    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if abs(W[i, j]) > 0.01:
                edges.append((var_names[i], var_names[j], "directed", float(W[i, j])))

    return {
        "method": "notears",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "weight_matrix": W.tolist() if hasattr(W, 'tolist') else W,
    }


def run_lingam(data, var_names):
    """DirectLiNGAM: non-Gaussianity based identifiability."""
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    log(f"  Running DirectLiNGAM ({data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()
    model = DirectLiNGAM()
    model.fit(data)
    elapsed = time.time() - t0

    B = model.adjacency_matrix_
    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if abs(B[i, j]) > 0.01:
                edges.append((var_names[j], var_names[i], "directed", float(B[i, j])))

    adj = B.tolist() if hasattr(B, 'tolist') else B
    order = model.causal_order_.tolist() if hasattr(model.causal_order_, 'tolist') else list(model.causal_order_)

    return {
        "method": "lingam",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency_matrix": adj,
        "causal_order": order,
    }


def select_top_heads(acts_clean, acts_corrupt, head_names, top_k=30):
    """Select heads with largest clean-vs-corrupt activation difference."""
    var_diff = np.var(acts_corrupt - acts_clean, axis=0)
    top_idx = np.argsort(-var_diff)[:top_k]
    selected_names = [head_names[i] for i in top_idx]
    log(f"  Selected top-{top_k} heads by variance: {selected_names[:10]}...")
    return acts_clean[:, top_idx], acts_corrupt[:, top_idx], selected_names


def run_task(model, task, args, device):
    log(f"Causal discovery: {task}")
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=args.num_examples)
    log(f"  {len(dataset)} examples")

    acts_clean, acts_corrupt, head_names = collect_head_activations(
        model, dataset, device, max_examples=args.num_examples or 200
    )

    sel_clean, sel_corrupt, sel_names = select_top_heads(
        acts_clean, acts_corrupt, head_names, top_k=args.top_k
    )

    data_combined = np.concatenate([sel_clean, sel_corrupt], axis=0)
    n_pairs = sel_clean.shape[0]
    c_indx = np.concatenate([np.zeros((n_pairs, 1)), np.ones((n_pairs, 1))], axis=0)

    results = {"task": task, "model": args.model, "top_k": args.top_k,
               "selected_heads": sel_names, "n_examples": n_pairs}

    for method in args.methods:
        log(f"  Method: {method}")
        try:
            if method == "pc":
                r = run_pc(data_combined, sel_names, alpha=args.alpha)
            elif method == "cdnod":
                r = run_cdnod(data_combined, c_indx, sel_names, alpha=args.alpha)
            elif method == "notears":
                r = run_notears(data_combined, sel_names)
            elif method == "lingam":
                r = run_lingam(data_combined, sel_names)
            else:
                log(f"    Unknown method: {method}")
                continue

            results[method] = r
            log(f"    {r['n_edges']} edges found in {r['elapsed_s']:.1f}s")

            if r["edges"]:
                log(f"    Top edges:")
                for edge in r["edges"][:10]:
                    log(f"      {edge[0]} -> {edge[1]} ({edge[2]})")

        except Exception as e:
            log(f"    FAILED: {e}")
            results[method] = {"method": method, "error": str(e)}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal discovery on head activations")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--methods", type=str, nargs="+", default=AVAILABLE_METHODS)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results_causal_discovery")
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

    all_results = {}
    for task in args.tasks:
        log(f"\n{'='*60}")
        t0 = time.time()
        results = run_task(model, task, args, device)
        elapsed = time.time() - t0
        log(f"  Total: {elapsed:.1f}s")
        all_results[task] = results

    out_path = out_dir / f"causal_discovery_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {out_path}")

    log("\n=== SUMMARY ===")
    for task, res in all_results.items():
        for method in args.methods:
            if method in res and "n_edges" in res[method]:
                log(f"  {task} / {method}: {res[method]['n_edges']} edges")
