"""Causal discovery on attention head activations for phonetic tasks.

Collects per-head outputs from GPT-2, runs causal discovery algorithms
to infer the causal DAG between heads — rather than assuming a graph.

Methods:
  - PC algorithm (constraint-based, conditional independence tests)
  - TPC (Temporal PC) — PC with layer-based tiered background knowledge
  - CD-NOD (exploits clean/corrupted distribution shift for edge orientation)
  - PCMCI+ (tigramite, treats layers as time dimension)
  - GES (Greedy Equivalence Search, score-based)
  - LiNGAM (non-Gaussianity for identifiability)

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_causal_discovery --tasks op4_oronym
    python -m lib.run_causal_discovery --tasks op4_oronym --methods pc cdnod
    python -m lib.run_causal_discovery  # all tasks, all methods
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.utils.cit import fisherz
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from transformer_lens import HookedTransformer

from lib.dataset import PhoneticDataset, PHONETIC_TASKS


MODEL_NAMES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
}

AVAILABLE_METHODS = ["pc", "tpc", "cdnod", "pcmci", "ges", "lingam"]


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


def _parse_layer(name):
    """Extract layer number from head name like 'a11.h0' -> 11."""
    m = re.match(r'a(\d+)\.h\d+', name)
    return int(m.group(1)) if m else 0


def run_tpc(data, var_names, alpha=0.05):
    """Temporal PC: PC with layer-based tiered background knowledge.

    Cross-tier edges are directed (lower layer -> higher layer).
    Within-tier edges remain undirected (discovered by PC).
    """
    log(f"  Running TPC (alpha={alpha}, {data.shape[0]} samples, {data.shape[1]} vars)")

    layers = [_parse_layer(name) for name in var_names]

    bk = BackgroundKnowledge()
    nodes = {name: GraphNode(name) for name in var_names}
    n = len(var_names)
    for i in tqdm(range(n), desc="    Building background knowledge"):
        for j in range(n):
            if i == j:
                continue
            if layers[i] > layers[j]:
                # Forbid higher-layer -> lower-layer direction
                bk.add_forbidden_by_node(nodes[var_names[i]], nodes[var_names[j]])

    t0 = time.time()
    cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True,
            uc_rule=0, uc_priority=2, verbose=False, show_progress=False,
            background_knowledge=bk)
    elapsed = time.time() - t0

    adj = cg.G.graph
    edges = []
    for i in range(n):
        for j in range(n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((var_names[i], var_names[j], "directed"))
            elif adj[i, j] == -1 and adj[j, i] == -1 and i < j:
                edges.append((var_names[i], var_names[j], "undirected"))

    return {
        "method": "tpc",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency": adj.tolist(),
    }


def run_pcmci(data, var_names, alpha=0.05):
    """PCMCI+: treats head activations as multivariate 'time series' over layers.

    Each sample is one observation. Variables are grouped by layer to define
    the temporal ordering. PCMCI+ handles contemporaneous edges (same layer)
    and skip-layer connections.
    """
    log(f"  Running PCMCI+ (alpha={alpha}, {data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()

    # tigramite expects (T, N) array
    dataframe = pp.DataFrame(data, var_names=var_names)
    parcorr = ParCorr(significance='analytic')
    pcmci_obj = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

    # tau_max: max lag = number of distinct layers - 1
    layers = [_parse_layer(name) for name in var_names]
    tau_max = max(layers) - min(layers) if len(set(layers)) > 1 else 1
    # Cap at reasonable value to keep runtime manageable
    tau_max = min(tau_max, 5)

    results_pcmci = pcmci_obj.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)
    elapsed = time.time() - t0

    graph = results_pcmci['graph']  # (N, N, tau_max+1) array of strings
    val_matrix = results_pcmci['val_matrix']
    n = len(var_names)

    edges = []
    for tau in range(tau_max + 1):
        for i in range(n):
            for j in range(n):
                link = graph[i, j, tau]
                if link == '-->':
                    edges.append((var_names[i], var_names[j], "directed", float(val_matrix[i, j, tau]), tau))
                elif link == 'o-o' and i < j:
                    edges.append((var_names[i], var_names[j], "undirected", float(val_matrix[i, j, tau]), tau))
                elif link == 'x-x' and i < j:
                    edges.append((var_names[i], var_names[j], "undirected", float(val_matrix[i, j, tau]), tau))

    return {
        "method": "pcmci",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "tau_max": tau_max,
    }


def run_ges(data, var_names):
    """GES: Greedy Equivalence Search (score-based)."""
    log(f"  Running GES ({data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()
    result = ges(data, score_func='local_score_BIC')
    elapsed = time.time() - t0

    adj = result['G'].graph
    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((var_names[i], var_names[j], "directed"))
            elif adj[i, j] == -1 and adj[j, i] == -1 and i < j:
                edges.append((var_names[i], var_names[j], "undirected"))

    return {
        "method": "ges",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency": adj.tolist(),
    }


def run_lingam(data, var_names):
    """DirectLiNGAM: non-Gaussianity based identifiability."""
    log(f"  Running DirectLiNGAM ({data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()
    model = DirectLiNGAM()
    model.fit(data)
    elapsed = time.time() - t0

    B = np.array(model.adjacency_matrix_) if not isinstance(model.adjacency_matrix_, np.ndarray) else model.adjacency_matrix_
    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if abs(B[i, j]) > 0.01:
                edges.append((var_names[j], var_names[i], "directed", float(B[i, j])))

    order = model.causal_order_
    if hasattr(order, 'tolist'):
        order = order.tolist()
    else:
        order = list(order)

    return {
        "method": "lingam",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency_matrix": B.tolist(),
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
            elif method == "tpc":
                r = run_tpc(data_combined, sel_names, alpha=args.alpha)
            elif method == "cdnod":
                r = run_cdnod(data_combined, c_indx, sel_names, alpha=args.alpha)
            elif method == "pcmci":
                r = run_pcmci(data_combined, sel_names, alpha=args.alpha)
            elif method == "ges":
                r = run_ges(data_combined, sel_names)
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
