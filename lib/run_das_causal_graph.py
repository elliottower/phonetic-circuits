"""Causal discovery on DAS-learned variables across layers.

Instead of running causal discovery on raw head activations, this projects
each example onto the DAS-learned causal direction at each layer, producing
a scalar "causal variable" per layer per example. Then runs LiNGAM / PC on
the resulting [N_examples x N_layers] matrix to discover the DAG connecting
layers in causal-variable space.

Key prediction: compound operations (Op1+Op4 chained) should show a multi-node
chain (shorten -> fuse -> output), while simple operations should show a single
edge.

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_das_causal_graph --retrain --device cuda
    python -m lib.run_das_causal_graph --das-dir results_das --device cuda
    python -m lib.run_das_causal_graph --tasks op1_hypocorism op4_oronym --retrain
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
from lib.run_das import collect_resids, train_das


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_das_scalars(model, dataset, layers, directions, device, max_pairs=200):
    """Project each example onto the DAS direction at each layer.

    Returns (n_examples, n_layers) matrix of scalar causal variables.
    """
    n_layers = len(layers)
    scalars_clean = []
    scalars_corrupt = []

    n = min(len(dataset), max_pairs)
    for i in tqdm(range(n), desc="Extracting DAS scalars", leave=False):
        clean, corrupted, (tid, fid) = dataset[i]
        clean_toks = model.to_tokens(clean).to(device)
        corrupt_toks = model.to_tokens(corrupted).to(device)

        hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

        with torch.no_grad():
            _, cache_c = model.run_with_cache(clean_toks, names_filter=hook_names)
            _, cache_s = model.run_with_cache(corrupt_toks, names_filter=hook_names)

        row_clean = []
        row_corrupt = []
        for l_idx, l in enumerate(layers):
            hook = f"blocks.{l}.hook_resid_post"
            U = directions[l].to(device)
            resid_c = cache_c[hook][0, -1, :]
            resid_s = cache_s[hook][0, -1, :]
            row_clean.append(float((U.T @ resid_c).squeeze()))
            row_corrupt.append(float((U.T @ resid_s).squeeze()))

        scalars_clean.append(row_clean)
        scalars_corrupt.append(row_corrupt)

    return np.array(scalars_clean), np.array(scalars_corrupt)


def train_das_directions(model, task, layers, device, k=1, n_train=50, n_steps=100):
    """Train DAS direction at each layer for a task. Returns {layer: U_tensor}."""
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=n_train)
    directions = {}
    for l in tqdm(layers, desc=f"Training DAS ({task})"):
        train_data = collect_resids(model, dataset, l, device, max_pairs=n_train)
        U = train_das(model, train_data, l, device, k=k, n_steps=n_steps)
        directions[l] = U.cpu()
    return directions


def load_das_directions(das_dir, task, model_name="gpt2", k=1):
    """Load previously saved DAS directions for a task."""
    das_dir = Path(das_dir)
    directions = {}

    for path in sorted(das_dir.glob(f"das_k{k}_{task}_{model_name}.json")):
        with open(path) as f:
            data = json.load(f)
        task_data = data.get(task, data)
        for layer_str, layer_data in task_data.get("layers", {}).items():
            if "direction" in layer_data:
                U = torch.tensor(layer_data["direction"])
                directions[int(layer_str)] = U

    return directions


def run_pc_on_scalars(data, var_names, alpha=0.05):
    """PC algorithm on scalar DAS variables."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    log(f"  PC on DAS scalars (alpha={alpha}, {data.shape[0]} samples, {data.shape[1]} vars)")
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


def run_lingam_on_scalars(data, var_names):
    """DirectLiNGAM on scalar DAS variables."""
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    log(f"  LiNGAM on DAS scalars ({data.shape[0]} samples, {data.shape[1]} vars)")
    t0 = time.time()
    model = DirectLiNGAM()
    model.fit(data)
    elapsed = time.time() - t0

    B = model.adjacency_matrix_
    if isinstance(B, list):
        B = np.array(B)

    edges = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if abs(B[i, j]) > 0.01:
                edges.append((var_names[j], var_names[i], "directed", float(B[i, j])))

    order_raw = model.causal_order_
    if hasattr(order_raw, 'tolist'):
        order = order_raw.tolist()
    else:
        order = list(order_raw)
    causal_order_names = [var_names[i] for i in order]

    return {
        "method": "lingam",
        "n_edges": len(edges),
        "edges": edges,
        "elapsed_s": elapsed,
        "adjacency_matrix": B.tolist() if hasattr(B, 'tolist') else B,
        "causal_order": order,
        "causal_order_names": causal_order_names,
    }


def run_task_causal_graph(model, task, layers, directions, args, device):
    """Run causal discovery on DAS scalars for one task."""
    log(f"\nCausal graph discovery: {task}")
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer,
                              num_examples=args.num_examples)
    log(f"  {len(dataset)} examples, {len(layers)} layers")

    scalars_clean, scalars_corrupt = extract_das_scalars(
        model, dataset, layers, directions, device, max_pairs=args.num_examples or 200
    )

    data_combined = np.concatenate([scalars_clean, scalars_corrupt], axis=0)
    var_names = [f"L{l}" for l in layers]

    log(f"  Data matrix: {data_combined.shape}")
    log(f"  Variables: {var_names}")

    results = {
        "task": task,
        "layers": layers,
        "var_names": var_names,
        "n_examples_clean": len(scalars_clean),
        "n_examples_total": len(data_combined),
        "scalar_stats": {
            f"L{l}": {
                "mean_clean": float(np.mean(scalars_clean[:, i])),
                "std_clean": float(np.std(scalars_clean[:, i])),
                "mean_corrupt": float(np.mean(scalars_corrupt[:, i])),
                "std_corrupt": float(np.std(scalars_corrupt[:, i])),
            }
            for i, l in enumerate(layers)
        },
    }

    methods = args.methods
    for method in methods:
        log(f"  Method: {method}")
        try:
            if method == "pc":
                r = run_pc_on_scalars(data_combined, var_names, alpha=args.alpha)
            elif method == "lingam":
                r = run_lingam_on_scalars(data_combined, var_names)
            else:
                log(f"    Unknown method: {method}")
                continue

            results[method] = r
            log(f"    {r['n_edges']} edges found in {r['elapsed_s']:.1f}s")
            for edge in r["edges"][:20]:
                weight_str = f" (w={edge[3]:.3f})" if len(edge) > 3 else ""
                log(f"    {edge[0]} -> {edge[1]}{weight_str}")

            if method == "lingam" and "causal_order_names" in r:
                log(f"    Causal order: {' -> '.join(r['causal_order_names'])}")

        except Exception as e:
            log(f"    FAILED: {e}")
            results[method] = {"method": method, "error": str(e)}

    return results


def compare_dag_complexity(all_results):
    """Compare discovered DAG structure across tasks."""
    log("\n=== DAG COMPLEXITY COMPARISON ===")
    log("(Compound ops should have more edges / longer chains than simple ops)")

    for task, res in sorted(all_results.items()):
        for method in ["lingam", "pc"]:
            if method not in res or "n_edges" not in res.get(method, {}):
                continue
            r = res[method]
            n_edges = r["n_edges"]
            chain_info = ""
            if method == "lingam" and "causal_order_names" in r:
                chain_info = f"  order: {' -> '.join(r['causal_order_names'])}"
            log(f"  {task} ({method}): {n_edges} edges{chain_info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal discovery on DAS scalar variables")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4, 6, 8, 9, 10, 11])
    parser.add_argument("--methods", type=str, nargs="+", default=["pc", "lingam"])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=100,
                        help="DAS training steps (only used with --retrain)")
    parser.add_argument("--das-dir", type=str, default=None,
                        help="Directory with saved DAS directions. If not set, uses --retrain.")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain DAS directions instead of loading saved ones")
    parser.add_argument("--output-dir", type=str, default="results_das_causal_graph")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = {"gpt2": "gpt2"}.get(args.model, args.model)
    layers = args.layers

    log(f"Loading model: {model_fullname} on {device}")
    model = HookedTransformer.from_pretrained(model_fullname, device=device)
    log(f"Model loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d={model.cfg.d_model}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for task in args.tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task}")
        log(f"{'='*60}")

        if args.retrain or not args.das_dir:
            log(f"Training DAS directions at {len(layers)} layers...")
            directions = train_das_directions(
                model, task, layers, device, k=args.k,
                n_train=min(args.num_examples or 200, 50),
                n_steps=args.n_steps,
            )
        else:
            directions = load_das_directions(args.das_dir, task, args.model, args.k)
            if not directions:
                log(f"  No saved directions for {task}, training fresh...")
                directions = train_das_directions(
                    model, task, layers, device, k=args.k,
                    n_train=min(args.num_examples or 200, 50),
                    n_steps=args.n_steps,
                )

        available_layers = [l for l in layers if l in directions]
        if len(available_layers) < 3:
            log(f"  Only {len(available_layers)} layers have directions, need at least 3. Skipping.")
            continue

        t0 = time.time()
        results = run_task_causal_graph(model, task, available_layers, directions, args, device)
        elapsed = time.time() - t0
        log(f"  Total: {elapsed:.1f}s")

        all_results[task] = results

        task_path = out_dir / f"das_causal_graph_{task}_{args.model}.json"
        with open(task_path, "w") as f:
            json.dump({task: results}, f, indent=2)
        log(f"  Saved incremental: {task_path}")

    out_path = out_dir / f"das_causal_graph_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {out_path}")

    compare_dag_complexity(all_results)

    log("\n=== SUMMARY ===")
    for task, res in sorted(all_results.items()):
        summary_parts = []
        for method in ["pc", "lingam"]:
            if method in res and "n_edges" in res.get(method, {}):
                summary_parts.append(f"{method}={res[method]['n_edges']} edges")
        log(f"  {task}: {', '.join(summary_parts)}")
