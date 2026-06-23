"""Distributed Alignment Search (DAS) for phonetic circuit tasks.

Finds linear subspaces encoding causal variables at each layer.
Trains a rotation matrix U, evaluates Interchange Intervention Accuracy (IIA).

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_das --tasks op4_oronym
    python -m lib.run_das --tasks op4_oronym --layers 8 9 10 --k 1
    python -m lib.run_das  # all tasks, all layers
"""
import argparse
import json
import time
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


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_resids(model, dataset, layer, device, max_pairs=200):
    hook_name = f"blocks.{layer}.hook_resid_post"
    data = []
    n = min(len(dataset), max_pairs)
    for i in tqdm(range(n), desc=f"Collecting L{layer}", leave=False):
        clean, corrupted, (tid, fid) = dataset[i]
        clean_toks = model.to_tokens(clean).to(device)
        corrupted_toks = model.to_tokens(corrupted).to(device)

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_toks, names_filter=[hook_name])
            _, corr_cache = model.run_with_cache(corrupted_toks, names_filter=[hook_name])

        data.append({
            "base_resid": clean_cache[hook_name][0, -1, :].clone(),
            "source_resid": corr_cache[hook_name][0, -1, :].clone(),
            "base_toks": clean_toks,
            "src_id": tid,
            "base_id": fid,
        })
    return data


def eval_iia(model, data, U, layer, device):
    hook_name = f"blocks.{layer}.hook_resid_post"
    proj = U @ U.T
    correct = 0
    total = 0

    for d in data:
        diff = d["source_resid"] - d["base_resid"]
        intervention = proj @ diff

        def make_hook(_interv):
            def hk(act, hook):
                new = act.clone()
                new[0, -1, :] += _interv
                return new
            return hk

        with torch.no_grad():
            logits = model.run_with_hooks(
                d["base_toks"],
                fwd_hooks=[(hook_name, make_hook(intervention))],
            )

        if logits[0, -1, d["src_id"]].item() > logits[0, -1, d["base_id"]].item():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def train_das(model, train_data, layer, device, k=1, n_steps=200, lr=1e-3):
    d_model = train_data[0]["base_resid"].shape[0]
    hook_name = f"blocks.{layer}.hook_resid_post"

    deltas = torch.stack([d["source_resid"] - d["base_resid"] for d in train_data])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    A = torch.nn.Parameter(Vh[:k].T.clone().to(device))
    optimizer = torch.optim.Adam([A], lr=lr)

    micro_batch = 10
    n_train = len(train_data)

    best_loss = float("inf")
    patience_counter = 0

    for step in range(n_steps):
        optimizer.zero_grad()
        step_loss = 0.0

        for mb_start in range(0, n_train, micro_batch):
            Q, _ = torch.linalg.qr(A)
            proj = Q @ Q.T
            mb_loss = torch.tensor(0.0, device=device)

            for d in train_data[mb_start:mb_start + micro_batch]:
                diff = d["source_resid"] - d["base_resid"]
                intervention = proj @ diff

                def make_hook(_interv):
                    def hk(act, hook):
                        new = act.clone()
                        new[0, -1, :] += _interv
                        return new
                    return hk

                logits = model.run_with_hooks(
                    d["base_toks"],
                    fwd_hooks=[(hook_name, make_hook(intervention))],
                )
                log_probs = logits[0, -1, :].log_softmax(dim=-1)
                mb_loss -= log_probs[d["src_id"]]

            (mb_loss / n_train).backward()
            step_loss += mb_loss.item() / n_train

        optimizer.step()

        if step_loss < best_loss:
            best_loss = step_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    Q, _ = torch.linalg.qr(A)
    return Q.detach()


def run_das_for_task(model, task, args, device):
    log(f"DAS: {task}")
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=args.num_examples)
    log(f"  {len(dataset)} examples")

    n_total = len(dataset)
    n_train = min(int(n_total * 0.5), 50)
    n_eval = min(n_total - n_train, 50)

    results = {"task": task, "model": args.model, "k": args.k, "layers": {}}

    for layer in args.layers:
        log(f"  Layer {layer}...")
        train_data = collect_resids(model, dataset, layer, device, max_pairs=n_train)

        eval_dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=args.num_examples)
        eval_dataset.df = eval_dataset.df.iloc[n_train:n_train + n_eval]
        eval_data = collect_resids(model, eval_dataset, layer, device, max_pairs=n_eval)

        # Baseline: random direction IIA
        d_model = train_data[0]["base_resid"].shape[0]
        random_U = torch.randn(d_model, args.k, device=device)
        random_U, _ = torch.linalg.qr(random_U)
        random_iia = eval_iia(model, eval_data, random_U, layer, device)

        # PCA baseline (no training)
        deltas = torch.stack([d["source_resid"] - d["base_resid"] for d in train_data])
        _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
        pca_U = Vh[:args.k].T.to(device)
        pca_iia = eval_iia(model, eval_data, pca_U, layer, device)

        # Trained DAS
        U = train_das(model, train_data, layer, device, k=args.k, n_steps=args.n_steps)
        das_iia = eval_iia(model, eval_data, U, layer, device)

        results["layers"][str(layer)] = {
            "random_iia": random_iia,
            "pca_iia": pca_iia,
            "das_iia": das_iia,
            "n_train": len(train_data),
            "n_eval": len(eval_data),
        }
        log(f"    Random IIA: {random_iia:.3f}  PCA IIA: {pca_iia:.3f}  DAS IIA: {das_iia:.3f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAS for phonetic tasks")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4, 6, 8, 9, 10, 11])
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results_das")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = MODEL_NAMES.get(args.model, args.model)

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

        t0 = time.time()
        results = run_das_for_task(model, task, args, device)
        elapsed = time.time() - t0
        log(f"  Done in {elapsed:.1f}s")

        all_results[task] = results

        task_path = out_dir / f"das_k{args.k}_{task}_{args.model}.json"
        with open(task_path, "w") as f:
            json.dump({task: results}, f, indent=2)
        log(f"  Saved incremental: {task_path}")

    out_path = out_dir / f"das_k{args.k}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {out_path}")

    log("\n=== SUMMARY ===")
    for task, res in all_results.items():
        best_layer = max(res["layers"].items(), key=lambda x: x[1]["das_iia"])
        log(f"  {task}: best DAS IIA = {best_layer[1]['das_iia']:.3f} at L{best_layer[0]}")
