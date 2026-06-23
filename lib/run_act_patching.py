"""Activation patching for phonetic circuit tasks.

Node patching: mean-ablate each attention head's output (hook_result),
measure logit-diff drop. One forward pass per head.

Edge patching: for each (writer, reader) pair, patch only the writer's
contribution to the reader's input. One forward pass per edge.

Pure Python, no Modal dependency.

Usage:
    python -m lib.run_act_patching --tasks op4_oronym
    python -m lib.run_act_patching --tasks op4_oronym --level edge
    python -m lib.run_act_patching --level node --all
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


def compute_logit_diff(logits, labels):
    diffs = []
    for i, (tid, fid) in enumerate(labels):
        diff = logits[i, -1, tid] - logits[i, -1, fid]
        diffs.append(diff.item())
    return diffs


def get_clean_corrupted_cache(model, clean_prompts, corrupted_prompts):
    clean_tokens = model.to_tokens(clean_prompts)
    corrupted_tokens = model.to_tokens(corrupted_prompts)

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens)
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    return clean_tokens, corrupted_tokens, clean_cache, corrupted_cache


def run_node_patching(model, task, args):
    log(f"Node patching: {task}")
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=args.num_examples)
    log(f"  {len(dataset)} examples")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_clean_prompts = []
    all_corrupted_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        clean, corrupted, label = dataset[i]
        all_clean_prompts.append(clean)
        all_corrupted_prompts.append(corrupted)
        all_labels.append(label)

    clean_tokens = model.to_tokens(all_clean_prompts)
    corrupted_tokens = model.to_tokens(all_corrupted_prompts)

    with torch.no_grad():
        clean_logits = model(clean_tokens)
    baseline_diffs = compute_logit_diff(clean_logits, all_labels)
    baseline_mean = sum(baseline_diffs) / len(baseline_diffs)
    log(f"  Baseline mean logit diff: {baseline_mean:.4f}")

    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    results = {}
    pbar = tqdm(total=n_layers * n_heads + n_layers, desc="Node patching")

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_result"
            corrupted_act = corrupted_cache[hook_name].clone()

            def patch_hook(value, hook, layer=layer, head=head, corrupted_act=corrupted_act):
                value[:, :, head, :] = corrupted_act[:, :, head, :]
                return value

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, patch_hook)],
                )
            patched_diffs = compute_logit_diff(patched_logits, all_labels)
            patched_mean = sum(patched_diffs) / len(patched_diffs)
            effect = baseline_mean - patched_mean

            results[f"a{layer}.h{head}"] = {
                "patched_logit_diff": patched_mean,
                "effect": effect,
                "effect_normalized": effect / baseline_mean if baseline_mean != 0 else 0,
            }
            pbar.update(1)

        hook_name = f"blocks.{layer}.hook_mlp_out"
        corrupted_mlp = corrupted_cache[hook_name].clone()

        def patch_mlp_hook(value, hook, corrupted_mlp=corrupted_mlp):
            return corrupted_mlp

        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_name, patch_mlp_hook)],
            )
        patched_diffs = compute_logit_diff(patched_logits, all_labels)
        patched_mean = sum(patched_diffs) / len(patched_diffs)
        effect = baseline_mean - patched_mean

        results[f"m{layer}"] = {
            "patched_logit_diff": patched_mean,
            "effect": effect,
            "effect_normalized": effect / baseline_mean if baseline_mean != 0 else 0,
        }
        pbar.update(1)

    pbar.close()

    results["_meta"] = {
        "baseline_mean_logit_diff": baseline_mean,
        "n_examples": len(dataset),
        "task": task,
        "model": args.model,
        "level": "node",
    }

    return results


def run_edge_patching(model, task, args):
    log(f"Edge patching: {task}")
    dataset = PhoneticDataset(task=task, tokenizer=model.tokenizer, num_examples=args.num_examples)
    log(f"  {len(dataset)} examples")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_clean_prompts = []
    all_corrupted_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        clean, corrupted, label = dataset[i]
        all_clean_prompts.append(clean)
        all_corrupted_prompts.append(corrupted)
        all_labels.append(label)

    clean_tokens = model.to_tokens(all_clean_prompts)
    corrupted_tokens = model.to_tokens(all_corrupted_prompts)

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    baseline_diffs = compute_logit_diff(clean_logits, all_labels)
    baseline_mean = sum(baseline_diffs) / len(baseline_diffs)
    log(f"  Baseline mean logit diff: {baseline_mean:.4f}")

    writers = []
    for layer in range(n_layers):
        for head in range(n_heads):
            writers.append((f"a{layer}.h{head}", f"blocks.{layer}.attn.hook_result", head))
        writers.append((f"m{layer}", f"blocks.{layer}.hook_mlp_out", None))

    readers = []
    for layer in range(n_layers):
        for head in range(n_heads):
            readers.append((f"a{layer}.h{head}", layer, head))
        readers.append((f"m{layer}", layer, None))

    results = {}
    total_edges = 0
    for w_name, w_hook, w_head in writers:
        w_layer = int(w_name[1:].split(".")[0].replace("a", "").replace("m", ""))
        for r_name, r_layer, r_head in readers:
            if r_layer > w_layer:
                total_edges += 1

    log(f"  {total_edges} edges to test")
    pbar = tqdm(total=total_edges, desc="Edge patching")

    for w_name, w_hook, w_head in writers:
        w_layer_str = w_name[1:].split("h")[0].rstrip(".")
        w_layer = int(w_layer_str) if w_layer_str.isdigit() else int(w_name[1:])

        if w_head is not None:
            clean_writer_out = clean_cache[w_hook][:, :, w_head, :].clone()
            corrupted_writer_out = corrupted_cache[w_hook][:, :, w_head, :].clone()
        else:
            clean_writer_out = clean_cache[w_hook].clone()
            corrupted_writer_out = corrupted_cache[w_hook].clone()

        writer_diff = corrupted_writer_out - clean_writer_out

        for r_name, r_layer, r_head in readers:
            if r_layer <= w_layer:
                continue

            if r_head is not None:
                resid_hook = f"blocks.{r_layer}.hook_resid_pre"
            else:
                resid_hook = f"blocks.{r_layer}.hook_resid_mid"

            def patch_edge_hook(value, hook, writer_diff=writer_diff):
                return value + writer_diff

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(resid_hook, patch_edge_hook)],
                )
            patched_diffs = compute_logit_diff(patched_logits, all_labels)
            patched_mean = sum(patched_diffs) / len(patched_diffs)
            effect = baseline_mean - patched_mean

            edge_key = f"{w_name}->{r_name}"
            results[edge_key] = {
                "patched_logit_diff": patched_mean,
                "effect": effect,
                "effect_normalized": effect / baseline_mean if baseline_mean != 0 else 0,
            }
            pbar.update(1)

    pbar.close()

    results["_meta"] = {
        "baseline_mean_logit_diff": baseline_mean,
        "n_examples": len(dataset),
        "task": task,
        "model": args.model,
        "level": "edge",
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation patching for phonetic tasks")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--level", type=str, default="node", choices=["node", "edge"])
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results_act_patching")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = MODEL_NAMES.get(args.model, args.model)

    log(f"Loading model: {model_fullname} on {device}")
    model = HookedTransformer.from_pretrained(model_fullname, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    log(f"Model loaded: {model.cfg.n_layers}L {model.cfg.n_heads}H d={model.cfg.d_model}")

    out_dir = Path(args.output_dir) / args.level
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task} ({args.level} patching)")
        log(f"{'='*60}")

        t0 = time.time()
        if args.level == "node":
            results = run_node_patching(model, task, args)
        else:
            results = run_edge_patching(model, task, args)
        elapsed = time.time() - t0

        out_path = out_dir / f"{task}_{args.model}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  Saved: {out_path} ({elapsed:.1f}s)")

        top = sorted(
            [(k, v) for k, v in results.items() if k != "_meta"],
            key=lambda x: abs(x[1]["effect"]),
            reverse=True,
        )[:10]
        log("  Top 10 by effect:")
        for name, info in top:
            log(f"    {name}: effect={info['effect']:.4f} (normalized={info['effect_normalized']:.2%})")
