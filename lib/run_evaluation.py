"""Run faithfulness evaluation on phonetic circuit tasks.

Loads an EAP-IG circuit (from run_attribution.py), sweeps circuit fractions,
and produces CMD/CPR metrics. Outputs a pickle compatible with plot.py.

Usage:
    python -m lib.run_evaluation \
        --task op4_oronym --model gpt2 \
        --circuit circuits/EAP-IG_patching_edge/op4_oronym_gpt2.json

    python -m lib.run_evaluation \
        --task op4_oronym op1_hypocorism --model gpt2 \
        --method EAP-IG --ablation patching --level edge
"""
import argparse
import os
import pickle
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference" / "mib"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference" / "mib" / "EAP-IG" / "src"))

from transformer_lens import HookedTransformer

from eap.graph import Graph
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from lib.dataset import PhoneticDataset, PHONETIC_TASKS
from lib.run_attribution import phonetic_logit_diff, MODEL_NAMES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faithfulness evaluation for phonetic circuits")
    parser.add_argument("--tasks", type=str, nargs="+", default=list(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--method", type=str, default="EAP-IG")
    parser.add_argument("--ablation", type=str, default="patching")
    parser.add_argument("--level", type=str, default="edge")
    parser.add_argument("--circuit", type=str, default=None,
                        help="Explicit circuit file (overrides auto-discovery)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_fullname = MODEL_NAMES.get(args.model, args.model)

    print(f"Loading model: {model_fullname}")
    model = HookedTransformer.from_pretrained(model_fullname, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating: {task}")
        print(f"{'='*60}")

        if args.circuit:
            circuit_path = args.circuit
        else:
            circuit_path = (
                f"circuits/{args.method}_{args.ablation}_{args.level}/"
                f"{task}_{args.model}.json"
            )

        if not Path(circuit_path).exists():
            print(f"  SKIP: circuit not found at {circuit_path}")
            continue

        print(f"  Loading circuit: {circuit_path}")
        if circuit_path.endswith(".json"):
            graph = Graph.from_json(circuit_path)
        elif circuit_path.endswith(".pt"):
            graph = Graph.from_pt(circuit_path)
        else:
            raise ValueError(f"Unknown circuit format: {circuit_path}")

        try:
            dataset = PhoneticDataset(
                task=task,
                tokenizer=model.tokenizer,
                num_examples=args.num_examples,
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  {len(dataset)} examples")
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

        out_dir = Path(args.output_dir) / f"{args.method}_{args.ablation}_{args.level}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{task}_{args.model}.pkl"

        with open(out_path, "wb") as f:
            pickle.dump(result, f)

        print(f"  CPR (area under): {area_under:.4f}")
        print(f"  CMD (area from 1): {area_from_1:.4f}")
        print(f"  Saved: {out_path}")
