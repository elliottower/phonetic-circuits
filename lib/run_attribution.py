"""Run EAP-IG attribution on phonetic circuit tasks.

Uses the MIB EAP-IG pipeline (reference/mib) with our PhoneticDataset.
Produces per-edge importance scores saved as JSON circuits.

Usage:
    python -m lib.run_attribution --task op4_oronym
    python -m lib.run_attribution --task op4_oronym --model gpt2 --method EAP-IG --ig-steps 5
    python -m lib.run_attribution --task op1_hypocorism op4_oronym --model gpt2
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference" / "mib"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference" / "mib" / "EAP-IG" / "src"))

from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node
from lib.dataset import PhoneticDataset, PHONETIC_TASKS


MODEL_NAMES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
}


def phonetic_logit_diff(
    circuit_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels,
    mean: bool = True,
    loss: bool = False,
    prob: bool = False,
):
    """Logit difference metric for phonetic tasks.

    labels is a list of (target_idx, foil_idx) pairs.
    """
    batch_size = circuit_logits.shape[0]
    diffs = []
    for i in range(batch_size):
        pos = input_length[i] - 1
        target_idx, foil_idx = labels[i]
        if prob:
            probs = torch.softmax(circuit_logits[i, pos], dim=-1)
            diff = probs[target_idx] - probs[foil_idx]
        else:
            diff = circuit_logits[i, pos, target_idx] - circuit_logits[i, pos, foil_idx]
        diffs.append(diff)
    result = torch.stack(diffs)
    if loss:
        result = -result
    if mean:
        return result.mean()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAP-IG attribution for phonetic tasks")
    parser.add_argument("--tasks", type=str, nargs="+", default=list(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--method", type=str, default="EAP-IG", choices=["EAP", "EAP-IG"])
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, default="patching",
                        choices=["patching", "zero", "mean", "mean-positional"])
    parser.add_argument("--level", type=str, default="edge", choices=["edge", "node"])
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="circuits")
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
        print(f"Task: {task}")
        print(f"{'='*60}")

        try:
            dataset = PhoneticDataset(
                task=task,
                tokenizer=model.tokenizer,
                num_examples=args.num_examples,
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  {len(dataset)} examples after token-length filtering")
        dataloader = dataset.to_dataloader(batch_size=args.batch_size)

        node_scores = args.level == "node"
        graph = Graph.from_model(model, neuron_level=False, node_scores=node_scores)

        metric_fn = partial(phonetic_logit_diff, mean=False, loss=False)

        method_map = {"EAP-IG": "EAP-IG-inputs", "EAP": "EAP"}
        eap_method = method_map.get(args.method, args.method)
        ig_steps = args.ig_steps if args.method == "EAP-IG" else None

        if node_scores:
            attribute_node(
                model, graph, dataloader, metric_fn,
                ablation_type=args.ablation,
                clean_metric=metric_fn,
            )
        else:
            attribute(
                model, graph, dataloader, metric_fn,
                method=eap_method,
                intervention=args.ablation,
                ig_steps=ig_steps,
            )

        save_dir = Path(args.output_dir) / f"{args.method}_{args.ablation}_{args.level}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{task}_{args.model}.json"
        graph.to_json(str(save_path))
        print(f"  Saved circuit: {save_path}")
