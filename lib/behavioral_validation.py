"""Behavioral validation: check GPT-2 accuracy on each phonetic task.

Gate: >70% accuracy before running circuit discovery.
Reports logit_diff, probability of target vs foil, and accuracy per task.

Usage:
    python -m lib.behavioral_validation
    python -m lib.behavioral_validation --tasks op4_oronym --model gpt2-medium
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from lib.dataset import PhoneticDataset, PHONETIC_TASKS


MODEL_NAMES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
}


def validate_task(
    model,
    task: str,
    num_examples: Optional[int] = None,
    device: str = "cpu",
):
    from transformer_lens import HookedTransformer

    try:
        dataset = PhoneticDataset(
            task=task,
            tokenizer=model.tokenizer,
            num_examples=num_examples,
        )
    except FileNotFoundError as e:
        return {"task": task, "error": str(e)}

    n_correct = 0
    total_logit_diff = 0.0
    total_prob_diff = 0.0
    n_total = len(dataset)

    for i in tqdm(range(n_total), desc=task, leave=False):
        clean, corrupted, (target_idx, foil_idx) = dataset[i]

        tokens = model.to_tokens(clean)
        with torch.no_grad():
            logits = model(tokens)

        last_logits = logits[0, -1]
        target_logit = last_logits[target_idx].item()
        foil_logit = last_logits[foil_idx].item()

        logit_diff = target_logit - foil_logit
        total_logit_diff += logit_diff

        probs = torch.softmax(last_logits, dim=-1)
        prob_diff = probs[target_idx].item() - probs[foil_idx].item()
        total_prob_diff += prob_diff

        if target_logit > foil_logit:
            n_correct += 1

    acc = n_correct / n_total if n_total > 0 else 0.0
    return {
        "task": task,
        "n_examples": n_total,
        "accuracy": acc,
        "mean_logit_diff": total_logit_diff / max(n_total, 1),
        "mean_prob_diff": total_prob_diff / max(n_total, 1),
        "gate_passed": acc > 0.70,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral validation for phonetic tasks")
    parser.add_argument("--tasks", type=str, nargs="+", default=sorted(PHONETIC_TASKS))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from transformer_lens import HookedTransformer
    model_name = MODEL_NAMES.get(args.model, args.model)
    print(f"Loading {model_name} on {device}")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    print(f"\n{'Task':<20} {'N':>5} {'Acc':>7} {'LogitDiff':>11} {'ProbDiff':>10} {'Gate':>6}")
    print("-" * 65)

    for task in args.tasks:
        result = validate_task(model, task, num_examples=args.num_examples, device=device)
        if "error" in result:
            print(f"{task:<20} {'--':>5} {'--':>7} {'--':>11} {'--':>10} {'SKIP':>6}")
            print(f"  {result['error']}")
        else:
            gate = "PASS" if result["gate_passed"] else "FAIL"
            print(
                f"{result['task']:<20} "
                f"{result['n_examples']:>5} "
                f"{result['accuracy']:>6.1%} "
                f"{result['mean_logit_diff']:>11.3f} "
                f"{result['mean_prob_diff']:>10.4f} "
                f"{gate:>6}"
            )
