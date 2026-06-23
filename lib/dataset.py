"""Phonetic circuits minimal-pair dataset loader.

Follows the same interface as MIB's HannaEAPDataset: each row is a
(clean, corrupted, label) triple stored in a CSV. This lets us plug
directly into the MIB EAP-IG attribution and faithfulness evaluation
pipeline without modification.

CSV format (one file per operation):
    clean,corrupted,target_idx,foil_idx
    "Bill Ding sounds like","Phil Ding sounds like",<token_id>,<token_id>

The six phonological operations (see slides/paper for definitions):
    op1_hypocorism   — Richard → Dick  (stored lookup)
    op2_clipping     — Timothy → Tim   (first-syllable truncation)
    op3_initialism   — J → Jay         (grapheme-to-phoneme name)
    op4_oronym       — Bill Ding → building  (cross-boundary fusion)
    op5_homophone    — Neil → kneel    (phoneme-identity detection)
    op6_folk_etym    — Mike Rowave → microwave  (reverse decomposition)
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader, Dataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

PHONETIC_TASKS = {
    "op1_hypocorism",
    "op2_clipping",
    "op3_initialism",
    "op4_oronym",
    "op5_homophone",
    "op6_folk_etym",
}


def collate_phonetic(xs):
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), labels


class PhoneticDataset(Dataset):
    """Minimal-pair counterfactual dataset for phonetic circuit tasks.

    Each task has a CSV in datasets/{task}.csv with columns:
        clean       — the prompt where the model should predict the target
        corrupted   — counterfactual prompt (one causal variable changed)
        target_idx  — token ID of the correct completion
        foil_idx    — token ID of the incorrect completion

    Evaluation metric: logit_diff(target) - logit_diff(foil) at final position.
    """

    def __init__(
        self,
        task: str,
        tokenizer=None,
        num_examples: Optional[int] = None,
        seed: int = 42,
        data_dir: Optional[Path] = None,
    ):
        if task not in PHONETIC_TASKS:
            raise ValueError(
                f"Unknown task {task!r}. Available: {sorted(PHONETIC_TASKS)}"
            )
        self.task = task
        self.tokenizer = tokenizer

        csv_path = (data_dir or DATASETS_DIR) / f"{task}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                f"Create the CSV with columns: clean,corrupted,target_idx,foil_idx"
            )

        self.df = pd.read_csv(csv_path)

        if tokenizer is not None:
            self.df = self.df[
                self.df.apply(
                    lambda r: len(tokenizer(r["clean"], add_special_tokens=False).input_ids)
                    == len(tokenizer(r["corrupted"], add_special_tokens=False).input_ids),
                    axis=1,
                )
            ]

        self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if num_examples and num_examples < len(self.df):
            self.df = self.df.head(num_examples)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (
            row["clean"],
            row["corrupted"],
            [int(row["target_idx"]), int(row["foil_idx"])],
        )

    def head(self, n: int):
        if n <= len(self.df):
            self.df = self.df.head(n)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_phonetic)
