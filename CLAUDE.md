# CLAUDE.md

## Project overview

Circuit discovery for phonological operations in GPT-2. Six tasks test
distinct phonological computations from orthographic input:

| Op | Name | Example | Computation |
|----|------|---------|-------------|
| 1 | Rhyming hypocorism | Richard → Dick | Stored lookup |
| 2 | Clipping | Timothy → Tim | First-syllable truncation |
| 3 | Initialism | J → Jay | Grapheme-to-phoneme |
| 4 | Oronym | Bill Ding → building | Cross-boundary fusion |
| 5 | Homophone | Neil → kneel | Phoneme-identity |
| 7 | Folk etymology | Mike Rowave → microwave | Reverse decomposition |

Op 4 (oronym) is the central novel finding — cross-boundary phoneme fusion.

## Commands

```bash
# Install
uv pip install -e ".[dev]"

# Behavioral validation (gate: >70% accuracy before circuit discovery)
uv run python -m lib.behavioral_validation --tasks op4_oronym --model gpt2

# EAP-IG attribution
uv run python -m lib.run_attribution --tasks op4_oronym --model gpt2

# Faithfulness evaluation (CMD/CPR curves)
uv run python -m lib.run_evaluation --tasks op4_oronym --model gpt2

# Plot CMD/CPR
uv run python -m lib.plot results/EAP-IG_patching_edge/op4_oronym_gpt2.pkl
```

## Structure

```
datasets/           CSV files per task (clean, corrupted, target_idx, foil_idx)
lib/
  dataset.py        PhoneticDataset loader (Hanna CSV format for MIB compatibility)
  behavioral_validation.py   Pre-circuit accuracy gate
  run_attribution.py         EAP-IG attribution via MIB pipeline
  run_evaluation.py          Faithfulness sweeps (CMD/CPR)
  plot.py                    CMD/CPR area charts
reference/
  mib/              MIB-circuit-track (submodule) — EAP-IG pipeline
  phonologybench/   PhonologyBench (submodule) — benchmark reference
  eap-ig-faithfulness/  Hanna et al. data + reference implementation
  papers/           Downloaded PDFs
paper/              LaTeX paper
slides/             Beamer slides
circuits/           EAP-IG output (gitignored)
results/            Evaluation pickles (gitignored)
```

## Dataset format

Each task is a CSV in `datasets/{task}.csv`:

```csv
clean,corrupted,target_idx,foil_idx
"Bill Ding sounds like","Phil Ding sounds like",2615,2147
```

- `clean`: prompt where model should predict target
- `corrupted`: counterfactual (one causal variable changed)
- `target_idx`: GPT-2 token ID of correct completion
- `foil_idx`: GPT-2 token ID of incorrect completion
- Metric: logit_diff at final position

## Guidelines

- Use `uv run` for all Python execution
- GPU experiments go on Modal or RunPod, not local (Intel Mac, no GPU)
- Follow MIB conventions for circuit format (JSON graphs)
- Cite BLiMP (Warstadt et al. 2020) for minimal-pair format
