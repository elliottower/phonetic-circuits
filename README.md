# Phonological Assembly Circuits in GPT-2

Discovering and validating the circuits GPT-2 uses for phonological computation — how a language model trained on text computes sound structure from orthography.

We identify circuits for six phonological operations (hypocorism, clipping, initialism, oronym resolution, homophone detection, folk etymology), show they share a common backbone with operation-specific specialist heads, and validate them with causal discovery and distributed alignment search (DAS).

## Slides

**[Slides](slides/phonetic_circuits_slides.pdf)** ([LaTeX source](slides/phonetic_circuits_slides.tex)) — 45 slides covering circuit localization, causal variable localization, and validation.

To compile:
```bash
cd slides
pdflatex phonetic_circuits_slides.tex
```

## Datasets

446 minimal pairs across 6 phonological operations, in BLiMP-style format:

| File | Operation | Examples | Description |
|------|-----------|----------|-------------|
| `datasets/op1_hypocorism.csv` | Rhyming hypocorism | 132 | Richard → Dick (irregular stored lookup) |
| `datasets/op2_clipping.csv` | Clipping | 62 | Timothy → Tim (first-syllable truncation) |
| `datasets/op3_initialism.csv` | Initialism | 68 | J → Jay (grapheme → phoneme name) |
| `datasets/op4_oronym.csv` | Oronym | 104 | Bill Ding → building (cross-boundary fusion) |
| `datasets/op5_homophone.csv` | Homophone | 52 | Neil → kneel (phoneme-identity detection) |
| `datasets/op7_folk_etym.csv` | Folk etymology | 64 | microwave → Michael Wave (reverse decomposition) |

All datasets are manually curated. See `datasets/MANUALLY_CURATED_EXAMPLES.md` for construction notes.

## Pipeline

The analysis pipeline runs on [Modal](https://modal.com/) (GPU) with pure-Python core logic:

| Script | What it does |
|--------|-------------|
| `lib/run_attribution.py` | EAP-IG circuit discovery per task |
| `lib/run_act_patching.py` | Node- and edge-level activation patching |
| `lib/run_das.py` | Distributed Alignment Search (rank-1 probes per layer) |
| `lib/run_das_composition.py` | DAS direction composition (AND/OR/NOT/TRANSFER across tasks) |
| `lib/run_causal_discovery.py` | PC and CD-NOD causal discovery on head activations |
| `lib/run_evaluation.py` | Faithfulness evaluation (CMD, CPR) |
| `lib/run_pipeline.py` | End-to-end pipeline (attribution → evaluation → plots) |
| `lib/behavioral_validation.py` | Behavioral validation (logit-diff, accuracy) |
| `lib/dataset.py` | Dataset loading and tokenization |
| `lib/plot.py` | Result visualization |

Each `*_modal.py` file is a thin Modal wrapper around the corresponding pure-Python script.

## Running

```bash
# Install
uv sync

# Full pipeline (requires GPU via Modal)
modal run --detach lib/run_pipeline_modal.py

# Individual experiments
modal run --detach lib/run_das_modal.py
modal run --detach lib/run_causal_discovery_modal.py
modal run --detach lib/run_act_patching_modal.py --level edge

# Local (CPU, for testing)
uv run python -m lib.behavioral_validation --device cpu
```

## Key findings

- **Shared backbone**: layers 8–11 attention heads appear across all six tasks (96–100% agreement between PC and CD-NOD causal discovery)
- **Specialist heads**: each operation recruits additional task-specific heads in layers 9–10
- **Faithful circuits**: all circuits recover full-model performance (CMD < 0.08, CPR ≈ 1.0)
- **Universal hub heads**: a11.h0 and a11.h1 appear in all 6 task circuits

## References

- Syed et al. (2023). [Attribution Patching](https://arxiv.org/abs/2310.10348) — EAP-IG method
- Geiger et al. (2024). [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://arxiv.org/abs/2303.02536) — DAS
- Spirtes et al. (2000). *Causation, Prediction, and Search* — PC algorithm
- Warstadt et al. (2020). [BLiMP](https://arxiv.org/abs/1912.00582) — minimal-pair evaluation format

## License

MIT
