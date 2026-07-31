# Evaluating LLM-Based and Hierarchy-Trained Embeddings for Pharmaceutical Demand Forecasting

Code artefact for the bachelor thesis of the same name (DHBW Ravensburg, 2026).

The project asks how much of the WHO **ATC** drug hierarchy is recoverable from frozen text
embeddings, how those compare to a **Lorentz** (hyperbolic) model trained directly on the
hierarchy, and whether either improves a demand-forecasting pipeline when appended as a
static product covariate. Nine configurations are evaluated throughout: **8 text**
(2 encoders × 4 input formats) + **1 Lorentz** reference.

| RQ | Goal | Question |
|---|---|---|
| RQ1 | Goal 1 | Does each embedding space carry the structure a hierarchy-aware representation should have? (Part A = text, Part B = Lorentz) |
| RQ2 | Goal 2 | How do the nine configurations compare under rank-based metrics that stay valid across Euclidean and hyperbolic geometry? |
| RQ3 | Goal 3 | Does an embedding block improve a fixed forecasting pipeline against a no-embedding control (ΔMASE)? |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate               # Windows
# source .venv/bin/activate          # macOS/Linux
pip install -r requirements.txt      # or requirements-lock.txt for the exact submitted environment
```

Python 3.14.2, CPU-only `torch`.

**The repository ships with its data.** The ATC ontology
([BioPortal](https://bioportal.bioontology.org/ontologies/ATC), version 2025_02_10), the
prepared node tables and all 24 embedding files (~440 MB) are included, so the intrinsic
evaluation runs out of the box — no OpenAI key and no 8–9 h of Lorentz training needed.
Only if you want to rebuild the embeddings yourself do you need a key in `.env` as
`OPENAI_API_KEY=...`.

Not included: the licensed IQVIA sales export and the Teva mapping files, and everything
derived from them. That is the whole of stage 4.

## Entry points

Four stages, one per thesis chapter. The order is strict — each reads only what an earlier
one wrote — but they can be run on separate days. Logs go to `logs/<stage>/`.

```bash
bash scripts/01_prepare_data.sh       # Ch. 5.1        ATC ontology            ~1 min
bash scripts/02_build_embeddings.sh   # Ch. 5.2 + 5.3  all embeddings          ~24 h
bash scripts/03_run_intrinsic.sh      # Ch. 6          intrinsic evaluation    ~6-8 h
bash scripts/04_run_extrinsic.sh      # Ch. 7          forecasting             ~7-8 h
```

**Stages 1–3 reproduce the entire intrinsic half of the thesis from public data.** Stage 4
is the only one that needs the licensed IQVIA sales export and the Teva mapping files;
these are confidential and not part of this repository. Without them stage 4 reports what
is missing and exits cleanly — nothing else depends on it.

| Switch | Effect |
|---|---|
| `SKIP_DIRECTED=1` (stages 2, 3) | Drops the ~15 h edge-direction ablation |
| `RUN_EXPLORATION=1` (stage 4) | Adds the descriptive dataset tables |
| `SKIP_SMOKE=1` (stage 4) | Drops the ridge-only pipeline check before the slow fits |
| `SKIP_PREP=1` (stage 4) | Series already built — run the forecasting only |

## Structure

```
data/raw/          ATC.csv (shipped); forecasting/ = licensed IQVIA + Teva files (not shipped)
data/prepared_data/ATC tables (shipped); sales chain (stage 4, not shipped)
data/embeddings/   stage 2 output, shipped: the representations under test
results/           stage 3 output, shipped: the intrinsic tables
scripts/           the four stages + two Python helpers
src/
  config.py        single source of truth for paths, seed, Lorentz hyperparameters
  data_prep/       prepare_atc.py (stage 1); sales/ = the sales chain (stage 4)
  embed.py         text-embedding CLI; backends in models/embedders/
  lorentz_training.py
  evaluation/      Goals 1 and 2
  forecasting/     Goal 3
  visuals/         the two Poincaré-disk figures + inspection tools
```

Module, function and result-column names follow the thesis question labels:

| Thesis | Module | Output |
|---|---|---|
| Goal 1, Part A — Q1.A1 isotropy, Q1.A2 neighbourhood precision | `evaluation/goal1a_text_internal.py` | `goal1a_results.csv` |
| Goal 1, Part B — Q1.B1 norm↔rank ρ, Q1.B2 HDC, Q1.B3 neighbourhood precision | `evaluation/goal1b_lorentz_internal.py` | `goal1b_*.csv` |
| Goal 2 — Q2.1 reconstruction, Q2.2 link prediction, Q2.3 probe, Q2.4 tree-distance ρ | `evaluation/goal2_cross_method.py` | `goal2_results.csv` |
| Goal 3 — the forecasting ablation | `forecasting/run.py` | `forecast_*.csv` |

Result columns carry the question label (`q21_map`, `q23_delta_f1`, `q24_stdc_rho`).
Forecasting outputs follow `forecast_<what>[_weighted][_by_unseen][_dim][_lpo].csv`.

Inside `results/`, the flat CSVs are the canonical tables and are identical to
`results/bidirectional/`; `results/directional/` holds the edge-direction ablation, which
writes the same filenames and would otherwise overwrite them.

## Reproducibility

`RANDOM_SEED = 42` drives Lorentz initialisation and negative sampling, the 90 %
transitive-closure split, every PCA fit, the probe's cross-validation, LightGBM, the
leave-product-out draw and the bootstrap. Re-running a stage overwrites the same files.

Everything shipped in `results/` was re-run against the code on 2026-07-31 and reproduces:
the Goal 1 and Goal 2 tables byte-identically, the forecasting per-series records with zero
deviation, and the bootstrap tables to 1e-16.

One exception: **`forecast_config_pair_tests.csv`**. The pair enumeration in
`bootstrap_config_pairs` changed after that file was produced, and because the bootstrap
advances a single RNG through all comparisons in sequence, a changed comparison order
shifts every interval. The ΔMASE point estimates and Holm-adjusted p-values still
reproduce exactly; the percentile CI bounds differ in the third decimal (thesis
`[−0.006, 0.003]` and `[−0.005, 0.002]`, current code `[−0.006, 0.002]` for both). Both
intervals still cross zero, so the conclusion the table supports is unaffected.

Note that the OpenAI embeddings come from a hosted model, and the probe's saga solver is
not run to convergence (±0.004 across runs) — no reported ordering depends on differences
of that size.
