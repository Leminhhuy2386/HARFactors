# HARFactors: what drives wearable-HAR accuracy (IEEE Sensors Letters)

Under one fixed, person-disjoint protocol (WHAR Arena v1, paper-era code), which factors move macro-F1 on people a
model has never seen: the data, the model, or the evaluation itself? The project follows the agent pipeline in
`agents/` (start with `agents/00.md`). State and open items: `RUN_REPORT.md`. Code layout: `ARCHITECTURE.md`.

## Results imported from Edge_SSM (reproduced here byte for byte; unverified until gate G5)

| Factor | Effect |
|---|---|
| Dataset | 76.6 % of all variance over 2,460 runs (10 models × 10 datasets × subject splits × 3 seeds) |
| **Within a dataset** | **test person 61 %**, training seed 14 %, model × person 11 %, model × dataset 7 %, **model 6 %** |
| Temporal memory vs none | +5.95 [+2.79, +9.10] F1, 10/10 datasets; memory type 1–2 points (Mamba − S4D −2.39, p 0.010) |
| Window laws on the recurrence trunk | +2.41 [+0.35, +4.47] |
| Channel count (30 datasets) | Random Forest margin over the best deep model +7.2 points per sd (p < 0.001) |
| Training subjects (30 datasets) | fewer subjects → the model choice matters more (−1.5 per sd, p 0.005) |
| Difficulty | no dataset property predicts it (R² 0.03); size does not predict accuracy (ρ −0.43, 14 deep models) |
| Evaluation noise | person ~8 points, memory type ~4, seed ~1.7; dropping ActRecTut (2 persons) reorders the top models |

## Setup and commands

```
uv sync                                        # Python 3.12, torch cu126, WHAR paper-era lock
uv run python scripts/run_analyses.py          # results/metrics/* from experiments/ (CPU, seconds)
uv run python scripts/make_figures.py          # figures/*.svg
uv run python -m src.evaluation.factor_analysis   # the main analysis alone
uv run python scripts/train.py --models s4d --datasets utd_mhad --seed 1   # only to add runs
```

Data are shared with the HAR reproduction (`data/raw/README.md`, `data/processed/README.md`). On 2026-10-14, after
the restructure, the analyses reproduced every imported output byte for byte. The one exception,
`results/metrics/arena/compare.*`, was regenerated from more runs than the stored copy: rows were added, none changed.
