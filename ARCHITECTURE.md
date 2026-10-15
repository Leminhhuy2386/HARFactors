# Architecture of the code

Layout: `agents/00_project_structure.md`. Locations: `src/utils/paths.py`.

## Data flow

```
experiments/whar_arena_v1/<model>/<dataset>/[seed-s/]split-XX.json   stored runs (S1, S2 of docs/preregistration.md)
experiments/external_har_paper_era/whar_arena_v1/...                 the 17 WHAR Arena models re-run (S3)
experiments/checkpoints/<model>/<dataset>/seed-<s>/split-XX.pt       weights of the 10 population models
docs/sources/arena_extracted.json                                    WHAR Arena paper, Figure 6 (quoted; docs/baselines_cited.json)
        │
        ├─ scripts/export_splits.py, build_data_records.py ─► data/processed/{harness_splits,splits,manifest}.json (G2)
        │
        ├─ scripts/run_confirmatory.py <analysis>   (src/evaluation/variance_components.py: REML + cluster bootstrap)
        │        ─► results/metrics/confirmatory/<analysis>.{json,md}   H1 (S2) and the 4 pre-registered sensitivities
        │
        ├─ scripts/run_analyses.py   (exploratory, imported from Edge_SSM)
        │        ─► results/metrics/{factors, contrasts_robustness, arena, synthesis, heldout, ...}
        │
        ├─ scripts/make_figures.py   (src/visualization/letter_figures.py, build_factor_figures.py)
        │        ─► figures/fig0x_*.pdf, tables/tab01_*.{csv,tex}, results/{figure,table}_data/
        │
        └─ scripts/verify_results.py (scripts/evaluate.py: predictions from checkpoints; src/evaluation/reml_numpy.py)
                 ─► experiments/<run_id>_verify_G5/, docs/verification_report.md, results/gates/G5_*.json
```

Every computed result has a sibling `<name>.provenance.json` (`src/utils/provenance.py`). Every analysis run has
`experiments/<run_id>/` and START/END lines in `experiments/LEDGER.md`. `scripts/audit.py` checks all of it.

## Modules

| Path | Role |
|---|---|
| `src/data/sets.py` | The three run sets of the pre-registration (paths only) |
| `src/evaluation/variance_components.py` | The pre-registered analysis: inclusion rule, REML fit, cluster bootstrap, decision (sealed at G4) |
| `src/evaluation/reml_numpy.py` | An independent REML implementation and the model-label permutation control (verification) |
| `src/evaluation/factor_analysis.py`, `contrasts_robustness.py` | Exploratory: sums-of-squares decomposition, data factors over 30 datasets, component contrasts (with and without daphnet) |
| `src/evaluation/compare_arena.py` | Access to our runs and the paper's table; other analyses import it |
| `src/evaluation/synthesis.py`, `seeds_summary.py`, `p5_summary.py`, `vs_benchmark.py`, `heldout_analysis.py` | Imported exploratory tables |
| `src/visualization/letter_figures.py` | The manuscript's figures and table |
| `src/utils/provenance.py` | Provenance records and ledger lines |
| `src/models/`, `src/training/run_whar.py` | The population's models and the paper-era harness (inference for verification; no training in this project) |
| `third_party/whar-*` | Pinned paper-era WHAR code (submodules at 144e006 / 9b65004 / 19b3019). Never edit |

Run modules from the project root with `uv run python -m src.<area>.<module>`, or through `scripts/`. On the cluster:
`source scripts/hpc/env.sh` and `scripts/hpc/submit.sh` (docs/hpc.md).
