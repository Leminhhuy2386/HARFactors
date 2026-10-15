# HARFactors: what drives wearable-HAR accuracy under person-disjoint evaluation (IEEE Sensors Letters)

Under one fixed, person-disjoint protocol (WHAR Arena v1, paper-era code), how much of the variance of macro-F1 on
unseen people comes from who is tested, from the model, and from the training seed? The project follows the agent
pipeline in `agents/` (start with `agents/00.md`). State: `RUN_REPORT.md`. Code: `ARCHITECTURE.md`. The
confirmatory analysis is pre-registered in `docs/preregistration.md` (frozen 2026-10-15, sealed at gate G4).

No model is trained in this project: every result comes from stored runs and their checkpoints (decision of
2026-10-15).

## Reproduce, command by command

On the cluster, first `source scripts/hpc/env.sh` and wrap each command in
`scripts/hpc/submit.sh <name> normal <walltime> -- ...` (docs/hpc.md). On the PC, `uv sync` once.

```
uv run --no-sync python scripts/check_environment.py          # gate G1
uv run --no-sync python scripts/export_splits.py              # harness splits of the cached datasets
uv run --no-sync python scripts/build_data_records.py         # data/processed/{splits,manifest}.json
uv run --no-sync python scripts/build_baselines_cited.py      # docs/baselines_cited.json (quoted Figure 6 cells)
uv run --no-sync python scripts/validate_data.py              # gate G2
uv run --no-sync python scripts/check_baseline_anchor.py      # gate G3
uv run --no-sync python scripts/run_analyses.py               # exploratory aggregates, with provenance
uv run --no-sync python scripts/seal_g4.py                    # gate G4 (once, before S2 is read)
uv run --no-sync python scripts/run_confirmatory.py H1 --workers 16          # the confirmatory test
uv run --no-sync python scripts/run_confirmatory.py sens1_S2_balanced_accuracy --workers 16   # and the sensitivities:
#   sens2_S1_macro_f1, sens2_S1_balanced_accuracy, sens3_S1_without_actrectut, sens4_S3_macro_f1
uv run --no-sync python scripts/make_figures.py               # figures/fig0x_*.pdf, tables/tab01_*
WHAR_DATASETS_DIR=<copy of the caches> uv run --no-sync python scripts/verify_results.py --workers 16   # gate G5
uv run --no-sync python scripts/audit.py --all                # the user's own check of everything above
uv run --no-sync python tests/test_variance_components.py     # tests of the analysis on synthetic data
```

S2 analyses refuse to run without the G4 seal, refuse to run twice, and log every read to `results/test_access.log`.
Re-running an S2 analysis therefore needs a declared deviation (docs/preregistration.md, § Deviations).

## Data

Shared with the HAR reproduction, read-only (`data/raw/README.md`, `data/processed/README.md`). On the cluster:
`/work/huy.leminh/code/Dr.Huy/HAR/datasets_paper_era`. Verification re-executes on a private copy of the caches,
because the harness may write normalised samples into the cache it reads.
