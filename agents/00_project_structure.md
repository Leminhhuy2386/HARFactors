# 00_project_structure.md — Canonical Layout

This file is the **single source of truth** for where files go.
Where any other prompt appears to suggest a different location, this file wins.

---

## Structure

```text
PROJECT_DIR/  (D:\Projects\HARFactors)
├── agents/                      # these prompts, adapted to HARFactors
├── configs/cloud_jobs/
├── data/raw/README.md, data/processed/README.md, data/synthetic/   # data shared in D:\Projects\HAR
├── src/
│   ├── models/                  # the 10 models of the run population (registry.py); classical/ references
│   ├── training/run_whar.py     # paper-era harness (only needed to add runs)
│   ├── evaluation/              # factor_analysis (main), compare_arena, synthesis, heldout, vs_benchmark, probes
│   ├── visualization/build_factor_figures.py
│   └── utils/paths.py
├── scripts/                     # run_analyses.py, make_figures.py, train.py, prepare_data.py, ledger_import.py, cloud/
├── experiments/                 # LEDGER.md; whar_arena_v1/ (2,460 runs used + others); probes; external_har_paper_era/
├── results/                     # gates/, metrics/<analysis>/, figure_data/, table_data/, predictions/
├── figures/, tables/, paper/, tests/
├── docs/                        # research_plan.md, protocol.md, experiment_record.md, sources/, history/
├── README.md, ARCHITECTURE.md, RUN_REPORT.md, CLAUDE.md, pyproject.toml, uv.lock, requirements.txt, .gitignore
└── third_party/whar-*           # pinned paper-era WHAR code
```

## Project-specific placement (adapted 2026-10-14)

The canonical layout above is followed, with these project-specific decisions:

- **Data are shared, not copied.** `data/raw/` and the paper-era caches live in `D:\Projects\HAR` (see the READMEs
  in `data/`). `data/processed/splits.json` and `manifest.json` *record* the harness's splits; they do not define them.
- **Per-run outputs** stay in the WHAR harness's own format under `experiments/whar_arena_v1/` (one json per model /
  dataset / seed / split) with weights in `experiments/checkpoints/`. These are not one `experiments/<run_id>/`
  directory per run because 4,950 runs were imported. New runs may use the harness format, with a START/END ledger
  line each; a `run_id` folder holds what the harness format cannot (config, provenance, predictions, logs).
- **Aggregates** go in `results/metrics/<analysis>/`, one folder per analysis script in `src/evaluation/`.
- **Imported, not produced here:** `experiments/external_har_paper_era/` (the HAR reproduction's re-runs) and
  `docs/sources/arena_extracted.json` (the paper's Figure 6). Never edit either.
- **Baselines in code:** S4D, Mamba and GRU are in `src/models/baselines.py` (shared skeleton); the classical references (1-NN DTW, segment statistics + trees) in `src/models/classical/`. The paper models of WHAR Arena come from `third_party/whar-models` (and their published numbers from `docs/sources/`).
- **Entry points:** `python scripts/<name>.py`, or `python -m src.<area>.<module>` from the project root.


---

## Rules

**`data/raw/` is read-only.** Never write, move, rename, or delete anything in it.

**`data/synthetic/` is quarantined.** It exists only to smoke-test the pipeline before real data is available. No file under `results/`, `figures/`, `tables/`, or `paper/` may ever be derived from it. Any run using it must set `synthetic: true` in its config, and `06_summary.md` must exclude such runs.

**Never overwrite a previous experiment.** Every training run gets a fresh `run_id`. If a `run_id` directory exists, pick a new one.

**Every result file carries provenance.** Each file under `results/` and under `experiments/*/metrics/` and `*/predictions/` has a sibling `<name>.provenance.json`. A result file without one does not exist, and downstream stages must ignore it. See `01_integrity.md`.

**Aggregation vs. raw.** Per-run outputs live under `experiments/<run_id>/`. Cross-run aggregates (mean over seeds, comparison tables) live under `results/`. Never write an aggregate into a run directory.

---

## Ownership

Each file has exactly one stage responsible for creating it.

| Artifact | Created by |
|---|---|
| `requirements.txt`, `.gitignore`, directory skeleton | 02_plan |
| `docs/research_plan.md`, `docs/research_checklist.md` | 02_plan |
| `results/gates/G1_environment.json` | 02_plan |
| `scripts/prepare_data.py`, `data/processed/`, `manifest.json` | 03_data |
| `docs/baselines_cited.json` | 03_data |
| `experiments/LEDGER.md`, `scripts/audit.py` | 03_data (extended by 04, 05, 06, 07) |
| `results/gates/G2_data.json`, `G3_baseline.json` | 03_data |
| `src/**`, `scripts/train.py`, `scripts/evaluate.py` | 04_research |
| `experiments/<run_id>/**` | 04_research |
| `tests/` | 04_research |
| `results/gates/G4_test_seal.json` | 04_research |
| `scripts/make_figures.py`, `figures/`, `tables/`, `results/figure_data/`, `results/table_data/` | 04_research |
| `docs/experiment_record.md` | 04_research |
| `ARCHITECTURE.md`, `README.md` | 04_research |
| `docs/verification_report.md`, `results/gates/G5_results_authenticity.json` | 05_verify |
| `docs/summary.md`, `docs/artifact_index.md` | 06_summary |
| `results/gates/G6_traceability.json` | 06_summary |
| `paper/manuscript.tex`, `paper/references.bib` | 07_paper |
| `results/gates/G7_compile_citations.json` | 07_paper |
| `docs/review_report_iter<N>.md` | 08_review |
| `RUN_REPORT.md` | whichever stage terminates the run |
