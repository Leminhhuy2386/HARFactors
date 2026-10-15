# Architecture of the code

Layout: `agents/00_project_structure.md`. Locations: `src/utils/paths.py`.

## Data flow

```
experiments/whar_arena_v1/<model>/<dataset>/[seed-s/]split-XX.json      (2,460 runs used: 10 models × 10 datasets × 3 seeds)
docs/sources/arena_extracted.json                                        (WHAR Arena paper, Figure 6: 30 datasets × 17 models)
        │  src/evaluation/compare_arena.py (ours, ours_seed_mean, PAPER, dataset properties)
        ▼
src/evaluation/factor_analysis.py ──► results/metrics/factors/{data.json, summary.md}
        │      variance decomposition (sums of squares over dataset, person = split, seed, model, interactions);
        │      data factors over 30 datasets (Spearman, OLS on standardized properties); paired component contrasts
        ▼
src/visualization/build_factor_figures.py ──► figures/{variance, data_factors, data_corr, components_*, eval_noise,
                                                        factor_map, ranking}.svg
```

## Modules

| Path | Role |
|---|---|
| `src/evaluation/factor_analysis.py` | The main analysis of the letter |
| `src/evaluation/compare_arena.py` | Access to our runs and the paper's table. Every analysis imports it |
| `src/evaluation/synthesis.py`, `seeds_summary.py`, `p5_summary.py`, `vs_benchmark.py`, `heldout_analysis.py` | Per-dataset tables, seed summaries, head-to-head, held-out agreement |
| `src/evaluation/person_probe.py`, `value_probe.py`, `error_analysis.py` | Probes of the person effect and of errors (`error_analysis` reads checkpoints from `experiments/checkpoints/`) |
| `src/models/`, `src/training/run_whar.py`, `scripts/train.py`, `scripts/cloud/` | The run population's models and the paper-era harness, to add runs if needed |
| `src/models/classical/` | Classical references (1-NN DTW, segment statistics + trees) |
| `third_party/whar-*` | Pinned paper-era WHAR code. Never edit |

Run modules from the project root with `uv run python -m src.<area>.<module>`, or through `scripts/`.
