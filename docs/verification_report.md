# Verification report (gate G5)

All 408 re-executed runs (360 of them from the confirmatory set S2) regenerate their stored metrics from their checkpoints within the declared tolerance, 373 of them exactly. An independent REML gives the same variance components for all 6 analyses, permuting model labels removes the model share, the training-split normalisation is a train-only fit, and the test-access log matches the G4 seal. 9 of 9 negative controls were caught by the auditor. The label-shuffle control collapses to chance: test macro-F1 4.9 against 8.7 by chance and 54.5 with the true labels.

Gate file: `results/gates/G5_results_authenticity_rerun6.json` (verdict PASS). Earlier G5 files are kept; `G5_results_authenticity.json` (FAIL) explains the first run.

## What was re-executed

- 408 runs, chosen before measuring: every run of S2 on the datasets with a cache on the cluster (up_fall, wear, uca_ehar: all 4 models x 3 seeds x every split) and a seeded sample of 6 runs on each of the 8 re-executable development datasets.
- Exact equality with the stored macro-F1 and accuracy: 373; within the tolerance: 408; outside: 0. Largest |delta macro-F1|: 0.349 points (float32 inference on a CPU node vs the original GPU).
- Not re-executable here, verified by their stored records only: S2's usc_had (unsorted parser, no matching cache); the six runs without a checkpoint (S4D on ActRecTut x2, Mamba on ActRecTut and UniMiB x4); dsads (below); motion_sense and daphnet (excluded from the analyses).
- Informative, not gating (since rerun 5, the user's decision): the S2 runs of pamap2, uma_fall, w_har, re-executed on caches rebuilt on the PC on 2026-10-15 (the stored runs used cloud-VM caches): 348 runs, 179 exact, 36 outside the gating tolerance; every test size equal: True. Per dataset: pamap2 108 runs (22 exact, 31 outside; dataset mean -0.068 F1 points; largest |delta| 0.67, 12 windows), uma_fall 120 runs (69 exact, 2 outside; dataset mean -0.002 F1 points; largest |delta| 0.46, 3 windows), w_har 120 runs (88 exact, 3 outside; dataset mean -0.002 F1 points; largest |delta| 3.25, 11 windows). The same splits, inputs that differ at float level: the dataset means agree, single runs differ slightly.
- dsads (informative, not gating): statnet seed 1 split 1: 5.3 vs stored 67.0, hystnet_nomem seed 0 split 1: 5.3 vs stored 64.3, lawnet seed 0 split 3: 5.3 vs stored 96.8, hystnet_sel_clock seed 0 split 4: 5.3 vs stored 78.8, recurrencenet seed 0 split 4: 5.2 vs stored 82.3, hystnet_nomem seed 2 split 6: 5.3 vs stored 88.1. On this cluster's dsads cache the checkpoints score at chance, so the cache's windows or channel order differ from the ones the stored runs used. dsads stays in S1 on the strength of its records.

## Checkpoints and predictions (Level 2)

- 408 checkpoints load strictly, match the recorded parameter count, are finite, differ from a fresh initialisation with the run's seed, and carry the run record's fingerprint. Problems: 0.
- Predictions of different seeds differ: 0 identical groups of 237.

## Aggregates (Level 3)

| Analysis | R (statsmodels) | R (numpy REML) | match |
|---|---|---|---|
| H1 | 8.1867 | 8.1922 | True |
| sens1_S2_balanced_accuracy | 4.0889 | 4.0867 | True |
| sens2_S1_balanced_accuracy | 1.8923 | 1.8926 | True |
| sens2_S1_macro_f1 | 3.2550 | 3.2561 | True |
| sens3_S1_without_actrectut | 3.9271 | 3.9270 | True |
| sens4_S3_macro_f1 | 0.2080 | 0.2081 | True |

Figure and table data regenerated in a temporary directory: results/figure_data/fig01_variance_components.json identical, results/figure_data/fig02_contrasts_channels.json identical, results/table_data/tab01_variance_components.csv identical.

## Leakage (Level 3.5)

- Subject-disjoint splits: asserted for all 13 cached datasets when the splits were exported (G2).
- Train-only normalisation: per-channel mean 0 and sd 1 on every re-executed training split; problems: 0.
- Test-access log: 80 entries, none before the seal (2026-10-15T16:49:10); the confirmatory reads are the two pre-registered S2 analyses.
- Label-shuffle control (experiments/20261016_055019_label_shuffle_control): S4D trained on permuted training labels, 6 runs on UP-Fall and UCA-EHAR. Mean test macro-F1 4.91 against a chance level of 8.74 and 54.49 for the true-label runs; validation 6.80 vs chance 8.72 vs 85.90. Collapsed to chance on every run: True.

## Negative controls (Level 5)

| Tampering | Check | Caught |
|---|---|---|
| one prediction value altered | --recompute | True |
| a provenance record deleted | --provenance | True |
| a ledger run directory removed | --ledger | True |
| a stored variance component edited | --seeds | True |
| a checkpoint replaced by a fresh initialisation | --reproduce (re-execution) | True |
| a value in results/table_data hand-edited | Level 3.4 regeneration | True |
| a metric file corrupted, figures regenerated | Level 3.4: figure data must change | True |
| one test window moved into train | split disjointness (Level 3.5.1) | True |
| normalisation fitted on train + test | train-only normalisation (Level 3.5.2) | True |
| model labels permuted within cells (S2) | model share 0.0000 (real 0.101; fit kept: numpy_reml) | True |
| model labels permuted within cells (S1) | model share 0.0000 (real 0.195; fit kept: statsmodels) | True |

Bootstrap refit check (informative): of 50 H1 resamples (seeded sample of 1000), statsmodels reached the maximum of the restricted likelihood in 50; largest shortfall 0.000; largest relative difference in R from the independent REML 0.000 (median 0.0001).

The traceability control (an edited number in a metric file) belongs to gate G6.

## Auditor

```
PASS  --ledger
PASS  --data
PASS  --synthetic
PASS  --recompute
PASS  --reproduce
PASS  --seeds
```

## Why this is a rerun

Rerun 1, approved by the user on 2026-10-16. The first G5 FAILed for four reasons. (1) 7 imported files under results/metrics had no provenance; they are regenerated with provenance by their own scripts (profile.json by the new src/evaluation/profile_models.py). (2) One run differed from its record by exactly 2 windows, the declared limit, and the float32 comparison counted it as above the limit; the window difference is now counted as an integer, as the tolerance states. (3) The recompute negative control changed one wrong prediction into another wrong one in a degenerate file (dsads, constant predictions), so no metric changed and --recompute passed; metric records now carry the prediction file's sha256, --recompute checks it, and the control alters a correct prediction in a non-degenerate file. (4) The provenance control was void because of (1); it now also requires the deleted file to be named. The tolerances are unchanged. Rerun 3 (approved by the user on 2026-10-16): adds the label-shuffle control of Level 3.5.3 (scripts/label_shuffle_control.py), which had not been run; nothing else changes. Rerun 5 (the user's decision of 2026-10-16): rerun 4 FAILed because 36 S2 runs of pamap2, uma_fall and w_har, re-executed on caches rebuilt on the PC (the stored runs used cloud-VM caches), exceeded the 2-window tolerance, while every dataset mean agrees within 0.1 F1 points and every test size is equal. These three datasets are now informative, like dsads; the gating set is that of rerun 3; both results are reported; no tolerance changed. Rerun 4 (approved by the user on 2026-10-16): pamap2, uma_fall and w_har received paper-era caches (built on the PC with scripts/prepare_data.py, split sizes checked, copied to the cluster), so their S2 runs are re-executed too; the set of re-executed cells grows accordingly and nothing else changes. Rerun 2: rerun 1 FAILed only because --recompute required the new hash field also in the 414 metric records of the first re-execution run (20261015_184242_evaluate_G5), written before the field existed, so even the untampered copy failed. Records without the field are now checked by recomputation only, records with it by hash and recomputation. Rerun 6 (approved by the user on 2026-10-16): rerun 5 FAILed on the S2 model-permutation control only. statsmodels' default optimizer reported convergence at a point whose restricted log-likelihood is 13.75 below the optimum (s2_M 16.3 instead of 0), depending on the BLAS thread count (docs/diagnostics/). The control now fits with statsmodels and with the independent REML and keeps the solution with the higher restricted likelihood, both recorded; the threshold is unchanged. A seeded sample of 50 H1 bootstrap refits is checked against the independent REML (informative).
