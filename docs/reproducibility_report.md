# Reproducibility of the stored runs

Of 3408 stored runs of S1 and S2, 3402 have a checkpoint, 3402 of those carry the registry's config and the record's config and fingerprint, 3402 were re-executed from their checkpoint, 2363 reproduce their record exactly and 3048 within the G5 tolerance (equal test size, at most 2 test windows and 2 macro-F1 points from the record). S3: 510 run records, no checkpoints.

| Set/dataset | runs | checkpoint | config ok | re-executed | exact | within tol. | max abs dF1 | max d windows |
|---|---|---|---|---|---|---|---|---|
| S1/actrectut_gestures | 60 | 56 | 56 | 56 | 56 | 56 | 0.00 | 0 |
| S1/daphnet | 300 | 300 | 300 | 300 | 121 | 264 | 1.74 | 5 |
| S1/dsads | 240 | 240 | 240 | 240 | 0 | 28 | 2.15 | 70 |
| S1/hapt | 300 | 300 | 300 | 300 | 290 | 300 | 0.07 | 1 |
| S1/opportunity | 120 | 120 | 120 | 120 | 109 | 120 | 0.05 | 2 |
| S1/sad | 300 | 300 | 300 | 300 | 294 | 300 | 0.20 | 1 |
| S1/uci_har | 300 | 300 | 300 | 300 | 296 | 300 | 0.16 | 1 |
| S1/unimib_shar | 300 | 298 | 298 | 298 | 263 | 298 | 0.35 | 1 |
| S1/utd_mhad | 240 | 240 | 240 | 240 | 240 | 240 | 0.00 | 0 |
| S1/wisdm | 300 | 300 | 300 | 300 | 126 | 291 | 0.78 | 5 |
| S2/motion_sense | 120 | 120 | 120 | 120 | 57 | 116 | 0.36 | 7 |
| S2/pamap2 | 108 | 108 | 108 | 108 | 22 | 77 | 0.67 | 12 |
| S2/uca_ehar | 120 | 120 | 120 | 120 | 120 | 120 | 0.00 | 0 |
| S2/uma_fall | 120 | 120 | 120 | 120 | 69 | 118 | 0.46 | 3 |
| S2/up_fall | 120 | 120 | 120 | 120 | 120 | 120 | 0.00 | 0 |
| S2/usc_had | 120 | 120 | 120 | 120 | 2 | 63 | 0.87 | 20 |
| S2/w_har | 120 | 120 | 120 | 120 | 88 | 117 | 3.25 | 11 |
| S2/wear | 120 | 120 | 120 | 120 | 90 | 120 | 0.06 | 1 |

The 6 runs without a checkpoint were retrained for one (run 20261016_230928_retrain_missing_checkpoints, results/metrics/reproducibility/retrained.json; outputs under that run's directory, the stored records untouched): 5 reproduce their record's macro-F1 exactly, the largest difference is 0.005 points, and 6 carry the record's config and fingerprint.

Excluded from the analyses by the pre-registered inclusion rule: S1/daphnet, S2/motion_sense. Methods: 'cache' = scripts/evaluate.py on a cache in the order the run was built with (inferred from the checkpoints where the parser lists files unsorted); 'per-checkpoint' = scripts/reconstruct_order.py reexecute (test subjects inferred per checkpoint, z-score on all non-test windows).

Re-execution runs: 20261016_064841_evaluate_G5, 20261016_145841_evaluate_full_S1, 20261016_153244_evaluate_usc_had, 20261016_154730_evaluate_dsads_inferred_order, reexecute_daphnet.json, reexecute_dsads.json, reexecute_motion_sense.json.
