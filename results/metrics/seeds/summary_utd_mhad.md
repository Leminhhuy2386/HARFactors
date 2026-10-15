# P4: multi-seed results on utd_mhad

Built by `seeds_summary.py`. Macro-F1 (%), WHAR Arena v1 protocol. A training seed changes only the model initialization and the batch order; the splits, validation windows and normalization are the protocol's (seed 0). Seeds are averaged within each subject split; comparisons are paired over splits.

Splits used: 8; seeds every model has on all of them: [0, 1, 2, 3, 4].

| Model | Params | Mean over splits (seed-averaged) | Std over splits | Mean seed std within a split | Mean per seed | Seed range |
|---|---|---|---|---|---|---|
| quantilenet | 20.8k | **73.99** | 13.3 | 2.87 | 74.3, 73.8, 73.3, 74.2, 74.5 | 73.3–74.5 |
| hystnet_sel_clock | 23.6k | **73.23** | 12.6 | 4.13 | 73.4, 76.5, 72.0, 73.1, 71.2 | 71.2–76.5 |
| s4d | 27.7k | **72.55** | 14.2 | 2.97 | 73.8, 74.0, 72.0, 71.2, 71.7 | 71.2–74.0 |
| mamba | 21.1k | **71.89** | 13.8 | 4.08 | 70.7, 69.6, 73.5, 74.6, 71.0 | 69.6–74.6 |
| hystnet_nomem | 20.0k | **67.12** | 11.2 | 3.50 | 67.6, 68.6, 66.6, 66.2, 66.6 | 66.2–68.6 |

Hardware per seed: seed 0 and 4 on the local GTX 1080 Ti, seeds 1–2 on Kaggle T4, seed 3 on a Colab GPU (see each result's `device`). Non-deterministic GPU kernels make results differ slightly across hardware; every seed ran all models on the same hardware.

## Model − s4d, paired over splits

| Model | Mean Δ | 95 % CI | Wins/losses | Wilcoxon p | Non-inferior (margin 1) |
|---|---|---|---|---|---|
| quantilenet | +1.45 | [-1.48, +4.37] | 4/4 | 0.461 | no |
| hystnet_sel_clock | +0.68 | [-3.26, +4.61] | 6/2 | 0.547 | no |
| mamba | -0.66 | [-3.52, +2.20] | 2/6 | 0.461 | no |
| hystnet_nomem | -5.42 | [-9.28, -1.57] | 1/7 | 0.016 | no |

## Model − mamba, paired over splits

| Model | Mean Δ | 95 % CI | Wins/losses | Wilcoxon p |
|---|---|---|---|---|
| quantilenet | +2.10 | [+0.37, +3.84] | 7/1 | 0.023 |
| hystnet_sel_clock | +1.34 | [-1.62, +4.29] | 5/3 | 0.250 |
| s4d | +0.66 | [-2.20, +3.52] | 6/2 | 0.461 |
| hystnet_nomem | -4.77 | [-7.74, -1.80] | 1/7 | 0.023 |

## Model − hystnet_nomem, paired over splits

| Model | Mean Δ | 95 % CI | Wins/losses | Wilcoxon p |
|---|---|---|---|---|
| quantilenet | +6.87 | [+4.26, +9.48] | 8/0 | 0.008 |
| hystnet_sel_clock | +6.10 | [+3.68, +8.52] | 8/0 | 0.008 |
| s4d | +5.42 | [+1.57, +9.28] | 7/1 | 0.016 |
| mamba | +4.77 | [+1.80, +7.74] | 7/1 | 0.023 |

## Per split (seed-averaged)

| Split | quantilenet | hystnet_sel_clock | s4d | mamba | hystnet_nomem |
|---|---|---|---|---|---|
| split-00 | 88.0 | 84.3 | 92.2 | 88.6 | 80.9 |
| split-01 | 43.9 | 45.3 | 44.2 | 42.6 | 42.3 |
| split-02 | 76.3 | 72.5 | 76.5 | 72.5 | 66.8 |
| split-03 | 68.1 | 68.2 | 63.6 | 63.5 | 64.9 |
| split-04 | 77.9 | 81.1 | 74.4 | 76.7 | 72.1 |
| split-05 | 80.8 | 81.0 | 80.9 | 80.6 | 72.8 |
| split-06 | 80.3 | 80.6 | 79.2 | 75.4 | 70.0 |
| split-07 | 76.6 | 72.8 | 69.4 | 75.3 | 67.1 |
