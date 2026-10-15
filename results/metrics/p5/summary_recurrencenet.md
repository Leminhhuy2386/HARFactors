# P5: the 5 models across WHAR datasets (paper-era code, WHAR Arena v1)

Built by `p5_summary.py`. Macro-F1 (%): per split the mean over training seeds 0–2, then the mean over splits. Pre-registered design in `RESEARCH_PLAN.md` §5b (P5).

| Dataset | Status | QuantileNet | HysteresisNet sel+clock | S4D | Mamba | No memory | RecurrenceNet |
|---|---|---|---|---|---|---|---|
| utd_mhad | 8 splits | 73.78 | **73.94** | 73.27 | 71.28 | 67.60 | 73.32 |
| actrectut_gestures | 2 splits | 36.07 | **43.57** | 43.43 | 36.70 | 36.46 | 40.37 |
| opportunity | 4 splits | 63.40 | **63.81** | 61.18 | 62.35 | 59.62 | 61.59 |
| unimib_shar | 10 splits | 59.74 | 55.50 | 60.50 | 59.11 | 50.02 | **60.78** |
| uci_har | 10 splits | 94.67 | 94.49 | 94.86 | 93.91 | 92.77 | **96.62** |
| sad | 10 splits | 85.69 | 87.59 | 87.32 | 86.33 | 84.12 | **92.93** |
| hapt | 10 splits | 82.11 | **84.33** | 83.57 | 82.37 | 81.25 | 83.32 |
| dsads | 8 splits | 81.31 | 83.04 | 84.44 | 79.60 | 79.88 | **85.55** |
| wisdm | 10 splits | 79.51 | 80.37 | **82.99** | 77.85 | 67.77 | 82.92 |
| daphnet | 10 splits | 40.71 | 40.29 | 41.36 | 39.57 | 41.11 | **42.65** |

## Ranking over the 10 complete datasets

Friedman χ² = 26.74, p = 6.4e-05. 
Nemenyi critical difference (α = 0.05): 2.38 ranks. Diagram: `results/p5/cd_recurrencenet.svg`.

| Model | Mean rank | Mean macro-F1 | Wins (rank 1) |
|---|---|---|---|
| RecurrenceNet | 2.00 | 72.01 | 5 |
| S4D | 2.50 | 71.29 | 1 |
| HysteresisNet sel+clock | 2.60 | 70.69 | 4 |
| QuantileNet | 3.80 | 69.70 | 0 |
| Mamba | 4.60 | 68.91 | 0 |
| No memory | 5.50 | 66.06 | 0 |

## Pairwise, over datasets (Wilcoxon signed-rank, Holm-corrected)

| A − B | Mean Δ | 95 % CI | Wins/losses | p | p (Holm) |
|---|---|---|---|---|---|
| QuantileNet − HysteresisNet sel+clock | -0.99 | [-3.09, +1.10] | 3/7 | 0.232 | 0.773 |
| QuantileNet − S4D | -1.59 | [-3.47, +0.28] | 2/8 | 0.0645 | 0.387 |
| QuantileNet − Mamba | +0.79 | [+0.04, +1.54] | 7/3 | 0.0488 | 0.342 |
| QuantileNet − No memory | +3.64 | [+0.61, +6.67] | 8/2 | 0.00977 | 0.107 |
| QuantileNet − RecurrenceNet | -2.31 | [-4.17, -0.44] | 2/8 | 0.0195 | 0.176 |
| HysteresisNet sel+clock − S4D | -0.60 | [-2.10, +0.90] | 5/5 | 0.557 | 0.773 |
| HysteresisNet sel+clock − Mamba | +1.78 | [-0.09, +3.66] | 9/1 | 0.0645 | 0.387 |
| HysteresisNet sel+clock − No memory | +4.63 | [+2.04, +7.22] | 9/1 | 0.00391 | 0.0508 |
| HysteresisNet sel+clock − RecurrenceNet | -1.31 | [-3.43, +0.80] | 4/6 | 0.232 | 0.773 |
| S4D − Mamba | +2.39 | [+0.67, +4.10] | 9/1 | 0.00977 | 0.107 |
| S4D − No memory | +5.23 | [+1.93, +8.54] | 10/0 | 0.00195 | 0.0293 |
| S4D − RecurrenceNet | -0.71 | [-2.26, +0.83] | 3/7 | 0.193 | 0.773 |
| Mamba − No memory | +2.85 | [+0.08, +5.61] | 8/2 | 0.0371 | 0.297 |
| Mamba − RecurrenceNet | -3.10 | [-4.74, -1.46] | 1/9 | 0.00391 | 0.0508 |
| No memory − RecurrenceNet | -5.95 | [-9.10, -2.79] | 0/10 | 0.00195 | 0.0293 |

## Pre-registered claims

| New − old | Mean Δ | 95 % CI | Non-inferior (margin 1) | Better (CI above 0) |
|---|---|---|---|---|
| QuantileNet − S4D | -1.59 | [-3.47, +0.28] | no | no |
| QuantileNet − Mamba | +0.79 | [+0.04, +1.54] | yes | yes |
| HysteresisNet sel+clock − S4D | -0.60 | [-2.10, +0.90] | no | no |
| HysteresisNet sel+clock − Mamba | +1.78 | [-0.09, +3.66] | yes | no |
| RecurrenceNet − S4D | +0.71 | [-0.83, +2.26] | yes | no |
| RecurrenceNet − Mamba | +3.10 | [+1.46, +4.74] | yes | yes |

## Context: best of the 17 WHAR baselines (paper-era re-run, seed 0)

| Dataset | Best WHAR model | Its macro-F1 | Best of our 5 |
|---|---|---|---|
| utd_mhad | attend_discriminate | 67.48 | 73.94 |
| actrectut_gestures | triple_cross_domain_attention | 61.42 | 43.57 |
| unimib_shar | cnn_har | 63.92 | 60.78 |
| uci_har | tinyhar | 96.07 | 96.62 |
