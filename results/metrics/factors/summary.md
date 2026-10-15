# What drives macro-F1: data factors and model components

## A. Where the variance of macro-F1 comes from (our 10 models × 10 datasets × splits × 3 seeds)

2460 runs. Sum of squares by source; shares of the total, and of the variance left once the dataset is fixed (what decides a ranking within a dataset).

| Source | Share of total | Share within a dataset | Typical size (RMS, F1 points) |
|---|---|---|---|
| dataset | 76.6 % | – | 17.15 |
| test person (split) | 14.4 % | 61.3 % | 7.45 |
| model | 1.5 % | 6.4 % | 2.41 |
| model x dataset | 1.7 % | 7.2 % | 2.55 |
| model x person | 2.6 % | 10.9 % | 3.14 |
| seed | 3.3 % | 14.2 % | 3.59 |
| sum of parts / total | 100.2 % | | |

## B. Data factors over the paper's 30 datasets (17 models, one run each)

Spearman ρ (p) between each dataset property and each outcome, n = 30:

| Outcome | log2 subjects | log2 channels | log2 classes | log2 samples/window |
|---|---|---|---|---|
| median deep F1 (level) | +0.07 (0.715) | +0.09 (0.646) | +0.06 (0.752) | -0.10 (0.617) |
| best − median deep (does the model matter) | -0.44 (0.016) | +0.38 (0.036) | +0.04 (0.830) | +0.10 (0.612) |
| RF − best deep (classical margin) | -0.24 (0.198) | +0.71 (0.000) | -0.09 (0.651) | +0.10 (0.601) |
| mean split sd of deep models (person spread) | -0.50 (0.004) | +0.58 (0.001) | +0.12 (0.538) | +0.19 (0.304) |

Correlations among the properties (Spearman ρ; large values mean their effects cannot be separated):

| | log2 subjects | log2 channels | log2 classes | log2 samples/window |
|---|---|---|---|---|
| log2 subjects | +1.00 | -0.50 | +0.11 | -0.16 |
| log2 channels | -0.50 | +1.00 | +0.04 | +0.10 |
| log2 classes | +0.11 | +0.04 | +1.00 | +0.02 |
| log2 samples/window | -0.16 | +0.10 | +0.02 | +1.00 |

Multiple regression (OLS, n = 30, predictors standardized: coefficient = change in F1 points per 1 sd of the property, others held fixed; p in brackets):

| Outcome | R² | log2 subjects | log2 channels | log2 classes | log2 samples/window |
|---|---|---|---|---|---|
| median deep F1 (level) | 0.03 | +1.48 (0.701) | +2.61 (0.497) | +0.43 (0.902) | -1.58 (0.642) |
| best − median deep (does the model matter) | 0.39 | -1.54 (0.004) | +0.39 (0.433) | +0.10 (0.818) | -0.05 (0.905) |
| RF − best deep (classical margin) | 0.55 | +3.07 (0.027) | +7.12 (0.000) | -2.21 (0.073) | +0.35 (0.764) |
| mean split sd of deep models (person spread) | 0.46 | -0.43 (0.542) | +2.44 (0.002) | +0.23 (0.723) | +0.79 (0.209) |

## C. Model components: controlled contrasts, paired on the same splits

Effect = a − b in F1 points; per dataset the mean over splits of the seed-averaged difference; across datasets the mean with a 95 % t-interval, datasets won, Wilcoxon p (no correction), and the between-dataset sd (heterogeneity). ★ = per-dataset 95 % interval over splits excludes 0.

| Component | a − b | Seeds | UTD-MH | UCI-HA | UniMiB | ActRec | Opport | SAD | HAPT | DSADS | WISDM | Daphne | Mean [95 % CI] | Won | p | sd between datasets |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Memory (any) vs none | RecurrenceNet − no memory | 3 | +5.7★ | +3.9★ | +10.8★ | +3.9 | +2.0 | +8.8 | +2.1 | +5.7★ | +15.2★ | +1.5 | +5.95 [+2.79, +9.10] | 10/10 | 0.002 | 4.4 |
| Memory (any) vs none | S4D − no memory | 3 | +5.7★ | +2.1★ | +10.5★ | +7.0 | +1.6 | +3.2 | +2.3 | +4.6★ | +15.2★ | +0.2 | +5.23 [+1.93, +8.54] | 10/10 | 0.002 | 4.6 |
| Memory (any) vs none | HysteresisNet − no memory | 3 | +6.3★ | +1.7★ | +5.5★ | +7.1★ | +4.2 | +3.5 | +3.1★ | +3.2★ | +12.6★ | -0.8 | +4.63 [+2.04, +7.22] | 9/10 | 0.004 | 3.6 |
| Memory (any) vs none | QuantileNet − no memory | 3 | +6.2★ | +1.9★ | +9.7★ | -0.4 | +3.8 | +1.6 | +0.9 | +1.4 | +11.7★ | -0.4 | +3.64 [+0.61, +6.67] | 8/10 | 0.010 | 4.2 |
| Memory (any) vs none | Mamba − no memory | 3 | +3.7★ | +1.1 | +9.1★ | +0.2 | +2.7 | +2.2 | +1.1 | -0.3 | +10.1★ | -1.5 | +2.85 [+0.08, +5.61] | 8/10 | 0.037 | 3.9 |
| Memory type (vs S4D) | RecurrenceNet − S4D | 3 | +0.0 | +1.8★ | +0.3 | -3.1 | +0.4 | +5.6 | -0.3 | +1.1 | -0.1 | +1.3 | +0.71 [-0.83, +2.26] | 7/10 | 0.193 | 2.2 |
| Memory type (vs S4D) | HysteresisNet − S4D | 3 | +0.7 | -0.4 | -5.0★ | +0.1 | +2.6 | +0.3 | +0.8 | -1.4 | -2.6★ | -1.1 | -0.60 [-2.10, +0.90] | 5/10 | 0.557 | 2.1 |
| Memory type (vs S4D) | QuantileNet − S4D | 3 | +0.5 | -0.2 | -0.8 | -7.4 | +2.2 | -1.6 | -1.5 | -3.1★ | -3.5★ | -0.6 | -1.59 [-3.47, +0.28] | 2/10 | 0.064 | 2.6 |
| Memory type (vs S4D) | Mamba − S4D | 3 | -2.0 | -0.9 | -1.4 | -6.7 | +1.2 | -1.0 | -1.2 | -4.8★ | -5.1★ | -1.8 | -2.39 [-4.10, -0.67] | 1/10 | 0.010 | 2.4 |
| Window statistics | stats (no memory) − no memory | 3 | -0.3 | -0.2 | +4.1★ | +19.2 | +2.4 | +9.0★ | +0.2 | +3.3 | +3.7★ | +0.4 | +4.18 [-0.09, +8.45] | 8/10 | 0.014 | 6.0 |
| Window statistics | RN+stats − RecurrenceNet | 3 | -1.1 | -0.3 | +1.0 | +21.4 | +2.3 | +3.0 | +1.2 | +0.6 | -1.0 | +0.1 | +2.70 [-2.08, +7.48] | 7/10 | 0.131 | 6.7 |
| Laws (lagged second-order) | laws (no memory) − no memory | 3 | +3.1 | +1.8★ | +8.6★ | +7.0 | +1.4 | +12.5★ | +0.2 | +7.4★ | +10.4★ | +2.0 | +5.46 [+2.40, +8.52] | 10/10 | 0.002 | 4.3 |
| Laws (lagged second-order) | RN+laws − RecurrenceNet | 3 | -0.4 | +0.1 | +3.6★ | +8.0 | +0.9 | +5.8★ | +0.9 | +4.2★ | -0.3 | +1.2 | +2.41 [+0.35, +4.47] | 8/10 | 0.020 | 2.9 |
| Laws (lagged second-order) | RN+laws − RN+stats | 3 | +0.7 | +0.4 | +2.6★ | -13.3 | -1.4 | +2.9 | -0.2 | +3.7★ | +0.7 | +1.1 | -0.29 [-3.74, +3.17] | 7/10 | 0.322 | 4.8 |
| Sensor grouping | laws (no memory) − laws, random groups | 1 | +2.4 | -0.7★ | +0.0 | -1.0 | -3.8 | +0.3 | -0.7 | +0.4 | +0.0 | -1.1 | -0.41 [-1.53, +0.71] | 3/10 | 0.312 | 1.6 |

Kill set only (UTD-MHAD, ActRecTut, SAD, Opportunity; 3 seeds):

| Component | a − b | UTD-MHAD | ActRecTut | SAD | Opportunity | Mean | Without ActRecTut |
|---|---|---|---|---|---|---|---|
| Which statistic | RN+mean − RecurrenceNet | -1.5 | -2.9 | -2.2 | +0.3 | -1.55 | -1.11 |
| Which statistic | RN+std − RecurrenceNet | -0.9 | +10.4 | +3.9 | +2.6 | +3.98 | +1.86 |
| Which statistic | RN+min/max − RecurrenceNet | -1.6 | +15.7 | +3.7 | +0.9 | +4.68 | +0.99 |
| Which statistic | RN+stats − RecurrenceNet | -1.1 | +21.4 | +3.0 | +2.3 | +6.38 | +1.39 |
| Normalization / pooling | RN, mean+std pooling − RecurrenceNet | -1.4 | +8.0 | +1.0 | -0.5 | +1.78 | -0.29 |
| Normalization / pooling | RN, no LayerNorm − RecurrenceNet | -3.1 | +6.2 | -2.7 | -0.7 | -0.08 | -2.17 |
| Normalization / pooling | RN, both − RecurrenceNet | -1.7 | +13.8 | -0.2 | -1.2 | +2.67 | -1.03 |

Which data properties go with each effect (Spearman ρ over the 10 datasets, p in brackets; descriptive):

| a − b | log2 subjects | log2 channels | log2 classes | samples/window (3 s × Hz) | log2 windows |
|---|---|---|---|---|---|
| RecurrenceNet − no memory | +0.36 (0.30) | -0.45 (0.20) | +0.48 (0.16) | -0.36 (0.31) | -0.10 (0.78) |
| S4D − no memory | +0.20 (0.58) | -0.54 (0.10) | +0.57 (0.08) | -0.47 (0.17) | -0.19 (0.60) |
| HysteresisNet − no memory | -0.13 (0.72) | -0.27 (0.45) | +0.34 (0.34) | -0.59 (0.07) | -0.22 (0.53) |
| QuantileNet − no memory | +0.40 (0.25) | -0.44 (0.20) | +0.17 (0.64) | -0.37 (0.29) | -0.04 (0.91) |
| Mamba − no memory | +0.40 (0.26) | -0.46 (0.18) | +0.13 (0.72) | -0.28 (0.44) | -0.15 (0.68) |
| RecurrenceNet − S4D | +0.07 (0.84) | +0.45 (0.19) | -0.34 (0.33) | +0.34 (0.33) | +0.19 (0.60) |
| HysteresisNet − S4D | -0.46 (0.18) | +0.49 (0.15) | -0.08 (0.83) | +0.12 (0.74) | -0.26 (0.47) |
| QuantileNet − S4D | -0.09 (0.81) | +0.10 (0.79) | -0.20 (0.58) | +0.43 (0.21) | -0.22 (0.53) |
| Mamba − S4D | +0.13 (0.72) | +0.34 (0.33) | -0.39 (0.26) | +0.36 (0.31) | -0.04 (0.91) |
| stats (no memory) − no memory | -0.15 (0.67) | +0.20 (0.59) | +0.01 (0.99) | -0.36 (0.31) | +0.05 (0.88) |
| RN+stats − RecurrenceNet | -0.44 (0.20) | +0.57 (0.08) | -0.08 (0.83) | -0.06 (0.87) | -0.14 (0.70) |
| laws (no memory) − no memory | +0.15 (0.67) | -0.12 (0.75) | +0.24 (0.50) | -0.27 (0.46) | +0.07 (0.85) |
| RN+laws − RecurrenceNet | -0.38 (0.28) | +0.45 (0.20) | +0.12 (0.75) | +0.03 (0.93) | -0.07 (0.85) |
| RN+laws − RN+stats | +0.28 (0.43) | -0.06 (0.87) | +0.23 (0.53) | +0.08 (0.82) | +0.33 (0.35) |
| laws (no memory) − laws, random groups | +0.16 (0.66) | -0.25 (0.48) | +0.74 (0.02) | -0.11 (0.76) | -0.17 (0.64) |

Model size (paper deep models, n = 14): Spearman(log params, mean F1) = -0.43 (p 0.12) over 30 datasets, -0.37 (p 0.19) over our 10.
