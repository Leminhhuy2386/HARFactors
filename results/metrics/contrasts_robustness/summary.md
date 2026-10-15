# Component contrasts without daphnet (exploratory)

Same contrasts and statistics as results/metrics/factors/summary.md § C, over the 9 datasets whose stored runs agree on split sizes (daphnet excluded: its split ids do not name the same test subjects across runs; results/gates/G2_data.json).

| Component | a − b | Seeds | 10 datasets: mean [95 % CI], won, p | 9 datasets: mean [95 % CI], won, p |
|---|---|---|---|---|
| Memory (any) vs none | RecurrenceNet − no memory | 3 | +5.95 [+2.79, +9.10], 10/10, 0.002 | +6.43 [+3.07, +9.80], 9/9, 0.004 |
| Memory (any) vs none | S4D − no memory | 3 | +5.23 [+1.93, +8.54], 10/10, 0.002 | +5.79 [+2.30, +9.27], 9/9, 0.004 |
| Memory (any) vs none | HysteresisNet − no memory | 3 | +4.63 [+2.04, +7.22], 9/10, 0.004 | +5.24 [+2.74, +7.74], 9/9, 0.004 |
| Memory (any) vs none | QuantileNet − no memory | 3 | +3.64 [+0.61, +6.67], 8/10, 0.010 | +4.09 [+0.83, +7.34], 8/9, 0.008 |
| Memory (any) vs none | Mamba − no memory | 3 | +2.85 [+0.08, +5.61], 8/10, 0.037 | +3.33 [+0.45, +6.22], 8/9, 0.012 |
| Memory type (vs S4D) | RecurrenceNet − S4D | 3 | +0.71 [-0.83, +2.26], 7/10, 0.193 | +0.65 [-1.11, +2.40], 6/9, 0.301 |
| Memory type (vs S4D) | HysteresisNet − S4D | 3 | -0.60 [-2.10, +0.90], 5/10, 0.557 | -0.55 [-2.25, +1.15], 5/9, 0.820 |
| Memory type (vs S4D) | QuantileNet − S4D | 3 | -1.59 [-3.47, +0.28], 2/10, 0.064 | -1.70 [-3.82, +0.42], 2/9, 0.098 |
| Memory type (vs S4D) | Mamba − S4D | 3 | -2.39 [-4.10, -0.67], 1/10, 0.010 | -2.45 [-4.40, -0.50], 1/9, 0.020 |
| Window statistics | stats (no memory) − no memory | 3 | +4.18 [-0.09, +8.45], 8/10, 0.014 | +4.60 [-0.14, +9.34], 7/9, 0.027 |
| Window statistics | RN+stats − RecurrenceNet | 3 | +2.70 [-2.08, +7.48], 7/10, 0.131 | +2.99 [-2.41, +8.39], 6/9, 0.129 |
| Laws (lagged second-order) | laws (no memory) − no memory | 3 | +5.46 [+2.40, +8.52], 10/10, 0.002 | +5.84 [+2.49, +9.19], 9/9, 0.004 |
| Laws (lagged second-order) | RN+laws − RecurrenceNet | 3 | +2.41 [+0.35, +4.47], 8/10, 0.020 | +2.55 [+0.23, +4.86], 7/9, 0.039 |
| Laws (lagged second-order) | RN+laws − RN+stats | 3 | -0.29 [-3.74, +3.17], 7/10, 0.322 | -0.44 [-4.36, +3.48], 6/9, 0.426 |
| Sensor grouping | laws (no memory) − laws, random groups | 1 | -0.41 [-1.53, +0.71], 3/10, 0.312 | -0.33 [-1.59, +0.93], 3/9, 0.469 |
