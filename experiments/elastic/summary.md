# Elastic 1-NN baselines against deep models (WHAR Arena v1 splits, no training)

Macro-F1 %, mean over splits. Rank = position if inserted into the paper's 17 models (1 = best). RecurrenceNet / S4D / no memory: our 3-seed means. Paper: Figure 6 (one run per cell).

| Dataset | 1-NN Euclid | Euclid, std | DTW | DTW, std | RecurrenceNet | S4D | No memory | Paper best | k-NN (tsfresh) | DTW, std − RecurrenceNet |
|---|---|---|---|---|---|---|---|---|---|---|
| UTD-MHAD | 55.4 (#12) | 62.1 (#8) | 71.0 (#1) | 74.3 (#1) | 73.3 | 73.3 | 67.6 | 68.2 CNN-HAR | 33.9 | +0.96 (6/8, p = 0.383) |
| ActRecTut | 34.1 (#15) | 52.2 (#6) | 58.3 (#2) | 69.3 (#1) | 40.4 | 43.4 | 36.5 | 61.5 Triple-Cross-Attn | 28.6 | +28.96 (2/2, p = nan) |
| UCI-HAR | 81.3 (#16) | 81.0 (#16) | 93.1 (#9) | 86.1 (#15) | 96.6 | 94.9 | 92.8 | 96.1 TinyHAR | 71.2 | -10.55 (0/10, p = 0.002) |
| UniMiB | 32.9 (#16) | 43.6 (#13) | 53.8 (#11) | 58.4 (#7) | 60.8 | 60.5 | 50.0 | 63.3 CNN-HAR | 23.4 | -2.41 (3/10, p = 0.064) |
| Opportunity | 47.0 (#18) | 29.4 (#18) | 49.5 (#18) | 31.7 (#18) | 61.6 | 61.2 | 59.6 | 74.7 MLP-HAR | 59.4 | -29.93 (0/4, p = 0.125) |
| SAD | 43.1 (#18) | 84.0 (#16) | 53.8 (#18) | 88.0 (#16) | 92.9 | 87.3 | 84.1 | 98.2 Random Forest | 82.1 | -4.98 (4/10, p = 0.232) |
| HAPT | 69.2 (#14) | 56.8 (#18) | 80.9 (#7) | 71.6 (#14) | 83.3 | 83.6 | 81.3 | 84.8 CNN-HAR | 66.3 | -11.69 (0/10, p = 0.002) |
| DSADS | 66.2 (#18) | 68.7 (#18) | 72.8 (#17) | 77.3 (#16) | 85.6 | 84.4 | 79.9 | 89.3 Random Forest | 70.9 | -8.25 (means only) |
| WISDM | 49.1 (#18) | 53.8 (#17) | 71.7 (#14) | 63.5 (#16) | 82.9 | 83.0 | 67.8 | 85.5 DANA | 51.2 | -19.42 (0/10, p = 0.002) |
| Daphnet | 32.0 (#15) | 31.7 (#16) | 34.9 (#14) | 32.5 (#15) | 42.7 | 41.4 | 41.1 | 45.4 TinierHAR | 31.1 | -10.13 (means only) |

Means over the 10 datasets: euclid 51.01, euclid_std 56.32, dtw 63.97, dtw_std 65.26, recurrencenet 72.01, paper_best 76.70
