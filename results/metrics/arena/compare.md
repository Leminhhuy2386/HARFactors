# Our models against the WHAR Arena paper table (Figure 6)

Paper numbers: Figure 6 of the WHAR Arena paper (macro-F1 %, mean over subject splits, one run). Ours: same paper-era code, data, splits and protocol, training seed 0 (the protocol run); the mean over seeds 0-2 is in a separate column. No new training for this comparison.

Datasets: the 10 of P5. Dev = used to screen rounds 6-7 (and UTD-MHAD for every earlier round); held-out = never used to choose anything. The paper's own mean (and the Leaderboard on the WHAR Arena website) is over 30 datasets; the means below are over these 10 only, which are easier on average (CNN-HAR 67.7 over 30, 73.2 over these 10, 65.0 over the other 20).

## 0. Is a paper cell comparable to one of ours?

Our harness re-ran all 17 Arena models on the 4 dev datasets (D:\Projects\HAR\paper-era):

| Dataset | Models | Spearman rho vs paper | Mean delta (ours - paper) | Mean abs delta |
|---|---|---|---|---|
| UTD-MHAD | 17 | 0.975 | -0.11 | 1.00 |
| UCI-HAR | 17 | 0.953 | +0.04 | 0.47 |
| UniMiB | 17 | 0.983 | -0.86 | 1.17 |
| ActRecTut | 17 | 0.897 | -1.55 | 2.55 |

Run-to-run noise between the paper's GPU and ours is about 1 point per cell (larger on ActRecTut, 2 splits). Classical models (RF, k-NN) match to the digit, so the splits are the same.

## 1. All 23 models over the 10 datasets

Ours at seed 0. Mean rank: per dataset among all 23, then averaged (1 = best). Top 3 = datasets where the model is in the first three. Params on the UTD-MHAD configuration.

| # | Model | Params | Mean F1 (10) | Dev (4) | Held-out (6) | Without ActRecTut (9) | Mean rank /23 | Top 3 | Ours, mean seeds 0-2 | Paper mean (30 ds) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TinyHAR | 185.1k | 73.13 | 67.89 | 76.63 | 75.59 | 5.60 | 4 |  | 67.61 |
| 2 | CNN-HAR | 72.0k | 73.18 | 69.94 | 75.34 | 75.51 | 6.20 | 4 |  | 67.72 |
| 3 | **RecurrenceNet** | 24.7k | 72.25 | 67.09 | 75.69 | 76.38 | 6.20 | 3 | 72.01 |  |
| 4 | DANA | 460.8k | 72.86 | 69.86 | 74.86 | 74.60 | 7.00 | 5 |  | 67.02 |
| 5 | Attend+Discriminate | 374.4k | 72.74 | 69.00 | 75.24 | 74.85 | 7.60 | 1 |  | 66.61 |
| 6 | Triple-Cross-Attn | 455.0k | 73.48 | 69.91 | 75.87 | 74.82 | 7.80 | 2 |  | 67.57 |
| 7 | TinierHAR | 8.0k | 71.39 | 64.94 | 75.68 | 74.30 | 7.95 | 1 |  | 66.73 |
| 8 | DeepConvLSTM-Attn | 477.4k | 71.20 | 66.39 | 74.40 | 73.27 | 9.70 | 1 |  | 64.67 |
| 9 | **S4D** | 27.7k | 71.25 | 68.10 | 73.35 | 74.31 | 10.30 | 1 | 71.29 |  |
| 10 | MLP-HAR | 217.2k | 69.31 | 60.20 | 75.38 | 72.15 | 11.60 | 2 |  | 64.23 |
| 11 | **HysteresisNet sel+clock** | 23.6k | 70.25 | 66.41 | 72.81 | 73.32 | 11.90 | 1 | 70.69 |  |
| 12 | **QuantileNet** | 20.8k | 68.86 | 65.76 | 70.92 | 72.59 | 12.40 | 1 | 69.70 |  |
| 13 | DeepConvLSTM | 460.8k | 68.97 | 63.12 | 72.88 | 71.28 | 12.50 | 0 |  | 63.61 |
| 14 | DynamicWHAR | 1.13M | 69.61 | 62.69 | 74.22 | 72.73 | 12.60 | 1 |  | 66.27 |
| 15 | MLP-Mixer | 621.1k | 69.21 | 63.61 | 72.95 | 72.70 | 12.90 | 0 |  | 62.46 |
| 16 | Random Forest | – | 65.30 | 55.57 | 71.79 | 67.90 | 13.40 | 3 |  | 65.10 |
| 17 | **Mamba** | 21.1k | 67.90 | 63.68 | 70.71 | 71.92 | 13.40 | 0 | 68.91 |  |
| 18 | DeepConvShallowLSTM | 402.8k | 61.83 | 48.47 | 70.74 | 63.49 | 14.90 | 0 |  | 59.96 |
| 19 | Guan-LSTM | 803.6k | 63.32 | 51.78 | 71.01 | 66.42 | 15.25 | 0 |  | 60.07 |
| 20 | **No memory (control)** | 20.0k | 65.63 | 61.95 | 68.09 | 68.72 | 15.90 | 0 | 66.06 |  |
| 21 | SA-HAR | 415.5k | 61.82 | 55.75 | 65.86 | 65.24 | 19.60 | 0 |  | 59.84 |
| 22 | SVM | – | 53.11 | 35.86 | 64.61 | 55.95 | 19.60 | 0 |  | 50.15 |
| 23 | k-NN | – | 51.83 | 39.29 | 60.18 | 54.40 | 21.70 | 0 |  | 46.25 |

## 2. Per dataset: our F1 and its rank among the 17 paper models

Cell: seed-0 F1 (rank if inserted into the paper's 17; 1 = above all of them).

| Dataset | Splits | Paper best | CNN-HAR | Paper median | RecurrenceNet | S4D | HysteresisNet sel+clock | QuantileNet | Mamba | No memory (control) |
|---|---|---|---|---|---|---|---|---|---|---|
| UTD-MHAD | 8 | 68.2 CNN-HAR | 68.2 | 61.0 | 74.6 (#1) | 73.8 (#1) | 73.4 (#1) | 74.3 (#1) | 70.7 (#1) | 67.6 (#2) |
| UCI-HAR | 10 | 96.1 TinyHAR | 96.0 | 93.0 | 96.7 (#1) | 94.8 (#6) | 94.3 (#7) | 94.7 (#6) | 93.5 (#8) | 92.7 (#10) |
| UniMiB | 10 | 63.3 CNN-HAR | 63.3 | 55.5 | 62.0 (#3) | 60.1 (#5) | 55.3 (#10) | 58.8 (#5) | 58.8 (#5) | 49.6 (#12) |
| ActRecTut | 2 | 61.5 Triple-Cross-Attn | 52.2 | 45.2 | 35.1 (#15) | 43.8 (#11) | 42.7 (#11) | 35.2 (#15) | 31.7 (#15) | 37.9 (#13) |
| Opportunity * | 4 | 74.7 MLP-HAR | 63.1 | 65.1 | 64.9 (#10) | 61.0 (#14) | 64.6 (#10) | 60.9 (#14) | 61.9 (#13) | 57.8 (#18) |
| SAD * | 10 | 98.2 Random Forest | 95.6 | 93.4 | 92.0 (#10) | 89.4 (#16) | 87.1 (#16) | 82.3 (#16) | 82.8 (#16) | 80.8 (#18) |
| HAPT * | 10 | 84.8 CNN-HAR | 84.8 | 79.4 | 83.2 (#3) | 82.1 (#3) | 84.4 (#2) | 82.7 (#3) | 82.8 (#3) | 81.1 (#6) |
| DSADS * | 8 | 89.3 Random Forest | 86.9 | 84.4 | 86.2 (#6) | 85.2 (#9) | 80.1 (#15) | 80.9 (#15) | 80.8 (#15) | 79.1 (#15) |
| WISDM * | 10 | 85.5 DANA | 81.4 | 83.0 | 84.3 (#4) | 82.5 (#11) | 80.9 (#13) | 78.4 (#13) | 76.5 (#13) | 69.1 (#15) |
| Daphnet * | 10 | 45.4 TinierHAR | 40.2 | 40.2 | 43.5 (#4) | 39.8 (#10) | 39.7 (#10) | 40.2 (#10) | 39.4 (#10) | 40.7 (#7) |

\* held-out dataset.

Same table, our models at the mean over seeds 0-2:

| Dataset | RecurrenceNet | S4D | HysteresisNet sel+clock | QuantileNet | Mamba | No memory (control) |
|---|---|---|---|---|---|---|
| UTD-MHAD | 73.3 (#1) | 73.3 (#1) | 73.9 (#1) | 73.8 (#1) | 71.3 (#1) | 67.6 (#2) |
| UCI-HAR | 96.6 (#1) | 94.9 (#6) | 94.5 (#7) | 94.7 (#6) | 93.9 (#8) | 92.8 (#10) |
| UniMiB | 60.8 (#5) | 60.5 (#5) | 55.5 (#10) | 59.7 (#5) | 59.1 (#5) | 50.0 (#12) |
| ActRecTut | 40.4 (#13) | 43.4 (#11) | 43.6 (#11) | 36.1 (#14) | 36.7 (#14) | 36.5 (#14) |
| Opportunity | 61.6 (#13) | 61.2 (#14) | 63.8 (#10) | 63.4 (#10) | 62.3 (#12) | 59.6 (#15) |
| SAD | 92.9 (#10) | 87.3 (#16) | 87.6 (#16) | 85.7 (#16) | 86.3 (#16) | 84.1 (#16) |
| HAPT | 83.3 (#3) | 83.6 (#3) | 84.3 (#2) | 82.1 (#3) | 82.4 (#3) | 81.3 (#6) |
| DSADS | 85.6 (#9) | 84.4 (#9) | 83.0 (#13) | 81.3 (#15) | 79.6 (#15) | 79.9 (#15) |
| WISDM | 82.9 (#11) | 83.0 (#9) | 80.4 (#13) | 79.5 (#13) | 77.9 (#13) | 67.8 (#15) |
| Daphnet | 42.7 (#6) | 41.4 (#7) | 40.3 (#9) | 40.7 (#7) | 39.6 (#10) | 41.1 (#7) |

## 3. Paired over the 10 datasets against the strongest paper models

Cell: mean difference in F1 points (ours - paper model), wins-losses over the 10 datasets, Wilcoxon signed-rank p with Holm correction within each row. TinierHAR is added as the only paper model at or below our size. "Best per dataset" is the strongest paper model on each dataset (an oracle pick, biased upward).

| Ours | Triple-Cross-Attn | CNN-HAR | TinyHAR | DANA | Attend+Discriminate | TinierHAR | Best per dataset |
|---|---|---|---|---|---|---|---|
| RecurrenceNet | -1.23 (8-2) p=1.00 | -0.93 (5-5) p=1.00 | -0.88 (4-6) p=1.00 | -0.61 (4-6) p=1.00 | -0.49 (8-2) p=1.00 | +0.87 (6-4) p=1.00 | -4.45 (2-8) p=0.45 |
| S4D | -2.23 (3-7) p=0.80 | -1.93 (2-8) p=0.50 | -1.88 (3-7) p=0.80 | -1.61 (3-7) p=0.80 | -1.49 (3-7) p=0.80 | -0.13 (3-7) p=0.80 | -5.45 (1-9) p=0.26 |
| HysteresisNet sel+clock | -3.23 (3-7) p=0.65 | -2.93 (2-8) p=0.50 | -2.88 (2-8) p=0.65 | -2.61 (3-7) p=0.70 | -2.49 (4-6) p=0.70 | -1.13 (2-8) p=0.70 | -6.45 (1-9) p=0.10 |
| QuantileNet | -4.62 (3-7) p=0.53 | -4.32 (1-9) p=0.29 | -4.27 (3-7) p=0.53 | -4.00 (2-8) p=0.53 | -3.88 (3-7) p=0.53 | -2.53 (3-7) p=0.53 | -7.85 (1-9) p=0.14 |
| Mamba | -5.59 (3-7) p=0.32 | -5.28 (1-9) p=0.12 | -5.24 (3-7) p=0.32 | -4.97 (2-8) p=0.32 | -4.85 (3-7) p=0.32 | -3.49 (3-7) p=0.32 | -8.81 (1-9) p=0.04 |
| No memory (control) | -7.85 (1-9) p=0.02 | -7.55 (1-9) p=0.02 | -7.50 (2-8) p=0.03 | -7.23 (1-9) p=0.03 | -7.11 (1-9) p=0.02 | -5.75 (1-9) p=0.03 | -11.07 (0-10) p=0.01 |

## 4. Our other models on the dev datasets (seed 0)

Models without the full 10-dataset run: earlier rounds (UTD-MHAD only, some also ActRecTut/UniMiB) and rounds 6-7 that stopped at stage 2. Cell: F1 (rank among the paper's 17).

| Model | Params | UTD-MHAD | UCI-HAR | UniMiB | ActRecTut |
|---|---|---|---|---|---|
| stylealignnet | 38.2k | 75.2 (#1) | 96.2 (#1) | 59.3 (#5) | 48.7 (#7) |
| recstat_mean | 25.8k | 74.3 (#1) |  |  | 38.7 (#13) |
| DualMemNet | 26.7k | 73.9 (#1) |  |  |  |
| lawnet_rec | 27.6k | 73.7 (#1) | 96.4 (#1) | 63.7 (#1) | 43.8 (#11) |
| isochron_rec | 36.1k | 73.6 (#1) |  |  | 53.8 (#3) |
| recstat_std | 25.8k | 73.3 (#1) |  |  | 46.0 (#9) |
| HysteresisNet mm | 21.3k | 72.1 (#1) |  |  |  |
| isochron_rec_uniform | 36.1k | 71.9 (#1) |  |  | 52.5 (#5) |
| HysteresisNet sel | 23.5k | 71.9 (#1) |  |  |  |
| HysteresisNet k6 | 26.7k | 71.8 (#1) |  |  |  |
| HysteresisNet sel k6 | 29.9k | 71.8 (#1) |  |  |  |
| recnet_meanstd_nonorm | 25.4k | 71.6 (#1) |  |  | 52.4 (#5) |
| recstatnet | 26.5k | 71.6 (#1) | 96.4 (#1) | 62.1 (#3) | 63.6 (#1) |
| RainflowNet | 23.5k | 71.2 (#1) | 95.6 (#4) | 58.2 (#7) | 32.5 (#15) |
| recstat_range | 26.0k | 71.1 (#1) |  |  | 52.7 (#4) |
| lawnet | 19.0k | 70.9 (#1) | 94.9 (#6) | 58.4 (#7) | 44.5 (#10) |
| recnet_meanstd | 25.6k | 70.9 (#1) |  |  | 49.3 (#7) |
| HysteresisNet clock | 20.4k | 70.7 (#1) |  |  |  |
| OrdinalNet | 18.7k | 70.5 (#1) | 96.3 (#1) | 56.3 (#9) | 40.7 (#13) |
| recnet_nonorm | 24.5k | 70.5 (#1) |  |  | 38.9 (#13) |
| HysteresisNet fixed r | 20.0k | 70.5 (#1) |  |  |  |
| HysteresisNet | 20.4k | 70.3 (#1) |  |  |  |
| TropicalNet | 26.4k | 70.3 (#1) |  |  |  |
| isochron_uniform | 16.4k | 70.1 (#1) |  |  | 58.9 (#2) |
| statnet | 17.9k | 69.6 (#1) | 92.1 (#12) | 53.9 (#11) | 47.2 (#8) |
| isochron | 16.4k | 69.5 (#1) |  |  | 54.7 (#3) |
| ReturnMapNet | 21.7k | 69.4 (#1) | 91.7 (#13) | 50.3 (#12) | 37.5 (#14) |
| lawnet_rand | 19.0k | 68.5 (#1) | 95.6 (#4) | 58.4 (#7) | 45.5 (#9) |
| levelnet_plane | 29.9k | 65.6 (#5) |  |  | 48.1 (#8) |
| bhnet | 20.2k | 65.0 (#5) |  |  |  |
| VortexNet | 17.0k | 63.4 (#8) |  |  |  |
| parliamentbody | 30.1k | 60.0 (#11) |  |  |  |
| levelnet | 20.5k | 58.7 (#12) |  |  | 48.3 (#7) |
| parliamentnet | 21.6k | 58.6 (#12) |  |  |  |
| Edge-SSM osc | 34.0k | 53.8 (#13) |  | 50.3 (#12) | 48.6 (#7) |
| Edge-SSM tau input | 33.2k | 53.2 (#13) |  | 30.7 (#16) | 33.0 (#15) |
| Edge-SSM no phi | 31.3k | 51.6 (#13) |  | 37.5 (#16) | 31.7 (#15) |
| Edge-SSM v2 | 33.6k | 51.6 (#13) |  | 42.4 (#13) | 28.6 (#16) |
| Edge-SSM tau fixed | 33.2k | 51.6 (#13) |  | 40.8 (#14) | 28.5 (#17) |
| GRU | 40.5k | 50.1 (#14) |  |  | 36.0 (#14) |
| stylealign_elastic | 13.5k | 49.8 (#14) |  |  | 34.1 (#15) |
| Edge-SSM-S | 5.3k | 47.8 (#14) |  | 33.1 (#16) | 35.9 (#14) |
| Edge-SSM EMA | 33.6k | 46.5 (#14) |  |  | 36.5 (#14) |
| forestnet | 29.1k | 4.4 (#18) |  |  |  |

Paper best on the dev datasets: UTD-MHAD 68.2, UCI-HAR 96.1, UniMiB 63.3, ActRecTut 61.5.

## 5. Where the classical Random Forest wins (paper table, all 30 datasets)

Random Forest on tsfresh minimal features minus the best of the 14 deep paper models, against the number of channels (paper Table 3). Spearman rho = 0.71 (p = 1.3e-05). Random Forest is the best model on 7 of 11 datasets with at least 20 channels and on 2 of 19 with fewer.

| Dataset | Channels | RF - best deep | Best model |
|---|---|---|---|
| wisdm | 3 | -13.9 | DANA |
| unimib_shar | 3 | -18.8 | CNN-HAR |
| hang_time | 3 | -5.4 | Guan-LSTM |
| hhar | 6 | -7.1 | CNN-HAR |
| utd_mhad | 6 | -17.0 | CNN-HAR |
| hapt | 6 | -17.7 | CNN-HAR |
| usc_had | 6 | -6.1 | MLP-Mixer |
| wisdm_19_phone | 6 | -5.4 | DynamicWHAR |
| wisdm_19_watch | 6 | -11.5 | DANA |
| harth | 6 | -16.8 | Triple-Cross-Attn |
| har70 | 6 | -11.7 | CNN-HAR |
| w_har | 7 | +0.4 | Random Forest |
| uca_ehar | 7 | -12.5 | DANA |
| uci_har | 9 | -11.4 | TinyHAR |
| daphnet | 9 | -11.6 | TinierHAR |
| wear | 12 | -2.3 | DynamicWHAR |
| actrectut_gestures | 15 | -19.6 | Triple-Cross-Attn |
| real_life_har | 15 | +7.6 | Random Forest |
| motion_sense | 18 | -2.0 | DynamicWHAR |
| mhealth | 23 | +1.5 | Random Forest |
| gotov | 28 | -4.5 | DynamicWHAR |
| hugadb | 38 | +3.3 | Random Forest |
| uma_fall | 39 | -2.0 | DynamicWHAR |
| up_fall | 42 | +11.6 | Random Forest |
| dsads | 45 | +1.5 | Random Forest |
| pamap2 | 52 | +0.8 | Random Forest |
| sad | 60 | +0.6 | Random Forest |
| real_world | 81 | +12.0 | Random Forest |
| real_disp | 117 | -3.4 | TinyHAR |
| opportunity | 133 | -4.1 | MLP-HAR |

## 6. Seen against unseen subjects (ours, seed 0)

Validation macro-F1 (20 % of the training subjects' windows, which overlap training windows by 50 %, so it overstates pure subject shift) minus test macro-F1 (unseen subjects).

| Model | Mean | UTD-MHAD | UCI-HAR | UniMiB | ActRecTut | Opportunity | SAD | HAPT | DSADS | WISDM | Daphnet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RecurrenceNet | 14.5 | 21.9 | 2.9 | 11.7 | 35.3 | 14.4 | 8.0 | 6.0 | 13.6 | 13.4 | 18.3 |
| S4D | – | – | 4.5 | 15.2 | – | 18.2 | 10.6 | 8.6 | 14.7 | 15.8 | 20.8 |
| HysteresisNet sel+clock | 16.3 | 23.4 | 5.0 | 16.6 | 27.7 | 14.6 | 12.9 | 6.3 | 19.8 | 17.1 | 19.6 |
| QuantileNet | 18.0 | 23.1 | 4.5 | 17.1 | 34.8 | 18.3 | 17.7 | 7.4 | 18.9 | 19.3 | 18.9 |
| Mamba | – | – | 5.8 | – | – | 17.3 | 17.1 | 7.1 | 19.1 | 22.2 | 22.3 |
| No memory (control) | – | – | 6.3 | 12.0 | 32.4 | 21.4 | 19.2 | 6.9 | 20.8 | 23.5 | 18.3 |

– = not recorded (the earliest seed-0 runs on UTD-MHAD, ActRecTut and part of UniMiB predate the validation score in the result file).
