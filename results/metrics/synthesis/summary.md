# Synthesis of rounds 1-13 (10 dev datasets)

3-seed means where seeds 0-2 exist, else seed 0 (marked). Probe = segment statistics + classical classifier (no deep training; DSADS and Daphnet not run). Paper = WHAR Arena Figure 6 (one run).

| Model | UTD-MHAD | UCI-HAR | UniMiB | ActRecTut | Opportunity | SAD | HAPT | DSADS | WISDM | Daphnet | Mean (10) | Mean w/o ActRecTut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RN + channel statistics | 72.2 | 96.3 | 61.8 | 61.7 | 63.9 | 95.9 | 84.5 | 86.1 | 81.9 | 42.8 | 74.70 | 76.15 |
| RN + LawNet | 72.9 | 96.7 | 64.4 | 48.4 | 62.5 | 98.8 | 84.2 | 89.8 | 82.7 | 43.8 | 74.42 | 77.31 |
| RecurrenceNet (RN) | 73.3 | 96.6 | 60.8 | 40.4 | 61.6 | 92.9 | 83.3 | 85.6 | 82.9 | 42.7 | 72.01 | 75.52 |
| S4D | 73.3 | 94.9 | 60.5 | 43.4 | 61.2 | 87.3 | 83.6 | 84.4 | 83.0 | 41.4 | 71.29 | 74.39 |
| HysteresisNet sel+clock | 73.9 | 94.5 | 55.5 | 43.6 | 63.8 | 87.6 | 84.3 | 83.0 | 80.4 | 40.3 | 70.69 | 73.70 |
| LawNet (memoryless + laws) | 70.7 | 94.6 | 58.6 | 43.5 | 61.1 | 96.7 | 81.5 | 87.3 | 78.2 | 43.2 | 71.52 | 74.63 |
| Statistics (memoryless + stats) | 67.3 | 92.6 | 54.1 | 55.6 | 62.0 | 93.1 | 81.5 | 83.2 | 71.5 | 41.5 | 70.24 | 71.86 |
| QuantileNet | 73.8 | 94.7 | 59.7 | 36.1 | 63.4 | 85.7 | 82.1 | 81.3 | 79.5 | 40.7 | 69.70 | 73.43 |
| Mamba | 71.3 | 93.9 | 59.1 | 36.7 | 62.3 | 86.3 | 82.4 | 79.6 | 77.9 | 39.6 | 68.91 | 72.48 |
| No memory (control) | 67.6 | 92.8 | 50.0 | 36.5 | 59.6 | 84.1 | 81.3 | 79.9 | 67.8 | 41.1 | 66.06 | 69.35 |
| StyleAlignNet (seed 0) | 75.2 | 96.2 | 59.3 | 48.7 | 65.8 | 93.5 | 83.3 | 88.0 | 83.9 | 43.5 | 73.74 | 76.52 |
| LawNet, random triples (seed 0) | 68.5 | 95.6 | 58.4 | 45.5 | 63.1 | 98.7 | 81.9 | 87.5 | 77.9 | 44.3 | 72.14 | 75.10 |
| probe root/et | 61.8 | 90.6 | 50.8 | 65.6 | 72.0 | 98.7 | 80.9 | – | 76.4 | – | – | – |
| probe tree_abs/et | 72.2 | 89.8 | 57.3 | 49.4 | 72.5 | 97.4 | 80.7 | – | 75.8 | – | – | – |
| probe tree_abs/lr | 66.6 | 92.4 | 46.9 | 56.8 | 62.9 | 98.2 | 80.0 | – | 77.6 | – | – | – |
| 1-NN DTW std | 74.3 | 86.1 | 58.4 | 69.3 | 31.7 | 88.0 | 71.6 | 77.3 | 63.5 | 32.5 | 65.26 | 64.81 |
| Paper best (17) | 68.2 | 96.1 | 63.3 | 61.5 | 74.7 | 98.2 | 84.8 | 89.3 | 85.5 | 45.4 | 76.70 | 78.40 |
| Paper RF | 51.2 | 84.7 | 44.5 | 41.9 | 70.6 | 98.2 | 67.1 | 89.3 | 71.6 | 33.8 | 65.30 | 67.90 |
| Paper CNN-HAR | 68.2 | 96.0 | 63.3 | 52.2 | 63.1 | 95.6 | 84.8 | 86.9 | 81.4 | 40.2 | 73.18 | 75.51 |
| Paper TinyHAR | 65.8 | 96.1 | 58.6 | 51.1 | 65.4 | 97.7 | 81.0 | 86.8 | 85.0 | 43.9 | 73.13 | 75.59 |

## Dataset properties (paper-era configs)

| Dataset | Kind | Hz | Subjects | Classes | Channels | Samples per 3-s window |
|---|---|---|---|---|---|---|
| UTD-MHAD | gesture | 50 | 8 | 27 | 6 | 150 |
| UCI-HAR | locomotion/ADL | 50 | 30 | 6 | 9 | 150 |
| UniMiB | locomotion/ADL | 50 | 30 | 17 | 3 | 150 |
| ActRecTut | gesture | 32 | 2 | 12 | 15 | 96 |
| Opportunity | locomotion, many sensors | 30 | 4 | 5 | 133 | 90 |
| SAD | locomotion/ADL | 50 | 10 | 7 | 60 | 150 |
| HAPT | locomotion/ADL | 50 | 30 | 12 | 6 | 150 |
| DSADS | locomotion/ADL | 25 | 8 | 19 | 45 | 75 |
| WISDM | locomotion/ADL | 20 | 36 | 6 | 3 | 60 |
| Daphnet | freezing of gait | 64 | 10 | 3 | 9 | 192 |

## Gain of each component per dataset (points of macro-F1)

| Contrast | UTD-MHAD | UCI-HAR | UniMiB | ActRecTut | Opportunity | SAD | HAPT | DSADS | WISDM | Daphnet | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| memory: RN − no memory | +5.7 | +3.9 | +10.8 | +3.9 | +2.0 | +8.8 | +2.1 | +5.7 | +15.2 | +1.5 | +5.95 |
| memory: S4D − no memory | +5.7 | +2.1 | +10.5 | +7.0 | +1.6 | +3.2 | +2.3 | +4.6 | +15.2 | +0.2 | +5.23 |
| statistics on memoryless: statnet − no memory | -0.3 | -0.2 | +4.1 | +19.2 | +2.4 | +9.0 | +0.2 | +3.3 | +3.7 | +0.4 | +4.18 |
| laws on memoryless: LawNet − no memory | +3.1 | +1.8 | +8.6 | +7.0 | +1.4 | +12.5 | +0.2 | +7.4 | +10.4 | +2.0 | +5.46 |
| statistics on RN: recstatnet − RN | -1.1 | -0.3 | +1.0 | +21.4 | +2.3 | +3.0 | +1.2 | +0.6 | -1.0 | +0.1 | +2.70 |
| laws on RN: lawnet_rec − RN | -0.4 | +0.1 | +3.6 | +8.0 | +0.9 | +5.8 | +0.9 | +4.2 | -0.3 | +1.2 | +2.41 |
| classical vs deep: probe root/et − RN | -11.5 | -6.0 | -10.0 | +25.2 | +10.5 | +5.8 | -2.5 | – | -6.6 | – | +0.61 |
| classical vs deep: probe tree_abs/et − RN | -1.2 | -6.9 | -3.4 | +9.0 | +10.9 | +4.4 | -2.6 | – | -7.1 | – | +0.39 |
| elastic vs deep: 1-NN DTW std − RN | +1.0 | -10.6 | -2.4 | +29.0 | -29.9 | -5.0 | -11.7 | -8.3 | -19.4 | -10.1 | -6.74 |
| RN − S4D | +0.0 | +1.8 | +0.3 | -3.1 | +0.4 | +5.6 | -0.3 | +1.1 | -0.1 | +1.3 | +0.71 |
| S4D − Mamba | +2.0 | +0.9 | +1.4 | +6.7 | -1.2 | +1.0 | +1.2 | +4.8 | +5.1 | +1.8 | +2.39 |

## Spearman correlation of each gain with dataset properties (n = 10, or 8 for the probe; descriptive only, no correction)

| Contrast | log2 subjects | log2 channels | samples/window (3 s × Hz) | classes |
|---|---|---|---|---|
| memory: RN − no memory | +0.36 (p 0.30) | -0.45 (p 0.20) | -0.36 (p 0.31) | +0.48 (p 0.16) |
| memory: S4D − no memory | +0.20 (p 0.58) | -0.54 (p 0.10) | -0.47 (p 0.17) | +0.57 (p 0.08) |
| statistics on memoryless: statnet − no memory | -0.15 (p 0.67) | +0.20 (p 0.59) | -0.36 (p 0.31) | +0.01 (p 0.99) |
| laws on memoryless: LawNet − no memory | +0.15 (p 0.67) | -0.12 (p 0.75) | -0.27 (p 0.46) | +0.24 (p 0.50) |
| statistics on RN: recstatnet − RN | -0.44 (p 0.20) | +0.57 (p 0.08) | -0.06 (p 0.87) | -0.08 (p 0.83) |
| laws on RN: lawnet_rec − RN | -0.38 (p 0.28) | +0.45 (p 0.20) | +0.03 (p 0.93) | +0.12 (p 0.75) |
| classical vs deep: probe root/et − RN | -0.56 (p 0.15) | +0.78 (p 0.02) | -0.35 (p 0.39) | -0.48 (p 0.23) |
| classical vs deep: probe tree_abs/et − RN | -0.93 (p 0.00) | +0.81 (p 0.02) | -0.11 (p 0.80) | -0.02 (p 0.95) |
| elastic vs deep: 1-NN DTW std − RN | -0.40 (p 0.26) | -0.06 (p 0.87) | +0.28 (p 0.44) | +0.65 (p 0.04) |
| RN − S4D | +0.07 (p 0.84) | +0.45 (p 0.19) | +0.34 (p 0.33) | -0.34 (p 0.33) |
| S4D − Mamba | -0.13 (p 0.72) | -0.34 (p 0.33) | -0.36 (p 0.31) | +0.39 (p 0.26) |

## Evaluation noise

Seed spread = sample std over seeds 0-2 of the dataset mean. Person spread = sample std over subject splits (seed 0), the paper's ±.

| Model | UTD-MHAD | UCI-HAR | UniMiB | ActRecTut | Opportunity | SAD | HAPT | DSADS | WISDM | Daphnet | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed spread, RecurrenceNet (RN) | 1.39 | 0.21 | 1.03 | 6.12 | 3.37 | 0.85 | 0.65 | 0.78 | 1.26 | 1.68 | 1.73 |
| seed spread, S4D | 1.11 | 0.17 | 0.40 | 2.97 | 0.79 | 2.00 | 1.33 | 0.74 | 1.15 | 1.43 | 1.21 |
| seed spread, RN + channel statistics | 1.14 | 0.16 | 0.89 | 2.05 | 2.50 | 0.64 | 1.38 | 0.42 | 1.94 | 1.51 | 1.26 |
| person spread, RecurrenceNet (RN) | 13.6 | 2.4 | 7.6 | 9.1 | 15.5 | 9.5 | 4.4 | 6.1 | 4.7 | 8.3 | 8.1 |
| person spread, RN + channel statistics | 14.1 | 2.4 | 7.3 | 0.3 | 14.1 | 5.4 | 7.6 | 6.2 | 9.1 | 5.1 | 7.2 |
| validation (seen people) − test (unseen), RN seed 0 | 21.9 | 2.9 | 11.7 | 35.3 | 14.4 | 8.0 | 6.0 | 13.6 | 13.4 | 18.3 | 14.5 |

## Size vs accuracy, paper deep models on these 10 datasets (n = 14)

Spearman(log params, mean F1) = -0.37 (p 0.19). Ours (20-38k params) for reference: RN 24.7k, recstatnet 26.5k, S4D 27.7k.

- Triple-Cross-Attn: 455.0k params, mean F1 73.48
- CNN-HAR: 72.0k params, mean F1 73.18
- TinyHAR: 185.1k params, mean F1 73.13
- DANA: 460.8k params, mean F1 72.86
- Attend+Discriminate: 374.4k params, mean F1 72.74
- TinierHAR: 8.0k params, mean F1 71.39
- DeepConvLSTM-Attn: 477.4k params, mean F1 71.20
- DynamicWHAR: 1127.5k params, mean F1 69.61
- MLP-HAR: 217.2k params, mean F1 69.31
- MLP-Mixer: 621.1k params, mean F1 69.21
- DeepConvLSTM: 460.8k params, mean F1 68.97
- Guan-LSTM: 803.6k params, mean F1 63.32
- DeepConvShallowLSTM: 402.8k params, mean F1 61.83
- SA-HAR: 415.5k params, mean F1 61.82
