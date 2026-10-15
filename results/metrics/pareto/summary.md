# Accuracy against cost (Pareto)

Mean macro-F1 over the 10 datasets at seed 0 (the paper ran once); cost = geometric mean over the 10 datasets of the measured cost on each dataset's input. ★ = on the Pareto front of all models for that cost. FLOPs count matmul/conv only (LSTM, FFT and scan work is missed), so they are listed without a front.

| Model | Source | F1 seed 0 | F1 3 seeds | Params | Params (UTD-MHAD) | MFLOPs / window | CPU ms / window |
|---|---|---|---|---|---|---|---|
| RN + statistics | ours | 74.69 | 74.70 | 28.7k ★ | 26.5k | 5.70 | 5.81 ★ |
| RN + laws | ours | 73.77 | 74.42 | 30.2k | 27.6k | 6.42 | 9.82 |
| StyleAlignNet | ours | 73.74 |  | 29.8k | 38.2k | 5.89 | 10.67 |
| Triple-Cross-Attn | paper | 73.48 |  | 410.7k | 455.0k | 34.47 | 3.52 ★ |
| CNN-HAR | paper | 73.18 |  | 133.2k | 72.0k | 20.33 | 4.78 |
| TinyHAR | paper | 73.13 |  | 191.9k | 185.1k | 17.79 | 5.10 |
| DANA | paper | 72.86 |  | 805.2k | 460.8k | 191.48 | 7.16 |
| Attend+Discriminate | paper | 72.74 |  | 640.2k | 374.4k | 328.19 | 30.56 |
| RecurrenceNet | ours | 72.25 | 72.01 | 24.9k ★ | 24.7k | 5.69 | 5.54 |
| LawNet (random groups) | ours | 72.14 |  | 21.4k ★ | 19.0k | 4.22 | 4.40 |
| LawNet | ours | 71.73 | 71.52 | 21.4k | 19.0k | 4.22 | 4.41 |
| TinierHAR | paper | 71.39 |  | 14.5k ★ | 8.0k | 1.23 | 3.54 |
| S4D | ours | 71.25 | 71.29 | 28.0k | 27.7k | 4.79 | 10.85 |
| DeepConvLSTM-Attn | paper | 71.20 |  | 826.6k | 477.4k | 176.35 | 7.06 |
| HysteresisNet | ours | 70.25 | 70.69 | 23.8k | 23.6k | 5.34 | 3.79 |
| Statistics (memoryless) | ours | 69.64 | 70.24 | 20.0k | 17.9k | 3.60 | 0.57 ★ |
| DynamicWHAR | paper | 69.61 |  | 2082.2k | 1127.5k | 66.54 | 3.17 |
| MLP-HAR | paper | 69.31 |  | 424.1k | 217.2k | 7.35 | 1.43 |
| MLP-Mixer | paper | 69.21 |  | 716.0k | 621.1k | 144.51 | 10.79 |
| DeepConvLSTM | paper | 68.97 |  | 805.2k | 460.8k | 170.66 | 6.91 |
| QuantileNet | ours | 68.86 | 69.70 | 20.9k | 20.8k | 4.60 | 12.39 |
| Mamba | ours | 67.90 | 68.91 | 21.3k | 21.1k | 4.36 | 4.28 |
| No memory | ours | 65.63 | 66.06 | 20.2k | 20.0k | 4.60 | 3.03 |
| Guan-LSTM | paper | 63.32 |  | 822.1k | 803.6k | 0.57 | 6.37 |
| DeepConvShallowLSTM | paper | 61.83 |  | 729.0k | 402.8k | 304.61 | 5.68 |
| SA-HAR | paper | 61.82 |  | 408.2k | 415.5k | 121.96 | 3.34 |
