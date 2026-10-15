# Accuracy against cost (Pareto)

Mean macro-F1 over the 10 datasets at seed 0 (the paper ran once); cost = geometric mean over the 10 datasets of the measured cost on each dataset's input. ★ = on the Pareto front of all models for that cost. FLOPs count matmul/conv only (LSTM, FFT and scan work is missed), so they are listed without a front.

| Model | Source | F1 seed 0 | F1 3 seeds | Params | Params (utd_mhad) | MFLOPs / window | CPU ms / window |
|---|---|---|---|---|---|---|---|
| RN + statistics | ours | 74.69 | 74.70 | 28.7k ★ | 26.5k | 5.70 | 5.00 ★ |
| RN + laws | ours | 73.77 | 74.42 | 30.2k | 27.6k | 6.42 | 8.93 |
| StyleAlignNet | ours | 73.74 |  | 29.8k | 38.2k | 5.89 | 9.06 |
| Triple-Cross-Attn | paper | 73.48 |  | 410.7k | 455.0k | 34.47 | 4.08 ★ |
| CNN-HAR | paper | 73.18 |  | 133.2k | 72.0k | 20.33 | 5.64 |
| TinyHAR | paper | 73.13 |  | 191.9k | 185.1k | 17.79 | 4.69 |
| DANA | paper | 72.86 |  | 805.2k | 460.8k | 191.48 | 6.15 |
| Attend+Discriminate | paper | 72.74 |  | 640.2k | 374.4k | 328.19 | 25.83 |
| RecurrenceNet | ours | 72.25 | 72.01 | 24.9k ★ | 24.7k | 5.69 | 4.76 |
| LawNet (random groups) | ours | 72.14 |  | 21.4k ★ | 19.0k | 4.22 | 4.29 |
| LawNet | ours | 71.73 | 71.52 | 21.4k | 19.0k | 4.22 | 4.31 |
| TinierHAR | paper | 71.39 |  | 14.5k ★ | 8.0k | 1.23 | 3.48 ★ |
| S4D | ours | 71.25 | 71.29 | 28.0k | 27.7k | 4.79 | 14.78 |
| DeepConvLSTM-Attn | paper | 71.20 |  | 826.6k | 477.4k | 176.35 | 6.16 |
| HysteresisNet | ours | 70.25 | 70.69 | 23.8k | 23.6k | 5.34 | 3.16 ★ |
| Statistics (memoryless) | ours | 69.64 | 70.24 | 20.0k | 17.9k | 3.60 | 0.53 ★ |
| DynamicWHAR | paper | 69.61 |  | 2082.2k | 1127.5k | 66.54 | 3.73 |
| MLP-HAR | paper | 69.31 |  | 424.1k | 217.2k | 7.35 | 1.48 |
| MLP-Mixer | paper | 69.21 |  | 716.0k | 621.1k | 144.51 | 5.45 |
| DeepConvLSTM | paper | 68.97 |  | 805.2k | 460.8k | 170.66 | 6.33 |
| QuantileNet | ours | 68.86 | 69.70 | 20.9k | 20.8k | 4.60 | 10.55 |
| Mamba | ours | 67.90 | 68.91 | 21.3k | 21.1k | 4.36 | 5.88 |
| No memory | ours | 65.63 | 66.06 | 20.2k | 20.0k | 4.60 | 2.72 |
| Guan-LSTM | paper | 63.32 |  | 822.1k | 803.6k | 0.57 | 7.25 |
| DeepConvShallowLSTM | paper | 61.83 |  | 729.0k | 402.8k | 304.61 | 6.02 |
| SA-HAR | paper | 61.82 |  | 408.2k | 415.5k | 121.96 | 3.95 |
