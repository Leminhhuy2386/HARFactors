# Accuracy against cost (Pareto)

Mean macro-F1 over the 8 held-out datasets at seed 0 (the paper ran once); cost = geometric mean over the 8 held-out datasets of the measured cost on each dataset's input. ★ = on the Pareto front of all models for that cost. FLOPs count matmul/conv only (LSTM, FFT and scan work is missed), so they are listed without a front.

| Model | Source | F1 seed 0 | F1 3 seeds | Params | Params (pamap2) | MFLOPs / window | CPU ms / window |
|---|---|---|---|---|---|---|---|
| DynamicWHAR | paper | 68.45 |  | 2301.8k ★ | 5653.6k | 103.73 | 4.21 ★ |
| TinyHAR | paper | 67.40 |  | 296.4k ★ | 429.9k | 39.39 | 8.34 |
| RN + laws | ours | 67.38 | 67.02 | 29.5k ★ | 35.3k | 10.72 | 10.16 |
| Triple-Cross-Attn | paper | 66.89 |  | 674.0k | 1978.2k | 79.93 | 6.82 |
| CNN-HAR | paper | 66.63 |  | 155.1k | 434.4k | 47.93 | 10.77 |
| RN + statistics | ours | 66.29 | 65.99 | 28.2k ★ | 33.3k | 10.09 | 5.86 |
| TinierHAR | paper | 65.66 |  | 16.3k ★ | 42.8k | 2.65 | 5.46 |
| Attend+Discriminate | paper | 65.42 |  | 679.6k | 1503.1k | 718.49 | 46.79 |
| DANA | paper | 65.15 |  | 862.8k | 1966.3k | 431.51 | 10.53 |
| MLP-HAR | paper | 65.12 |  | 886.7k | 2384.0k | 24.82 | 1.95 ★ |
| RecurrenceNet | ours | 63.77 | 63.87 | 24.7k | 25.7k | 10.08 | 5.63 |
| DeepConvLSTM-Attn | paper | 63.10 |  | 882.0k | 1982.9k | 404.75 | 11.25 |
| DeepConvLSTM | paper | 62.77 |  | 862.8k | 1966.3k | 396.36 | 10.20 |
| S4D | ours | 61.86 | 62.27 | 27.8k | 29.7k | 8.43 | 25.45 |
| SA-HAR | paper | 61.65 |  | 407.6k | 412.0k | 247.28 | 5.92 |
| MLP-Mixer | paper | 60.60 |  | 896.5k | 1547.1k | 322.96 | 7.48 |
| Guan-LSTM | paper | 59.45 |  | 816.8k | 847.1k | 1.25 | 11.34 |
| DeepConvShallowLSTM | paper | 57.81 |  | 795.1k | 1908.3k | 745.08 | 10.89 |
