# Held-out confirmation (RESEARCH_PLAN.md §5p)

Complete datasets: 8 of 8 (PAMAP2 (9 splits), UP-Fall (10 splits), UMA-Fall (10 splits), USC-HAD (10 splits), MotionSense (10 splits), WEAR (10 splits), UCA-EHAR (10 splits), W-HAR (10 splits)). Pending: none.

## Per dataset (macro-F1 %, mean over splits; ours: 3-seed mean, paper: one run)

| Model | PAMAP2 | UP-Fall | UMA-Fall | USC-HAD† | MotionSense† | WEAR | UCA-EHAR | W-HAR | Mean |
|---|---|---|---|---|---|---|---|---|---|
| RN + laws | 60.3 | 52.5 | 52.7 | 76.9 | 93.7 | 86.8 | 62.6 | 50.6 | 67.02 |
| RN + statistics | 58.6 | 51.8 | 50.0 | 72.7 | 91.6 | 85.8 | 64.0 | 53.5 | 65.99 |
| RecurrenceNet | 52.6 | 49.8 | 46.6 | 73.3 | 91.2 | 85.7 | 61.7 | 50.1 | 63.87 |
| S4D | 54.5 | 42.8 | 42.9 | 76.2 | 91.9 | 84.4 | 59.1 | 46.3 | 62.27 |
| TinyHAR (paper) | 62.2 | 61.0 | 50.2 | 78.1 | 91.2 | 86.5 | 58.0 | 52.2 | 67.40 |
| CNN-HAR (paper) | 62.1 | 55.0 | 50.0 | 76.2 | 91.5 | 86.9 | 58.7 | 52.7 | 66.63 |
| Triple-Cross-Attn (paper) | 61.9 | 58.7 | 52.4 | 74.3 | 91.3 | 88.5 | 58.1 | 49.8 | 66.89 |
| Random Forest (paper) | 65.1 | 72.8 | 54.2 | 72.7 | 91.5 | 87.3 | 47.6 | 53.1 | 68.02 |
| best paper model | 65.1 Random Forest | 72.8 Random Forest | 56.2 DynamicWHAR | 78.8 MLP-Mixer | 93.5 DynamicWHAR | 89.7 DynamicWHAR | 60.1 DANA | 53.1 Random Forest | |

† parser lists files unsorted: subject numbering may differ from the paper's machine; compared by means only.

## Primary (pre-registered)

1. RN + laws − S4D (3 seeds): +4.75 [+1.85, +7.65], 8–0 of 8, Wilcoxon p 0.008 → better than S4D.
2. RN + laws (seed 0) − paper models, non-inferiority margin −1 on the top three:
   - DynamicWHAR: -1.07 [-5.08, +2.94], 4–4 of 8, Wilcoxon p 0.641
   - Random Forest: -0.64 [-9.40, +8.11], 3–5 of 8, Wilcoxon p 0.742
   - TinyHAR: -0.02 [-3.67, +3.63], 4–4 of 8, Wilcoxon p 0.844 — non-inferiority not shown
   - Triple-Cross-Attn: +0.49 [-2.72, +3.70], 4–4 of 8, Wilcoxon p 0.844 — non-inferiority not shown
   - CNN-HAR: +0.76 [-1.57, +3.08], 5–3 of 8, Wilcoxon p 0.742 — non-inferiority not shown
   - TinierHAR: +1.73 [-1.07, +4.53], 7–1 of 8, Wilcoxon p 0.148
   - Attend+Discriminate: +1.97 [-0.29, +4.22], 6–2 of 8, Wilcoxon p 0.109
   - DANA: +2.24 [-0.05, +4.52], 6–2 of 8, Wilcoxon p 0.055
   - MLP-HAR: +2.27 [-1.56, +6.09], 6–2 of 8, Wilcoxon p 0.195
   - DeepConvLSTM-Attn: +4.28 [+2.02, +6.55], 7–1 of 8, Wilcoxon p 0.016
   - DeepConvLSTM: +4.62 [+2.37, +6.86], 7–1 of 8, Wilcoxon p 0.016
   - SA-HAR: +5.73 [+1.43, +10.03], 7–1 of 8, Wilcoxon p 0.023
   - MLP-Mixer: +6.78 [+1.20, +12.36], 7–1 of 8, Wilcoxon p 0.016
   - Guan-LSTM: +7.94 [+4.70, +11.17], 8–0 of 8, Wilcoxon p 0.008
   - DeepConvShallowLSTM: +9.57 [+3.30, +15.84], 7–1 of 8, Wilcoxon p 0.016
   - SVM: +19.51 [+10.46, +28.55], 8–0 of 8, Wilcoxon p 0.008
   - k-NN: +22.00 [+14.74, +29.25], 8–0 of 8, Wilcoxon p 0.008

## Secondary

- RN + statistics − S4D (3 seeds): +3.73 [+0.17, +7.29], 6–2 of 8, Wilcoxon p 0.055
- RN + statistics (seed 0) − TinyHAR: -1.11 [-5.32, +3.10], 2–6 of 8, Wilcoxon p 0.383 — non-inferiority not shown
- RN + statistics (seed 0) − CNN-HAR: -0.34 [-3.30, +2.63], 2–6 of 8, Wilcoxon p 0.461 — non-inferiority not shown
- RN + statistics (seed 0) − Triple-Cross-Attn: -0.60 [-4.56, +3.35], 2–6 of 8, Wilcoxon p 0.641 — non-inferiority not shown
- RN + laws − RecurrenceNet (3 seeds): +3.15 [+1.00, +5.30], 8–0 of 8, Wilcoxon p 0.008
- RecurrenceNet − S4D (3 seeds): +1.61 [-1.18, +4.40], 5–3 of 8, Wilcoxon p 0.250
- All 18 datasets (includes the 10 development datasets): RN + laws − S4D +3.85 [+2.10, +5.59], 16–2 of 18, Wilcoxon p 0.000
