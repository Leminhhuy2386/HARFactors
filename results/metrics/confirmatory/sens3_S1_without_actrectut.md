# sens3_S1_without_actrectut: sensitivity 3

Set S1, outcome macro_f1 (x 100), 2100 runs, 8 datasets (utd_mhad, uci_har, unimib_shar, opportunity, sad, hapt, dsads, wisdm), 70 splits. Excluded: daphnet (split sizes disagree across runs (pre-registered inclusion rule)), actrectut_gestures (pre-registered sensitivity subset).

REML (default optimizer), variance components (F1 points squared):

| Component | Estimate | 97.5 % interval |
|---|---|---|
| M | 5.510 | [3.696, 7.845] |
| MD | 5.141 | [3.928, 12.152] |
| MP | 6.502 | [0.643, 11.206] |
| P | 67.365 | [22.794, 99.313] |
| e | 16.959 | [12.000, 22.392] |

R = s2_P / (s2_M + s2_MD + s2_MP) = 3.927, 97.5 % interval [1.632, 5.797]
pi_P = 0.664, 97.5 % interval [0.435, 0.722]
Bootstrap: 1000 resamples, 1000 converged, 0 dropped (0.0 %), 4 refitted with Powell; seed 20261015.

