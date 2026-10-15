# sens2_S1_balanced_accuracy: sensitivity 2

Set S1, outcome balanced_accuracy (x 100), 2160 runs, 9 datasets (utd_mhad, uci_har, unimib_shar, actrectut_gestures, opportunity, sad, hapt, dsads, wisdm), 72 splits. Excluded: daphnet (split sizes disagree across runs (pre-registered inclusion rule)).

REML (default optimizer), variance components (F1 points squared):

| Component | Estimate | 97.5 % interval |
|---|---|---|
| M | 5.548 | [3.726, 8.139] |
| MD | 9.629 | [8.960, 19.207] |
| MP | 5.545 | [0.539, 8.299] |
| P | 39.213 | [14.336, 53.982] |
| e | 15.275 | [10.863, 19.928] |

R = s2_P / (s2_M + s2_MD + s2_MP) = 1.892, 97.5 % interval [0.689, 2.294]
pi_P = 0.521, 97.5 % interval [0.298, 0.569]
Bootstrap: 1000 resamples, 1000 converged, 0 dropped (0.0 %), 2 refitted with Powell; seed 20261015.

