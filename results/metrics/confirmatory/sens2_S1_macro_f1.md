# sens2_S1_macro_f1: sensitivity 2

Set S1, outcome macro_f1 (x 100), 2160 runs, 9 datasets (utd_mhad, uci_har, unimib_shar, actrectut_gestures, opportunity, sad, hapt, dsads, wisdm), 72 splits. Excluded: daphnet (split sizes disagree across runs (pre-registered inclusion rule)).

REML (default optimizer), variance components (F1 points squared):

| Component | Estimate | 97.5 % interval |
|---|---|---|
| M | 5.917 | [4.177, 8.637] |
| MD | 7.769 | [6.614, 15.189] |
| MP | 6.667 | [0.597, 11.066] |
| P | 66.247 | [23.308, 93.448] |
| e | 17.606 | [12.339, 23.409] |

R = s2_P / (s2_M + s2_MD + s2_MP) = 3.255, 97.5 % interval [1.344, 4.473]
pi_P = 0.636, 97.5 % interval [0.414, 0.699]
Bootstrap: 1000 resamples, 1000 converged, 0 dropped (0.0 %), 1 refitted with Powell; seed 20261015.

