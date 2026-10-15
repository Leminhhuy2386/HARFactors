# Person-level headroom on fixed features (linear probe, no network)

Features: level3+stats (value_probe.py). Macro-F1 %, per test subject then mean over subjects and splits. Each adapted score is paired with the base model on exactly the windows it is scored on (shots are removed from scoring); the gain column is that paired difference.

| Dataset | base | unlabelled (gain) | 1 shot/class (gain) | 3 shots/class (gain) | unlabelled + 1 shot (gain) |
|---|---|---|---|---|---|
| utd_mhad | 70.5 | 71.7 (+1.2) | 77.1 (+10.6) | 64.4 (+11.3) | 75.7 (+9.5) |
| actrectut_gestures | 62.9 | 66.7 (+3.8) | 65.1 (+6.4) | 64.8 (+10.6) | 66.9 (+8.1) |
| uci_har | 91.8 | 95.7 (+3.9) | 92.8 (+1.0) | 93.9 (+2.1) | 96.0 (+4.2) |
| unimib_shar | 54.5 | 53.4 (-1.1) | 56.2 (+1.7) | 57.7 (+3.9) | 54.9 (+0.6) |
| opportunity | 65.3 | 72.4 (+7.0) | 68.2 (+2.9) | 73.7 (+8.4) | 74.6 (+9.3) |
| sad | 99.3 | 99.7 (+0.3) | 99.8 (+0.4) | 99.8 (+0.4) | 99.5 (+0.2) |
| hapt | 78.3 | 76.5 (-1.8) | 70.6 (+0.3) | 50.8 (+0.3) | 70.5 (-0.2) |
| wisdm | 61.2 | 60.1 (-1.1) | 62.4 (+1.2) | 62.8 (+1.7) | 63.2 (+2.0) |
| mean (8) | 73.0 | (+1.5) | (+3.1) | (+4.9) | (+4.2) |
