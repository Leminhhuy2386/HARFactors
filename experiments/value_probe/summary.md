# Round 13 pre-test: value-domain features vs per-channel statistics (linear probe, no network)

Macro-F1 %, mean over splits. Probe: logistic regression on standardized per-channel features, fitted on the training subjects. Deep models: seed 0 of the stored runs, for context.

| Dataset | stats | level | level3 | level+stats | level3+stats | ET level3 | ET level3+stats | recstatnet | RecurrenceNet | statnet |
|---|---|---|---|---|---|---|---|---|---|---|
| utd_mhad | 63.1 | 63.4 | 71.0 | 65.5 | 70.5 | 70.7 | 72.9 | 71.6 | 74.6 | 69.6 |
| actrectut_gestures | 65.7 | 58.8 | 62.0 | 61.3 | 62.9 | 63.3 | 68.3 | 63.6 | 35.1 | 47.2 |
| uci_har | 85.8 | 92.9 | 92.3 | 92.8 | 92.3 | 92.0 | 91.9 | 96.4 | 96.7 | 92.1 |
| unimib_shar | 35.7 | 51.8 | 52.5 | 55.5 | 55.4 | 56.9 | 57.8 | 62.1 | 62.0 | 53.9 |
| opportunity | 62.4 | 67.8 | 65.5 | 67.7 | 65.3 | 72.4 | 72.3 | 62.8 | 64.9 | 62.4 |
| sad | 96.7 | 99.2 | 99.2 | 99.3 | 99.3 | 99.2 | 99.2 | 96.6 | 92.0 | 93.0 |
| hapt | 78.2 | 79.0 | 83.5 | 80.6 | 83.3 | 82.7 | 83.0 | 84.0 | 83.2 | 80.5 |
| wisdm | 71.8 | 73.0 | 71.8 | 74.0 | 73.4 | 76.6 | 76.7 | 82.2 | 84.3 | 71.6 |
| mean (8) | 69.9 | 73.2 | 74.7 | 74.6 | 75.3 | 76.7 | 77.8 | | | |
