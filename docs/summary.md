# Summary of results (stage 06)

Every number below is printed exactly as `docs/summary_values.json` derives it from its artifact; `scripts/audit.py --traceability` re-derives each one. The manuscript may print no other number.

## 1. Confirmatory test (pre-registered, docs/preregistration.md)

- H1, S2 macro-F1: 828 runs, 69 splits of 7 held-out datasets (motion_sense excluded by the inclusion rule).
- Variance components (F1 points²): test-subject group 127.7, model 4.8, model × dataset 4.0, model × split 6.9, seed 11.7.
- R = 8.19 (97.5 % interval 2.85–16.90); pi_P = 82 % (64–89 %).
- Pre-registered decision: **success**; secondary criterion (pi_P lower bound >= 40 %) met.

## 2. Sensitivity analyses (pre-registered; they do not decide H1)

| Analysis | runs | R [97.5 % interval] | pi_P % [interval] |
|---|---|---|---|
| H1, confirmatory (S2, macro-F1) | 828 | 8.19 [2.85, 16.90] | 82 [64, 89] |
| S2, balanced accuracy | 828 | 4.09 [2.05, 6.03] | 70 [53, 76] |
| S1, macro-F1 | 2160 | 3.26 [1.34, 4.47] | 64 [41, 70] |
| S1, balanced accuracy | 2160 | 1.89 [0.69, 2.29] | 52 [30, 57] |
| S1 without ActRecTut | 2100 | 3.93 [1.63, 5.80] | 66 [43, 72] |
| S3, 17 architectures | 510 | 0.21 [0.06, 0.36] | 17 [6, 26] |

S3's R is the conservative R′ = s2_P / (s2_M + s2_MD + s2_e) (one run per split).

## 3. Exploratory findings (labelled exploratory in the letter)

- Sums-of-squares shares within a dataset on S1 (the finding that motivated H1): test-subject group 61.3 %, seed 14.2 %, model × split 10.9 %, model × dataset 7.2 %, model 6.4 %.
- Component contrasts over the 9 datasets of S1 whose runs pair by person (10-dataset values in summary_values.json):

| Contrast | mean [95 % CI] | won | p |
|---|---|---|---|
| RecurrenceNet − no memory | +6.43 [+3.07, +9.80] | 9/9 | = 0.004 |
| S4D − no memory | +5.79 [+2.30, +9.27] | 9/9 | = 0.004 |
| HysteresisNet − no memory | +5.24 [+2.74, +7.74] | 9/9 | = 0.004 |
| QuantileNet − no memory | +4.09 [+0.83, +7.34] | 8/9 | = 0.008 |
| Mamba − no memory | +3.33 [+0.45, +6.22] | 8/9 | = 0.012 |
| RecurrenceNet − S4D | +0.65 [-1.11, +2.40] | 6/9 | = 0.301 |
| HysteresisNet − S4D | -0.55 [-2.25, +1.15] | 5/9 | = 0.820 |
| QuantileNet − S4D | -1.70 [-3.82, +0.42] | 2/9 | = 0.098 |
| Mamba − S4D | -2.45 [-4.40, -0.50] | 1/9 | = 0.020 |

- Data factors over the WHAR Arena paper's 30 datasets (re-analysis of published single runs; correlational):
  - RF − best deep vs channels: +7.1 points per sd (p < 0.001), R² 0.55;
  - best − median deep vs subjects: -1.5 per sd (p = 0.004), R² 0.39;
  - median deep F1 (difficulty): R² 0.03.

## 4. Anchors and verification

- G3: Random Forest re-run vs Figure 6: utd_mhad 51.20 vs 51.20, uci_har 84.69 vs 84.69, unimib_shar 44.49 vs 44.49, actrectut_gestures 41.89 vs 41.89.
- G5 PASS: 408 runs re-executed from checkpoints (360 of them in S2), 373 equal to the stored metrics exactly.
- Informative: 348 S2 runs re-executed on caches rebuilt on another machine (pamap2, uma_fall, w_har): 179 exact, 36 outside the gating tolerance; dataset means within 0.07 F1 points; largest single-run difference 3.25.
- Label-shuffle control (S4D, 6 runs on UP-Fall and UCA-EHAR, permuted training labels): test macro-F1 4.9 against chance 8.7 and 54.5 with true labels; validation 6.8 vs 8.7 vs 85.9; collapsed on every run: True.

## 5. Excluded runs (agents/08_review.md, check 9)

- The ledger imports 4950 WHAR-harness runs. The letter analyses S1 (2460 runs) and S2 (948 runs), and S3 (510 re-runs of the benchmark's models, in experiments/external_har_paper_era).
- The other imported runs are Edge_SSM's development history: models outside the population, and training seeds 3-4 on UTD-MHAD for five models. The letter does not use them; some enter the imported exploratory aggregates (results/metrics/factors: the kill-set and sensor-grouping contrasts; synthesis, p5, seeds).
- Excluded by the pre-registered inclusion rule: daphnet in S1 (S1 analyses use 2160 of 2460 runs) and motion_sense in S2 (H1 uses 828 of 948 runs).
- Analysis runs in the ledger: development runs (none wrote to results/), crashed verification runs (recorded with status=crashed), and the G5 runs whose gates are kept (G5 FAIL, rerun1 FAIL, rerun2 PASS).

## 6. Limitations carried to the manuscript

- One protocol (WHAR Arena v1); 3 training seeds (template asks >= 5; user's decision); the 10 population models share one skeleton; the 30-dataset part re-analyses published single runs; data factors are correlational.
- Re-execution covers the datasets with a cache on the cluster; runs without a checkpoint and datasets without a cache are verified by their stored records only (docs/verification_report.md).
- The S2 per-dataset means were known before the pre-registration; no variance component of S2 was.
- The manuscript was drafted with IEEEtran [journal,9pt] in place of IEEE_lsens.cls.

## 7. Pre-registration outcome

H1: success. No deviation from docs/preregistration.md in the confirmatory analysis.
