# Pre-registration: the confirmatory analysis of HARFactors

Written and approved by the user on 2026-10-15, before any confirmatory analysis code exists and before the held-out
runs below have been analysed for this question. Referenced from `docs/research_plan.md` § Pre-registration.

This file is frozen. Its sha256 on 2026-10-15 is recorded in `RUN_REPORT.md` and will be sealed in
`results/gates/G4_test_seal.json`. Any later change is appended under § Deviations, with its date and reason, and the
manuscript discloses it.

## What is confirmatory and what is not

- **One confirmatory hypothesis: H1.** Everything else in the letter is exploratory and is labelled so: claims 2–4 of
  `docs/research_plan.md`, and claim 1 as measured on the 2,460 development runs (it is where H1 came from).
- **No new training** (user's decision, 2026-10-15). H1 is tested on runs that already exist. A test on new runs of
  all 10 models (540 runs on UCA-EHAR, UP-Fall and WEAR) was considered and not chosen.

## Data

| Set | Runs | Role | Already seen |
|---|---|---|---|
| S2 held-out | RecurrenceNet, S4D, RN+stats, RN+laws × 8 datasets (PAMAP2 9 splits; UP-Fall, UMA-Fall, USC-HAD, MotionSense, WEAR, UCA-EHAR, W-HAR 10 each; 79 splits) × seeds 0–2 = 948 runs | **Confirmatory test of H1** | Per-dataset means and paired model differences (`results/metrics/heldout/summary.md`). No variance decomposition of S2 has been computed |
| S1 development | 10 models × 10 datasets (82 splits) × seeds 0–2 = 2,460 runs | Sensitivity | Everything: S1 produced the exploratory finding |
| S3 independent architectures | the 17 WHAR Arena models (14 deep, 3 classical) re-run by the HAR reproduction × UTD-MHAD, UCI-HAR, UniMiB-SHAR, ActRecTut (30 splits) × 1 run = 510 runs | Sensitivity: architectures that are not built on our shared skeleton | Per-dataset means; no variance decomposition |

- Files: `experiments/whar_arena_v1/<model>/<dataset>/[seed-<s>/]split-XX.json` (S1, S2) and
  `experiments/external_har_paper_era/whar_arena_v1/<model>/<dataset>/split-XX.json` (S3).
- All three sets are imported runs, unverified until gate G5.
- The S2 runs were produced for the held-out confirmation of the RILA-Net study. The letter says so, and says that
  their means were known when this test was registered.

## Terms

- **Test-subject group:** the subjects of one split (leave-subjects-out, k = 10 groups). A group is one person only
  when the dataset has at most 10 subjects. A different split also means different training and validation data.
- The manuscript says "test-subject group", not "person".

## H1

> Within a dataset, the test-subject group explains more of the variance of macro-F1 than all model-related sources
> together.

- **Outcome:** the harness macro-F1 of each run (`metrics.macro_f1`, × 100; absent classes count as 0).
- **Model.** Fitted by REML with statsmodels (version 0.15.0, as locked in `uv.lock`), `MixedLM` with one group and
  crossed variance components:

  `F1 = dataset (fixed) + P[dataset:split] + M[model] + MD[model:dataset] + MP[model:dataset:split] + e`

  The residual `e` is the variation between training seeds.
- **Estimand:** R = σ²_P / (σ²_M + σ²_MD + σ²_MP).
- **Interval.**
  - Percentile 97.5 % interval from 1,000 cluster-bootstrap resamples.
  - Each resample draws, within every dataset, as many splits as the dataset has, with replacement.
  - A drawn split keeps all its models and seeds, and each draw gets its own label.
  - Generator: `numpy.random.default_rng(20261015)`.
- **Success:** the lower bound of the 97.5 % interval is above 1.
- **Failure:** the point estimate of R is at most 1. The letter then takes the negative-result branch: the
  test-subject group does not dominate the architecture, and the letter is written up that way.
- **Inconclusive:** R > 1, but the interval contains 1. Reported as inconclusive. No data, subset or specification is
  added to resolve it.
- **Secondary** (reported; does not decide H1):
  - π_P = σ²_P / (σ²_P + σ²_M + σ²_MD + σ²_MP + σ²_e);
  - criterion: the lower bound of its 97.5 % interval is at least 0.40.
- **Level.** 97.5 % was set when two confirmatory hypotheses were planned (Bonferroni over 2). It is kept, as approved,
  now that there is one, which makes the test conservative.
- **Exploratory reference** (S1, sums of squares, `results/metrics/factors/summary.md`):
  - R ≈ 61.3 / (6.4 + 7.2 + 10.9) = 2.5, and π_P ≈ 0.61;
  - REML separates seed noise from the interactions, so its estimates can move in either direction.

## Rules fixed now

- **Inclusion.** A dataset enters only if two conditions hold for every split:
  - `source.train_size`, `source.val_size` and `source.test_size` are equal across all models and seeds, so every
    model was tested on the same subjects and windows;
  - every run has `protocol_hash` e8739ef2c5b36c99.

  Datasets that fail are excluded and listed. This check reads no metric. USC-HAD and MotionSense, whose paper-era
  parsers list files unsorted, need it most.
- **Completeness.** S2 has 948 of 948 runs (counted 2026-10-15). If a run turns out missing or invalid, the fit uses
  the remaining runs, and the count is reported. Nothing is re-run.
- **Fitting.**
  - Each fit uses statsmodels' default optimizers. A fit that does not converge is refitted once with Powell's
    method. A bootstrap resample that still fails is dropped and counted.
  - If more than 5 % of the resamples are dropped, H1 is reported as inconclusive.
  - Variance components estimated at 0 are kept as 0. If the denominator of R is 0, R counts as above 1.
- **Sealing (gate G4).**
  - The analysis code is written and debugged on S1 and on synthetic data only.
  - Before it reads any S2 metric, the sha256 of the code and of this file are written to
    `results/gates/G4_test_seal.json`.
  - The code then runs on S2 once.
- **No re-specification.** H1 gets no other outcome, covariate, transformation, subset or estimator. No threshold
  changes after the results are seen.

## Sensitivity analyses

These are pre-specified and reported whatever they show. They do not decide H1. Every interval uses the bootstrap of
H1.

1. **S2 with balanced accuracy** (`metrics.balanced_accuracy`) instead of macro-F1. It averages recall over the
   classes present in the test split, so it has no per-split ceiling.
2. **S1** with the H1 model, for macro-F1 and for balanced accuracy.
3. **S1 without ActRecTut**, which has only 2 splits.
4. **S3**, which has one run per split, so seed noise and model × split cannot be separated. The model is
   `F1 = dataset (fixed) + P + M + MD + e`, and the conservative ratio R′ = σ²_P / (σ²_M + σ²_MD + σ²_e) is
   reported.

## Exploratory, and labelled so in the letter

- **H2, temporal memory vs none** (+5.95 [+2.79, +9.10] on S1).
  - It is not tested on held-out data, because S2 contains no memoryless model.
  - It is an architecture result, the subject of the RILA-Net paper. Here it describes the population.
- **H3, data factors over the WHAR Arena paper's 30 datasets.**
  - This is a re-analysis of published single runs.
  - The OLS specification in `src/evaluation/factor_analysis.py` (4 standardized predictors, 4 outcomes) is frozen:
    no covariate or outcome is added.
  - Wording: "associated with", never causal.
- **H4, evaluation noise and the fragility of rankings:** descriptive.

## Template fields (`agents/02_plan.md`, Step 5)

| Item | Value |
|---|---|
| Primary metric | Harness macro-F1 per run (absent classes count as 0) |
| Primary baseline | Not applicable: H1 compares variance components, not methods |
| Success criterion | Lower bound of the 97.5 % cluster-bootstrap interval of R above 1, on S2 |
| Failure criterion | Point estimate of R at most 1 on S2 → negative-result branch |
| Secondary claims | π_P lower bound ≥ 0.40; sensitivity analyses 1–4; H2–H4 exploratory |
| Max improvement iterations | 0: no re-specification |

## Deviations

None.
