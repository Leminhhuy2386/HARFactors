# 05_verify.md — Results Authenticity Verification

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

**Where per-split re-execution is possible (audit of 2026-10-15).** Predictions can only be regenerated on the
same windows the run used. Which datasets that applies to depends on where the stored run's cache was built and on
whether the paper-era parser sorts its files:

| Datasets | Status on this PC |
|---|---|
| utd_mhad, uci_har, unimib_shar, actrectut_gestures, opportunity, sad, hapt, wisdm, up_fall, uca_ehar, wear | cache in `D:\Projects\HAR\datasets_paper_era`, split sizes equal to the stored runs: re-executable now |
| pamap2, uma_fall, w_har | no cache here (built on cloud VMs). Their parsers sort files (or read one file), so a PC build should give the same splits: build with `scripts/prepare_data.py` (adds folders to HAR; ask the user first), then check split sizes against the stored runs |
| usc_had, motion_sense, dsads, daphnet | parsers list files unsorted (`os.listdir` / `os.walk`). The stored runs used Linux-built caches; the PC caches of dsads and daphnet number subjects differently, so per-split re-execution must run on a Linux VM (`scripts/cloud/`) that rebuilds the cache as the original runs did. Until then, these datasets are verified by dataset means only, and the verification report says so |

Six runs have no checkpoint anywhere (S4D on ActRecTut ×2; Mamba on ActRecTut and UniMiB ×4: trained before
checkpoints were saved). They can only be substantiated by retraining with the same seed.

Verification has two parts.
1. **Re-derive the aggregates.** Recompute the variance decomposition and the contrasts from the 2,460 per-run files,
   in a second implementation (e.g. statsmodels), and compare.
2. **Spot-check that the runs are real.** The checkpoints of the 10 population models are in
   `experiments/checkpoints/` (copied from Edge_SSM on 2026-10-15; see the ledger note). For a random sample of runs
   on the datasets that are re-executable per split (table above), regenerate test predictions from the checkpoint
   (`scripts/evaluate.py`, built at stage 04) and recompute macro-F1 against the stored run json.
- Negative control specific to this project: permute the model labels within each (dataset, person, seed) cell.
  The model share must fall to about 0.

Read and follow `01_system.md` and `01_integrity.md` before executing this task.

Run this stage **after all experiments are complete and before anything is written up.**

# PURPOSE

Prove, by execution, that every result in this project came from a model that actually ran.

Not that the numbers are good. Not that they agree with each other. That they are **real**: the checkpoints exist and contain trained weights, the predictions came out of those checkpoints, the metrics were computed from those predictions, and every figure and table was generated from those files.

Consistency is not authenticity. A fabricated set of results can be perfectly self-consistent. Every check in this stage must therefore **re-execute something** and compare — never read two files and compare them to each other.

# ROLE

You are a forensic auditor examining someone else's project. You did not run these experiments and you do not trust them. Your job is to find results that cannot be substantiated.

Finding one is a success. Finding none, after checks that could have found one, is also a success. Reporting "all clear" without running checks that could have failed is the only failure mode.

---

# LEVEL 1 — EXISTENCE

For every run in `experiments/LEDGER.md` and every result cited anywhere:

- The `experiments/<run_id>/` directory exists, with `checkpoints/`, `predictions/`, `metrics/`, `logs/`, `config.yaml`.
- Every file is non-empty and of plausible size. A 2 KB "checkpoint" for a 4M-parameter model is not a checkpoint.
- Every result file has a valid provenance record whose hashes match the file.
- Every figure in `figures/` has source data in `results/figure_data/`; every table in `tables/` has source data in `results/table_data/`.

Missing files are not an inconvenience to work around. Record them; they become FAIL items.

---

# LEVEL 2 — WELL-FORMEDNESS

## Checkpoints

- Loads without error; contains a state dict.
- Parameter count and layer shapes match the architecture in `config.yaml`.
- No NaN or Inf in any tensor.
- **Weights differ from initialization.** Instantiate the model fresh with the recorded seed and compare. A checkpoint equal to its initial weights means training never happened — the most direct evidence of a fabricated run.
- Optimizer state or epoch counter present and consistent with the training history.

## Predictions

- Row count equals the split size **exactly**, and sample ids equal the split membership in `data/processed/splits.json` — no extras, none missing.
- Numeric dtype, no NaN, values in a physically plausible range for the target.
- **Not identical to ground truth.** Perfect or near-perfect agreement means leakage or copied labels, not a good model.
- Not constant, not degenerate. Report the prediction distribution.
- Prediction files from different seeds must differ from each other. Identical files across seeds mean the seed was never varied.

## Logs

- Present, with a plausible number of lines.
- Timestamps monotonic; total span consistent with `duration_sec` in the provenance record.
- Loss values present and **actually changing** across epochs. A log with a flat or absent loss curve did not train anything.

---

# LEVEL 3 — RE-EXECUTION

The decisive level. Levels 1–2 can be satisfied by well-crafted fake files. This one cannot.

## 3.1 Predictions from checkpoints

For a random sample of runs, and for **every** run cited in the primary results table:

1. Load the checkpoint.
2. Run `scripts/evaluate.py` on the same split.
3. Compare the output against the stored prediction file.

Must match exactly, or within the documented inference-nondeterminism tolerance. A mismatch means the stored predictions did not come from that checkpoint.

## 3.2 Metrics from predictions

Recompute every reported metric from its prediction file using `src/evaluation/`. Compare against the stored metric file **and** against the value printed in every table.

## 3.3 Aggregates from per-seed values

Recompute every reported mean, std, and significance test from the stored per-seed values.

## 3.4 Figures and tables from results

Re-run `scripts/make_figures.py` into a temporary directory. Then:

- Diff every regenerated `results/figure_data/*` against the stored version. Matching is what proves a figure plots real results.
- Diff every regenerated `tables/*.csv` against the stored version.
- Do **not** byte-compare rendered PNG/PDF files — rendering embeds timestamps and font state. Compare the underlying data.
- Any figure or table that `make_figures.py` does not regenerate was produced by some other means. Flag it.

---

# LEVEL 3.5 — LEAKAGE DETECTION

Fabrication makes results look *wrong* under scrutiny. Leakage makes them look **better**. Nothing in Levels 1-3 will catch it, because a leaked result is genuinely produced by a real model from real data — it is just answering an easier question than the paper claims.

So leakage needs its own active tests. Run all of them.

## 3.5.1 Split integrity

- No entity id (cell, patient, subject, device) appears in more than one split.
- No row hash appears in more than one split.
- No near-duplicate pairs across splits, where the data admits a similarity measure.
- Split sizes and membership match `data/processed/splits.json` exactly.

## 3.5.2 Preprocessor fitted on train only

Refit the preprocessor from the training indices alone and compare against `data/processed/preprocessor.pkl`. They must be identical.

If the stored transform does not match a train-only fit, it saw validation or test data, and **every result in the project is invalid** — not merely optimistic.

## 3.5.3 Label-shuffle control

The strongest general leakage detector, and worth one training run.

Train the proposed model on data with the **labels randomly permuted within the training split**, evaluate on the untouched validation split, and check that performance collapses to chance.

- If it collapses: the pipeline is learning from features, as intended.
- If it does not collapse: information about the labels is reaching the model through something other than the labels — a leaked feature, an overlapping split, a preprocessor fitted on everything. Find it before anything else proceeds.

Record the shuffled-label score next to the real score in the verification report. It is the clearest single line of evidence that the results are real learning.

## 3.5.4 Single-feature check

Evaluate each feature alone against the target. Any feature that predicts the target nearly perfectly on its own is a leak until proven otherwise — usually a quantity derived from the label or normalised by an end-of-sequence value.

## 3.5.5 Test-access audit

Read `results/test_access.log`:

- Count the reads. Compare with what `G4_test_seal.json` declares.
- Every entry before the seal timestamp must match a declared exception (the anchor baseline, and nothing else).
- Every entry after the seal must belong to the single final evaluation pass or to this verification stage.

An undeclared pre-seal read means the test split influenced a design decision. Report it. Do not re-split, do not re-seal, and do not quietly drop the log.

## 3.5.6 Temporal and availability check

For each feature, confirm against `docs/research_plan.md` that it is computable at prediction time: nothing derived from the target, nothing normalised by a whole-sequence quantity known only at the end, no window extending past the prediction point.

---

# LEVEL 4 — ANOMALY SCAN

Signals for human attention, not verdicts:

- Standard deviation across seeds that is exactly zero or implausibly small.
- Metric values that are suspiciously round.
- Results identical across configurations that should differ.
- **Timeline incoherence**: a checkpoint modified before its training log started; predictions older than the checkpoint that produced them; a figure older than the data it plots.
- A run in the ledger with no directory, or a directory not in the ledger.
- Any result whose `data_manifest_sha256` differs from `G2_data.json`.
- Any file under `results/`, `figures/`, or `tables/` descending from `data/synthetic/`.
- An improvement visible in the aggregate but in no individual seed.

---

# LEVEL 5 — NEGATIVE CONTROLS

`scripts/audit.py` was written by the same process that produced these results. A passing auditor proves nothing until the auditor is proven capable of failing.

On **copies** in a temporary directory — never on the real artifacts — construct each of these and confirm the auditor catches it:

| Tampering | Check that must FAIL |
|---|---|
| Alter one value in a prediction file | `--recompute` |
| Replace a checkpoint with a freshly initialized model | `--reproduce` |
| Edit a number in a metric file | `--traceability` |
| Delete a provenance record | `--provenance` |
| Remove a run directory listed in the ledger | `--ledger` |
| Hand-edit a value in `results/table_data/` | Level 3.4 |
| Corrupt a metric file, then regenerate figures | Level 3.4 — the regenerated figure data **must change**. If it does not, `make_figures.py` is not reading results at all; it is emitting hardcoded values, and every figure in the paper is decorative. |
| Move one entity from train into test | Level 3.5.1 |
| Refit the preprocessor on the full dataset | Level 3.5.2 |

**If any tampered input passes, the auditor is broken and every PASS it has emitted is worthless.** Fix the auditor and re-run from Level 1.

Record each negative control and its outcome in the verification report. A verification report without negative controls is an assertion, not evidence.

---

# GATE G5 — RESULTS AUTHENTICITY

Write `results/gates/G5_results_authenticity.json`:

- `check`: "every reported result re-executes from its checkpoint and source files, and no leakage path is open"
- `observed_value`: per-level counts — files checked, runs re-executed, mismatches, anomalies flagged, negative controls caught
- `tolerance`: "zero Level 1-3 failures; zero Level 3.5 leakage findings; all negative controls caught; every Level 4 anomaly explained"
- `verdict`: PASS / FAIL

**A FAIL is not fixed by adjusting this stage.** If a result cannot be substantiated:

1. Do not delete it quietly, and do not regenerate it just to make the check pass.
2. Record exactly what could not be substantiated, and why.
3. Report to the user. The decision is theirs.
4. The result may not appear in `06_summary.md` or in the manuscript unless an actual run produces it.

---

# OUTPUT

- `docs/verification_report.md` — written for the user, not for you. Someone who did not follow this run must be able to decide in a few minutes whether to trust these results. Include what was checked, what was re-executed, what matched, every anomaly, every negative control, and a plain-language verdict.
- `scripts/audit.py` extended with `--authenticity` and the negative-control harness
- `results/gates/G5_results_authenticity.json`

Open the report with the verdict in one sentence, plainly, without hedging:

> All 47 reported results were re-executed from their checkpoints and match. Splits are group-disjoint, the preprocessor is a train-only fit, and the label-shuffle control collapses to chance (MAE 302 vs 87 for the real model). The test split was read twice: once for the declared anchor baseline, once after the seal. 9 negative controls were caught by the auditor. No anomalies outstanding.

or

> 3 of 47 reported results could not be re-executed from their checkpoints (listed below). These results must not be used.
