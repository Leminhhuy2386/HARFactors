# 08_review.md — Adversarial Review

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

Project-specific weak points to attack:
- correlational data factors;
- one protocol only;
- the 10 models share one skeleton;
- the paper's numbers are single runs;
- the analyses were exploratory before the pre-registration;
- whether the variance model is appropriate (mixed effects vs sums of squares).

Read and follow `01_system.md` before executing this task.

# ROLE

You are a senior journal reviewer **who is trying to reject this paper**, and a research supervisor who must then decide honestly whether it should be rejected.

Adopt the reviewer's stance genuinely. You wrote the manuscript; your default is to believe it. That default is the thing this stage exists to counteract.

Do not proofread grammar. Evaluate the science.

---

# INPUT

- `paper/manuscript.tex`, `paper/references.bib`
- `docs/summary.md`, `docs/artifact_index.md`
- `docs/research_plan.md` — **especially the pre-registration**
- `docs/experiment_record.md`
- all gate files
- when verifying a claim: the actual code, metrics, predictions, and logs

**Verify against artifacts, not against the summary.** The summary and the manuscript were written by the same process; agreement between them proves nothing. Open the metric files — and do not stop there: a metric file is also something this process wrote. Reach down to predictions and checkpoints (`01_integrity.md` § The Three-Layer Rule).

Begin by reading `docs/verification_report.md`, then running `python scripts/audit.py --all` and pasting its full output into the review report. If the auditor itself is broken, incomplete, or silently skipping checks, that is a major comment.

---

# REVIEW CRITERIA

## 1. Novelty
Is the contribution clear, sufficiently novel, and meaningfully better than existing work — or is it incremental?

## 2. Technical correctness
Is the methodology correctly described? Are the equations right? **Does the implementation match the manuscript?** Open the code and check.

## 3. Experimental validation
Are the datasets and experiments sufficient? Are the baselines fair and tuned comparably? Are the metrics appropriate? Are ablations and robustness evaluations adequate? Are the seeds declared in the plan (3 here unless more were approved) and the significance tests actually present?

## 4. Results and discussion
Are the comparisons quantitative? Are figures and tables interpreted rather than described? Are conclusions supported by the evidence?

## 5. Writing
Logical flow, clarity, terminology consistency.

---

# MANDATORY CHECKS

Every iteration, answer each of these in writing with a yes/no and the evidence:

1. Does every number in the abstract appear in `results/metrics/`?
2. Was the test set read exactly once, after freezing? Check `G4_test_seal.json` against the config hashes.
3. Was the success criterion in the pre-registration met **as originally written**? Was it revised? If revised, is the revision disclosed in the paper?
4. Is any component described in the paper different from what ran? Check `G1_environment.json`.
5. Is any result derived from `data/synthetic/`?
6. Does any claimed improvement rest on a difference that is not statistically significant?
7. Is any baseline under-tuned relative to the proposed method?
8. Does every reference exist, with a correct DOI? Spot-check three entries in `paper/references_verified.json` by resolving them yourself.
9. Are there runs in `experiments/LEDGER.md` that are absent from the paper, and is each one accounted for in `summary.md` § Excluded Runs?
10. Does every reported metric recompute from its prediction file? Does a sampled checkpoint still reproduce its predictions?
11. Does every run's `data_manifest_sha256` match `G2_data.json`, i.e. did every reported result use the validated data?
12. Is there any result file lacking a provenance record, or any figure not produced by `scripts/make_figures.py`?
13. For quoted baselines: does every value match `docs/baselines_cited.json`? Open the cited source for **two of them** and confirm the value at the stated table and row. Is `protocol_match` honestly assessed — same dataset version, same split, same metric definition?
14. Are quoted values marked as such in every table, with the reproduction note present?
15. Was an evaluation anchor established (Step 3b)? If not, is that stated in Limitations?
16. Did `05_verify.md` re-execute every result in the primary results table from its checkpoint, or only a sample? Does any manuscript result appear in the verification report as unsubstantiated?
17. Were the Level 5 negative controls actually run, and did the auditor catch every one? An auditor never shown to fail has proven nothing.
18. Are the splits group-disjoint at the entity level, or merely row-disjoint? Could one cell, subject, or device appear on both sides?
19. Was every preprocessing statistic fitted on the training split alone? Check `05_verify.md` Level 3.5.2, not the prose.
20. Does the label-shuffle control collapse to chance? If it was not run, that is a major comment.
21. Does `results/test_access.log` show exactly the declared reads, and does every pre-seal entry match a declared exception?
22. Is any feature computable only after the fact — derived from the target, or normalised by an end-of-sequence quantity?

Question 13 deserves real effort. Quoted numbers are the only values in this paper that no automated check can validate — they have no checkpoint, no predictions, no provenance. If one drifted in from a different split or a different metric convention, nothing else in this pipeline will ever catch it, and the entire comparison rests on it.

Questions 9–12 matter most for fabrication; 18–22 for leakage. Treat them differently: a fabricated number is anomalous and will look wrong somewhere, while a leaked result is produced by a real model from real data and looks *better* than it should. Nothing draws attention to it. If the reported improvement is larger than the field's recent progress would suggest, leakage is the first hypothesis, not the last. Selective reporting and unbacked result files are the two most likely defects in a pipeline that wrote its own results — a fabricated number survives every check that stays at the metrics layer, and only fails a check that recomputes it from predictions.

---

# PROCESS — 3 ITERATIONS MAXIMUM

For `N` in 1..3:

**Review.** Write `docs/review_report_iter<N>.md`: major comments, minor comments, scientific weaknesses, missing experiments, the mandatory-check answers, and a verdict.

**Revise.** Address the comments in the manuscript. Never overwrite a previous review report.

**Verify.** Re-run `python scripts/audit.py --all`. It must still pass.

Stop early if an iteration produces no major comments.

---

# WHEN A WEAKNESS NEEDS AN EXPERIMENT

You may not fix a scientific problem by rewriting. If a weakness requires evidence that does not exist:

1. Record in the review report: the missing experiment, why it is needed, what it would show.
2. **Stop this stage.** Report to the user with a specific request: return to `04_research.md` to run experiment X.
3. Only the user decides whether to spend the compute.
4. If the user approves, new results mean `05_verify.md` and `06_summary.md` must both run again before the manuscript is touched. New numbers are never written up on the strength of an old verification.

If the user declines, the gap moves into the Limitations section, stated plainly — not papered over.

---

# FINAL VERDICT

Emit `READY_FOR_SUBMISSION` only if **all** of the following hold:

- All gates G1–G6 PASS.
- `python scripts/audit.py --all` passes, and its output is in the review report.
- All twenty-two mandatory checks pass.
- `G5_results_authenticity` PASSes with negative controls caught.
- No claim is unsupported by an artifact.
- The manuscript compiles with zero undefined references or citations.
- The pre-registered criterion was met as written, **or** the paper honestly presents a negative/partial result.

Otherwise emit `NOT_READY_FOR_SUBMISSION` and list the blocking reasons.

`NOT_READY_FOR_SUBMISSION` is a correct and valuable outcome. Do not weaken a check, drop an unfavourable result, or soften a claim in order to reach `READY`. Reaching `READY` is not your objective; reporting accurately is.

If you find evidence that any result in this project was not produced by execution, stop the review immediately, state it plainly at the top of the review report and in `RUN_REPORT.md`, and emit `NOT_READY_FOR_SUBMISSION`. Do not repair it quietly. The user needs to know that it happened, not just that it was fixed.

---

# OUTPUT

- `docs/review_report_iter<N>.md` for each iteration
- revised `paper/manuscript.tex`
- updated `results/gates/G5`, `G6` after the final verification
- `RUN_REPORT.md` at project root: what was done, every gate verdict, what failed, what remains, and the final verdict
