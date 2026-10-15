# 06_summary.md — Scientific Summary and Traceability

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

Claims to trace:
- the variance shares;
- the component contrasts;
- the data-factor regressions;
- the noise floor.
Mark the 30-dataset results as re-analysis of quoted numbers.

Read and follow `01_system.md` before executing this task.

Check all gates in `results/gates/`. If any FAILed, stop.

**`G5_results_authenticity.json` must PASS.** Any result that `05_verify.md` could not substantiate is excluded from this summary — it does not exist for the purposes of writing up. Read `docs/verification_report.md` before starting.

# ROLE

You are a senior research scientist consolidating the completed work into the knowledge base the manuscript will be written from.

**Do not write the manuscript in this stage.**

---

# INPUT

- `docs/research_plan.md` (including the pre-registration)
- `docs/experiment_record.md`
- `docs/research_checklist.md`
- `src/`, `scripts/`
- `experiments/**`, `results/**`, `figures/`, `tables/`
- all gate files

Use actual experimental results only. Exclude every run marked `synthetic: true`.

---

# SUMMARY STRUCTURE

Write `docs/summary.md` with these sections.

## 1. Research Overview
Topic, objective, motivation, research problem.

## 2. Research Gap and Contribution
Limitations of existing methods; motivation for the approach; contributions.

**Only claim contributions the results support.** Check each claimed contribution against the pre-registration. If the success criterion was not met, say so here — do not restate the intended contribution as an achieved one.

## 3. Proposed Method
Framework, architecture, pipeline, mathematical formulation, implementation details.

Describe **what ran**, including any substitution recorded in `G1_environment.json`.

## 4. Experimental Setup
Datasets, preprocessing, split protocol, training configuration, evaluation protocol, baselines, seeds, statistical tests, hardware/software.

State which baseline numbers were **measured here** and which were **quoted** from published work, whose protocol was adopted, and whether `03_data.md` Step 3b established an evaluation anchor. If it did not, say so — this carries into the paper's Limitations.

## 5. Results and Analysis
Main results, baseline comparison, ablations, robustness, error analysis.

For each result: the actual values with mean ± std and significance, the figure/table reference, and what it means scientifically.

Include results that do not favour the proposed method.

## 6. Scientific Discussion
Why the method works or does not; advantages; limitations; practical implications; future work.

## 7. Pre-registration Outcome
State plainly: was the success criterion met, partially met, or not met? Quote the criterion as written in the plan and the value achieved. Note any deviation from the plan and why.

---

# ARTIFACT INDEX

Write `docs/artifact_index.md`: a table mapping every claim to its evidence.

| Claim | Value | Metric file | Run id(s) | Script | Figure/Table |
|---|---|---|---|---|---|
| Proposed method beats baseline B on test MAE | 87.3 ± 4.1 vs 112.6 ± 6.0 | `results/metrics/test_main.json` | `2026...seed1..5` | `scripts/evaluate.py` | `tables/tab02_main_results.tex` |

Every row must have a real, existing file path.

---

# GATE G5 — TRACEABILITY

Do not eyeball this, and **do not stop at the metrics layer**. Agreement between `summary.md` and a metric file proves nothing: you wrote both. The check must reach down to predictions and checkpoints (`01_integrity.md` § The Three-Layer Rule).

Extend `scripts/audit.py` with `--traceability`, and run `--all`. It must verify:

1. Every numeric claim in `docs/summary.md` matches the value in the file named by `artifact_index.md`.
2. Every referenced path exists and carries a valid provenance record.
3. **Every cited metric recomputes from its prediction file.**
4. **For a random sample of cited runs, the checkpoint re-generates the stored predictions.**
5. Every reported mean ± std recomputes from the stored per-seed values.
6. Every cited run's `data_manifest_sha256` equals the value in `G2_data.json`.
7. No cited run has `synthetic: true`.
7b. Every quoted baseline number in `summary.md` matches `docs/baselines_cited.json`, is marked as quoted, and has `protocol_match: exact` if it appears in the primary comparison.
8. Every figure in `figures/` has matching source data in `results/figure_data/`.
9. **Every run in `experiments/LEDGER.md` either appears in `docs/summary.md` or is listed in `docs/summary.md` § Excluded Runs with a stated reason.**

Write `results/gates/G6_traceability.json` with per-check counts and the verdict. PASS requires zero mismatches on every item.

**If a number mismatches, fix the summary, never the result.** If a claim cannot be traced to an artifact, delete the claim.

Check 9 is the one that catches selective reporting. Add a `## Excluded Runs` section to `summary.md` listing every run that exists but is not reported, with the reason (crashed, superseded, exploratory). "It looked bad" is a reason that must be written down, not a reason to omit.

---

# OUTPUT

- `docs/summary.md`, including § Excluded Runs and § Pre-registration Outcome
- `docs/artifact_index.md`
- `scripts/audit.py` extended with `--traceability`
- `results/gates/G6_traceability.json`

These must let another researcher understand and reproduce the project without reading the manuscript.
