# Review addendum 2, 2026-10-16: label-shuffle control

The user approved a small label-shuffle control (mandatory check 20, `agents/05_verify.md` Level 3.5.3) on 2026-10-16.
Stages 05 and 06 were then re-run before the manuscript was touched, as `agents/08_review.md` requires.

## Experiment

- **Run:** `scripts/label_shuffle_control.py`, ledger run `20261016_055019_label_shuffle_control`.
- **Labels:** recorded as `synthetic=true`, because the labels are manipulated. No output goes into `results/`.
- **Design:** S4D, seed 0, splits 0-2 of UP-Fall and UCA-EHAR, 6 runs on a CPU node, trained with the paper-era
  harness. Only the training labels are permuted (a seeded permutation of the rows), so the class weights are
  unchanged. Validation and test keep their true labels.
- **Tolerance, declared in the script before running:**
  - shuffled-label macro-F1 ≤ chance + 5 points, on validation and on test;
  - the stored true-label run ≥ shuffled + 20 points on validation.
- **Result:** every run collapsed.

  | | Shuffled labels | Chance | True labels (stored runs) |
  |---|---|---|---|
  | Mean test macro-F1 | 4.9 | 8.7 | 54.5 |
  | Mean validation macro-F1 | 6.8 | | |

## Gates re-run

| Gate | File | Verdict |
|---|---|---|
| G5 | G5_results_authenticity_rerun3.json | PASS; adds Level 3.5.3, nothing else changed |
| G6 | G6_traceability_rerun2.json | PASS, 247 values; the tampering control was caught |
| G7 | G7_compile_citations_rerun4.json | PASS: 3 pages, abstract 140 words, 2 figures, IEEE_lsens |

`scripts/audit.py --all` (Slurm job 86807): 10/10 PASS.

## Manuscript

- **Methods:** the control's result is reported (\val markers to G5 rerun3).
- **Limitations:** the sentence on the missing control is removed.

## Mandatory checks still not passing

- **Check 16:** 360 of 828 confirmatory runs were re-executed. pamap2, uma_fall, usc_had and w_har have no cache on
  the cluster.
- **Formalities:**
  - authors, affiliations, subject category, graphical abstract;
  - suggested reviewers, ORCID, EDICS;
  - the scope risk.

## Verdict

Still `NOT_READY_FOR_SUBMISSION`, now for check 16 and the formalities only.
