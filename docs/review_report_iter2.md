# Review report, iteration 2 (stage 08)

Inputs:
- the manuscript as revised after iteration 1 (G7 rerun1: 3 pages, 0 problems);
- G6 rerun1 (239 values);
- `scripts/audit.py --all`, Slurm job 86786.

## Auditor output (job 86786)

```
PASS  --provenance
PASS  --ledger
PASS  --data
PASS  --synthetic
PASS  --recompute
PASS  --reproduce
PASS  --seeds
PASS  --authenticity
PASS  --traceability
PASS  --manuscript
```

No check is skipped any more (M5 of iteration 1 resolved).

## Iteration 1 comments: status

| Comment | Status |
|---|---|
| M1, paired vs unpaired comparisons | resolved: the Discussion separates the absolute level (σ²_P) from paired comparisons (σ²_MP, σ²_e) and points to Table I |
| M2, scope of "comparable" | resolved: defined in the Discussion |
| M3, label-shuffle control not run | disclosed in Limitations; still an open experiment for the user to decide |
| M4, re-execution fractions | resolved: 360 of 828 confirmatory runs; DSADS named |
| M5, --authenticity | resolved: implemented and PASS |
| m1, uncorrected exploratory p | resolved (Methods) |
| m2, evaluation anchor | resolved (Methods: Random Forest 84.69 against 84.69 on UCI-HAR) |
| m3, excluded runs | resolved: `docs/summary.md` § 5 |
| m5, author block and official class | open: needs the authors and `IEEE_lsens.cls` |
| m6, sensor-system rules | resolved: three rules in the Discussion |

## New comments

**M6 (major, fixable by rewriting). The intervals are about test subjects, not about networks.**
- The pre-registered bootstrap resamples splits within datasets, so the 97.5 % interval reflects the sampling of
  test-subject groups.
- S2 has four networks. σ²_M rests on four levels, and they are not a sample of all networks.
- **Required:** say in Limitations that the intervals treat the networks as fixed and that a different network
  population can give a different R. S3, where R′ < 1, is the evidence for this.

**m7 (minor). Abstract, last sentence.**
- "the subjects a wearable study records bound what an accuracy difference can show" predates M1's distinction.
- **Required:** align it with the Discussion. The absolute level depends mostly on who is tested; comparisons must be
  paired by split and must report the spread across subjects.

## Mandatory checks

These are unchanged from iteration 1 except checks 9 and 16:
- **Check 9: now yes.** See `docs/summary.md` § 5.
- **Check 16: still "no".** Only part of the confirmatory set was re-executed. This is disclosed in the paper.
- **Still open:**
  - check 13: two Figure 6 cells to be confirmed against the paper PDF;
  - check 20: the label-shuffle control was not run; disclosed.

## Verdict for this iteration

`NOT_READY_FOR_SUBMISSION`. Revise M6 and m7, then check again in iteration 3.
