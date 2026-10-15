# Review report, iteration 1 (stage 08)

Reviewer stance: trying to reject. Inputs: `paper/manuscript.tex` (rendered 2026-10-16), the gates G1-G7,
`docs/verification_report.md`, the artifacts and the code.

## Auditor output (`scripts/audit.py --all`, Slurm job 86782, 2026-10-16)

```
PASS  --provenance
PASS  --ledger
PASS  --data
PASS  --synthetic
PASS  --recompute
PASS  --reproduce
PASS  --seeds
PASS  --traceability
PASS  --manuscript
SKIPPED --authenticity  (not implemented yet: added by a later stage)
```

The auditor silently skips `--authenticity`. This is major comment M5.

## Major comments

**M1. The headline conflates paired and unpaired comparisons.** σ²_P is the variance between test-subject groups.
- It decides how much an *absolute* macro-F1 level depends on who is tested, so it governs comparisons across studies
  or across different splits.
- In a comparison paired by split, it cancels. There the noise is the model × split interaction σ²_MP plus the seed
  term σ²_e, and on S2 both are of the same order as the model terms.
- The Discussion currently says a few splits "cannot carry a gain of one or two points" as though σ²_P were the
  obstacle. That is misleading.
- **Required:** state that σ²_P dominates absolute levels and unpaired comparisons, and that paired comparisons are
  limited by σ²_MP and σ²_e, which do not shrink with more models but do shrink with more splits and seeds.
- Fixable by rewriting. The numbers are already in Table I.

**M2. The claim's scope after S3 must be in the headline, not only in the body.** The abstract already reports the
reversal.
- The title and the conclusion phrase it as "comparable networks". That is adequate only if "comparable" is defined.
- **Required:** define it once. These are networks that share one skeleton and differ in single components, and
  whose dataset means differ by a few points.

**M3. Label-shuffle control not run** (mandatory check 20).
- No network was trained in this project (user's decision, 2026-10-15), and the label-shuffle control needs one
  training run per tested configuration.
- What weakens the concern for this letter:
  - the claim is about variance shares between test-subject groups, not about the level of accuracy;
  - the protocol's validation leak (validation windows from the training subjects) does not reach the test subjects,
    whose windows are never seen;
  - the splits are subject-disjoint by construction (G2, `export_splits.py` asserts it);
  - the normalisation is a train-only fit (G5 L3.5.2).
- It is still a missing experiment. **Stop condition** (agents/08_review.md): either the user approves a small
  label-shuffle experiment (back to stage 04, then 05 and 06), or the gap goes into Limitations, stated plainly.
- Until the user decides, it goes into Limitations.

**M4. Re-execution covers 43 % of the confirmatory runs** (mandatory check 16).
- 360 of 828 S2 runs were re-executed, namely all those whose datasets have a cache on the cluster. pamap2, uma_fall,
  usc_had and w_har (468 runs) rest on their records.
- Within S1, dsads's checkpoints score at chance on the cluster's dsads cache. That cache was built differently (G5
  informative), so dsads rests on its records too.
- The manuscript says only "the remaining runs are verified by their records".
- **Required:** state the fractions and the dsads finding.

**M5. `--authenticity` is not implemented.** The auditor must be runnable without trusting the author. A check that
prints SKIPPED is incomplete.
- **Required:** implement it. It should confirm the latest G5 gate (PASS, every negative control caught, zero
  mismatches) and re-execute a fresh sample of checkpoints.

## Minor comments

- m1. The exploratory p-values (Wilcoxon over datasets, OLS) are uncorrected for multiplicity, and the text does not
  say so. Add it.
- m2. The evaluation anchor (G3: the Random Forest re-run equals Figure 6 to rounding on four datasets) supports the
  claim that the harness is the benchmark's. Mention it in one clause.
- m3. "Excluded runs" are not accounted for in `docs/summary.md` (mandatory check 9). The ledger holds 4,950
  imported runs, of which the letter analyses S1 (2,460) and S2 (948). The others are Edge_SSM's development history
  (other models and seeds 3-4 on UTD-MHAD). Add a § Excluded runs with counts.
- m4. Fig. 2b labels three datasets by name. The OLS annotation repeats numbers that are in the text, which is fine.
- m5. The author block is a placeholder. The class is a drafting substitute (IEEEtran 9 pt for `IEEE_lsens.cls`), and
  at 3 pages the length is under the limit but not final. Both block submission; neither is a scientific issue.
- m6. Scope risk: IEEE Sensors Letters declines "data analysis" papers without impact on sensor systems. The framing
  (number of subjects to record, channel count) is the right one. The Discussion could state the practical rule for
  sensor studies more concretely, without new numbers.

## Mandatory checks

| # | Check | Answer | Evidence |
|---|---|---|---|
| 1 | Every abstract number in results/metrics | yes | 7 datasets, 828 runs, 82 %, 8.19 [2.85, 16.90]: `results/metrics/confirmatory/H1.json`; "17", "three", "four" are design constants |
| 2 | Test set (S2) read once, after freezing | yes, with declared exceptions | `results/test_access.log`: S2 read by H1 and sens1 after the seal (16:49:41 > 16:49:10); G4 hashes are checked by run_confirmatory; earlier heldout_analysis reads of S2 means are declared in G4; the G5 re-reads are logged |
| 3 | Pre-registered criterion met as written | yes, not revised | lower bound 2.847 > 1; the 97.5 % level was kept |
| 4 | A component different from what ran | no (the drafting LaTeX class only) | G1 rerun2 substitutes |
| 5 | Results from data/synthetic | no | audit --synthetic |
| 6 | An improvement resting on a non-significant difference | no confirmatory one; exploratory contrasts carry uncorrected p (m1) | contrasts_robustness |
| 7 | An under-tuned baseline | not applicable (no method claim); S3 uses the benchmark's published configurations | G3 |
| 8 | References exist (3 spot checks) | yes | Unpaywall (10.1145/2750858.2807551), IDEAS/RePEc (10.1111/j.1467-9868.2007.00593.x), KIT library (10.1145/3714394.3756254), resolved 2026-10-16 by web search: doi.org and Crossref are blocked from the cluster |
| 9 | Ledger runs absent from the paper accounted for | **no** (m3) | to add |
| 10 | Metrics recompute from predictions; a checkpoint reproduces | yes, for the re-executed set | audit --recompute, --reproduce; G5 rerun2 |
| 11 | data_manifest_sha256 equals G2 | yes | audit --data |
| 12 | Result files without provenance; figures outside make_figures | none in results/; figures/pareto*.svg are drawn by src/evaluation/pareto.py and not used in the paper | audit --provenance |
| 13 | Quoted values match baselines_cited; two checked against the paper | **partly**: 510/510 match the authors' own snapshot of Figure 6; the paper PDF is not yet reachable from the cluster (requested through the PC) | docs/baselines_cited.json |
| 14 | Quoted values marked | yes: Fig. 2b's caption says "published single runs"; no table holds quoted values | manuscript |
| 15 | Evaluation anchor | yes, G3 PASS; not yet stated in the paper (m2) | G3 |
| 16 | Every primary-table result re-executed | **no**: 360 of 828 S2 runs, plus 48 S1 runs; the rest on records (M4) | G5 rerun2 |
| 17 | Negative controls run and caught | yes, 9/9, plus model-label permutation | G5 rerun2 |
| 18 | Splits group-disjoint at the subject level | yes | export_splits.py assertions; G5 control |
| 19 | Preprocessing fitted on train only | yes (G5 L3.5.2: train mean about 1e-7, sd 1) | G5 rerun2 |
| 20 | Label-shuffle control | **not run** (M3) | user decision: no training |
| 21 | test_access.log shows exactly the declared reads | yes | G5 access-log audit |
| 22 | A feature computable only after the fact | no: raw windows; the analysis factors are design labels | research_plan § leakage |

## Verdict for this iteration

`NOT_READY_FOR_SUBMISSION`. Blocking:
- M3, the label-shuffle control: it needs the user's decision, otherwise it is disclosed in Limitations;
- check 13: two Figure 6 cells still have to be confirmed against the paper PDF;
- m5, the author block and the official class.

M1, M2, M4, M5, m1-m3 and m6 are fixable now by rewriting and by completing the auditor.
