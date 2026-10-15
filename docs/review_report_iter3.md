# Review report, iteration 3 (stage 08, final)

Inputs:
- the manuscript after iteration 2;
- G7 rerun2 (3 pages in the drafting class, 0 problems);
- `scripts/audit.py --all`, Slurm job 86787: all 10 checks PASS, none skipped.

## Iteration 2 comments: status

- M6, the scope of the intervals: resolved. Limitations now say that the intervals resample test-subject groups and
  treat the networks as fixed, and that S3 shows another population of networks can reverse R.
- m7, the abstract: resolved. The abstract now says comparisons must be paired by split and report the spread.

No new major comment. A reviewer could still ask for:
- a second protocol;
- more seeds;
- networks from different design families with S2-style held-out data.

Each is stated as a limitation, and none can be supplied by rewriting.

## Gates (latest file of each)

| Gate | File | Verdict |
|---|---|---|
| G1 | G1_environment_rerun2.json | PASS (tolerance changed after the first run; declared) |
| G2 | G2_data.json | PASS |
| G3 | G3_baseline.json | PASS |
| G4 | G4_test_seal.json | PASS (seal; template form not applicable) |
| G5 | G5_results_authenticity_rerun2.json | PASS (first run and rerun1 FAIL, kept) |
| G6 | G6_traceability_rerun1.json | PASS |
| G7 | G7_compile_citations_rerun2.json | PASS (drafting class) |

## Mandatory checks that do not pass

| # | Check | State | What would resolve it |
|---|---|---|---|
| 13 | two quoted Figure 6 cells confirmed against the paper | pending: 510/510 equal the authors' snapshot; the PDF is unreachable from the cluster | the WHAR Arena PDF (arXiv:2606.13194), fetched on the PC |
| 16 | every confirmatory run re-executed | partial: 360 of 828 runs (the datasets cached on the cluster); 468 rest on their records | caches for pamap2, uma_fall, w_har (adding them to HAR needs the user's approval) and usc_had (unsorted parser) |
| 20 | label-shuffle control | not run: no training (user's decision of 2026-10-15) | a small training experiment (stage 04, then 05 and 06 again), if the user approves |

The other 19 checks pass (iteration 1 table; check 9 passes since iteration 2).

## Submission formalities (not scientific)

- The author block is a placeholder.
- The class is IEEEtran [journal,9pt] in place of `IEEE_lsens.cls`. After the swap, check the 4-page limit; it is
  3 pages now.
- Scope risk: IEEE Sensors Letters declines AI/ML or data-analysis papers that have no measurable impact on sensor
  systems (Information for Authors; Out-of-Scope Topics, 31 July 2025). The letter is framed around sensing-study
  design (which subjects to record, pairing, channel count). The editor may still judge it out of scope.

## Final verdict

`NOT_READY_FOR_SUBMISSION`

Blocking reasons:
1. Mandatory check 20: no label-shuffle control (disclosed in Limitations; it needs a training experiment, the user's
   decision).
2. Mandatory check 13: the quoted Figure 6 cells are not yet confirmed against the paper itself.
3. Mandatory check 16: only 43 % of the confirmatory runs were re-executed from checkpoints (disclosed).
4. Formalities: the author block and the official IEEE Sensors Letters class.

What does **not** block:
- the evidence for the pre-registered claim: H1 success, verified by re-execution of 360 S2 runs, an independent REML
  and 9/9 negative controls;
- the honesty of the text: the S3 reversal, the S1 balanced-accuracy interval that includes 1, the exclusions and the
  partial re-execution are all stated.
