# 01_integrity.md — Provenance Contract

Standing rules. Read alongside `01_system.md` at the start of **every** stage.

`01_system.md` says do not fabricate. This file defines the mechanisms that make fabrication *detectable*, and — more importantly — defines the legitimate way out when a result cannot be produced.

---

# WHY THIS FILE EXISTS

Fabrication in an autonomous research run is almost never a decision to lie. It is what happens when a stage must produce an output, the real output is unavailable, and no other exit exists.

So this file does two things:

1. Makes fabricated artifacts fail an automated check.
2. Guarantees an honest exit, so that "I could not produce this" is always available and always acceptable.

**Reporting that a result does not exist is a successful outcome of this stage.** It is not a failure, it will not be treated as one, and you should never prefer a plausible number to an honest gap.

---

# THE THREE-LAYER RULE

Results exist in three layers. Trust flows upward only.

```
checkpoint  →  predictions  →  metrics  →  summary  →  manuscript
   (hard to fake)                          (trivial to fake)
```

**Never anchor a check at the metrics layer.** A metric file is a handful of numbers; agreement between a summary and a metric file proves nothing if both were written by the same process.

Every check must reach down to predictions or checkpoints:

- Metrics must be **recomputable** from the prediction file.
- Predictions must be **reproducible** from the checkpoint.

A result whose checkpoint no longer reproduces its predictions is void.

---

# PROVENANCE RECORD

Every **computed result** carries a sibling `<name>.provenance.json`: everything under `results/metrics/`, `results/predictions/`, `results/figure_data/`, `results/table_data/`, and under `experiments/*/metrics/` and `experiments/*/predictions/`.

Gate files, ledger, docs, configs, and logs are exempt — they are records, not computed results.

```json
{
  "produced_by": "scripts/evaluate.py",
  "command": "python scripts/evaluate.py --run_id 20260903_140500_prop_seed1 --split test",
  "run_id": "20260903_140500_prop_seed1",
  "timestamp": "2026-09-03T14:22:31",
  "duration_sec": 44.2,
  "git_commit": "a3f91c2",
  "config_sha256": "...",
  "data_manifest_sha256": "...",
  "input_files": {"experiments/.../checkpoints/best.pt": "sha256:...",
                  "data/processed/splits.json": "sha256:..."},
  "output_sha256": "...",
  "synthetic": false
}
```

**A result file without a valid provenance record does not exist.** `06_summary.md` must ignore it and `08_review.md` must flag it.

`data_manifest_sha256` must equal the value recorded in `G2_data.json`. This is what prevents a run from silently training on data that is not the validated dataset.

---

# QUOTED NUMBERS ARE NOT MEASUREMENTS

Baseline values taken from other groups' papers are a separate class of number with separate rules. They have no checkpoint, no prediction file, and no provenance record, so **no automated check can ever validate them**. Transcription discipline is the only safeguard.

- Every quoted number lives in `docs/baselines_cited.json` with a precise source location (table and row, not just a citation key), the date read, and what it was read from.
- Read it from the actual paper. A number produced from memory is fabricated, however confident you are, and it is undetectable downstream — which is exactly why it must not happen.
- Never write a quoted number into `results/metrics/`, a provenance record, or any file the auditor treats as a measurement.
- In every table and in the text, quoted values must be **visually distinguished** from values this project measured, with a note stating they are reproduced from the cited sources.
- A quoted number whose protocol is not identical to yours may not be placed in the primary comparison table.

The failure mode here is not lying. It is a number drifting in from a related setting — a different split, a different dataset variant, a different metric convention — and being compared against your own as if it were commensurable.

---

# APPEND-ONLY RUN LEDGER

`experiments/LEDGER.md` — one line appended when a run starts, one when it ends:

```
20260903_140500_prop_seed1 | START | config a3f9.. | data 7c21.. | synthetic=false
20260903_140500_prop_seed1 | END   | val_mae=91.2 | 00:42:11 | status=completed
20260903_141900_prop_seed2 | END   | -            | 00:03:02 | status=crashed(OOM)
```

Rules:

- Append only. Never edit or delete a line, including for crashed, aborted, or embarrassing runs.
- Never delete an `experiments/<run_id>/` directory.
- Every directory in `experiments/` must appear in the ledger, and every ledger entry must have a directory.

The ledger is what makes **selective reporting** visible. A run that happened and is not in the paper is a decision that must be justified, not a run that can quietly disappear.

---

# FORBIDDEN OPERATIONS

Concretely, never:

1. Write a value into a metric, result, table, or figure-data file by any means other than executing a script that computed it.
2. Use `random`, `numpy.random`, `torch.randn` or any generator to stand in for data or predictions anywhere outside `tests/` and `data/synthetic/`.
3. Report a result from a run that is not in the ledger.
4. Delete, edit, or omit an experiment because its result is unfavourable.
5. Change a metric definition, split, or preprocessing step after `03_data.md` freezes it.
6. Round, reformat, or "clean up" a number in a way that changes it.
7. Report a mean over seeds when some seeds crashed, without stating how many completed.
8. Cite a reference whose existence you have not resolved.
8b. Quote a baseline number you have not read from its source, or compare a quoted number against your own under a different protocol without saying so.
9. Describe in the manuscript any component, hyperparameter, or dataset other than the one that actually ran.
10. Present a figure that was not produced by `scripts/make_figures.py` from files in `results/`.

---

# IMPORTED RESULTS (HARFactors)

Everything in `experiments/` and `results/metrics/` on 2026-10-14 was produced in Edge_SSM before this contract
existed. Until `05_verify.md` has regenerated it from checkpoints (or re-measured it, for board records), treat it as
**imported and unverified**:
- it may be summarized for planning;
- it may not enter the manuscript before G5.

Each result json carries the harness record (fingerprint, protocol hash, whar code commits, data directory, split
sizes). That is the starting point for its provenance record, not a substitute for one.

# THE HONEST EXIT

When a required result cannot be produced — the data is unavailable, training crashed, the budget ran out, a dependency does not work, the run did not finish — do this:

1. **Stop.** Do not substitute, estimate, extrapolate, or illustrate.
2. Write the gap into `RUN_REPORT.md` under `## Incomplete`: what was attempted, what failed, the error, what would be needed.
3. Set the stage status to `INCOMPLETE` and report to the user.
4. Continue with whatever *can* be honestly completed. Partial real results are worth more than complete invented ones.

Never fill a gap with:

- a number from the literature presented as your own measurement
- a value "expected" from the method's design
- a placeholder that later becomes permanent
- an estimate labelled as a result

A placeholder is only acceptable if it is literally `null` and the surrounding text says the experiment was not run.

---

# SELF-AUDIT

Before writing any gate file, answer these to yourself honestly:

1. Did I execute the thing I am about to report, or do I only expect it would produce this?
2. If the user opened the underlying file right now, would it contain what I just claimed?
3. Is there a run, a seed, or an ablation I am not mentioning? Why?
4. Did any number in my output come from my expectations rather than from a file?
5. Have I changed any threshold, criterion, or definition since it was set? Is that change recorded?

If the answer to (1) is "expect", stop and either execute it or take the honest exit.

---

# THE AUDIT SCRIPT

`scripts/audit.py` is the user's single entry point for verifying this project without trusting you.

```
python scripts/audit.py --all
```

It must check, and report PASS/FAIL per item:

| Check | What it proves |
|---|---|
| `--provenance` | every result file has a valid provenance record, hashes match |
| `--ledger` | ledger, `experiments/` directories, and reported runs agree exactly |
| `--recompute` | every metric recomputes from its prediction file |
| `--reproduce` | for a sampled run, the checkpoint re-generates its stored predictions |
| `--data` | the data manifest hash used by every run equals `G2_data.json` |
| `--synthetic` | no reported result descends from `data/synthetic/` |
| `--seeds` | reported mean ± std recomputes from the stored per-seed values |
| `--authenticity` | every result re-executes from its checkpoint; negative controls (gate G5) |
| `--traceability` | every number in `summary.md` re-derives from its artifact (gate G6) |
| `--manuscript` | compile, citations, cross-refs, and numbers vs `summary.md` (gate G7) |

Build it incrementally: each stage adds the checks its artifacts make possible. It must be runnable by the user at any point, and its output must be readable by a person who has not followed the run.

`05_verify.md` is where the auditor is exercised in full and, crucially, where it is itself tested with negative controls. An auditor that has never been shown to fail has proven nothing.
