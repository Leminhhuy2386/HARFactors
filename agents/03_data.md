# 03_data.md — Data Validation and Baseline Reproduction

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

Two kinds of input.
- **Our runs:** measurements with the harness record, imported.
- **The WHAR Arena paper's Figure 6:** 510 quoted cells, one run each.

For G3, every quoted cell needs an entry in `docs/baselines_cited.json` with its location: Figure 6, row = dataset,
column = model. The source file is `docs/sources/arena_extracted.json`, matched 510/510 to whar-arena's
paper-benchmark.json. Read a sample of cells from the paper figure itself.
- The anchor: the 17 models re-run with this code on 4 datasets (`experiments/external_har_paper_era/`) against
  Figure 6. CNN-HAR is equal digit for digit, and the rank correlation is 0.95–0.98.
- Data-factor covariates (channels, subjects, classes, rate) come from the paper-era dataset configs. Record their
  source per dataset.

Read and follow `01_system.md` before executing this task.

This stage exists for one reason: **to anchor the project to something outside itself.**

Everything after this stage is written by you and checked against other things written by you. This is the last point at which an external fact can contradict the project. Take it seriously.

Check `results/gates/G1_environment.json` first. If it FAILed, stop.

---

# ROLE

You are a research engineer responsible for obtaining the data, proving it is what the plan says it is, and demonstrating that this codebase can reproduce a result someone else already published.

**Do not implement the proposed method in this stage.**

---

# STEP 1 — RESOLVE THE DATA

Locate the dataset named in `docs/research_plan.md`.

Do exactly one of:

**(a) The data exists locally.** Record the absolute path.

**(b) The data must be downloaded.** Download it to `data/raw/`. Record the source URL and the download date.

**(c) The data cannot be obtained.** STOP. Write `G2_data.json` with `verdict: FAIL` and report to the user what is missing and where you looked.

**You may not proceed by generating data.** If you build synthetic data to smoke-test the pipeline, it goes in `data/synthetic/`, every run using it sets `synthetic: true`, and no result derived from it may enter `results/`, `figures/`, `tables/`, or `paper/`. Generating data and continuing as if the problem were solved is the single most damaging thing you can do in this project.

---

# STEP 2 — VALIDATE THE DATA (gate G2)

Do not trust a successful download. Verify:

- File count and total size
- SHA-256 of the archive or of a stable manifest of file hashes
- Schema: expected fields present, dtypes correct
- Shape: number of samples/units, sequence lengths, value ranges
- Missing values, duplicates, obvious corruption
- **At least one descriptive statistic compared against a value published in the dataset paper or official documentation** — e.g. number of cells, mean cycle life, class balance, split sizes

Build the frozen splits. **If any baseline number will be quoted from a published paper, the split must be that paper's split** — adopt it, do not invent one (see Step 3a). Write the split membership to `data/processed/splits.json` and hash it. **This file is now immutable.**

Write `data/processed/manifest.json`: the SHA-256 of every processed data file, plus a single `data_manifest_sha256` over the sorted list. **Every training and evaluation run must record this value in its provenance record.** This is what makes it impossible for a run to quietly train on data that is not the validated dataset.

Write `results/gates/G2_data.json`:

- `reference_value` / `reference_source`: the published statistic and where it comes from
- `observed_value`: what you measured
- `tolerance`: declared before measuring
- `verdict`: FAIL if the data does not match its own documentation

A mismatch here means you have the wrong data, a corrupted copy, or a misunderstanding of the format. All three are worth stopping for.

---

# STEP 3 — BASELINE ANCHOR (gate G3)

Baseline numbers may be **quoted from the original papers** rather than re-run. This is standard practice and is the default for this project.

It is valid under exactly one condition: **your evaluation protocol must be identical to theirs.** If it is not, the comparison is not a comparison — it is two different measurements printed next to each other, and the difference between them is uninterpretable.

Quoting also removes the pipeline's only external anchor. Nothing downstream ever executes against a known-correct value again. So this step does two things: it proves protocol identity, and it re-establishes a cheap anchor.

## 3a — Protocol identity (mandatory, no compute)

For every quoted number, establish that the following are the same as in the source paper:

- dataset and dataset version
- **the exact split** — same cells/samples in train/val/test, not merely the same ratio
- metric definition, including units, aggregation, and whether it is computed per-sample or per-cell
- preprocessing that affects the metric (filtering, outlier removal, truncation)
- the evaluation subset — full test set vs. a reported subset

**Adopt their protocol; do not build your own and then quote their numbers against it.** Where the source publishes a split file or cell list, use that file. Where it does not, reconstruct it from the paper's description and record the reconstruction.

Classify each quoted number as `exact`, `partial`, or `unknown`. Anything not `exact` may not be used for the primary comparison — it goes in a clearly separated table with its differences stated.

## 3b — Cheap anchor (strongly recommended)

Re-run **one** baseline. Pick the cheapest one available — a linear/ridge model, a statistical predictor, a published trivial baseline. In most benchmarks this costs seconds to minutes, not GPU-hours.

Compare against its published value with a tolerance declared beforehand (default ±10% relative).

This is not about the baseline. It is a test of **your own data loading, splits, and metric code**, using a number someone else computed. If a ridge regression lands 40% off its published value, something in your evaluation stack is wrong — and without this check you would discover it never, while every subsequent number silently inherits the error.

### The test-split conflict, and how to resolve it

Published baseline numbers are usually reported on the **test** split, but `04_research.md` seals the test split until gate G4. These collide. Resolve it explicitly — do not resolve it silently:

1. **Preferred**: if the source also publishes a validation-split number, anchor against that. No test read, no conflict.
2. **Otherwise**: take one **pre-registered test read for the anchor baseline only**. It is permitted because the anchor must be a baseline with **no tuning** — a ridge, a linear fit, a statistical predictor — so there is nothing about it you could adjust in response to what you see.

If you take route 2, the following are binding:

- Record it in `G3_baseline.json` as `"test_read": {"scope": "anchor baseline only", "date": ..., "justification": ...}`, and repeat it in `G4_test_seal.json`.
- The anchor's test value may **not** influence any decision about the proposed method — not the architecture, not a hyperparameter, not the loss, not which ablations to run.
- Do not read the test split for anything else. The seal still holds for the proposed method and for every quoted-protocol comparison.
- Disclose it in the manuscript's Experimental Setup.

A declared, bounded exception is sound practice. An undeclared one is contamination.

A mismatch here is the cheapest bug you will ever find. A mismatch discovered by a reviewer is the most expensive.

## 3c — Write the gate

`results/gates/G3_baseline.json` records:

- `mode`: `"quoted"` or `"quoted+anchor"`
- protocol-match classification for every quoted number
- if 3b was run: published value, source, observed value, tolerance, verdict
- if 3b was **skipped**: `"anchor": "none"` and `"known_weakness": "the evaluation pipeline has never been validated against an external value"`

Skipping 3b does not FAIL the gate. It is your decision to make. But it is recorded, `06_summary.md` carries it forward, `07_paper.md` states it in Limitations, and `08_review.md` reports it. An unvalidated evaluation pipeline is a real weakness; it just has to be a *disclosed* one.

## 3d — The quoted-numbers register

Write `docs/baselines_cited.json`, one entry per quoted number:

```json
{
  "id": "chen2023_transformer",
  "method": "Transformer-RUL",
  "value": 112.6,
  "metric": "MAE (cycles)",
  "dataset": "<name+version>",
  "split": "<which split, per their protocol>",
  "source": {"citation_key": "chen2023", "doi": "10.xxxx/yyyy",
             "location": "Table 3, row 4, p. 12"},
  "protocol_match": "exact",
  "protocol_differences": [],
  "read_on": "2026-09-03",
  "read_from": "publisher PDF"
}
```

These numbers are **not your measurements** and must never be treated as such:

- Each must have a precise source location — table and row, not just a citation key.
- Each must be read from the actual paper, not recalled. A number you cannot open and point to is fabricated, regardless of how confident you are.
- No quoted number may enter `results/metrics/` or carry a provenance record. It lives only in `docs/baselines_cited.json`.
- `audit.py --recompute` skips them, so nothing else will ever catch an error here. Transcription is the only safeguard, so transcribe carefully and record where from.

---

# STEP 3.5 — LEAKAGE PREVENTION BY CONSTRUCTION

Leakage is not caught by inspecting results — a leaked model looks *better*, not stranger. It has to be prevented when the splits and features are built, which is here.

Every rule below is a property of the code you write in this stage. State in `docs/research_plan.md` how each one is satisfied.

## 1. Split by unit, not by sample

Splits must be disjoint at the level of the **underlying entity** — cell, patient, subject, device, session — not at the level of the row.

If one entity contributes many rows (windows, cycles, segments, augmented copies), every row from that entity goes to exactly one split. Row-level random splitting puts near-duplicate rows on both sides and inflates every metric.

Record the grouping key in `splits.json`.

## 2. Split before anything else touches the data

Order is fixed and non-negotiable:

```
raw → split → [fit preprocessing on train] → transform each split → augment train only
```

Never: resample, augment, oversample, deduplicate, or balance **before** splitting. Duplicated or interpolated rows straddling the split boundary are leakage that no later check will find.

## 3. Fit preprocessing on train only

Every statistic learned from data — scaler mean/std, min/max, normalisation constants, imputation values, PCA basis, vocabulary, feature-selection ranking, class weights, outlier thresholds — is computed on the **training split only** and then applied unchanged to validation and test.

Persist the fitted transform to `data/processed/preprocessor.pkl` with the train indices it was fitted on. `05_verify.md` recomputes it from those indices and compares.

Fitting a scaler on the full dataset is the single most common leak in published ML work. It is silent, it always helps the numbers, and it invalidates every result.

## 4. Features must be available at prediction time

For every feature, ask: at the moment this prediction would be made in deployment, does this value exist?

Forbidden without exception:

- Any feature computed from the target, or from a quantity the target is derived from.
- Any feature normalised by a whole-sequence quantity that is only known at the end (total lifetime, final value, sequence length used as a divisor).
- Any statistic aggregated over a window that extends past the prediction point.
- Any identifier that correlates with the label through collection order.

Write the feature list into `docs/research_plan.md` with a one-line availability justification for each. A feature you cannot justify is removed, not kept with a note.

## 5. Validation carved from train, never from test

If the split protocol is inherited from a published paper (Step 3a), the validation split is carved out of **their training portion**. Taking validation from their test portion contaminates the comparison and voids the quoted numbers.

## 6. No test-time statistics

At evaluation, nothing may be computed *from* the test split and used to process it: no test-set normalisation, no per-batch standardisation over test data, no batch-norm in training mode, no transductive step, no threshold tuned on test.

---

# STEP 4 — FREEZE

At the end of this stage the following are immutable for the rest of the project:

- `data/processed/splits.json`
- the metric definitions in `src/evaluation/`
- the preprocessing in `scripts/prepare_data.py`

Any later change to these invalidates G3 and every result produced after it. If a change is genuinely necessary, re-run G3 and record why.

---

# OUTPUT

- `scripts/prepare_data.py` — deterministic, re-runnable
- `data/processed/` including `splits.json`, `manifest.json`, `preprocessor.pkl`
- `src/data/`, `src/evaluation/` (metrics only)
- baseline implementation under `src/models/baselines/`
- one full baseline run under `experiments/<run_id>/`
- `data/processed/manifest.json`
- `docs/baselines_cited.json`
- `experiments/LEDGER.md` — created here, append-only from now on
- `scripts/audit.py` — first version: `--provenance`, `--ledger`, `--data`, `--synthetic` (see `01_integrity.md`)
- `results/gates/G2_data.json`
- `results/gates/G3_baseline.json`

If Step 3b was run, that baseline is the first run to carry a full provenance record; verify `python scripts/audit.py --provenance --ledger --data` passes on it before leaving this stage. If the contract does not hold for one run, it will not hold for forty.
