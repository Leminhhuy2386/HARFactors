# 04_research.md — Implementation and Experiments

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

No model training is needed for the main claims. The "experiments" are the analyses in
`src/evaluation/factor_analysis.py`. Every new analysis is a run, with an entry in `experiments/LEDGER.md` and a
provenance record. **G4 in its template form does not apply:** there is no held-out test split to seal, because the
analysis is descriptive over all stored runs. Write G4 as `not_applicable`, with this reason, and seal instead the
*analysis code and pre-registration* before the confirmatory analysis runs. [adapted: under WHAR Arena v1 every split has its own **test subjects**, never seen in training or validation.
What can be "sealed" is a set of datasets, not a split. Data that informed a design decision is labelled
*development* data, and the manuscript says so.]

Read and follow `01_system.md` before executing this task.

Check `results/gates/G1_environment.json`, `G2_data.json`, `G3_baseline.json`. If any FAILed, stop.

# ROLE

You are a research engineer implementing the plan, running the experiments, and producing every artifact the analysis and manuscript stages will need.

Follow `docs/research_plan.md`. Do not change the research direction unless the user asks.

---

# INPUT

- `docs/research_plan.md`, `docs/research_checklist.md`
- the frozen splits, metrics, and preprocessing from `03_data.md`
- `HPC_SKILL` if available

---

# IMPLEMENTATION

Build under `src/` per `00_project_structure.md`. Three entry points:

## `scripts/train.py`

Dataset loading, model definition, training, validation, checkpointing, logging.

Saves to `experiments/<run_id>/`: best and final checkpoint, frozen config + hash, seed, git commit, training history, full log, wall-clock and peak memory.

Appends START and END lines to `experiments/LEDGER.md` — including when the run crashes.

Launched **detached**, per `00.md` § Long-running jobs: write `experiments/<run_id>/job.json` at launch, append the START line at launch rather than at completion, and never relaunch a run whose job is still alive. Writes a provenance record for every output, with `data_manifest_sha256` read from the data actually loaded (per `01_integrity.md`).

**Reads the training and validation splits only.** It must not be able to open the test split — enforce this in code, not by intention.

Checkpoint selection and early stopping use validation only. A checkpoint chosen by test performance is leakage regardless of what the metric file says.

## `scripts/evaluate.py`

Loads a checkpoint, runs inference, computes metrics, writes predictions and metric files.

Does not train. Takes the split as an explicit argument.

**Every read of the test split appends one line to `results/test_access.log`** — timestamp, caller, run_id, purpose. Append-only, never edited. The loader itself writes this line, so it cannot be forgotten or bypassed by calling the loader directly.

This log is the only thing that makes the test seal enforceable rather than aspirational. Without it, "the test set was read once" is a claim; with it, it is a record.

Applies no transform fitted on the split being evaluated: load `data/processed/preprocessor.pkl` and apply it unchanged.

Must be **deterministic given a checkpoint**: re-running it on the same checkpoint and split reproduces the same prediction file. `audit.py --reproduce` depends on this. If any inference-time nondeterminism is unavoidable, document it and provide a tolerance.

Metrics are computed from the prediction file and never anywhere else, so that `audit.py --recompute` can re-derive every metric independently.

## `scripts/make_figures.py`

Generates publication-quality figures and tables from saved artifacts **only**.

Must not train, must not run inference, must not modify results.

For every figure and table it must also write the underlying numbers to `results/figure_data/` and `results/table_data/`. A figure whose source numbers are not on disk is not finished.

Naming: `figures/fig0x_short_description.{png,pdf}` at 300 DPI; `tables/tab0x_short_description.{csv,tex}`. PDF is what the manuscript includes.

## `tests/`

At minimum:

- metric implementations verified against hand-computed values
- splits verified **group-disjoint**: no entity id appears in two splits, and no row hash appears in two splits
- the preprocessor verified to be fitted on train indices only
- a one-batch overfit test proving the model can learn
- a test asserting `train.py` cannot open the test split
- a test asserting the test loader appends to `results/test_access.log`

---

# EXPERIMENTS

Run, in this order, all on **validation**:

1. Proposed method
2. All baselines from the plan
3. Ablations from the plan
4. Robustness evaluations from the plan

Every comparative configuration runs with the seeds fixed in the plan [adapted: 3 training seeds for the imported runs; 5 only if the user approves seeds 3-4]. Report mean ± std. Use the significance test named in the plan.

Store the **per-seed raw values**, not only the aggregate — `audit.py --seeds` recomputes mean and std from them. If a seed crashed, it stays in the ledger and the reported aggregate states how many seeds completed.

No run may generate its data. Outside `tests/` and `data/synthetic/`, no random generator may stand in for data or predictions (`01_integrity.md` § Forbidden Operations).

Record for each configuration: accuracy metrics, computational cost (params, FLOPs or wall-clock, memory), and failure cases.

---

# ITERATION — BOUNDED

If validation results do not meet the pre-registered success criterion, you may iterate at most **3 times**.

Each iteration:

1. State the hypothesis for why it underperforms.
2. Make one coherent change (architecture, features, loss, training strategy, data processing).
3. Re-run on validation.
4. Append the outcome to `docs/experiment_record.md` — including iterations that made things worse.

Rules:

- Iterate on **validation only**. The test split stays sealed.
- Never modify metrics, splits, or preprocessing to improve a number.
- Never drop a baseline, a seed, or an unfavourable ablation from the record.
- After 3 iterations, stop regardless of outcome.

**If the pre-registered failure criterion is met after 3 iterations, take the negative-result branch.** Notify the user, record it in `docs/experiment_record.md`, and continue to `05_verify.md` — a negative result is verified exactly like a positive one, then written up as a negative or limitation study. Do not keep iterating, and do not adjust the criterion.

---

# GATE G4 — TEST-SET SEAL

Only when **all** of the following hold:

- the architecture is final
- all hyperparameters are final
- all ablations are complete on validation
- the iteration budget is closed

Then, exactly once:

1. Record the SHA-256 of every final config and of `data/processed/splits.json`.
1b. Carry forward any pre-registered anchor-baseline test read from `G3_baseline.json`, so the seal record states exactly what had already been read and why. If none was taken, state that this is the first test read in the project.
1c. **Read `results/test_access.log` and reproduce it in the gate file.** Every pre-seal entry must correspond to a declared exception. An undeclared pre-seal read means the test split informed a design decision: the test results are contaminated. Say so in the gate and in the manuscript — do not re-split to make the problem disappear.
2. Run `scripts/evaluate.py` on the test split for the proposed method, all baselines, all seeds.
3. Write `results/gates/G4_test_seal.json` with the timestamp, the config hashes, the seeds, and a statement that this was the first and only test-set read.

**After G4, no model, hyperparameter, or preprocessing change is permitted.** If one becomes necessary, the test results are void: say so explicitly in the gate file and in the manuscript.

---

# ANALYSIS

Do not only report numbers. For each significant result, explain what it means and why the method produces it.

Identify: strengths, weaknesses, failure cases, where baselines beat the proposed method, and where the improvement is not statistically significant. Negative and neutral findings are part of the record.

---

# OUTPUT

- `src/`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/make_figures.py`, `tests/`
- `experiments/<run_id>/` for every run
- `results/metrics/`, `results/predictions/`, `results/figure_data/`, `results/table_data/`
- `figures/`, `tables/`
- `results/gates/G4_test_seal.json`
- `experiments/LEDGER.md`, complete and append-only
- `scripts/audit.py` extended with `--recompute`, `--reproduce`, `--seeds`. Before leaving this stage, `python scripts/audit.py --provenance --ledger --data --synthetic --recompute --reproduce --seeds` must pass. `--traceability`, `--manuscript`, and `--authenticity` do not exist yet; `--all` runs only the checks implemented so far and must say which ones it skipped.
- `ARCHITECTURE.md`, `README.md` (how to reproduce every result, command by command)
- **`docs/experiment_record.md`** — the required handoff artifact for `06_summary.md`, containing:
  - every configuration run, with its `run_id`
  - all validation results
  - all test results
  - every iteration and its outcome, including failures
  - any deviation from `docs/research_plan.md`, with justification
  - computational cost

Do **not** proceed to writing anything up. The next stage is `05_verify.md`, which audits these results for authenticity before a single sentence is written about them.
