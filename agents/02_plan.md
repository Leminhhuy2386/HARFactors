# 02_plan.md — Research Plan and Pre-Registration

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

The analyses already exist, and they were **exploratory**. Before any confirmatory analysis,
write `docs/research_plan.md` § Pre-registration. Example: "the person share of within-dataset variance stays above
40 % under a mixed-effects model with random intercepts for dataset, person and seed". The existing numbers then
count as the exploratory finding that motivates it, and the manuscript says so.
- Journal: **IEEE Sensors Letters**. Check its LaTeX template and its page and figure limits now.
- G1:
  - `uv sync`;
  - the paper-era guard;
  - `uv run python scripts/run_analyses.py` reproduces `results/metrics/` exactly;
  - pdflatex compiles the template stub.

Read and follow `01_system.md` before executing this task.

# ROLE

You are a senior research scientist designing a publication-quality research plan.

**Do not write model code or run experiments in this stage.** You will create the directory skeleton, `requirements.txt`, and the environment gate — nothing else.

---

# INPUT

- `IDEA_FILE`
- Existing project files, datasets, and constraints
- `COMPUTE` budget
- Related literature

Develop a complete understanding of the problem before proposing a solution.

---

# STEP 1 — LITERATURE REVIEW

Review representative and recent work. Identify state-of-the-art methods, their limitations, and the gap this project addresses.

Focus on a meaningful contribution rather than an incremental one.

For every work you intend to cite later, record: authors, year, venue, title, DOI/arXiv id. **Verify each one exists.** Unverified entries do not enter `references.bib`.

---

# STEP 2 — TARGET JOURNAL (mandatory)

Decide and record:

- Journal name
- LaTeX template / document class and where to obtain it
- Page or word limit
- Required manuscript structure
- Reference style

If the user has not specified a journal, propose one with justification and record the choice. **This decision is now fixed** — `07_paper.md` will follow it. A plan without a named journal is incomplete.

---

# STEP 3 — RESEARCH DESIGN

Produce:

- Research background, problem statement, research gap
- Research objectives and scientific hypothesis
- Proposed methodology, with a scientific justification for **each** design choice (architecture, learning strategy, representation, loss, evaluation protocol)
- Expected innovations and contributions
- Potential limitations

State plainly what practical cost or limitation the method reduces, and for whom. A method that improves a metric without a use case is not a contribution.

---

# STEP 4 — FROZEN EXPERIMENTAL PROTOCOL

Define, and do not change later:

- Dataset(s), version, and exact source
- Preprocessing
- **Split protocol**: how train/val/test are formed, split ratios, and the rule that makes the split deterministic (seed or explicit cell/sample list). Splits are frozen here.
  If baseline numbers will be quoted from published papers, **the split protocol is inherited from those papers, not chosen by you.** Name the source whose protocol you are adopting.
- Baseline methods. State for each whether its number will be **quoted** from the literature or **measured** here, and record the source for every quoted one
- At least one cheap baseline that can be re-run as an evaluation-pipeline anchor (see `03_data.md` Step 3b), or an explicit decision not to have one
- Evaluation metrics, with exact definitions
- Number of seeds (≥ 5) and the significance test to be used
- Statistical analysis plan
- Ablation studies
- Robustness evaluation
- Computational requirements, checked against `COMPUTE`

If the plan does not fit `COMPUTE`, cut scope **now** and say what was cut. Do not plan experiments you cannot run.

---

# STEP 5 — PRE-REGISTRATION (mandatory)

Before any experiment exists, commit in writing to what would count as failure.

Record in `docs/research_plan.md` under `## Pre-registration`:

| Item | Value |
|---|---|
| Primary metric | |
| Primary baseline | |
| Success criterion | e.g. "≥ 8% relative MAE reduction vs. baseline B, p < 0.05 over 5 seeds" |
| Failure criterion | e.g. "< 3% relative improvement, or p ≥ 0.05" |
| Secondary claims | with their own criteria |
| Max improvement iterations | 3 |

**The success criterion may not be revised after results are seen.** If it is revised, the revision, its date, and its reason must be recorded, and the manuscript must disclose it.

If the failure criterion is met, the project takes the **negative-result branch**: the work is written up as a negative or limitation study (`06_summary.md` and `07_paper.md` both support this). It is not iterated until it wins.

---

# STEP 6 — SCAFFOLD AND ENVIRONMENT GATE

Create the directory skeleton from `00_project_structure.md`, plus `requirements.txt` and `.gitignore`.

If the project is not a git repository, initialise one and make an initial commit — provenance records reference a commit hash, and without version control there is no way to tell which code produced a result. If the user does not want version control, record `"vcs": "none"` in `G1_environment.json` and note that code-level traceability is unavailable.

Then verify the environment **by executing**, not by assuming:

- Python version
- Every core dependency: install it, import it, run a one-line smoke test
- Accelerator: device name, VRAM, and whether it actually runs a tensor op
- **Every non-standard architectural component named in the plan** (e.g. a specific SSM/attention/conv kernel): import it and run one forward pass at the intended shape
- LaTeX toolchain: does the target document class compile a minimal file? If LaTeX is unavailable, **report it to the user now**. Discovering it at `07_paper.md` wastes the entire run, and the fix (installing a distribution, or switching to a different output) is a decision for the user to make before any compute is spent.

Write `results/gates/G1_environment.json`:

- `check`: "all planned components import and execute on the available hardware"
- `observed_value`: the version/device table, plus a list of any component that failed
- `tolerance`: "zero planned components missing without a recorded substitute"
- `verdict`: PASS only if every planned component either works or has an **explicitly recorded substitute**

If a component cannot run, do **not** silently substitute. Record the substitution in the gate file and update `docs/research_plan.md` to name the component that will actually be used. The manuscript will describe the substitute, not the original.

---

# OUTPUT

- `docs/research_plan.md`
- `docs/research_checklist.md`
- `results/gates/G1_environment.json`
- directory skeleton, `requirements.txt`, `.gitignore`

The plan is the implementation blueprint for `03_data.md` and `04_research.md`. Do not implement the proposed method.
