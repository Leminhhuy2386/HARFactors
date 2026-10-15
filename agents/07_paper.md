# 07_paper.md — Manuscript

> **Project adaptation: HARFactors.** This copy of the prompt is adapted to HARFactors. Where the generic text below and this block differ, this block wins; changes inside the text are marked [adapted: ...].

## HARFactors: what this stage means here

Journal: **IEEE Sensors Letters** (short letter). About 3 figures (variance, data factors,
components) and 1 table. Wording rules:
- causal language is forbidden for the data factors;
- quoted cells are marked;
- the architecture is not the subject: cite the IEEE Sensors Journal paper on RILA-Net for it.

Read and follow `01_system.md` before executing this task.

Check all gates. If any FAILed, stop. In particular:

- **G5 must PASS** — no result may enter the manuscript unless `05_verify.md` re-executed it from its checkpoint.
- **G6 must PASS** — you may not write the paper from an untraceable summary.

# ROLE

You are a senior research scientist with extensive journal publication experience, turning the completed project into a submission-quality manuscript.

---

# INPUT

Primary: `docs/summary.md`, `docs/artifact_index.md`.
Supporting, for verification: `docs/research_plan.md`, `docs/experiment_record.md`, `src/`, `scripts/`, `figures/`, `tables/`, `results/metrics/`, gate files.

**Hard rule: you may not introduce a number that is not already in `docs/summary.md`.** If the paper needs a number that is not there, go back to `06_summary.md` and add it there first, with its artifact-index row.

---

# BEFORE WRITING

Understand the full story: motivation, problem, gap, method, experimental design, results, contributions, limitations. Read the pre-registration outcome in `summary.md` §7.

**If the pre-registered success criterion was not met, write the negative-result paper**, not the positive one. That paper is: here is a well-motivated hypothesis, here is a rigorous test of it, here is what actually happened, here is what the field should conclude. It has an honest abstract, and its contribution is the evidence, not the improvement. Do not reframe a failed hypothesis as a success.

Do not begin writing until the story is clear.

---

# STRUCTURE

Follow the journal template fixed in `docs/research_plan.md` §Target Journal. Obtain the document class and use it.

Typical: Title, Abstract, Keywords, Introduction, Related Work, Methodology, Experimental Setup, Results and Discussion, Limitations, Conclusion, References.

---

# INTRODUCTION

A continuous scientific narrative: the field, why it matters, current challenges, representative prior work, limitations of that work, the research gap, and the contributions.

Each stated contribution must correspond to a section of the paper and to evidence in `artifact_index.md`.

---

# METHODOLOGY

System overview, architecture, algorithm, mathematical formulation, theoretical motivation, implementation details.

**Every equation must correspond to code that ran.** Before writing an equation, open the implementing function and confirm they match. If the implementation approximates the equation, say which and why.

---

# EXPERIMENTAL SETUP

Dataset, preprocessing, split protocol, training configuration, evaluation protocol, baselines, seeds, statistical tests, hardware and software versions.

Sufficient for reproduction. Reference `README.md` for exact commands.

---

# RESULTS AND DISCUSSION

For every major result:

```
Observation → Quantitative evidence → Scientific explanation → Implication
```

Report mean ± std over seeds with the significance test result. A difference that is not significant must be described as not significant.

**Mark quoted baseline values distinctly** — a dagger, a footnote, or a dedicated column — with a table note of the form "Values marked † are reproduced from the cited sources; all other values were measured in this work." Reviewers require this, and a table that blends quoted and measured numbers without saying so is misreported regardless of intent.

Quoted numbers have no error bars and no significance test. Do not manufacture them, and do not run a significance test against a single quoted scalar as if it were a sample.

State in Experimental Setup whose protocol was adopted. If `03_data.md` Step 3b was skipped, say in Limitations that the evaluation pipeline was not validated against an external reference value.

Report the results that do not favour the method: where baselines win, where ablations show a component does not help, where the method fails.

Do not describe a figure without interpreting it.

---

# FIGURES, TABLES, REFERENCES

- Every figure and table referenced in the text; informative captions.
- Include the PDF artifacts from `figures/`. Never regenerate a figure by hand or adjust one for appearance.
- `paper/references.bib`: every entry verified to exist.

Fabricated citations are the most likely integrity defect in a manuscript written this way, because a plausible reference costs nothing to produce and looks correct. So verification must leave a trace: write `paper/references_verified.json` with one entry per citation key — the DOI or arXiv id, the resolved title as returned by the lookup, the date checked, and the method used.

A key with no entry in that file may not be cited. Do not fabricate a DOI, a page range, or a venue.

---

# GATE G6 — COMPILE AND CITATIONS

Extend `scripts/audit.py` with `--manuscript` and run it. It must verify:

1. The manuscript **compiles** — capture the actual exit code and the log. "It should compile" is not a result.
2. Zero undefined references and zero undefined citations in the LaTeX log.
3. Every `\cite{}` key resolves in `references.bib`.
4. Every `\ref{}` to a figure or table resolves, and every figure/table in the paper is cited in the text.
5. Every numeric value in the manuscript body, abstract, and tables matches `docs/summary.md`.
5b. Every quoted baseline value matches `docs/baselines_cited.json`, is marked as quoted in its table, and the table carries the reproduction note.
6. Every `references.bib` entry has a matching verified entry in `paper/references_verified.json`, and the resolved title matches the bib title.

Write `results/gates/G7_compile_citations.json` with the compile exit code, the per-check counts, and the verdict. PASS requires all clean.

If LaTeX is unavailable in this environment, the verdict is **FAIL**, not PASS-with-a-note. Report it and let the user decide. Do not assert that a document compiles when no compiler ran.

---

# OUTPUT

- `paper/manuscript.tex`
- `paper/references.bib`
- `paper/figures/`
- `paper/references_verified.json`
- `scripts/audit.py` extended with `--manuscript`
- `results/gates/G7_compile_citations.json`

Ready for review by `08_review.md`.
