# HARFactors

Paper project for **IEEE Sensors Letters**: what drives wearable-HAR accuracy under person-disjoint evaluation. The
answer so far: who is tested matters far more than the architecture. Split off from `D:\Projects\Edge_SSM` on
2026-10-14 and self-contained. The only outside resource is the shared WHAR data in `D:\Projects\HAR`.

## Current state (handoff of 2026-10-16): read this first

- **Pipeline run to stage 08 on the HPC copy (2026-10-15/16).** All gates G1-G7 PASS; the final review verdict is
  **NOT_READY_FOR_SUBMISSION**. Details: `RUN_REPORT.md` § Final state, and `docs/review_report_iter3.md`.
  - H1 (pre-registered): **success**. R = 8.19 [2.85, 16.90] on 828 held-out runs.
  - **S3 reversal:** across the 17 WHAR architectures, R' = 0.21.
  - Manuscript: `paper/manuscript.tex`, rendered from `paper/manuscript_src.tex` by `scripts/render_manuscript.py`.
    Never hand-edit numbers.
  - Open items:
    - (label-shuffle control: done 2026-10-16, collapsed to chance);
    - (two Figure 6 cells: done 2026-10-16);
    - re-execution of the 468 S2 runs without a cache on the HPC;
    - the author block;
    - (`IEEE_lsens.cls`: in place 2026-10-16).
    - Submission formalities: authors, subject category, graphical abstract, reviewers, ORCID, EDICS.
- **Expansion of 2026-10-16 (user request: many more figures; page limit waived for the draft):** new exploratory
  analyses (split spread, validation vs test, learned memory parameters, embeddings) and Figs. 3-9, Tables II-III in
  `paper/manuscript_src.tex`. G5 rerun5 FAILed on the S2 model-permutation control (statsmodels optimizer stopped at
  a non-maximum); with the user's approved fix, G5 rerun6, G6 rerun4, G7 rerun6 and `audit --all` PASS (10 pages).
  Figures restyled (deeper validated palette, Fig. 3 as a dataflow diagram): G7 rerun7 PASS.
  Restore the venue limits in `paper/format_limits.json` before submission.
- **What exists (history).** Imported from Edge_SSM:
  - the 2,460-run population (10 models × 10 datasets × 3 seeds) with its checkpoints;
  - the exploratory factor analysis and its figures.

  These were verified at G5 (rerun 2) where re-executable.
- **Audited 2026-10-15** (`RUN_REPORT.md` § Audit):
  - independent: 0 accesses to other projects;
  - HAR data unchanged;
  - 47/47 required artifacts present;
  - `scripts/run_analyses.py` reproduces every number byte for byte.
- **Stage 02 (done):**
  - **Pre-registered on 2026-10-15** (`docs/preregistration.md`; frozen; sha256 in `RUN_REPORT.md`).
    - H1 is the only confirmatory hypothesis, tested on the 948 held-out runs.
    - There is no new training.
    - Every other result stays exploratory.
  - Then `docs/research_checklist.md`, `git init` and gate G1.
- **Decided 2026-10-15:** keep 3 seeds and state it as a limitation in the letter. No more seeds are run.
- **Reproducibility (2026-10-16, user request):** every stored S1/S2 run with a checkpoint (3,402) was re-executed;
  the listing order of the cloud VMs was recovered from the checkpoints where needed (`scripts/reconstruct_order.py`),
  and the six runs without a checkpoint were retrained for one (5 exact). `docs/reproducibility_report.md`;
  `configs/` holds every network's config. S3 stays records only.
- **Known gaps:**
  - 6 runs had no checkpoint (S4D actrectut ×2, Mamba actrectut/unimib ×4): retrained 2026-10-16, checkpoints in
    the retrain run's directory (the stored records stay the analysed ones);
  - `tests/test_edge_ssm.py` fails on `isochron_uniform` (not in the population).
- **Framing:**
  - no claims about RILA-Net's merit (that is the IEEE Sensors Journal paper);
  - data factors are correlational;
  - the 30-dataset part re-analyses the paper's single runs.

## On the HPC (copy uploaded 2026-10-15)

- A copy of this project lives at `/work/huy.leminh/code/Dr.Huy/HAR/HARFactors` (login `ssh hpc-gw`, Slurm).
  The shared data mirror is `/work/huy.leminh/code/Dr.Huy/HAR`: 16 paper-era caches (13 uploaded 2026-10-15, plus
  pamap2, uma_fall and w_har copied 2026-10-16; the 10 development datasets
  among them) and the raw files of all 30 WHAR datasets.
- **The environment is installed** (2026-10-15, `scripts/hpc/setup.sh` with hash-checked local wheels). Jobs run on the compute nodes
  (hpc21-24 GPU, dgx A100, hpc01-20 CPU). `hpc-gw` is for Claude Code and internet access only.
- Read `docs/hpc.md` first:
  - one-time `bash scripts/hpc/setup.sh` on the login node;
  - then `source scripts/hpc/env.sh` and submit with `scripts/hpc/submit.sh` (sbatch);
  - never run work on the login node.
- The PC copy is the reference. Bring results produced on the cluster back to the PC with their ledger lines, and
  never edit the same file on both sides.

## How to work here

- Work through the agent pipeline. Start every session with `agents/00.md` (parameters, status, resume point, facts).
- Run one stage per session. Read `agents/01_system.md` and `agents/01_integrity.md` first.
- Where files go: `agents/00_project_structure.md`. Code: `ARCHITECTURE.md`. State: `RUN_REPORT.md`. Plan:
  `docs/research_plan.md`.

## Rules

- **The architecture is not the subject.** Make no claim about RILA-Net's merit; that is `D:\Projects\RILA` (IEEE
  Sensors Journal), which this letter cites.
- **Correlational evidence stays correlational.** The data factors over 30 datasets are correlational (subjects and
  channels correlate, ρ −0.50). Never phrase them causally.
- **The 30-dataset part re-analyses published single runs** (`docs/sources/arena_extracted.json`). Say so.
- **Pre-register first.** The existing analyses were exploratory; pre-register a confirmatory analysis before
  running it.
- **Paper-era WHAR code only** (`third_party/`, 144e006 / 9b65004 / 19b3019). Never edit `third_party/`.
- **Data:** shared in `D:\Projects\HAR` (paper-era caches only). Never edit or delete anything there.
- **Splits** come from the harness. Never write your own split, and never change `cfg.seed`.
- **Pairing.** Compare models paired by split and seed.
- **Selection.** Never choose variants by test scores.
- **Imported results** are unverified until gate G5.
- **Machine:** GTX 1080 Ti (Pascal), float32 only. Run `uv` from PowerShell.
- **Language:** chat in Vietnamese. Write files, comments and commits in English.
