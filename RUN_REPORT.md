# Run report: HARFactors

**State on 2026-10-14:** split off from Edge_SSM with its runs and analyses, then restructured to
`agents/00_project_structure.md`. No gate written yet. Resume at `agents/02_plan.md`: pre-register the
confirmatory analyses before running any new one (see `agents/00.md` § Project status).

## Gates

G1-G7: not written. G4 is planned as `not_applicable` in its template form: no held-out split is sealed, because the
analysis is descriptive over all stored runs. The analysis code and the pre-registration are sealed instead.

## Checked on 2026-10-14 (outside the gate protocol)

- `uv.lock` versions are identical to the WHAR paper-era environment, and the paper-era guard passes.
- `scripts/run_analyses.py` and `scripts/make_figures.py` reproduce 5,215 imported files byte for byte.
  `results/metrics/arena/compare.*` gained rows for later runs.
- A one-epoch run on the shared caches works (`scripts/train.py`).

## Audit of 2026-10-15 (independence and completeness)

- **I/O audit.** A Python audit hook recorded every file opened, listed, created or removed, and every process
  started, while these ran: all imports of `src/`, `scripts/run_analyses.py`, `scripts/make_figures.py`, a test file,
  and a 1-epoch training on utd_mhad, uca_ehar and dsads.
  - Result: 0 accesses to another project.
  - Reads outside the project were limited to the shared paper-era caches in `D:\Projects\HAR\datasets_paper_era`.
    HAR's raw files were not even needed, and HAR's latest-code cache was never read.
  - Writes outside the project: only matplotlib's font cache. The harness's `makedirs(exist_ok)` calls touched
    existing HAR folders without changing them.
- **HAR data unchanged.** A manifest (size, mtime) of `HAR/datasets` and `HAR/datasets_paper_era` (42,296 entries),
  taken before and after all checks: 0 added, 0 removed, 0 changed.
- **Static scan.** No code or config refers to Edge_SSM, another paper project, the old `edge_mcu` venv or the old
  build trees. Mentions of Edge_SSM that remain are provenance notes in docstrings and documents.
- **Environment.** `uv lock --check` passes. `third_party/whar-*` are at 144e006 / 9b65004 / 19b3019 and clean.
- **Code integrity.** Every module was compared with its Edge_SSM source. The only differences are the intended edits
  (imports, locations, the fixes listed here).
- **Reproduction.** The analyses and figures, re-run, still equal the imported outputs. The exceptions are the
  regenerated `arena/compare` (rows added for later runs) and, where noted, JSON key order or a "Built by" line.
- **Completeness:** 47/47 required artifacts present. The variance population (10 models × 10 datasets × 3 seeds) is complete.
- **Fixed in this audit:**
  - the checkpoints of the 10 population models were missing (needed by `error_analysis` and G5). They were added from Edge_SSM; see the ledger note;
  - the Edge_SSM MCU batch files were removed from `configs/cloud_jobs`;
  - the runner label was changed;
  - the cloud runner accepts `module` jobs;
  - `docs/research_plan.md` now points at `results/metrics/factors/data.json`.

## HPC copy (2026-10-15)

- **Copy.** This project is copied to `/work/huy.leminh/code/Dr.Huy/HAR/HARFactors` (login `hpc-gw`). The shared data mirror
  `/work/huy.leminh/code/Dr.Huy/HAR` holds all the WHAR data the paper-era protocol uses: the 13 paper-era caches
  (`datasets_paper_era/`, 28.5 GiB; the 10 development datasets plus uca_ehar, up_fall, wear) and the raw files of all
  30 datasets (`datasets/<id>/data`; hard links for the 13 cached ones). HAR's latest-code processed folders were not
  sent.
- **Verification.**
  - The project files are sha256-identical to the PC copy: 16,448 files over both projects.
  - The data match the PC in file count and size for every dataset, caches and raw: 33,532 files, 65.5 GiB,
    0 mismatches.
- **How it was uploaded.** By tar over SSH: the PC's D: drive is an HDD and the upload was bound by it, at about
  30 MB/s for large files and much slower for small files. Small-file trees were first packed into one tar on the SSD.
  See `docs/hpc.md`.
- **Environment: not installed.** The user's request was the upload only. An environment install was started and
  then stopped at the user's request, and its traces in the project were removed. Things learned on the way, for
  whoever sets it up:
  - the site firewall blocks `download-r2.pytorch.org`, which serves the torch wheel;
  - PyPI's file host answers at below 1 MB/s from both the gateway and the PC, while `pypi.nvidia.com` is fast;
  - left on the cluster, outside the project:
    - `~/.local/bin/uv`;
    - a uv-managed Python 3.12;
    - a partial uv cache (`/work/huy.leminh/.cache/uv`, 3.6 GB);
    - hash-checked wheels for torch, nvidia-* and triton, with `requirements-linux.txt`
      (`/work/huy.leminh/.cache/whl`, about 3.3 GB).
  - `scripts/hpc/setup.sh` has not completed a run yet.
- **GPUs.** On 2026-10-15 all 4 GPUs (hpc21-24) were busy and the DGX was down. CPU nodes hpc01-20 were partly free.

## Stage 02 (started 2026-10-15, on the HPC copy)

- **The confirmatory analysis is decided with the user and pre-registered** in `docs/preregistration.md`.
  - **H1 is the only confirmatory hypothesis.** Within a dataset, the test-subject group explains more variance
    than all model-related sources together, by REML.
  - **Data.** It is tested on the 948 held-out runs (S2).
  - **Sensitivity analyses:** the 2,460 development runs, balanced accuracy and the 17 independent architectures.
  - **No new training.** A 540-run option was considered and not chosen.
- **Seeds (user's decision, 2026-10-15):** keep 3 seeds, the template asks for ≥ 5. The letter states this as a
  limitation. No more seeds are run.
  - sha256 of `docs/preregistration.md` as written (CRLF): `91487bd455f70451c2be613dc80ba581b8d1d64636f14ac92471e43c6f7cf3c4`. Check it on the PC with
    `Get-FileHash docs\preregistration.md`.
- **These files were written on the HPC copy.** Copy them to `D:\Projects\HARFactors` before editing any of them
  there:
  - `docs/preregistration.md` (new);
  - `docs/research_plan.md`, `RUN_REPORT.md`, `agents/00.md`, `agents/01_system.md` and `CLAUDE.md`.
- **Still owed for G1:**
  - `docs/research_checklist.md`;
  - literature with verified ids;
  - the IEEE Sensors Letters template and its limits;
  - `git init`;
  - the environment checks.

## Stages 03-05 (2026-10-15, on the HPC copy)

- **G2 PASS, G3 PASS.**
- **G4 sealed** the pre-registration and the analysis code (commit 761d6bd).
- **H1 (confirmatory): success.**
  - R = 8.19, 97.5 % interval [2.85, 16.9]; pi_P = 82 % [64, 89]; 828 runs on 7 held-out datasets.
  - Sensitivity analyses: `results/metrics/confirmatory/`.
  - With balanced accuracy on S1, the interval includes 1.
  - Across the 17 WHAR architectures (S3) the network dominates: R' = 0.21.
- **G5 FAIL** (`results/gates/G5_results_authenticity.json`; kept, never edited).
  - **What passed:**
    - 408/408 re-executed runs agree with their records (373 exactly; largest difference 0.35 F1 points);
    - an independent REML gives the same R for all 6 analyses;
    - permuting model labels removes the model share;
    - no leakage path; the S2 access log matches the seal.
  - **What failed:**
    1. 7 imported Edge_SSM files under `results/metrics/` have no provenance: `pareto/`, `pareto_heldout/`,
       `profile.json`, `p5/*_recurrencenet`. The letter does not use them.
    2. One run differs from its record by exactly 2 windows, which is the declared limit. The comparison used float32
       values and so counted it as above the limit (an implementation bug).
    3. A negative control altered one prediction in a degenerate file (dsads, constant predictions), changing one
       wrong class to another wrong class. No metric changed, so `--recompute` passed. The auditor misses changes that
       leave the metrics unchanged.
    4. The provenance negative control was void, because the untampered copy already failed on item 1.
  - **Informative:** on this cluster's dsads cache the stored dsads checkpoints score at chance (macro-F1 about
    0.053). dsads runs (in S1) can be verified only by their records.
  - **Waiting for the user's decision** on the fixes before a G5 rerun.

## Incomplete / open decisions

- Data for G5:
  - 8 of the 10 datasets are re-executable per split on this PC.
  - dsads and daphnet need a Linux VM, because their parsers are unsorted.
  - Six runs (S4D, Mamba) have no checkpoint.
  - Details: `agents/05_verify.md`.
- The confirmatory analysis is pre-registered (2026-10-15, `docs/preregistration.md`) but has not been run.
  - Its code must be sealed at G4 before it reads the held-out runs.
- Checkpoints of the 10 population models were added on 2026-10-15 (`experiments/checkpoints/`, ledger note), so
  G5 can regenerate predictions here.
- Figures are SVG; `fig0x_*.pdf` are due at stage 04.
