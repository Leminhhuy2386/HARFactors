# Run report: HARFactors

## Final state (2026-10-16, end of this run)

Final verdict of stage 08: **`NOT_READY_FOR_SUBMISSION`** (`docs/review_report_iter3.md`). The science is done and
verified. What blocks submission is listed under § Open items.

### Gates (latest file of each; earlier files kept and never edited)

| Gate | Verdict | File | Note |
|---|---|---|---|
| G1 environment | PASS | G1_environment_rerun2.json | Tolerance changed after the first run (line endings, last-digit floats), declared. The first run and rerun1 FAILed |
| G2 data | PASS | G2_data.json | S1/S2/S3 complete; ActRecTut sizes equal the authors' files; UTD-MHAD 861 sequences |
| G3 baseline | PASS | G3_baseline.json | The Random Forest re-run equals Figure 6 to rounding |
| G4 seal | PASS | G4_test_seal.json | Pre-registration and code sealed at commit 761d6bd; prior reads of S2 means declared |
| G5 authenticity | PASS | G5_results_authenticity_rerun6.json | 408 runs re-executed, 373 exactly; independent REML; 9/9 negative controls; permutation control on the better of two REML fits (since rerun 6); 50/50 sampled H1 bootstrap refits at the maximum; label-shuffle control collapsed (6/6). Rerun5 FAILed (optimizer, see below); rerun4 FAILed (rebuilt caches); the first run and rerun1 FAILed (auditor gap, float boundary, 7 imported files without provenance), fixed with the user's approval |
| G6 traceability | PASS | G6_traceability_rerun6.json | 553 values re-derive (expanded manuscript, reproducibility values); the tampering control was caught |
| G7 manuscript | PASS | G7_compile_citations_rerun9.json | Expanded draft, restyled figures, verification paragraph on the full re-execution: 10 pages, 9 figures, 3 tables, abstract 140 words, 0 problems (page and figure limits waived, `paper/format_limits.json`) |

### Results

- **H1 (pre-registered):** success.
  - S2, 828 runs on 7 held-out datasets: R = 8.19 [2.85, 16.90] (97.5 %); pi_P = 82 % [64, 89].
  - Secondary criterion (pi_P lower bound ≥ 40 %) met.
- **Sensitivity analyses:**

  | Analysis | R | Interval |
  |---|---|---|
  | S2, balanced accuracy | 4.09 | [2.05, 6.03] |
  | S1, macro-F1 | 3.26 | [1.34, 4.47] |
  | S1, balanced accuracy | 1.89 | [0.69, 2.29], includes 1 |
  | S1 without ActRecTut | 3.93 | [1.63, 5.80] |
  | **S3, 17 WHAR architectures** | **R' = 0.21** | **[0.06, 0.36]: the network dominates** |

- **Exploratory:**
  - memory vs none: S4D +5.79 [+2.30, +9.27], 9/9 datasets;
  - Mamba − S4D: −2.45;
  - Random Forest margin vs channel count: +7.1 per sd (p < 0.001).
- Every number is in `docs/summary.md` / `docs/summary_values.json`. The manuscript prints no other.

### Reproducibility of every stored run (2026-10-16, user request)

The user asked for every result to be reproducible, with checkpoints and configs for every network. Report:
`docs/reproducibility_report.md`; numbers: `results/metrics/reproducibility/summary.json`.

- **Inventory.** S1 and S2 hold 3,408 stored runs; 3,402 have a checkpoint, and every one of those carries the
  registry's config and its record's config and fingerprint. `configs/` now has one file per network (10 population
  networks, 17 WHAR Arena models of S3) and the protocol (digest e8739ef2c5b36c99, equal to every record's).
- **Re-execution.** All 3,402 re-executed from their checkpoints: 2,363 reproduce their record exactly, 3,048 within
  the G5 tolerance. Of the 2,988 analysed runs (daphnet and motion_sense excluded), 2,982 re-executed, 2,185 exact,
  2,668 within tolerance, none more than 3.25 F1 points apart. The eight S1 datasets with a matching cache: 1,914 runs,
  1,905 within tolerance (9 WISDM runs 3-5 windows apart).
- **Why some runs do not match bit for bit.** Several whar-datasets parsers number activities and subjects, and order
  recordings, as `os.listdir` returns them. Most stored runs were built on Colab/Kaggle VMs, which list files in ext4
  hash order with a per-disk seed (two Colab VMs gave two orders for one dataset), so that order cannot be recreated:
  no download would help (the data on the cluster are complete).
  - DSADS: the order was recovered from the checkpoints (each network is near-perfect on its training subjects, so a
    Hungarian match of its outputs gives its labels and the least accurate subject its test subject; unanimous over 80
    checkpoints), the cache rebuilt in that order (`scripts/reconstruct_order.py`), and the Kaggle runs re-executed on
    it (median |dF1| 0.32). Only the order of the 60 files within a recording stays unknown.
  - Runs from several VMs per run set (DSADS on Colab, Daphnet, MotionSense): per-checkpoint re-execution, the test
    group identified by the record's test size where that is unique (Daphnet 300/300) and the z-score fitted on every
    non-test window.
  - USC-HAD (ids from the files' metadata): cache built on the cluster, split sizes equal for all 10 splits; the test
    sets are identical, only the validation draw (and so the train-only z-score) differs. The same holds for PAMAP2,
    UMAFall and w-HAR on the PC-built caches.
- **Six runs without a checkpoint** were retrained with the same harness, config, cache, split and seed
  (`scripts/retrain_missing.py`; outputs under that run's directory; the stored records untouched). Five reproduce
  their record's macro-F1 exactly; the sixth (S4D, ActRecTut, split 1) is 0.005 points apart.
  - Disclosure: that job was submitted without a GPU (`--gres=gpu:0`), but Slurm does not hide a node's GPUs from such
    a job, and PyTorch used hpc22's RTX A5000 for the 32-minute run. `scripts/hpc/submit.sh` now sets
    `CUDA_VISIBLE_DEVICES=''` whenever no GPU is requested.
- **Audit fix.** `audit --reproduce` re-generated sampled prediction files on the default cache; it now uses the cache
  recorded in the run's config (a sampled DSADS file had been made on the rebuilt cache).
- **Gates.** G6 rerun5/rerun6 and G7 rerun8/rerun9 PASS after the verification paragraph was rewritten with these
  numbers (rerun6/rerun9 add the retrained runs); `audit --all` PASS. S3 (510 runs) stays records only: retraining it
  was proposed and not approved.
- **Disk.** Private caches outside the project: harfactors_verify 67 GB, harfactors_rebuilt 22 GB,
  harfactors_rebuilt_colab 6.6 GB (normalised copies per test group); safe to delete, they rebuild from HAR.

### G5 rerun5 FAIL (2026-10-16): the optimizer in the model-permutation control

- **Run.** Slurm 86864 on hpc22 (Intel Xeon Gold 6226R, CPU cores of a GPU node, 8 threads), after the expansion
  below. Re-execution passed (408 of 408 gating runs within tolerance, 373 exact, 0 mismatches); the independent REML
  agrees on every analysis; 9 of 9 negative controls caught; the rebuilt-cache datasets are informative.
- **What failed.** The S2 model-permutation control: with the model labels shuffled within cells, the network-related
  share should vanish (criterion ≤ 0.03); it came out 0.110 (rerun3/4: 0.0011). The chain stopped at the PASS guard,
  so no G6 or G7 file was written, and `docs/summary*` were restored to the G5 rerun3 / G6 rerun2 state.
- **Diagnosis** (`docs/diagnostics/`, read-only script and its output):
  - statsmodels' default optimizer reported convergence at M = 16.3, a point whose restricted log-likelihood is
    13.75 worse than the optimum (-l_R 1881.20 against 1867.45). At the optimum M = MD = MP ≈ 0, as the independent
    numpy REML finds.
  - Whether the default optimizer stops there depends on the BLAS thread count: with 4 threads on the same node it
    failed to converge, the Powell fallback found the optimum (share 0.0011). Other permutation seeds give shares
    from 0.002 to 0.099 for the same reason.
  - The real-data fits are not affected: on every analysis the independent REML matches statsmodels (H1: R 8.19 in
    both). The bootstrap resamples are fitted by statsmodels only; they were not checked against the independent
    implementation.
- **Resolved (the user approved the fix on 2026-10-16).** The control now fits with statsmodels and with the
  independent REML and keeps the solution with the higher restricted likelihood (both recorded); the threshold is
  unchanged. A seeded sample of 50 of the 1,000 H1 bootstrap refits was checked against the independent REML: statsmodels
  reached the maximum in all 50 (largest shortfall 8e-7; R within 0.03 %). G5 rerun6 PASS (Slurm 86918, hpc22), then
  G6 rerun4 PASS, G7 rerun6 PASS and `audit --all` PASS on every check.

### Expansion of 2026-10-16: more figures, exploratory analyses (user request)

The user asked for many more figures (sensors, experiments, detailed network structure) and a deeper analysis of the
embeddings and other characteristics. Exceeding the page count is acceptable for this draft (the user will shorten it).
Everything below is descriptive or exploratory; no network was trained. Details: `docs/experiment_record.md`
(§ HARFactors, 2026-10-16).

- **New analyses** (each with provenance and ledger runs):
  - `split_spread`: split means vs network means per dataset. In 15 of 16 datasets the split means vary more than
    the network means (median ratio of sds 2.6); the exception is ActRecTut (two subjects). Also validation
    (training subjects) vs test (unseen subjects): validation is higher in 2,307 of 2,388 runs, by a median of 17.1
    points per dataset (4.6 to 47.8), and picks the test-best network in 3 of 13 datasets.
  - `model_anatomy`: parameters per part (all ten equal to G1); memory parameters of 766 S1 checkpoints against
    freshly initialised networks. They barely move: quantile levels by a median of 0.0047 (97 % by < 0.05),
    recurrence radii by 0.014, S4D decay rates by 0.028 in log10; play widths and S4D time constants keep their
    initial distributions.
  - `embedding_analysis` (inference only, seed-0 checkpoints, 11 datasets, 754 network-split cases; regenerated
    macro-F1 equal to the stored one in 686, at most 0.20 points apart): the test group's class-conditional shift in
    the embedding goes with lower split F1 (pooled within-dataset rho -0.46, p < 0.001; negative in 10 of 10
    datasets), more closely than the same shift in sensor features (rho -0.32); a linear probe identifies the
    training subject at a median 3.2 times chance from the embeddings (4.0 from window statistics).
- **New figures and tables** (`src/visualization/detail_figures.py`): Fig. design (pipeline, run grid, UCI-HAR test
  groups), sensors (18 datasets), architecture (skeleton, block, ten networks), memories (operators on a synthetic
  input; learned vs initial parameters), split spread (with network x split heatmaps), validation vs test,
  embeddings (t-SNE case, shift vs F1, subject probes); Table of datasets and table of networks.
- **Manuscript:** new sections Datasets and sensors, Networks, Embedding analysis, and results on spread, validation,
  memories and embeddings; six new references, verified (`paper/references_verified.json`). Ten pages.
- **Format limits:** page and figure limits waived for the working draft (`paper/format_limits.json`); the
  150-word abstract limit stays. Restore them before submission.
- **Jobs:** the CPU partitions were full, so the embedding and figure jobs ran on idle CPU cores of the GPU nodes
  without a GPU (`--gres=gpu:0`; `docs/hpc.md` allows the `gpu` partition for regenerating predictions).
- **Figures restyled** (user request, 2026-10-16): the dataviz reference hues at their deeper steps (validated with
  the dataviz validator on the light surface: worst adjacent CVD delta E 8.4, normal-vision 19.3), black ink for text
  and axes, thicker marks, no faded layers, the seed term in neutral grey; colour now also encodes whether an
  interval clears its threshold (Figs. 1 and 2, stated in the captions). The network figure was redrawn as a
  dataflow diagram (tensor shapes on the edges, the residual block with its skip connection, a table of the memory
  update rules). Numbers unchanged; G7 rerun7 and `audit --all` PASS.
- **Gates:** the pending rerun5 chain (Slurm 86823) was cancelled before it started, so that its negative controls
  would not see the embedding runs' open ledger lines; G5 rerun5, G6, G7 and the audit ran once afterwards (Slurm
  86864), with nothing else running.

### G5 rerun4 FAIL (2026-10-16, after adding the pamap2, uma_fall and w_har caches)

- **Caches.** Copied from the PC: sha256 OK for 853 files; the harness reproduces all 29 split sizes.
- **Re-execution.** The S2 runs of pamap2, uma_fall and w_har were re-executed: 762 cells (Slurm job 86815).
  - 720 of 756 gating runs are within the declared tolerance (≤ 2 windows, ≤ 2 F1 points).
  - **36 are outside, all on the three new datasets**: pamap2 31, uma_fall 2, w_har 3.
  - The test sizes are identical. The differences are small and of both signs.

    | Dataset | Mean ΔF1 | Max \|ΔF1\| | Largest gap |
    |---|---|---|---|
    | pamap2 | −0.07 | 0.67 | 12 windows of about 1,700 |
    | uma_fall | −0.002 | 0.46 | |
    | w_har | −0.002 | 3.25 | a 250-window test split |

  - Likely cause: the stored runs of these datasets used caches built on cloud VMs, while the caches now on the cluster
    were built on the PC on 2026-10-15. The splits are the same; the inputs differ at float level.
- **Verdict.** `G5_results_authenticity_rerun4.json` FAIL is kept, and the tolerance was not changed. `audit
  --authenticity` now reports it.
- **Products of the same job chain.** G6_traceability_rerun3 and G7_compile_citations_rerun5 were produced
  automatically from a summary based on the FAIL gate. They are superseded and not valid. `docs/summary*`,
  `docs/artifact_index.md` and `paper/manuscript.*` were restored to the G5 rerun3 / G6 rerun2 / G7 rerun4 state.
- **Waiting for the user's decision** on how to treat these three datasets.

### Open items (the user's decisions)

1. ~~Label-shuffle control~~: **resolved 2026-10-16** (approved by the user).
   - S4D, 6 runs on permuted training labels: test macro-F1 4.9 against 8.7 by chance and 54.5 with the true labels.
   - G5 rerun3 and G6 rerun2 PASS. See `docs/review_addendum_2026-10-16b.md`.
2. ~~Figure 6 spot check~~: **resolved 2026-10-16.** Four cells match the paper
   (`docs/sources/figure6_spot_checks.json`, `docs/review_addendum_2026-10-16.md`).
3. **Re-execution of the remaining S2 runs** (check 16; 468 runs).
   - pamap2, uma_fall and w_har need paper-era caches added to HAR (the user's approval).
   - usc_had needs a cache built as the stored runs were (unsorted parser).
4. **Submission formalities:**
   - the author block and affiliations;
   - the subject category ("Sensor Applications" set, to be confirmed);
   - the graphical abstract (Fig. 1, replaceable);
   - at submission: 2 suggested reviewers, ORCID and the EDICS category;
   - the Sensors Letters scope risk (data-analysis papers may be declined without review).
   - `IEEE_lsens.cls` is **in place since 2026-10-16** (G7 rerun3).
   - PC downloads: `/work/huy.leminh/.cache/harfactors_scratch/pc_downloads/2026-10-16/` (36 files, SOURCES.txt).
     They stay outside the project (copyright).
5. **dsads:** on the HPC cache its checkpoints score at chance. Its runs (in S1) rest on their records.
6. ~~G5 rerun5 FAIL~~: **resolved 2026-10-16** (fix approved by the user; G5 rerun6 PASS).
7. **Expanded manuscript** (2026-10-16): rendered and checked (G6 rerun4, G7 rerun6 PASS): 10 pages, 9 figures,
   3 tables. The page and figure limits are waived for the draft; the user will shorten it, and the venue limits
   must be restored in `paper/format_limits.json` before submission.

### Copy back to the PC

The HPC copy is now the working copy: git repository, environment, gates, results.
- The PC copy has none of the 2026-10-15/16 changes.
- Copy the HPC project back as a whole (it is a git repository; e.g. clone it), rather than file by file.
- Do not edit the PC copy in the meantime.

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
