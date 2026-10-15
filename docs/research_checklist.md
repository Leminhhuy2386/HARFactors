# Research checklist (HARFactors, IEEE Sensors Letters)

One line per deliverable of the pipeline in `agents/`.

- Mark an item done only when the file it names exists and its check has run.
- State lives in `RUN_REPORT.md`; gates are in `results/gates/`.

## Stage 02: plan (gate G1)

- [x] Confirmatory analysis pre-registered before it exists: `docs/preregistration.md` (2026-10-15, frozen, sha256 in
      `RUN_REPORT.md`).
- [x] Decisions recorded:
  - no new training;
  - 3 seeds, disclosed as a limitation;
  - dsads and daphnet verified by dataset means unless a Linux VM is approved.
- [ ] Literature with verified ids, in `docs/research_plan.md` § Related work and `paper/references_verified.json`.
- [ ] Target journal fixed in `docs/research_plan.md` § Target journal: format, limits, class, reference style.
- [ ] Version control: `git init` and an initial commit.
- [ ] `results/gates/G1_environment.json`. It covers the imports, the device, the paper-era guard, a forward pass of
      each population model, loading checkpoints, the data pipeline with the shared cache unchanged, the analyses
      reproducing byte for byte, and a LaTeX stub.

## Stage 03: data and baseline anchor (gates G2, G3)

- [ ] `data/processed/splits.json` and `manifest.json`. They *record* the harness splits (split sizes and test
      subjects) for the datasets of S1 and S2; they do not define them.
- [ ] `docs/baselines_cited.json`: the 510 quoted cells of the WHAR Arena paper's Figure 6, each with its location.
- [ ] Protocol identity of the quoted cells (`exact` or `partial`), and the evaluation-stack anchor (the 17 models
      re-run on 4 datasets).
- [ ] Leakage rules of `agents/03_data.md` Step 3.5, stated for the WHAR protocol in `docs/research_plan.md`.
- [ ] `scripts/audit.py`: `--provenance`, `--ledger`, `--data`, `--synthetic`.
- [ ] `results/gates/G2_data.json`, `results/gates/G3_baseline.json`.

## Stage 04: confirmatory analysis (gate G4)

- [ ] Analysis code for H1 and the sensitivity analyses, written and tested on S1 and on synthetic data only.
- [ ] Inclusion check of S2 (split sizes and protocol hash, no metric read).
- [ ] `results/gates/G4_test_seal.json`: sha256 of the code and of `docs/preregistration.md`, sealed before S2's
      metrics are read.
- [ ] One run on S2, then the sensitivity analyses, each with a provenance record.
- [ ] Figures as PDF (`figures/fig0x_*.pdf`) with `results/figure_data/`; tables with `results/table_data/`.
- [ ] `docs/experiment_record.md` updated.

## Stage 05: verification (gate G5)

- [ ] A second implementation of the variance decomposition.
- [ ] Predictions regenerated from checkpoints for sampled runs on the datasets that can be re-executed per split;
      metrics recomputed from them.
- [ ] Negative controls, including permuting model labels within (dataset, split, seed) cells.
- [ ] `docs/verification_report.md`, `results/gates/G5_results_authenticity.json`.

## Stage 06: summary (gate G6)

- [ ] `docs/summary.md` and `docs/artifact_index.md`. Every number traces to a file.
- [ ] `results/gates/G6_traceability.json`.

## Stage 07: manuscript (gate G7)

- [ ] `paper/manuscript.tex`, `paper/references.bib`, `paper/references_verified.json`, `paper/figures/`.
- [ ] `results/gates/G7_compile_citations.json`.

## Stage 08: review

- [ ] `docs/review_report_iter<N>.md` and the revised manuscript, ending with a verdict.
