# Paper plan: IEEE Sensors Letters

On 2026-10-14 the results were split into three papers:
- **P1 (`D:\Projects\RILA`, IEEE Sensors Journal):** the architecture.
- **P2 (this project):** what drives accuracy.
- **P3 (`D:\Projects\CortexHAR`, IEEE Embedded Systems Letters):** deployment.

P2 cites P1 (as under review) for the models it uses as a population. It must not repeat P1's main claim.

## Working title

"Who Is Tested Matters More Than the Architecture: What Drives Wearable Activity-Recognition Accuracy Under
Person-Disjoint Evaluation"

## Claims

These are exploratory findings (Edge_SSM, 2026-10-12). The one confirmatory test is pre-registered in § Pre-registration.

1. **Under a fixed person-disjoint protocol, the test person dominates.**
   - Within a dataset, the test person explains 61 % of the variance of 2,460 runs.
   - The training seed explains 14 %; the model main effect 6 %, about a tenth of the person's share.
   - Typical size: person 7.4 points, seed 3.6, model 2.4.
2. **Temporal memory is the only architectural component worth several points.**
   - Memory vs none: +5.95 [+2.79, +9.10], 10/10 datasets.
   - Which memory: 1–2 points (Mamba − S4D −2.39, p 0.010; RecurrenceNet / HysteresisNet / QuantileNet vs S4D: n.s.).
   - Window statistics: large on single datasets only (ActRecTut +21, +0.6 on the other 9).
3. **The data predict when classical models win.**
   - Over the WHAR Arena paper's 30 datasets, the Random Forest's margin over the best deep model rises with the
     channel count: +7.2 points per sd, p < 0.001, with the other properties held fixed.
   - With fewer training subjects, the gap between the best and the median deep model grows: −1.5 per sd, p 0.005.
   - Nothing predicts how hard a dataset is (R² 0.03). Model size does not predict accuracy (ρ −0.43).
   - The held-out confirmation agrees: RILA-Net loses only on UP-Fall and PAMAP2 (42–52 channels), where the Random
     Forest is strongest.
4. **Rankings are fragile.**
   - Person spread ~8 points, memory type ~4, seed ~1.7.
   - Removing the one 2-person dataset reorders the top models.
   - A tuned classical baseline (segment statistics + ExtraTrees ~74.5) is far above the paper's Random Forest (65).
   - Recommendation: report person-level spread and seeds, and compare paired by split.

## Evidence

| Item | File |
|---|---|
| Variance decomposition | `results/metrics/factors/data.json`, `results/metrics/factors/summary.md`, `figures/variance.svg` |
| Component contrasts | `results/metrics/factors/` §C, `figures/components_forest.svg`, `components_heatmap.svg` |
| Data factors (30 datasets) | `results/metrics/factors/`, `figures/data_factors.svg`, `data_corr.svg` |
| Evaluation noise | `figures/eval_noise.svg`, `results/metrics/seeds/`, `experiments/person_probe/` |
| Overview figure | `figures/factor_map.svg`, `ranking.svg` |
| Held-out agreement | `results/metrics/heldout/summary.md` |
| Classical references | `experiments/tree_probe/`, `experiments/elastic/` |

## Format and missing items

1. A letter: check the current IEEE Sensors Letters limits (pages and figures) before writing. Plan on about 3
   figures (variance, data factors, components) and 1 table.
2. Methods paragraph: the protocol (`docs/protocol.md`).
   - State that the 30-dataset part re-analyses the WHAR Arena paper's published single runs.
   - Our 2,460 runs cover 10 datasets × 10 models × 3 seeds.
3. Make the variance model explicit: the sum-of-squares decomposition in `src/evaluation/factor_analysis.py`
   (exploratory) and the pre-registered mixed-effects (REML) model (§ Pre-registration).
4. Print-ready figures (vector PDF, ≥ 8 pt).

## Related work (verified 2026-10-15; `paper/references_verified.json`, notes in `docs/literature_notes.md`)

- **Benchmark and data.**
  - WHAR Arena [burzer2026arena] defines the person-disjoint protocol used here: k = 10 subject groups, test only
    on unseen subjects, mean and sd over splits. It finds the state of the art "distributed rather than dominated
    by a single architecture". It reports one run per split, with no seed replicates.
  - WHAR Datasets [burzer2025whar] provides the caches and splits.
- **Evaluation and people.**
  - Random cross-validation of segmented sensor data is optimistic [hammerla2015lets].
  - How training and test data are partitioned across people decides the reported accuracy
    [lockhart2014limitations].
  - Activity patterns differ between people [qian2021latent], and so do devices [stisen2015smart].
  - Background on the activity recognition chain: [bulling2014tutorial].
- **Variance in benchmarks.** Comparisons must account for data-split and seed variation
  [bouthillier2021accounting]. Classifiers are compared over multiple datasets with rank tests [demsar2006statistical,
  wilcoxon1945individual, holm1979simple].
- **Models in the population.** S4 and S4D [gu2022efficiently, gu2022parameterization], Mamba [gu2023mamba]. The WHAR
  baselines include DeepConvLSTM [ordonez2016deep], deep and recurrent HAR models [hammerla2016deep], and Random
  Forest [breiman2001random] on tsfresh features [christ2018tsfresh].
- **Statistics.** REML [patterson1971recovery] fitted with statsmodels [seabold2010statsmodels]; the cluster bootstrap
  [field2007bootstrapping].
- **Gap.**
  - No study separates, under one fixed person-disjoint protocol, how much of macro-F1 variance comes from the test
    subjects, the model and the training seed.
  - WHAR Arena has a single run per split, so it cannot separate seed noise from the person effect.

## Target journal (fixed 2026-10-15)

- **Journal:** IEEE Sensors Letters, a regular letter.
- **Length:** at most 4 pages, which must include at least one column of references. The source is the journal's
  Information for Authors (ieee-sensorsletters.org); papers longer than this go to IEEE Sensors Journal.
- **Class:** `IEEE_lsens.cls` (IEEE / Overleaf template `bare_lsens.tex`, 9 pt, two columns).
  - The class could not be downloaded from the cluster: the TLS-inspecting firewall blocks Overleaf, IEEE and arXiv.
  - Drafting uses `IEEEtran.cls [journal,9pt]`, a substitute recorded in `results/gates/G1_environment_rerun2.json`.
  - Before submission, swap in `IEEE_lsens.cls` and check the page count again. **Done 2026-10-16:** the 2023 class
    from the IEEE Template Selector, fetched on the PC. The letter is 3 pages; the abstract is 140 words (limit 150);
    a graphical abstract is required.
- **Budget:** about 3 pages of body, 3 figures (variance shares, data factors, component contrasts) and 1 table.
- **Structure:** title, abstract, Index Terms, Introduction, Methods, Results, Discussion and limitations,
  Conclusion, References.
- **Reference style:** IEEE numeric. Every entry is verified (`paper/references_verified.json`).

## Data records and leakage prevention (stage 03, `agents/03_data.md` Step 3.5)

The analysed data are stored run records. Their windows, splits and normalisation were produced by the paper-era
harness. How each leakage rule is satisfied:

1. **Split by unit.** Splits are by subject (LKSOSplitter, k = 10 subject groups).
   - `scripts/export_splits.py` regenerates the splits of the 13 cached datasets and asserts that no test subject has
     a window in train or validation.
   - Each split's test subjects are in `data/processed/harness_splits.json`: 1 to 4 subjects per split.
2. **Split before anything else touches the data.**
   - Windows are cut per session, and whole subjects are then assigned to folds.
   - The z-score is fitted only after splitting.
3. **Preprocessing is fitted on train only.**
   - The z-score is fitted on the training split's windows (`PostProcessingPipeline(cfg, ..., split.train_indices)`).
   - Class weights for the loss come from the training frequency.
   - The validation windows are 20 % of the training subjects' windows, drawn per window. This is a property of the
     protocol, inherited and kept (`docs/protocol.md`). It affects early stopping only; test subjects are never seen.
4. **Features available at prediction time.** The inputs are raw 3-s windows of the sensor channels. No input is
   derived from labels, from whole-session quantities, or from future samples beyond the window.
5. **Validation is carved from train, never from test**, as the protocol prescribes.
6. **No test-time statistics.**
   - Evaluation runs the model in eval mode, with the training-split normalisation.
   - The models' LayerNorm acts per sample.
7. **At the level of this letter's analysis:**
   - the outcome is each run's test macro-F1, and the factors are design labels (dataset, split, model, seed);
   - the confirmatory set S2 is sealed until gate G4 (`docs/preregistration.md`).
   - **Inclusion.** By the pre-registered inclusion rule, a dataset whose stored runs disagree on split sizes is
     excluded, because its split ids do not name the same test subjects across runs (S1: daphnet; S2: motion_sense;
     `results/gates/G2_data.json`).

## Reviewer risks

| Risk | Answer |
|---|---|
| "Correlational: subjects and channels are correlated (ρ −0.50)" | Multiple regression with both; say "associated with", not "causes"; show the partial effects. |
| "One protocol only" | The point is a fixed, published protocol (WHAR Arena v1). Name other protocols as future work. |
| "The 10 models are mostly yours" | They share one skeleton, which is what makes the component contrasts controlled. The 30-dataset part uses the paper's 17 independent models. |
| "Single runs in the paper's numbers" | Our seed analysis bounds the seed noise (~1.7 points), and the conclusions hold well above it. |

## Pre-registration

Pre-registered on 2026-10-15 and approved by the user: `docs/preregistration.md`. The file is frozen, its sha256
is recorded in `RUN_REPORT.md`, and changes go only under its § Deviations. In short:
- **H1 is the only confirmatory hypothesis.** Within a dataset, the test-subject group explains more variance of
  macro-F1 than all model-related sources together: R = σ²_P / (σ²_M + σ²_MD + σ²_MP), by REML.
- **Data.** H1 is tested on the 948 held-out runs (4 models × 8 datasets × 3 seeds). No new training.
- **Criteria.**
  - Success: the lower bound of the 97.5 % cluster-bootstrap interval of R is above 1.
  - Failure: R is at most 1, which sends the letter to the negative-result branch.
- **Sensitivity analyses** (they do not decide H1): balanced accuracy, the 2,460 development runs, and the 17
  independent WHAR Arena architectures.
- **Exploratory, and labelled so:** claims 2–4 (memory, data factors, evaluation noise) and claim 1 as measured on
  the development runs.
