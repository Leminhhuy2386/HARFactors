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
