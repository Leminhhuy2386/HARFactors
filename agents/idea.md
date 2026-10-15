# IDEA: Who is tested matters more than the architecture

## Core idea

Under one fixed, published, person-disjoint protocol (WHAR Arena v1), measure which factors move macro-F1 on people a
model has never seen.

- **Data factors:** channel count, number of training subjects, classes, sampling.
- **Model factors:** temporal memory and its type, window-level side paths, size.
- **Evaluation factors:** which person is tested, the training seed.

## Evidence already in hand (computed in Edge_SSM, reproduced here byte for byte; still to be verified by 05_verify)

- Variance decomposition of 2,460 runs (10 models × 10 datasets × subject splits × 3 seeds):
  - the dataset explains 76.6 % of all variance;
  - within a dataset: test person 61 %, seed 14 %, model × person 11 %, model × dataset 7 %, model main effect 6 %.
- Controlled contrasts in one shared skeleton:
  - temporal memory vs none +5.95 [+2.79, +9.10], 10/10 datasets;
  - Mamba − S4D −2.39 (p 0.010);
  - window laws on the recurrence trunk +2.41 [+0.35, +4.47].
- Over the paper's 30 datasets:
  - the Random Forest's margin over the best deep model grows with the channel count (+7.2 points per sd, p < 0.001);
  - fewer training subjects make the model choice matter more (−1.5 per sd, p 0.005);
  - nothing predicts difficulty (R² 0.03).
- Noise floor: person ~8 points, memory type ~4, seed ~1.7.

## Central question

How much of a reported HAR improvement can the architecture explain, compared with who happens to be tested?

## Limits to respect

- The data factors are correlational (subjects and channels correlate, ρ −0.50).
- The 30-dataset part re-analyses the WHAR Arena paper's single runs.
- The 10 models share one skeleton: that makes the contrasts controlled, but they are not 10 independent designs.
