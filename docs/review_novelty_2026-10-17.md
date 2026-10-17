# Review of the manuscript, 2026-10-17: accuracy of the claims and novelty

This is an external-style review, outside the stage-08 iterations. It reviews `paper/manuscript.tex` as rendered on
2026-10-16 (10 pages, 9 figures, 3 tables), with the most weight on novelty. Line numbers refer to
`paper/manuscript_src.tex`.

The reviewer ran some checks of their own:
- script `/work/huy.leminh/.cache/harfactors_scratch/review_2026-10-17/review_checks.py`, Slurm job 87058 on hpc02;
- output in `review_checks_out.json` next to the script;
- they read S1 and S3 run records and `docs/sources/arena_extracted.json` only. **S2 was not read**;
- they wrote nothing under `results/`, and they are not in the ledger.

These checks are exploratory reviewer checks. Anything taken from them into the letter must go through stages 04-06
with provenance.

## 1. Verdict

| Aspect | Assessment |
|---|---|
| Rigour, verification, transparency | Very strong, well above the HAR norm: pre-registration, sealed code, independent REML, negative controls, label-shuffle control, full re-execution |
| Correctness of the confirmatory result | Sound. H1 success on S2 is correctly computed and honestly bounded |
| Novelty of the headline | **Low to moderate.** The phenomenon is known qualitatively. What is new is the formal crossed decomposition with a pre-registered held-out test. The size of R is largely set by the choice of near-equivalent networks |
| Novelty that is present but under-sold | Moderate: (i) an error budget for paired comparisons; (ii) on S3, the network × dataset interaction is the largest term; (iii) the benchmark's validation protocol misranks networks |
| Accuracy of the text | 7 statements need correcting or qualifying (§ 3). Two of them change the interpretation |
| Fit to venue | 10 pages and 9 figures against 4 pages and 5 figures. High scope risk for IEEE Sensors Letters (AI/ML or data analysis without impact on the sensor system) |

## 2. Novelty

### 2.1 What is already known (prior art the letter must engage with)

| Work | What it already shows | Cited? |
|---|---|---|
| Jordão et al. 2018, arXiv:1806.05226 | Under leave-one-subject-out evaluation, accuracy varies widely across subjects. The CIs are so wide that the methods are "statistically equivalent" | No |
| Barshan & Yurtman, Comput. J. 59(9) 2016, doi 10.1093/comjnl/bxv093 | Quantifies inter-subject variation on DSADS (one of S1's datasets) | No |
| Lockhart & Weiss 2014; Hammerla & Plötz 2015 | The partitioning of people and windows changes the reported accuracy | Yes |
| Bouthillier et al. 2021 | General ML: data-sampling variance dominates seed variance, and benchmarks must randomise over sources of variation | Yes |
| Varoquaux, NeuroImage 180 (2018) 68-77 | Small test samples give error bars of about ±10 %, larger than typical method gains | No |
| WHAR Arena (Burzer et al. 2026) | Reports the SD over splits, warns that small differences "should not be overinterpreted", and finds the SOTA "distributed" | Yes |
| Iwasawa et al., IJCAI 2017, doi 10.24963/ijcai.2017/268 | User identity can be decoded from the features of HAR networks | No (relevant to § Embeddings) |
| Gholamiangonabadi (thesis, Western Univ., ir.lib.uwo.ca/etd/7722); probably IEEE Access 2020 (not verified) | Model selection must be subject-disjoint (LOSO for model selection) | No |
| Dehghani et al. 2019, arXiv:1904.02666 | Subject cross-validation, and the bias from overlapping windows | No (arXiv only) |
| Int. J. Med. Inform. 2026, S1386505626002042 (read from a search summary only; full text not reached) | GLMM variance decomposition of predictive variance across split strategies (skeleton HAR and rehabilitation data) | No: the closest methodological precedent; read it and set the letter apart from it |

Consequence: the sentence "their sizes in wearable HAR are unknown" (l. 67) overstates. They were not *decomposed*,
but the dominance of subject spread over method differences was already shown by Jordão (2018) and the Arena SDs.
Reviewers in this community will call the headline "expected".

### 2.2 Why the headline R = 8.19 is largely by construction

- Three of S2's four networks are RecurrenceNet variants: RecurrenceNet, RN+stats and RN+laws, plus S4D. Their
  differences were known to be 1.6-4.8 points (`results/metrics/heldout/summary.md`). σ²_M + σ²_MD = 8.8 is therefore
  small by design.
- With σ_P ≈ 11 points between test groups (typical of LOSO-style splits), R > 1 was close to certain. The
  pre-registration protects against analytic flexibility. It does not make H1 a risky prediction.
- **Reviewer check A (S3 subsets, exploratory, 200 bootstrap resamples):**

  | S3 population | n | σ²_P | σ²_M | σ²_MD | σ²_e | R′ [97.5 %] |
  |---|---|---|---|---|---|---|
  | all 17 (manuscript) | 510 | 37.5 | 100.2 | 51.0 | 29.2 | 0.21 [0.06, 0.37] |
  | 14 deep (no RF, SVM, k-NN) | 420 | 42.7 | 37.5 | 56.4 | 29.9 | **0.34 [0.10, 0.65]** |
  | without SVM and k-NN | 450 | 42.9 | 38.0 | 52.6 | 29.5 | 0.36 [0.10, 0.67] |
  | top 8 by Arena mean on the other 26 datasets | 240 | 55.2 | 20.1 | 8.1 | 15.2 | 1.28 [0.24, 2.37] |
  | top 5 by Arena mean on the other 26 datasets | 150 | 53.5 | 3.6 | 12.6 | 12.6 | 1.86 [0.36, 3.81] |

  - The top-k sets are chosen on datasets outside S3, not on the scores analysed.
  - Among the published *deep* networks, the network still dominates. The subject term overtakes only for the five
    most competitive ones, and even there the interval includes 1.
  - So the claim holds for "networks within a few points of each other". That is close to a tautology, and it is much
    narrower than the title ("How much of … accuracy is the network?").

### 2.3 Novelty that is present but under-sold

1. **An error budget for comparisons.** Computed by the reviewer from the S2 point estimates in Table I:

   | Quantity | Formula | SD (F1 points) |
   |---|---|---|
   | true difference between two networks in a dataset | √(2(σ²_M+σ²_MD)) | 4.2 |
   | noise, unpaired, one split, one seed (e.g. two papers with different hold-out subjects) | √(2(σ²_P+σ²_MP+σ²_e)) | 17.1 |
   | noise, paired, one split, one seed | √(2(σ²_MP+σ²_e)) | 6.1 |
   | noise, paired, 10 splits × 1 seed | √(2(σ²_MP+σ²_e)/10) | 1.9 |
   | noise, paired, 10 splits × 3 seeds | √(2(σ²_MP/10+σ²_e/30)) | 1.5 (minimum detectable difference ≈ 4.1 at 80 % power) |
   | floor with infinitely many seeds, 10 splits | √(2σ²_MP/10) | 1.2 |
   | SE of a dataset mean over 10 test groups, as an estimate for new people | √(σ²_P/10) | 3.6 |

   - This turns "average several seeds" into a sizing rule: beyond about 3 seeds the network × test-group term is the
     floor, so more *subjects or test groups* are needed, not more seeds.
   - A typical difference between comparable networks (about 4 points) sits at the detection limit of a 10 × 3
     design.
   - No HAR paper gives this. It is the most original practical contribution available from the existing data.
2. **S3 re-read.** For the 14 deep networks, the network × dataset interaction (σ²_MD = 56) is the largest term. It
   exceeds both the test-subject term (43) and the overall network term (37.5). "Which architecture suits which
   dataset" outweighs both "who is tested" and "which architecture overall". This gives a quantitative account of the
   Arena's "distributed state of the art", and it is new.
3. **The validation protocol of the benchmark.**
   - The harness draws validation windows at random from the training subjects' windows
     (`third_party/whar-datasets/src/whar_datasets/splitting/splitter.py:25-35`, a permutation of window indices). With
     50 % overlap, validation windows share samples with training windows.
   - Early stopping, and any model selection, therefore runs on a near-resubstitution estimate.
   - Quantifying that this inflates the score by 17 points and misranks networks *within the standard benchmark* is
     actionable, for the Arena and for anyone who uses it. It needs a reliability ceiling, though (§ 3, item 6).

### 2.4 Parts with little novelty or off the subject

- **What the memories learned (Fig. 6, § What the memories learned).** It is off-topic for a letter about evaluation,
  and it brushes against the rule that the architecture is not the subject. Parameters that barely move can just
  reflect the learning rate or the parameterisation. Cut it.
- **Networks and architecture (Fig. 5, Table III).** Reduce to one sentence and a reference to `configs/`.
- **Memory vs none (Fig. 2a).** An architecture result, and misdescribed (§ 3, item 1). Cut it or move it to
  supplementary material.
- **Embeddings (Fig. 8).**
  - The link between shift and F1 in the classifier's own input space is close to circular.
  - The sensor-space version (ρ = −0.32) is the informative one.
  - The subject probe repeats Iwasawa et al. (2017).
  - At most one sentence.
- **Regressions over 30 datasets.** Keep only the channel → RF-margin association (§ 3, item 7).

## 3. Accuracy of statements: what must change

1. **l. 332, "S4D gains … over the same network without memory": false.**
   - The no-memory control is the 32×3 play-width-0 skeleton (19,995 parameters).
   - S4D uses its own 64×2 blocks (27,675 parameters).
   - The contrast confounds memory with width, depth and size.
   - RecurrenceNet − no memory is the same-skeleton contrast (+6.43 on 9 datasets,
     `results/metrics/contrasts_robustness/summary.md`). It names RILA-Net's family, though, which is another reason
     to cut the section.
2. **l. 404, "a weak baseline is told apart from a strong network easily": not supported.** Without SVM and k-NN,
   R′ = 0.36, and with the 14 deep networks only, 0.34 (§ 2.2). The reversal comes from heterogeneity among the deep
   networks, mostly σ²_MD, not from the weak baselines.
3. **l. 187, disclosure.** `docs/preregistration.md` § Data says that S2's per-dataset means *and paired model
   differences* were known. The manuscript names only the means. The paired differences bear on the denominator of R,
   so say so.
4. **"Who is tested" against the split; and the level the claim applies to (abstract l. 48, Discussion).**
   - (a) A split also changes the training and validation subjects. P is the split term, not a pure property of the
     test subjects. Say this once.
   - (b) Under the Arena protocol every subject is tested once, so a *dataset mean over all k groups* is not
     re-drawn by "who is tested" the way a single split is.
   - The claim "reported accuracy depends more on who is tested than on the network" therefore holds for single-split
     hold-out reports, for the per-user spread at deployment, and for inference to new people (SE ≈ 3.6 points). It
     does not hold for paired benchmark means.
   - The abstract should say which.
5. **l. 299, the cause of the validation gap.** Validation windows are random windows of the training sessions
   (overlap 50 %), so the gap mixes "subjects seen" with "windows overlapping the training windows" (Hammerla &
   Plötz 2015). It cannot be attributed to the people alone.
6. **l. 303-306, the evidence on validation ranking lacks a noise ceiling.** Reviewer check B, on the 6 S1 datasets
   with validation recorded (S2 not read):

   | Median over 6 S1 datasets | Value |
   |---|---|
   | ρ(validation means, test means) | 0.52 |
   | ρ(test, seed 0 against test, seeds 1-2; same splits) | 0.87 |
   | ρ(test, even splits against test, odd splits; different subjects) | 0.72 |
   | best by validation = best on test | 1 of 6 |
   | best on even splits = best on odd splits | 1 of 6 |
   | test points lost: choosing by validation / by seed 0 | 3.26 / 0.65 |

   - The claim survives: validation ranks worse than an independent test estimate.
   - "Best on validation is best on test in 3 of 13" carries no information, because even independent halves of the
     test pick the same winner only 1 time in 6.
   - Report ρ against the ceilings instead.
7. **l. 336-340 and 406-407, the regressions over 30 datasets.**
   - There are 16 frozen coefficient tests with no correction.
   - Under Holm, only channels → RF margin (p = 1e-5) and channels → spread over splits (p = 0.0016) survive.
   - Subjects → best − median gap (p = 0.0041) fails: the Holm threshold is 0.0036.
   - The "best − median" outcome is an order statistic. Across the 30 datasets it tracks the mean split SD (ρ = 0.60,
     p < 0.001, check C). Datasets with few subjects also have fewer splits, so their network means are noisier and
     their maximum is inflated.
   - Drop "with few recorded subjects the choice of network matters more" (l. 406-407), or test it against a noise
     null.

Smaller points:
- **Abstract.** The interval resamples test groups with the four networks fixed. Say "for these networks" in the
  abstract, not only in Limitations.
- **Abstract.** The sentence "The ordering holds with a ceiling-free metric and on development datasets, where that
  metric leaves it unresolved" is hard to parse. Split it into two claims.
- **Title.** It promises a general answer that S3 contradicts.
- **Embeddings, l. 413.** The sentence "in the space the classifier reads" makes the circularity explicit. Lead with
  the sensor-space result.

## 4. Recommended reframing for a 4-page letter

- **Title**, for example: "Who Is Tested, Which Network, Which Dataset: An Error Budget for Person-Disjoint Wearable
  HAR Comparisons".
- **Keep**, in order:
  1. H1, pre-registered, for comparable networks. One figure: the variance shares, with S3 beside them.
  2. S3, including the deep-only population: the network × dataset interaction dominates among published deep
     networks. This needs a new exploratory analysis through the pipeline.
  3. The error-budget table: paired against unpaired, splits against seeds, the floor, and the minimum detectable
     difference.
  4. Validation in the benchmark protocol: the inflation and the misranking against a reliability ceiling. This is
     actionable for WHAR Arena users.
- **Cut:** Figs. 3, 5, 6 and 8, Table III, the memory contrasts, and most of the verification prose. Replace them with
  one sentence and a reference to a repository or supplement.
- **Venue.** The scope risk at IEEE Sensors Letters is high. The sensing content (channels, devices) is correlational.
  If the authors stay there, put channel count and modality inside the decomposition, as dataset-level covariates of
  σ²_P. Otherwise consider an ISWC short-paper track or a UbiComp/ISWC workshop such as HASCA, where evaluation
  methodology for wearables is in scope. Check the current calls; they were not checked here.

## 5. What the reviewer's checks do not settle

- Check A used 200 bootstrap resamples, not 1,000, and S3 has one run per split. σ²_e absorbs the network × split
  interaction, and S3 covers 4 datasets.
- Check B covers the 6 S1 datasets whose runs recorded validation. The manuscript's 13 datasets include S2, which was
  not read here.
- The error budget uses point estimates. σ²_M from 4 networks is very imprecise.
- The ScienceDirect 2026 paper was read only as a search summary.
- Gholamiangonabadi et al., IEEE Access 2020, is unverified (only the thesis was found).

## Addendum, same day: what the full analyses showed (correction to item 3.2)

The reviewer's quick check used 200 resamples and five populations. The full exploratory analysis
(`results/metrics/network_population/data.json`, 1,000 resamples, 16 populations) refines item 3.2.

- The best k networks, ranked on the 26 Arena datasets outside S3, keep R′ above 1 for k = 3 to 12. No interval
  excludes 1.
- From the 13th network on, every interval lies below 1. The networks that enter there are DeepConvShallowLSTM,
  Guan-LSTM and SA-HAR, then SVM and k-NN.

The manuscript's original sentence ("a weak baseline is told apart from a strong network easily") was therefore
**half right**:
- weak networks do cause the reversal;
- they are not only the classical baselines, because the 14 deep networks alone still give R′ = 0.34.

The revised text states both. The validation check on all 13 datasets, S2 included, gives:
- a median ρ of +0.18 between validation and test, against +0.80 between seeds and +0.80 between split halves;
- the best network by validation is the best on test in 3 of 13 datasets, against 7 of 13 between split halves.
