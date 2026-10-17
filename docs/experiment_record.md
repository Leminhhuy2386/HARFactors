# Development record behind the factor analysis

> Extracted unchanged from Edge_SSM's RESEARCH_PLAN.md on 2026-10-14, when this project was split off.
> Paths inside refer to Edge_SSM's layout. Here: the WHAR code is `third_party/whar-*` (same commits); data
> caches are `D:\Projects\HAR\datasets_paper_era` (shared); runs are `experiments/whar_arena_v1/`;
> aggregates are `results/metrics/<analysis>/`; figures are `figures/`; model code is `src/models/`; the
> harness is `src/training/run_whar.py`; the paper's Figure 6 numbers are `docs/sources/arena_extracted.json`;
> the 17 re-runs are `experiments/external_har_paper_era/`. The full original plan is in `docs/history/`.

### 5a. Done

| ID | Experiment | Result | Where |
|---|---|---|---|
| D1 | Edge-SSM v2 on WHAR (UTD, UniMiB, ActRecTut) and 5 ablations | negative: 51.6 on UTD; τ does not learn | `RESULTS.md`, `EDGE_SSM_SUMMARY.md` §6 |
| D2 | Round 1: HysteresisNet, VortexNet, no-memory and fixed-width controls | HysteresisNet 70.3; no-memory 67.6 | `NEW_ARCHITECTURES.md` |
| D3 | Tempo robustness: 10 models × 8 splits | new families robust; S4D collapses | `results/warp/summary_utd_mhad.md` |
| D4 | Round 2: 4 HysteresisNet variants (protocols A/B), QuantileNet, TropicalNet | QuantileNet 74.3; HysteresisNet sel+clock 73.4 | `NEW_ARCHITECTURES.md` round 2 |
| D5 | QuantileNet hard comparator (splits 0, 4) | no loss | `results/quantile_hard.json` |
| D6 | Round 3: PathQuantileNet, DualMemNet, DualMemNet-path (protocol B) | no gain; dev→held-out reversal | `NEW_ARCHITECTURES.md` round 3 |
| D7 | Literature checks of 6 candidate primitives | see §2 | `NEW_ARCHITECTURES.md` |
| D8 | Reports and structure figures | — | `report/`, `docs/arch/`, `RESULTS_SUMMARY.md` |
| D9 | Round 4: P4 multi-seed, P1–P3 robustness, P10 synthetic tasks (PC + Colab + Kaggle) | QuantileNet 74.0, best; tie with S4D, beats Mamba; play memory rate-independent; tempo gains overstated; quantile not outlier-robust | `NEW_ARCHITECTURES.md` round 4 |

### 5b. To do — the claim "better than old architectures"

Primary endpoints, fixed in advance:
- **Accuracy:** mean rank of macro-F1 across datasets, with a non-inferiority margin of 1 point against S4D.
- **Robustness:** area under accuracy against warp level.
- **Edge cost:** measured latency and energy per window on the MCU.

| ID | Experiment | Question | Design | Success criterion | Status | Next action |
|---|---|---|---|---|---|---|
| P1 | **Physically consistent tempo warp** | robust | Faster by s: gyro ×s, dynamic acceleration ×s², gravity unchanged (0.3 Hz low-pass), plus within-gesture σ warps | Area under the warp curve > S4D and Mamba (paired, p < 0.05) | **DONE** (partly met) | `robust_test.py`, 5 seeds. HysteresisNet sel+clock 55.2 vs S4D 45.9 (8/0, p = 0.008) and Mamba 51.4 (7/1, p = 0.016): met for HysteresisNet. QuantileNet 48.8: above S4D, below Mamba (0/8). Every model collapses at ×0.5 and ×2 (22–34 %); within-gesture physical warps do not separate the memories. Gyro offset uses the split's train mean/std |
| P2 | **Sampling-rate change and irregular sampling** | robust | Resample to 25, 33.3, 75, 100 Hz (anti-aliased polyphase); drop 10/30/50 % of samples; jitter ±0.45 samples | HysteresisNet nearly invariant; old models drop | **DONE** (met) | HysteresisNet 73.1–73.5 at every rate (clean 73.2), 72.0 with half the samples dropped. S4D 43.9 at 25 Hz, 43.7 with half dropped. Mamba −2, QuantileNet −6 at the rates |
| P3 | **Noise, outliers, sensor bias** | robust | Gaussian noise 0.1/0.25/0.5, spikes ±5 on 1/3/10 % of samples, offset ±0.25/0.5/1 per channel (z units) | QuantileNet degrades least on outliers | **DONE** (not met) | S4D is best on noise (66.9) and spikes (59.2); QuantileNet 56.8 on spikes (−2.4 vs S4D, 1/7, p = 0.023); HysteresisNet most fragile (40.2). An offset of 1 std breaks every model (< 10 %) |
| P4 | **Multi-seed on UTD-MHAD** (primary for the UTD claim) | accuracy | 5 seeds × 8 splits for QuantileNet, HysteresisNet sel+clock, S4D, Mamba, no-memory | Paired over splits (average the seeds within a split) | **DONE** | `run_whar.py --seed` changes only the shuffle generator and the init; the splitter stays at `cfg.seed` 0. Results in `<model>/<dataset>/seed-<s>/` (a subdirectory instead of `split-00_s1.json`, so older globs ignore it). QuantileNet 73.99, sel+clock 73.23, S4D 72.55, Mamba 71.89, no memory 67.12. QuantileNet − S4D +1.45 [−1.5, +4.4], 4/4, p = 0.46; − Mamba +2.10, 7/1, p = 0.023. `results/seeds/summary_utd_mhad.md` |
| P5 | **Multi-dataset benchmark** (primary) | accuracy | 10 WHAR datasets (11 selected, mhealth unprocessable by the paper-era code) × 3 training seeds (0–2) × 5 models (QuantileNet, HysteresisNet sel+clock, S4D, Mamba, no memory), paper-era code and protocol. Friedman + Wilcoxon-Holm over datasets; critical-difference diagram | Mean rank better than S4D/Mamba, or non-inferior within 1 point | **DONE** (not met) | Mean rank S4D 1.80, HysteresisNet sel+clock 2.00, QuantileNet 3.00, Mamba 3.70, no memory 4.50 (Friedman p = 0.0004). New − S4D: QuantileNet −1.59 [−3.47, +0.28], HysteresisNet −0.60 [−2.10, +0.90]; new − Mamba: +0.79 [+0.04, +1.54], +1.78 [−0.09, +3.66]. Non-inferior to Mamba only. `results/p5/summary.md`; runs: `cloud/make_p5_jobs.py` |
| P6 | **Stronger baselines, equal tuning** | accuracy | MiniRocket and QUANT (quantile features; `uv add aeon`), TinyHAR, CNN-HAR, LinOSS or minGRU, TCN; equal tuning budget chosen on held-out subjects; tiers ~5k / 20k / 100k params | On the accuracy–params–latency Pareto front | TODO (UTD part) / with P5 | Add the baselines to `registry.py` (or a sklearn path for MiniRocket/QUANT) |
| P7 | **Real tempo variation** | robust | A dataset with the same activity at different speeds, or train/test split by gesture duration per subject | Better than S4D when test speed differs from train | TODO | Find a candidate dataset and check its labels; may need P5 approval |
| P8 | **STM32 deployment** (primary for the edge claim) | cost | C or INT8 for both memories; latency per sample and per window, energy, RAM/flash, INT8 accuracy; compare S4D (recurrent), CNN-HAR, TinyHAR via TFLite Micro / X-CUBE-AI | Faster and lower energy at equal accuracy | TODO | Use the skills `stm32-tinyml-deploy` and `stm32-build-flash-windows`. The memories are custom ops: write C kernels (play: 2 add + 2 compare; tracker: 1 compare + 1 add) |
| P9 | **Streaming / early classification** | use | Macro-F1 after the first 25 / 50 / 75 % of the window, from the 200 UTD checkpoints (`early_test.py`; HysteresisNet's clock keeps full-window positions) | Earlier accurate decisions than S4D/CNN | **DONE** (not met) | At 50 %: Mamba 52.6, no memory 48.2, S4D 35.7, QuantileNet 33.8, HysteresisNet sel+clock 30.1 (full window 71.9 / 67.1 / 72.5 / 74.0 / 73.2). The new families trail Mamba by 19–23 points at 50 % (0/8 splits); QuantileNet passes S4D only at 75 % (+4.9, 7/1). All models were trained on full windows only. `results/early/summary_utd_mhad.md` |
| P10 | **Memory swap and mechanism tests** | why | Same skeleton, swap memory (SSM / quantile / play / none); 4 synthetic tasks × 6 models × 3 seeds (`synthetic_test.py`) | Each wins where predicted, loses where predicted | **DONE** (synthetic) | Confirmed: play wins on speed-varied shapes (S4D 34.6 at 2×); S4D wins the frequency control and pure play is at chance (24.4). Refuted: quantile memory is not the most outlier-robust (Mamba 71.2 vs QuantileNet 46.1 at 30 % spikes). The UTD memory swap at equal size is P4 itself (all share the skeleton) |

**P5 pre-registration (written 2026-10-09, before any P5 result).**
- *Datasets.* From the paper-era `BENCHMARK_DATASET_IDS` (31), keep those that download automatically with the
  paper-era code (`HAR/ATLAS.md` §2, "tự động"), have no 🔴/🟠 data defect listed there, and cost at most 2.2 M
  window-samples (windows × samples per 3-s window). That gives 11: utd_mhad (done), actrectut_gestures, mhealth,
  opportunity, unimib_shar, uci_har, sad, hapt, dsads, wisdm, daphnet. Excluded by cost only, possible second wave:
  motion_sense, pamap2, wear, usc_had. ⚪ metric caps (daphnet 66.7) are shared by every model and kept.
- *Found while running (2026-10-09).* **mhealth is out:** the paper-era code itself cannot preprocess it ("number of
  activity_ids 13 does not match number of activities 12", input validation fails), and that code must stay unmodified,
  so it fails the rule's first condition. **opportunity** failed on the Colab VM only because `playwright`, a declared
  dependency of the paper-era whar-datasets used by its downloader, was not installed there (the Kaggle VMs built it).
  It is rerun for seeds 0–2 on one Colab VM with playwright (`cloud/jobs/colab_p5_opportunity.json`); merging keeps the
  first copy of each file, so the Kaggle duplicates of seeds 0–1, which arrive later, are discarded. P5 therefore has 10 datasets.
- *Runs.* Seeds 0–2 per dataset (UTD-MHAD: its existing seeds 0–2), 5 models, unchanged configs, protocol unchanged.
  `EDGE_SSM_FAST=1` for every new run: same math, gradients equal up to float rounding (checked against the
  reference on UTD, `results/fastcheck/`). One (dataset, seed) per machine. Datasets not cached locally are downloaded
  and preprocessed on the VMs by the paper-era code; nothing under `D:\Projects\HAR` changes.
- *Primary analysis.* Per dataset: mean over splits of the seed means. Across the datasets: mean rank of the 5 models,
  Friedman test, Wilcoxon-Holm for each pair; QuantileNet and HysteresisNet sel+clock − S4D and − Mamba with a 95 % t-interval
  over datasets; non-inferiority if the lower bound is above −1 point. The 17 WHAR baselines exist for only 4 datasets and
  are context, not part of the test.

**Recommended order** (P1–P5, P9 and P10 done on 2026-10-09):
1. Decide with the user how to continue the search (P5 showed no new family beats S4D). Options: P6 (stronger
   old baselines at equal size, needs approval); a round-6 primitive aimed at the gaps P5/P9 exposed; or a P5 second
   wave (motion_sense, pamap2, wear, usc_had) if more datasets are wanted.
2. P8.
3. P7 and P9.
4. Optional, protocol B: a readout that keeps outlier robustness (quantile pooling instead of mean), since P3/P10
   showed the instantaneous path and the mean over time carry the spikes past the quantile memory.

### 5h. Round 11 (2026-10-11): evaluate ideas before building (no training)

User: analyse ideas thoroughly before implementing. Full write-up: `NEW_ARCHITECTURES.md` round 11. Tools:
`error_analysis.py` (confusions and oracle headroom from saved checkpoints) and `style_probe.py` (1-NN with and
without per-window tempo/gain fitting).
- Largest addressable errors of RecurrenceNet:
  - gait type, level vs stairs vs jogging: WISDM +13.8, SAD +6.0;
  - null class: Opportunity +11.4, ActRecTut +7.1;
  - diffuse person-specific style on gestures.
  - Time-reversal pairs and drawing gestures add at most ~1 point.
- 1-NN with DTW and per-window standardization, no learning:
  - Gestures: UTD-MHAD 74.3 (RecurrenceNet 73.3) and ActRecTut 69.0 (RecurrenceNet 40.4; paper best 61.5).
  - Locomotion and posture: SAD 77.7 (RecurrenceNet 88.9 on the same splits) and Opportunity split 0 12.0 (posture erased).
- Verdicts:
  - DrawNet and the arrow-of-time net: do not build (headroom about 1 point; prior art close).
  - Analysis-by-synthesis: the only idea the data supports, provided the design keeps posture and gait shape. Partly new.
- Decision (user, 2026-10-11): first C, then A.

**C: the elastic-invariance gap** (`elastic_baseline.py`, `elastic_summary.py` -> `results/elastic/`). This is 1-NN
against every training window of each split, with 4 fixed distances (no choice on test scores), on all 10 dev datasets.
- DTW is computed in a band, row by row on the GPU (prefix-sum/cummin form); it equals the plain recursion to 1e-13.
- PC: UTD, ActRecTut, Opportunity, SAD, UCI, HAPT, UniMiB.
- Colab: DSADS, Daphnet, WISDM (`cloud/jobs/colab_elastic.json`). Their caches are built on the VM, so they are in Linux
  listdir order.
- DSADS and Daphnet are compared by dataset means only. 1-NN on leave-subject-out folds does not depend on fold order,
  but split indices may not match P5's, because listdir order can differ between machines.
- **Result (all 10 datasets, done 2026-10-11).** Mean macro-F1 over the 10:
  - 1-NN Euclid 51.0, Euclid + standardization 56.3, DTW 64.0, DTW + standardization 65.3.
  - For comparison: RecurrenceNet 72.0 (3 seeds), the paper's k-NN on tsfresh features 51.8.
  - DTW + standardization ranks #1 of the paper's 17 models on UTD-MHAD (74.3) and ActRecTut (69.3; paper best 61.5).
  - It ranks #14-18 on locomotion: Opportunity 31.7, WISDM 63.5, DSADS 77.3, SAD 88.0.
  - Reading: tempo/gain invariance pays only on gestures. On locomotion, per-window standardization erases the posture and
    gait-intensity cues the deep models use. The paper's k-NN understates what nearest-neighbour methods can do.

**A: StyleAlignNet** (`edge_ssm/round11.py`, tests `tests/test_round11.py`, configs fixed before any result).
- Architecture:
  - RecurrenceNet's path, unchanged, plus an elastic path.
  - Elastic path: an 8-dim linear embedding, averaged over 5-sample patches.
  - It makes a closed-form per-window gain/offset fit, then a banded soft-DTW (band 30 % of the tokens, gamma 0.5)
    against 2 learned programs per class.
  - Its class evidence is added to the logits. The head also sees the fitted style (offset, log gain) and the smallest
    cost (null-class evidence).
  - Size: 38.2k params on UTD-MHAD (RecurrenceNet 24.7k; programs scale with the class count), ~29-31k on SAD /
    Opportunity / ActRecTut.
- Control: `stylealign_elastic`, the elastic path alone, 13.5k.
- Literature check: partly new.
  - Learned class prototypes with (soft) DTW: DP-DTW (CVPR 2021), teNN (2024), DTWNet (2019).
  - Not found: the gain/offset fit with style features, null evidence from the smallest cost, fusion with a deep path,
    learned soft-DTW templates for IMU HAR.
  - Named StyleAlignNet, since "analysis-by-synthesis / motor programs" refers to Nair & Hinton 2005.
- Predictions:
  - gestures: ActRecTut well above RecurrenceNet (its 1-NN analogue gives 69.0 vs 40.4), UTD-MHAD at least
    RecurrenceNet;
  - locomotion: SAD and Opportunity within 2 points of RecurrenceNet, because the deep path keeps posture and gait shape;
  - the control: near the 1-NN DTW numbers on gestures, far below on locomotion.
- Kill rule (seed 0, local PC + Colab):
  - kill set: UTD-MHAD, ActRecTut, SAD, Opportunity; RecurrenceNet seed 0 there: 74.59, 35.08, 91.97, 64.91, mean 66.64;
  - stop unless the mean is at least 66.64 and neither SAD nor Opportunity is more than 2 points below RecurrenceNet;
  - a survivor runs the other 6 dev datasets, then seeds 1-2, then held-out datasets.
- **Kill-set result (seed 0, PC): passed.**

  | Dataset | StyleAlignNet | RecurrenceNet | Diff |
  |---|---|---|---|
  | UTD-MHAD | 75.22 | 74.59 | +0.63 |
  | ActRecTut | 48.72 | 35.08 | +13.64 |
  | SAD | 93.52 | 91.97 | +1.55 |
  | Opportunity | 65.83 | 64.91 | +0.92 |
  | Mean | 70.82 | 66.64 | +4.18 |

  - It is above RecurrenceNet on all 4, and the locomotion conditions hold.
  - Caveats: it is larger (38.2k vs 24.7k on UTD-MHAD). ActRecTut stays far below 1-NN DTW (69.3) and below round 8's
    statistics hybrid (63.6, seed 0).
- Control (elastic path alone, seed 0): UTD-MHAD 49.84, ActRecTut 34.07, Opportunity 63.76; its second SAD half died of
  RAM, rerun pending.
  - The learned elastic path does not reproduce 1-NN DTW (74.3 / 69.3 on the gestures).
  - So StyleAlignNet's gains come from the deep path plus the window-level features (fitted offset/gain), the same kind of
    signal as round 8's channel statistics.
  - On the same kill set, round 8's statistics hybrid scores 73.66 against StyleAlignNet's 70.82.
- Stage 2 (other 6 dev datasets, seed 0), done 2026-10-11: UCI-HAR 96.16 (RN 96.74), UniMiB 59.25 (61.96), HAPT 83.28
  (83.21), DSADS 87.98 (86.24), WISDM 83.91 (84.30), Daphnet 43.5 (43.5).
  - Mean of 10: **73.74, +1.49 vs RecurrenceNet** (passes the stage-2 rule).
  - Level with LawNet + recurrence (73.77) and below the statistics hybrid (74.69).
  - Seeds 1-2 held back pending the user's decision: it costs ~3.5x RecurrenceNet, and the ablation points to window-level
    statistics as the shared lever.

### 5i. Round 11b (2026-10-11): what carries the gain, and does it hold over seeds

User: do (3) the ablation, together with (2) seeds 1-2 of the round-8 models that pass, and compare with the benchmark.
Pre-registered before any result of these runs:
- **(3) Ablation of the window-level side path** on RecurrenceNet's trunk (same fusion at the head as `recstatnet`,
  `edge_ssm/round8.py` `ChannelStats(stats=...)`):
  - `recstat_mean` (per-channel mean: offset / posture), `recstat_std` (intensity), `recstat_range` (min and max),
    against `recstatnet` (all four) and RecurrenceNet (none).
  - Representative kill set: UTD-MHAD, ActRecTut, SAD, Opportunity. Seeds 0-2 on one Colab VM, together with
    `recstatnet` seeds 1-2.
  - Predictions:
    - the mean carries most of the gain on posture-heavy data (Opportunity, SAD);
    - range/std carry it on gestures (ActRecTut);
    - no single statistic reaches the full set on all four.
- **(2) Seeds 1-2** of `recstatnet`, `lawnet_rec`, `lawnet` and `statnet` (the control for prediction 1) on all 10 dev datasets:
  - Kaggle T4: DSADS + Daphnet, WISDM.
  - Colab: UCI-HAR + UniMiB; the 4 kill-set datasets on the ablation VM.
  - PC, once StyleAlignNet's stage 2 frees it: HAPT, SAD (if not done elsewhere).
  - Hardware differs from the references on some cells (noise ~1 point per cell); recorded per result file.
- **Ablation result (3 seeds, kill set mean):**

  | Variant | Mean of 4 | − RecurrenceNet |
  |---|---|---|
  | full statistics | 73.43 | +6.38 |
  | range | 71.73 | +4.68 |
  | std | 71.04 | +3.99 |
  | mean | 65.50 | −1.55 |
  | RecurrenceNet | 67.05 | |

  - The lever is per-channel amplitude/intensity (std, min/max); the channel mean (posture) does not help.
  - Prediction "the mean carries posture-heavy data": refuted on all four.
  - Prediction "range/std carry gestures": confirmed (ActRecTut 56.1 / 50.7 vs 40.4).
  - Every side path costs ~1 point on UTD-MHAD.
- **Mechanism test (a/b), pre-registered 2026-10-11 before any result.** RecurrenceNet with one change each
  (`edge_ssm/round6.py`):
  - `recnet_meanstd`: (a) mean + std over time of the last hidden states at the head.
  - `recnet_nonorm`: (b) no per-time-step LayerNorm in the blocks.
  - `recnet_meanstd_nonorm`: both.
  - Hypotheses:
    - H1: the per-time-step LayerNorm hides intensity from the nonlinear path, and mean pooling keeps only offsets.
    - H2: per-channel intensity is lost in the first channel mixing (C -> 32) whatever the normalization.
  - Kill set, 3 seeds, one Colab VM.
  - Reading rule:
    - if (a) or (b) reaches a mean of at least 71.3 over the 4 datasets (2/3 of the way from RecurrenceNet 67.05 to the
      statistics hybrid 73.43), H1 is supported;
    - if both stay below 69, H2 is favoured;
    - in between is inconclusive.
  - **Result (3 seeds, done 2026-10-11): H2 favoured by the rule.**
    - Colab L4. The Opportunity cache and 4 SAD jobs were OOM-killed on a T4 fallback VM and rerun on L4
      (`cloud/jobs/colab_r11d_ab2.json`).

    | Variant | UTD-MHAD | ActRecTut | SAD | Opportunity | Mean of 4 | Seed means |
    |---|---|---|---|---|---|---|
    | (a) mean + std pooling | 71.97 | 48.35 | 93.88 | 61.12 | 68.83 | 69.5 / 70.2 / 66.7 |
    | (b) no LayerNorm | 70.19 | 46.59 | 90.19 | 60.93 | 66.97 | 65.0 / 67.6 / 68.3 |
    | (a) + (b) | 71.61 | 54.13 | 92.76 | 60.38 | 69.72 | 70.1 / 70.7 / 68.4 |
    | RecurrenceNet | 73.32 | 40.37 | 92.93 | 61.59 | 67.05 | 66.6 / 67.0 / 67.6 |
    | statistics hybrid | 72.22 | 61.73 | 95.91 | 63.87 | 73.43 | 73.7 / 74.0 / 72.6 |

    - Both (a) and (b) stay below 69. (a) + (b) is outside the rule and gives 69.72, still far from 71.3.
    - Changing normalization and pooling inside the trunk recovers at most 2.7 of the hybrid's 6.4 points, almost all
      of it on ActRecTut (54.1 vs 61.7). Removing LayerNorm alone costs SAD 2.7 points.
    - Reading: the gain needs per-channel amplitude statistics taken before the channels are mixed. The trunk does not
      recover them, even with std pooling and no per-step normalization.
    - Limit: this test does not separate information loss in the mixing from a trunk that could represent the
      statistics but does not learn them. For C ≤ 32 the first linear map can be injective, so the second reading stays
      open.
- **Analysis**, paired over the 10 datasets with 3-seed means: P5 statistics against RecurrenceNet and S4D, plus
  placement in the WHAR Arena paper table (`compare_arena.py`), with the elastic 1-NN baselines and StyleAlignNet
  (seed 0) as context.
- **Seeds 1-2 result (done 2026-10-11; `benchmark11.py` -> `results/benchmark11/summary.md`).**
  - Hardware: Kaggle T4 for DSADS / Daphnet / WISDM, Colab for the rest.
  - 3-seed means over the 10 dev datasets:

    | Model | 3-seed mean | − S4D [95 % t-interval], wins/losses, Wilcoxon p (Holm) | − RecurrenceNet |
    |---|---|---|---|
    | statistics + recurrence (`recstatnet`) | 74.70 | +3.41 [−0.79, +7.62], 8/2, p 0.020 (0.078) | +2.70 [−2.08, +7.48], 7/3, p 0.131 |
    | LawNet + recurrence (`lawnet_rec`) | 74.42 | **+3.12 [+0.58, +5.67], 8/2, p 0.010 (0.049)** | +2.41 [+0.35, +4.47], 8/2, p 0.020 (0.098) |
    | RecurrenceNet | 72.01 | +0.71 [−0.83, +2.26], 7/3, p 0.193 | |
    | LawNet | 71.52 | +0.23, 4/6, p 0.695 | −0.49, 4/6 |
    | S4D | 71.29 | | −0.71, 3/7 |
    | statistics, memoryless (`statnet`) | 70.24 | −1.05, 4/6 | −1.77, 3/7 |
    | no memory (control) | 66.06 | −5.23, 0/10, p 0.002 | −5.95, 0/10 |

  - The seed-0 order holds over seeds: recstatnet 74.69 -> 74.70, lawnet_rec 73.77 -> 74.42.
  - `lawnet_rec` is the only model whose gain over S4D survives Holm (p 0.049). recstatnet has the larger mean but
    is less consistent: ActRecTut +21, while it loses about 1 point on UTD-MHAD.
  - Caveat: these 10 datasets are the ones round 8 was screened on, so the p-values are optimistic. Confirmation needs
    the held-out datasets.
  - Under the clarified goal (§1), both are RecurrenceNet plus a window-level side path, i.e. known parts in a known
    pattern. They give an engineering gain, not a new family.
- **Arena placement (seed 0, field of 27 = 17 paper models + 10 of ours and the elastic baselines).**
  - Mean rank: StyleAlignNet 6.30, LawNet + recurrence 6.80, statistics + recurrence 7.10 (highest mean F1, 74.69),
    TinyHAR 7.50, RecurrenceNet 8.20, CNN-HAR 8.70.
  - No paired difference against the 3 strongest paper models (Triple-Cross-Attn, CNN-HAR, TinyHAR) survives Holm.

### 5n. Strategic evaluation before choosing a direction (2026-10-12, direction-3 session; no network trained)

User: think and evaluate thoroughly before proposing a direction, to avoid running experiment after experiment without
effect.

**1. What 14 rounds say about the size of possible gains (from §5l's factor analysis and our screens).**
- Within a dataset:
  - the test person carries 61 % of the variance, the seed 14 % and the model 6 %;
  - model effects have an RMS of 2.4 points, against 3.6 for seeds and 7.5 for persons.
- Memory type moves results by ±1-3 points, about seed noise. Bigger or more flexible networks do not help, and
  LevelNet shows they can hurt by fitting the training people.
- Robustness capabilities vanish once the baselines get the standard deployment pipeline (§5k).
- Ceiling of combining the strengths we already have: taking, per dataset, the better of fixed features + ExtraTrees
  and recstatnet gives 79.7 over the 8 datasets, against 77.8 and 77.4 for each alone. So a model that combined both
  perfectly would gain about +2.
- Conclusion: under the fixed cross-person protocol, any new architecture can be expected to move the mean by about
  1-3 points. That is inside the noise of 10 datasets and 3 seeds, and not a basis for a Nature-family claim. The
  limit is the data regime (2-36 subjects), not a lack of ideas.

**2. Where the headroom is: the person (`person_probe.py` -> `results/person_probe/summary.md`).**
- Method: the linear probe on the fixed value-domain features. Gains are paired: each adapted score is compared with
  the unadapted model on exactly the windows it is scored on.

  | Access to the new person | Mean gain (8 datasets) | Few-subject gesture / many-channel datasets | Many-subject locomotion datasets |
  |---|---|---|---|
  | unlabelled windows only (per-person standardization) | +1.5 | Opportunity +7.0, ActRecTut +3.8 | UCI +3.9; UniMiB, WISDM, HAPT −1 to −2 |
  | 1 labelled window per class | +3.1 | UTD-MHAD +10.6, ActRecTut +6.4, Opportunity +2.9 | +0.3 to +1.7 |
  | 3 labelled windows per class | +4.9 | UTD-MHAD +11.3, ActRecTut +10.6, Opportunity +8.4 | +0.3 to +3.9 |
  | unlabelled + 1 per class | +4.2 | Opportunity +9.3, UTD-MHAD +9.5 | UCI +4.2 |

- Reading:
  - Where the person matters most (few training subjects, gestures, many channels), even this crude adapter gains
    6-11 points: 3-5 times the effect of any architecture change so far.
  - Where many training subjects exist, the model already generalizes and little is left.
  - The seen-vs-unseen gap of 14.5 points overstates the headroom, because validation windows share trials with
    training.

**3. Options compared.**

| Option | Expected effect | Can it be shown convincingly? | Novelty potential | Feasible now? | Nature-family fit |
|---|---|---|---|---|---|
| A. Keep searching architectures under the fixed protocol | +1-3 (14 rounds of evidence) | no: noise-level | partly new at best | yes | poor |
| B. A family designed for adaptation to the wearer (person-general vs person-specific structure; learning from 1-3 examples or unlabelled data, on the MCU, without backprop) | +6-11 on few-subject data with a crude adapter; more if the structure is better | yes, paired within person | open (needs a literature check; HAR personalization and MCU on-device training exist) | yes (PC + NUCLEO hardware-in-the-loop); energy only as an estimate | plausible (Nature Electronics precedent: on-device adaptive wearable learning) |
| C. Write up the findings (person dominates; classical value-domain features ≥ deep; robustness claims vanish under fair pipelines) | certain, results exist | yes | as an analysis | yes (overlaps with the other session's synthesis) | uncertain; strong field venue likely |
| D. Hardware-first system paper | unknown | needs a power meter and an IMU | no mechanism yet | no | weak without B |

**4. Recommendation: B, behind three cheap gates, with C in parallel.** Each gate can stop B before any architecture is
built.
- *Gate 1 (done, above):* is there person headroom? Yes, on few-subject and many-channel data.
- *Gate 2 (literature, no compute):* what exists for HAR personalization (few-shot, test-time adaptation, AdaBN,
  meta-learning) and for MCU on-device learning (TinyTL, on-device training under 256 KB, one-shot rules)? Which claim
  is still open?
- *Gate 3 (minutes, checkpoints only):* how much do existing deep models gain with the standard adapters? Options are
  retraining the head on k shots and AdaBN-style normalization from unlabelled windows.
  - If they get most of the headroom, a new structure must win on something else: no backprop, MCU memory and
    compute, or unlabelled-only adaptation. Otherwise B stops.
- Only after the gates: design the adaptation-native family, pre-register, screen.
- **Needs the user's decision:** B changes the protocol. Pretraining stays subject-disjoint, but the evaluation uses a
  few windows (or unlabelled windows) of the test person, so the 17 paper models are no longer comparable as they
  stand; their adapted versions would be the baselines.

### 5p. Paper for IEEE Sensors Journal: held-out confirmation (pre-registered 2026-10-12, before any held-out run)

User decisions, 2026-10-12, in the round-11b/13 session:
- full paper for IEEE Sensors Journal on the best family;
- the 8 held-out datasets below;
- paper baselines taken from Figure 6, not re-run;
- MCU measurements after the held-out results.

**Held-out set.** Start from the 20 WHAR Arena datasets outside P5. Drop:
- datasets with a 🔴/🟠 data defect in `HAR/ATLAS.md`: real_disp, har70, real_world, hugadb;
- mhealth, which the paper-era code cannot process;
- datasets with more than 50k windows (cost): hhar, real_life_har, gotov, wisdm_19_phone, wisdm_19_watch, harth,
  hang_time.

That leaves exactly 8: PAMAP2, UP-Fall, UMA-Fall, USC-HAD, MotionSense, WEAR, UCA-EHAR, W-HAR. None of them was used
to choose any model or setting. The ⚪ metric caps (PAMAP2 92.3, W-HAR 88.9) apply to every model alike and are kept.

**Models, frozen as registered in `edge_ssm/registry.py`; no change of any setting:**
- `lawnet_rec` (RN + laws): primary;
- `recstatnet` (RN + statistics): secondary;
- `recurrencenet` and `s4d`: references.

Seeds 0-2, `EDGE_SSM_FAST=1`, protocol hash e8739ef2c5b36c99, paper-era code only.

**Where the data come from.**
- Cloud VMs build the caches from fresh paper-era downloads, as for P5.
- UCA-EHAR needs a manual download, so it is built on the PC from HAR's raw files (`prepare_local_datasets.py`;
  its parser sorts its files).
- Two parsers list files without sorting (MotionSense `os.listdir`, USC-HAD `os.walk`). Subject numbering, and so
  split membership, may differ from the paper's machine. For these two, comparisons with the paper use dataset means
  only. Comparisons among our own models stay paired on our splits.

**Analyses, fixed now.** Over the 8 held-out datasets, paired over datasets:
1. `lawnet_rec` − S4D on 3-seed means: mean, 95 % t-interval, wins, Wilcoxon. Claim "better than S4D" only if the
   lower bound is above 0.
2. `lawnet_rec` (seed 0, as the paper ran once) against each paper model: mean difference with a 95 % t-interval.
   - Non-inferiority to the three strongest paper models (TinyHAR, CNN-HAR, Triple-Cross-Attn; the top three by
     paper mean on the 10 dev datasets, fixed before these runs) with a 1-point margin: lower bound above −1.
3. Efficiency: parameters and latency of `lawnet_rec` against those three on the held-out shapes (Pareto, `pareto.py`).

Secondary:
- `recstatnet` under the same tests;
- `lawnet_rec` − RN: does the laws path still help on new data;
- all 18 datasets together, reported separately and labelled as including the development data.

Everything is reported whatever the outcome. A failed run is rerun with the same configuration; nothing else may change.

**Result (all 8 datasets × 4 models × 3 seeds complete, 2026-10-13; `heldout_analysis.py` -> `results/heldout/summary.md`,
`pareto.py --heldout` -> `results/pareto_heldout/`).**

| Model | PAMAP2 | UP-Fall | UMA-Fall | USC-HAD† | MotionSense† | WEAR | UCA-EHAR | W-HAR | Mean |
|---|---|---|---|---|---|---|---|---|---|
| RN + laws (3 seeds) | 60.3 | 52.5 | 52.7 | 76.9 | 93.7 | 86.8 | 62.6 | 50.6 | **67.02** |
| RN + statistics | 58.6 | 51.8 | 50.0 | 72.7 | 91.6 | 85.8 | 64.0 | 53.5 | 65.99 |
| RecurrenceNet | 52.6 | 49.8 | 46.6 | 73.3 | 91.2 | 85.7 | 61.7 | 50.1 | 63.87 |
| S4D | 54.5 | 42.8 | 42.9 | 76.2 | 91.9 | 84.4 | 59.1 | 46.3 | 62.27 |
| TinyHAR (paper) | 62.2 | 61.0 | 50.2 | 78.1 | 91.2 | 86.5 | 58.0 | 52.2 | 67.40 |
| CNN-HAR (paper) | 62.1 | 55.0 | 50.0 | 76.2 | 91.5 | 86.9 | 58.7 | 52.7 | 66.63 |
| Triple-Cross-Attn (paper) | 61.9 | 58.7 | 52.4 | 74.3 | 91.3 | 88.5 | 58.1 | 49.8 | 66.89 |
| Random Forest (paper) | 65.1 | 72.8 | 54.2 | 72.7 | 91.5 | 87.3 | 47.6 | 53.1 | 68.02 |

† parser lists files unsorted, so these two are compared with the paper by means only.

- **Primary 1, confirmed:** RN + laws − S4D is **+4.75 [+1.85, +7.65], better on 8 of 8, Wilcoxon p 0.008**.
- **Primary 2, not shown:** non-inferiority to the three strongest paper models with a 1-point margin.
  - Point estimates are ties: − TinyHAR −0.02 [−3.67, +3.63], − Triple-Cross-Attn +0.49 [−2.72, +3.70], − CNN-HAR
    +0.76 [−1.57, +3.08].
  - The lower bounds are below −1 with 8 datasets.
- **Primary 3, efficiency:** on the held-out shapes (geometric mean over the 8 datasets), RN + laws has 29.5k parameters
  and sits on the parameter Pareto front.
  - TinyHAR: 296k parameters, +0.02 F1 at seed 0.
  - DynamicWHAR: 2.3M parameters, +1.07.
  - Our PyTorch-eager CPU latency (11.2 ms) is not competitive. MCU measurements are pending.
- **Secondary:**
  - RN + laws − RN +3.15 [+1.00, +5.30], 8/8, p 0.008: the law path generalizes to new data.
  - RN + statistics − S4D +3.73 [+0.17, +7.29], 6/2, p 0.055.
  - RN + statistics against the top three: −1.11 / −0.34 / −0.60, non-inferiority not shown.
  - RecurrenceNet − S4D +1.61 (5/3, n.s.).
  - All 18 datasets (includes the development data): RN + laws − S4D +3.85 [+2.10, +5.59], 16/2.
- **Where it loses:**
  - UP-Fall: 42 channels, 17 subjects; Random Forest 72.8 against 52.5.
  - PAMAP2: 52 channels; Random Forest 65.1 against 60.3.
  - Both fit the channel-count factor of §5l and `results/factors/`: many-channel data favour the classical family.
- **Reading for the paper:**
  - Confirmed on untouched data: a large, consistent gain over S4D and over its own trunk.
  - Accuracy level with the best published models at about a tenth of their size. "Non-inferior" cannot be claimed
    with 8 datasets.

Run notes (no change to models, settings or analysis):
- Kaggle's weekly GPU quota was spent, so the runs moved to Colab A, B and C and the PC.
- UP-Fall's paper-era parser fails under Colab's Python 3.13. Its patched `rglob` rejects `case_sensitive`
  (TypeError in `parsing_step._filtered_rglob`). UP-Fall therefore runs on the PC (Python 3.12) from a local cache
  built from HAR's raw files. Its parser sorts its files.

**Paper plan:** `paper/PLAN.md`.
- Outline: method; WHAR Arena protocol; dev and held-out results; Pareto; ablations; factor analysis; MCU
  measurements.
- Claims:
  - SOTA-level accuracy at 5-15× fewer parameters;
  - an RQA-style recurrence layer;
  - a GL(3)-invariant per-window law path.
- Threats:
  - partial novelty: Ravì 2017, SELFY, Dong 2021;
  - development-set reuse;
  - Opportunity and WISDM losses.

## HARFactors, 2026-10-15: the confirmatory analysis (stage 04)

Every run below is in `experiments/LEDGER.md`, with START and END lines and a directory `experiments/<run_id>/`.
No model was trained: every analysis reads the stored runs.

### Configurations run

| run_id | What | Set, outcome | Result |
|---|---|---|---|
| 20261015_164309_run_analyses | Exploratory aggregates regenerated with provenance (+ contrasts without daphnet) | S1, quoted table | identical to the imported files except line endings and last-digit floats (G1 rerun 2) |
| 20261015_164940_vc_H1 | **H1, the confirmatory test** | S2, macro-F1 | R = 8.187 [2.847, 16.905], pi_P = 0.824 [0.640, 0.894]: **success** |
| 20261015_164941_vc_sens4_S3_macro_f1 | Sensitivity 4 | S3, macro-F1 | R' = 0.208 [0.059, 0.355], pi_P = 0.172 |
| (see ledger) vc_sens1_S2_balanced_accuracy | Sensitivity 1 | S2, balanced accuracy | R = 4.089 [2.046, 6.031], pi_P = 0.696 |
| (see ledger) vc_sens2_S1_macro_f1, vc_sens2_S1_balanced_accuracy, vc_sens3_S1_without_actrectut | Sensitivities 2 and 3 | S1 | `results/metrics/confirmatory/` |

All intervals are percentile 97.5 % intervals from 1,000 cluster-bootstrap resamples (seed 20261015). Every resample
of every analysis converged, and none needed Powell's method.

### Development before the seal

The analysis code was developed on synthetic data (`tests/test_variance_components.py`, 8 tests) and on S1.
- **Development runs** wrote nothing to `results/`:
  - an S1 point fit and 16 resamples, stopped because the workers oversubscribed the node's cores;
  - an S1 point fit and 32 resamples;
  - two diagnostics, one timing sparse vs dense design matrices and one timing single resamples.
- **Point estimates seen on S1 during development:**
  - s2_P 66.2, s2_M 5.9, s2_MD 7.8, s2_MP 6.7, s2_e 17.6, so R = 3.26;
  - development interval from 32 resamples: [1.52, 4.09].
- **Seal.** G4 sealed the code at commit 761d6bd (`results/gates/G4_test_seal.json`). Afterwards S2 was read only by
  the two pre-registered S2 analyses (`results/test_access.log`).

### Deviations and decisions, with reasons

1. **G1 tolerance changed after the first run.** It was "byte for byte" and became "identical modulo line endings,
   JSON floats to relative 1e-9".
   - Reason: the files were written on Windows (CRLF), and the PC's and the node's LAPACK differ in the last digits
     of the OLS coefficients.
   - All three G1 files are kept. Reported to the user.
2. **Inclusion rule** (pre-registered) applied:
   - daphnet is excluded from S1 and motion_sense from S2;
   - their stored runs disagree on split sizes, so their split ids do not name the same test subjects across runs.
   - The exploratory component contrasts pair models by split; on daphnet that pairing is not by person. They were
     therefore repeated without daphnet (`results/metrics/contrasts_robustness/`), and the conclusions are unchanged.
3. **Implementation choices that do not change the estimator:**
   - dense variance-component design matrices (statsmodels `use_sparse=False`, the default; sparse was 3x slower with
     identical estimates);
   - one BLAS thread per bootstrap worker (`threadpoolctl`).
4. **Prior reads of S2 declared in G4.** The imported exploratory `heldout_analysis` recomputed S2's per-dataset
   means during the G1 reproduction checks. It computes no variance component.
5. **No label-shuffle control** (`agents/05_verify.md` 3.5.3): it needs training, which the user ruled out.

### Computational cost

All analyses ran on CPU nodes (hpc01/02, Intel, 8-16 cores): the H1 bootstrap took 4 min 25 s and the S3 bootstrap
47 s. The S1 bootstraps took about 40 min each with 8 workers.

## HARFactors, 2026-10-16: descriptive and exploratory expansion (more figures)

The user asked for many more figures (sensors, experiments, detailed network structure) and for a deeper analysis of
the embeddings and other characteristics; exceeding the page count is acceptable for this draft (the user will
shorten it). Nothing here is confirmatory, and no network was trained.

### Configurations run

- `src/evaluation/split_spread.py` (a `scripts/run_analyses.py` step): per dataset of S1 and S2 that passes the
  inclusion rule, the means of splits, of networks and of (network, split) cells, from the stored runs.
- `src/evaluation/model_anatomy.py`: parameter counts per part with the reference input of gate G1 (checked equal to
  G1), sizes per dataset from the run records, the memory operators on a synthetic input (illustration), and the
  memory parameters of every S1 checkpoint (seeds 0-2) of eight datasets against the same networks freshly
  initialised.
- `src/evaluation/embedding_analysis.py`: seed-0 checkpoints of every split of eleven datasets (eight of S1 with ten
  networks, up_fall, wear and uca_ehar of S2 with four), inference only, on the cluster caches that reproduce the
  stored runs; class-conditional shift of the test group in the embedding and in a sensor-feature space, linear
  probes for subject and activity, one t-SNE case fixed in the code before the run (UCI-HAR, S4D, split 0).
- `src/visualization/detail_figures.py`: Figs. 3-8 and Tables II-III of the manuscript, from saved results only.

### Deviations and decisions, with reasons

1. **Format limits.** The page and figure limits of IEEE Sensors Letters are waived for the working draft
   (`paper/format_limits.json`, read by `scripts/check_manuscript.py` and `scripts/audit.py --manuscript`); the
   150-word abstract limit stays. Restore the limits before submission.
2. **G5 rerun5 chain postponed.** The pending chain (Slurm 86823) was cancelled before it started: run while the
   embedding jobs write to the ledger, its negative controls would see START lines without END. G5 rerun5, G6, G7 and
   the audit run once, after the new analyses, with nothing else running.
3. **Where the jobs ran.** The CPU partitions were full (all of hpc01-08 allocated for days). The embedding analysis
   regenerates predictions from checkpoints, which `docs/hpc.md` allows on the `gpu` partition; it ran there on idle
   CPU cores without a GPU (`--gres=gpu:0`), as one job of 16 cores (the partition's QOS allows one running job).
4. **Aggregation code amended during the run.** `aggregate()` gained the regeneration check, a per-dimension shift and
   per-network correlations while dataset runs were in progress; `analyse_dataset()` was not touched, so every
   dataset file comes from the same analysis code. The file was replaced atomically after a syntax check.
5. **Learned memory parameters, first version superseded.** The first `model_anatomy` run read every checkpoint on
   disk (including extra seeds of some networks) and had no initial values to compare with; the second reads seeds
   0-2 only and compares with freshly initialised networks.

## HARFactors, 2026-10-16: reproducibility of every stored run

The user asked that every result be reproducible and that every network have its checkpoint and config.

### Configurations run

- `scripts/evaluate.py` on every S1 run with a checkpoint on the eight datasets whose cluster cache matches (1,914),
  on USC-HAD's 120 S2 runs (cache built on the cluster with `scripts/prepare_data.py`), and on DSADS's 240 S1 runs (cache
  rebuilt in the listing order inferred from the checkpoints).
- `scripts/reconstruct_order.py`: `infer` (labels and test subjects from every checkpoint of a group), `build` (cache
  rebuilt with `os.listdir` returning that order, from hard links to HAR's raw files) and `reexecute` (per checkpoint,
  for run sets built on several VMs: DSADS on Colab, Daphnet, MotionSense).
- `scripts/retrain_missing.py`: the six runs without a checkpoint, retrained in the run's own directory.
- `src/evaluation/reproducibility.py`: the inventory and `docs/reproducibility_report.md`; `scripts/export_configs.py`:
  `configs/`.

### Deviations and decisions, with reasons

1. **No download.** The original archives were first considered, to read their member order; the user pointed out
   that the data are complete on the cluster, and the listing order on ext4 depends on a per-disk hash seed anyway
   (a zero-seed half-MD4 hypothesis was tested and rejected), so the order was recovered from the checkpoints instead.
2. **Test-group identification.** Per checkpoint, the least accurate subjects identify the test group when the network
   is near-perfect on its training subjects (DSADS, MotionSense) but not on Daphnet (macro-F1 about 42 %); the record's
   test size, a structural fact of the split, now identifies the group where one subset matches it.
3. **Retraining.** The pre-registration rules out new training for the analyses; the six runs were retrained only to
   give them checkpoints, in a separate directory, and the stored records stay the analysed ones. The job used a GPU
   it had not been allocated (Slurm does not hide a node's GPUs from a job with `--gres=gpu:0`); `submit.sh` now hides
   them.
4. **Audit.** `--reproduce` now re-generates a sampled prediction file on the cache its run used.
5. **S3** (17 WHAR Arena models, 510 runs) stays records only; retraining it (10-20 GPU hours) was not approved.


## HARFactors, 2026-10-17: revision after the novelty review (user request)

The user asked to state the novelty and the significant contributions clearly and to fix the errors listed in
`docs/review_novelty_2026-10-17.md` (page limits still waived). No network was trained; every new number comes from the
stored runs, the confirmatory outputs or the Arena table. None of these analyses was pre-registered; the manuscript
marks them (r).

### Configurations run

- `src/evaluation/network_population.py` (ledger `20261017_061812_network_population`, Slurm 87063): S3's model refitted
  on the 14 deep networks and on the best k networks, k = 3-17, ranked by mean published macro-F1 on the 26 Arena
  datasets outside S3 (the ranking reads no S3 score). Point estimates keep the better of statsmodels and the
  independent REML (they agree within 0.8 % in R on every population); intervals use H1's bootstrap (1,000 resamples,
  seed 20261015).
- `scripts/run_analyses.py --only validation_reliability factor_multiplicity error_budget` (ledger
  `20261017_061800_run_analyses`): the validation-vs-test reliability (S1 and S2, read logged in
  `results/test_access.log`), Holm's correction over the 16 frozen OLS coefficient tests and the order-statistic check,
  and the error budget from the point estimates of H1 and of S1's sensitivity analysis.
- `src/visualization/contribution_figures.py`: Fig. 5 of the rendered letter (`figures/fig10_population_budget.pdf`) and
  Table 4 (`tables/tab04_error_budget.tex`).

### Results that changed the text

1. The S3 reversal is caused by the weakest networks: R' stays above 1 (point estimate) for the best 3-12 networks
   (no interval excludes 1) and every interval lies below 1 from the 13th on (DeepConvShallowLSTM, Guan-LSTM, SA-HAR,
   then SVM, k-NN). The 14 deep networks alone give R' = 0.34, with network x dataset as the largest term. The review's
   item 2 (that the "weak baseline" sentence was unsupported) was only half right: weak networks do cause the
   reversal, but they are not only the classical baselines. The text now says exactly this.
2. Validation ranks networks far worse than the test ranks itself: median Spearman +0.18 against +0.80 (seeds) and
   +0.80 (split halves) on 13 datasets.
3. Of the 16 OLS tests, 2 survive Holm (channels with the RF margin and with the split spread); the subjects-gap
   association does not (adjusted p 0.058), and the gap tracks the split spread (rho +0.60).

### Text corrections

- S4D vs the no-memory control: no longer "the same network"; width, depth and size are stated.
- S2 disclosure: paired network differences were known too.
- P is the effect of the subject partition (mainly the test group); validation windows can overlap training windows.
- The embedding result leads with the sensor-feature space; the embedding-space correlation is called partly circular.
- New references (verified, `paper/references_verified.json`): Jordao et al. 2018, Barshan & Yurtman 2016, Varoquaux
  2018, Iwasawa et al. 2017.

### Later the same day: why these networks (user request)

- **Question.** Why were the four S2 networks used, and not only the benchmark's networks? The user also asked not to
  cite the companion study.
- **Answer in the letter (Section II-D).** It states the requirements of a confirmatory estimate and shows that only
  S2 meets them among the stored runs; the benchmark has one run per split and published means only. The section also
  states that three S2 networks share one memory, so S2 measures side paths and one alternative memory.
- **Provenance.** The S2 runs are disclosed as produced earlier by the authors for a held-out evaluation of these
  networks, with their known means and paired differences.
- **`src/evaluation/s2_positioning.py`** (exploratory, (r)) locates the four networks against the published benchmark
  on H1's datasets:
  - means 58.0-63.2;
  - median within-dataset range 7.2 points, against 27.0 for the 17 published networks;
  - ranks 5-13 of 17 if inserted.

  It compares three-seed means with published single runs, as a location and not a ranking.
- **Risk to raise with the user.** If the companion study publishes the same S2 results, an editor may ask for the
  overlap to be disclosed at submission (IEEE prior-publication policy), even without a citation.

### Later the same day: companion-study citation and overlap (user request)

The user first asked not to cite the RILA-Net study, then decided to cite it and declare the overlap. At the user's
request the RILA repository (private GitHub `Leminhhuy2386/RILA`, commit 1742ef6) was cloned read-only to
`/work/huy.leminh/.cache/harfactors_scratch/rila_ref`, outside this project. Nothing in it was run or changed.

**What it showed:**
- RILA-Net is RN + laws.
- The companion manuscript's four primary models are S2's four networks, and its confirmatory results come from the
  same 948 S2 runs.
- It reports the component contrasts (memory vs none, memory type, side paths) and the learned recurrence radii.
- It cites this letter as `@unpublished{harfactors}` under an earlier title.
- After its G5 it retrained 176 runs it could not re-execute. Compared with this project's records, these differ in:
  - 3 runs of H1's datasets: recstatnet/w_har x2 (mean -1.44 F1), recurrencenet/usc_had x1 (-4.29);
  - 24 DSADS seed-2 runs of three networks (split-level differences up to 28 points: the retrained splits name other
    subjects);
  - MotionSense and Daphnet runs, both excluded here.

**Changes to the letter:**
- `le2026rila` added: unpublished, in preparation for IEEE Sensors Journal, verified against the companion
  manuscript's own title page.
- The memory contrasts (Fig. 2a and their paragraph) and the section "What the memories learned" (former Fig. 9)
  were removed, and the companion study is cited instead.
- Fig. 2 is now the channel-count panel only (`figures/fig02_channels.pdf`); the superseded
  `fig02_contrasts_channels` outputs and figure data were deleted.
- Section II-D and the network description cite the companion study.
- The verification paragraph states the record differences.
- `paper/cover_letter.md` declares the overlap.
