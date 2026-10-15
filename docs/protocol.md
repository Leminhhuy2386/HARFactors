# Data and protocol (WHAR Arena v1, paper-era code)

> Extracted unchanged from Edge_SSM's RESEARCH_PLAN.md on 2026-10-14, when this project was split off.
> Paths inside refer to Edge_SSM's layout. Here: the WHAR code is `third_party/whar-*` (same commits); data
> caches are `D:\Projects\HAR\datasets_paper_era` (shared); runs are `experiments/whar_arena_v1/`;
> aggregates are `results/metrics/<analysis>/`; figures are `figures/`; model code is `src/models/`; the
> harness is `src/training/run_whar.py`; the paper's Figure 6 numbers are `docs/sources/arena_extracted.json`;
> the 17 re-runs are `experiments/external_har_paper_era/`. The full original plan is in `docs/history/`.

## 2a. Data and protocol: where the data comes from, and how close it is to WHAR Arena

**Dataset: UTD-MHAD, inertial part.**
- 8 subjects, 27 actions (swipes, waves, drawing shapes, sports moves, walk, jog, sit/stand, lunge, squat).
- 6 channels: 3-axis accelerometer `Ax, Ay, Az` and 3-axis gyroscope `GyroX, GyroY, GyroZ`, at 50 Hz.
- The wearable sensor is on the right wrist, or the right thigh for the leg actions (per the dataset's own description).

**Where it comes from.**
- **Raw data:** the original UTD-MHAD inertial `.mat` files (`aA_sS_tT_inertial.mat`), as named on the dataset page
  https://personal.utdallas.edu/~kehtar/UTD-MHAD.html. The paper-era whar-datasets (`cfg_utd_mhad.py`, 144e006)
  downloads them from the Google Drive link in that config and parses them.
- **Cache:** this project downloaded nothing. It reads the cache that the earlier HAR reproduction built at
  `D:\Projects\HAR\datasets_paper_era\utd_mhad` (`data/`, `sessions/`, `windows/`, `samples/`, `metadata/`).
  Every run logs `Hash is up to date` for each step, so the cache is reused unchanged.
- **The 17 baselines:** their results come from the same cache and code (`D:\Projects\HAR\paper-era\env\results`).

**Protocol: WHAR Arena v1 (`whar-arena/evaluation/protocol.yaml`, paper-era 19b3019), applied by the harness's own code.**

| Item | Value | How it is applied here |
|---|---|---|
| Window | 3 s (150 samples at 50 Hz), 50 % overlap, native rate | `training._configure_dataset_cfg` (overrides the dataset config's own 2 s) |
| Normalization | global z-score, fitted on the training subjects only | `PostProcessingPipeline` |
| Splits | k = 10 subject groups → 8 splits for 8 subjects (one test subject per split) | `LKSOSplitter` (see "How the splits are made") |
| Validation | 20 % of the training subjects' windows, drawn at random per window (not per subject) | `Splitter._get_train_val_indices` |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, batch 64, cosine LR, ≤ 100 epochs | `training._fit_torch_split` |
| Loss / stopping | class-weighted cross-entropy (train frequency); early stop on validation macro-F1, patience 10, best epoch restored | `training._fit_torch_split` |
| Metric | macro-F1 over all 27 activities (harness definition: absent classes count as 0) | `training._classification_metrics` |
| Seed | `cfg.seed = 0`, RNG reseeded before the model is built, as `TorchAdapter` does | `run_whar.fit` |

**How the splits are made.** The protocol ships a rule, not split files. The paper-era code regenerates the splits
on every run, and they come out the same each time. Source: `whar-datasets\src\whar_datasets\splitting\splitter_lkso.py`
and `splitter.py`.
1. Subject ids are sorted. Subject *i* goes to fold *i* mod min(k, #subjects). For UTD-MHAD, split-0*i* tests subject index *i*.
2. Test = all windows of that fold's subjects.
3. Train/val = the remaining windows, permuted by one `np.random.RandomState(cfg.seed)`; the first 20 % become validation.
   That RNG is shared by the folds in order, so a fold's validation set depends on the folds before it.
   Always generate all splits through `run_whar.prepare_dataset` (it does, even with `--splits`). Never build one fold
   alone, write a custom splitter, or change `cfg.seed`.
4. `cfg.seed` is 0: the `WHARConfig` default, since the protocol has no `seed` key. The same seed also drives the
   shuffle generator and the model init (P4).

**UTD-MHAD facts that shape every score** (counted from `datasets_paper_era\utd_mhad\metadata`):
- **Dropped trials.** 861 trials (27 actions × 8 subjects × 4 trials, 3 missing). 174 of them are shorter than 3 s
  and give no window, so they are never used. The other 687 trials give 783 windows: 598 trials give 1 window,
  82 give 2, and 7 give 3.
- **Activities per test subject.** Activities with at least one window, by subject index 0–7: 27, **14**, 24, 21,
  26, 27, 26, 27.
- **Ceiling per split.** Harness macro-F1 averages over all 27 classes, and an absent class scores 0. The best
  possible score per split is therefore 100, **51.9**, 88.9, 77.8, 96.3, 100, 96.3, 100 (mean 88.9). Split-01 scores
  of 42–46 for every model are near that ceiling, not failures.
- **What this means for comparisons.** Most of the split-to-split spread (std ≈ 13) comes from the ceiling.
  - Compare models only paired by split.
  - Never compare means over different sets of splits. Splits 4–7 average higher than 0–7 for every model,
    e.g. QuantileNet 80.9 vs 74.3.

**Verified equal.** Train/val/test sizes and split identifiers match the WHAR baseline result files for all
8 UTD-MHAD splits (e.g. split-00: 536 / 134 / 113, `group_kfold_0`), and for ActRecTut (334 / 83 / 381,
305 / 76 / 417). Every model, ours and WHAR's, is therefore tested on the same subjects and windows.

**Differences from running `whar-arena` itself.** None change the data, the splits or the metric.
1. **Our own outer loop.** `whar_arena.training.run_benchmark` only builds models from the whar-models registry,
   so `run_whar.py` repeats its outer loop (same calls, same order) and builds our models from `edge_ssm/registry.py`.
2. **Tensor dataloaders.** Each split is read into tensors once, instead of through `TorchAdapter.__getitem__`
   per window. Batch size, shuffle and generator seed are the same, so the batches are identical. This is only for speed.
3. **Deterministic mode with `warn_only=True`,** as the paper-era runner used for the WHAR baselines.
4. **Results location and fingerprint.** Results are written to `Edge_SSM/results/`, with our own fingerprint and
   extra fields (validation F1, params, device).
5. **Beyond the protocol:**
   - Tempo warps of the test windows (`warp_test.py`).
   - Variant selection by protocols A/B (§6).
   WHAR Arena itself has neither: its models use fixed published configurations.

**Code comparison with the HAR paper-era runner** (`D:\Projects\HAR\paper-era\env\run_benchmark_paper_era.py`).
That runner states that it uses whar-datasets 144e006, whar-models 9b65004 and whar-arena 19b3019, unmodified
except for `warn_only=True` and the results path. `HAR\ATLAS.md` §5 note 1 says this code reproduces the authors'
split sizes exactly, and CNN-HAR digit for digit. `run_whar.py` matches it as follows.

| | HAR runner | `run_whar.py` |
|---|---|---|
| whar-arena source | `paper-era\whar-arena\src` | same path (`ARENA_SRC`) |
| whar-datasets / whar-models | paper-era checkouts (env `paper-era\env`) | same checkouts as path deps; `uv.lock` copied from that env |
| Data cache | `datasets_paper_era` | same (`DATASETS_DIR`) |
| Protocol and config | `load_protocol`, `_configure_dataset_cfg` | same functions, unpatched |
| Seeding | `_seed_everything` + `warn_only=True` | same |
| Splits | `PreProcessingPipeline` → `LKSOSplitter`, `PostProcessingPipeline` per split | same calls, same order |
| Training and metric | `_fit_torch_split` (`_evaluate`, metrics inside) | same function |

- **Seeding order.** In `_run_split`, `TorchAdapter.__init__` seeds torch, numpy and random with `cfg.seed` (0, from
  the protocol). `_infer_dataset_shape` then draws only from the adapter's own generator, and the model is built
  afterwards. `run_whar.fit` reseeds the global RNGs right before the build, so the model sees the same RNG state.
- **Guard.** `check_paper_era()` refuses to run unless all three repos are at the paper commits with clean trees,
  the protocol hash is `e8739ef2c5b36c99`, and the data directory is not the latest-code cache `HAR\datasets`.
  The commits it found are written to each result file under `whar_code`.

**Inherited quirks** (shared with the paper and the 17 baselines, so keep them):
- **Macro-F1 over all classes, absent classes as 0.** This gives the per-split ceilings above. A macro-F1 over the
  classes present could be reported as a *secondary* metric, but never in place of the harness one.
- **Validation from the training subjects, drawn per window.**
  - It shares subjects, and sometimes trials, with train.
  - Windows overlap by 50 %, so two windows of one trial can share 75 samples across train and val. Only 185 of
    the 783 UTD windows come from multi-window trials, so the main leak is the shared subjects.
  - Validation F1 is therefore 93–98 % and does not predict test F1 (§6).
  - The latest whar code purges train/val overlaps ("strict split"; `D:\Projects\HAR\ATLAS.md` §5). Do not adopt it here.
- **The test set is clean.** Test subjects never appear in train or val.

This project stays on the paper-era code so the 17 WHAR baselines and the paper's numbers stay comparable.

## 6. Methodology notes (lessons already paid for)

- **The bar is the same network without memory (67.6), and S4D (73.8), not the WHAR table.** On 3 s
  UTD-MHAD windows, which values occur already carries most of the label.
- **Validation F1 does not predict test F1 here.** Validation windows come from the training
  subjects, drawn per window (§2a), and score 93–98 %.
  - Use **protocol B**: choose on test F1 of splits 0–3, judge only on splits 4–7.
  - Even that reversed in round 3, because split-to-split spread (std ≈ 13) exceeds the gaps
    between architectures (≈ 1 point). More datasets add more statistical power than more seeds.
  - Much of that spread is the per-split F1 ceiling (§2a). Splits 0–3 hold the low ceilings, 51.9 and 77.8.
    A protocol-B score on splits 4–7 is only comparable with other models' scores on the same splits.
- **Learned scales (τ, r) stay at their initial values** under this protocol (AdamW 1e-3 with
  weight decay on everything). Freezing them changes nothing.
- **Rate-independent quantile trackers (PathQuantileNet) lost 3.5 points.** Part of the
  quantile memory's value is its per-sample, rate-dependent dynamics.
- **Reproducibility checks:**
  - Training is deterministic for a given seed. After refactors, QuantileNet split 0 and
    HysteresisNet split 0 reproduced to the last digit (0.8352, 0.8368).
  - `warp_test.py` uses the same `fit()` and writes the run_whar result when it is missing.
- **Results are skipped by fingerprint** (protocol, model config, dataset, split, epochs,
  patience). Changing a config in `registry.py` invalidates old results for that id, so add a
  **new id** instead of editing one that has results.
- **One seed is not enough on UTD-MHAD** (round 4). Within a split the seed std is 3–4 points, and a
  model's mean over splits moves by up to 5 points between seeds (HysteresisNet sel+clock 71.2–76.5).
  Compare architectures on seed-averaged splits.
- **Test the property you claim with the physics of the claim.** Stretching only the time axis made the
  tempo robustness look like +29 to +40 over S4D; scaling angular rates and accelerations as a real faster
  gesture does reduced it to +9, with every model collapsing. Device-rate changes (same signal, other
  sampling rate) are the case where rate independence really pays.
- **A robust memory does not make a robust network.** The quantile trackers move one step per spike, yet
  QuantileNet is not more spike-robust than S4D: the block passes u straight to the output and the readout
  averages over time.
- **Local PC limits:** 16 GB RAM. About 6 concurrent training processes is the limit; at 8–9 some died with
  exit code 116 or `CUDA error: out of memory` although VRAM was free. Send extra parallel work to Colab/Kaggle.
