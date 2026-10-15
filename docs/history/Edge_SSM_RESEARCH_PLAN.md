# Research plan: new memory architectures vs. old ones

Handoff document for any session that continues this work. Read **§1–§3** first, pick the next
`TODO` in **§5**, and update its row and **§8** when you finish.

_Last updated: 2026-10-10 (round 8 started, §5e)._

---

## 1. Goal and standing decisions

**Goal (user, 2026-10-08).** Find a model family whose core computation is new — not a
modification of MLP, CNN, RNN, attention, SSM/Mamba, KAN, CfC/LTC or spiking networks — that
beats the old families. Edge (microcontroller) deployment is the intended use.

**Clarified by the user, 2026-10-11 (applies from then on).** "New" means a new *architecture
family*: a network structure that differs from every family in `RESULTS_SUMMARY.md` §0 "Cấu trúc
mạng" (MLP, KAN, CNN, Transformer, RNN/LSTM/GRU, CfC/LTC, SNN, SSM/S4D/Mamba, and the project's own
HysteresisNet/QuantileNet in their shared SSM-style skeleton). Known layers such as convolution,
attention, LSTM/GRU or SSM blocks **may be used freely** as parts. The novelty must be in how the
network is structured, not in avoiding these layers. Stacking known layers in a known pattern (for
example conv then LSTM, or conv plus attention) is still an existing family. Sessions before
2026-10-11 read the goal as "no conv/attention/RNN/SSM at all"; their results stand as run.

Decisions the user made; do not reverse them without asking:

| Date | Decision |
|---|---|
| 2026-10-08 | Edge-SSM (the original per-edge idea) is dropped; look for completely new families. |
| 2026-10-08 | **One dataset first (UTD-MHAD).** Expand to more datasets only if a model looks good **and the user approves**. |
| 2026-10-08 | Use the GPU; Colab/Kaggle are allowed when local is too slow (ask before uploading data). |
| 2026-10-09 | Results go into `RESULTS_SUMMARY.md` (Vietnamese) and the HTML report, **not** into `D:\Projects\HAR\ATLAS.md`; the HAR project must not be modified. |
| 2026-10-09 | The user wants structure figures and an HTML page for checking results. |
| 2026-10-09 | **Only the WHAR code version of the paper** (whar-datasets 144e006, whar-models 9b65004, whar-arena 19b3019) and its data cache `D:\Projects\HAR\datasets_paper_era`, never the latest GitHub version. The paper's tables and the 17 baseline results come from that version. `run_whar.py` enforces it (§2a, Guard). |
| 2026-10-09 | Protocol questions are answered by reading code: this project against `D:\Projects\HAR\paper-era` and `HAR\ATLAS.md`, which already document and validate the versions. Do not re-run WHAR models to prove conformance. |
| 2026-10-10 | Round 8 (§5e): look for models fast on the 10 P5 datasets as dev; new held-out datasets come later. The raw files in `D:\Projects\HAR\datasets` may be used directly (no copy), but nothing there may be deleted. They are re-processed with the paper-era code into `datasets_paper_era_local/` through hard links; HAR's latest-code cache is never read. |
| 2026-10-10 | **Screen fast, stop early.** A candidate is stopped as soon as it is clearly worse than the bar on a few datasets; then try another model. Round 8 (LawNet) was stopped this way after 5 of 10 dev datasets. |
| 2026-10-11 | **Known layers are allowed; the network structure must be new.** Conv, attention, LSTM/GRU and SSM layers may be used; the family must differ structurally from those in `RESULTS_SUMMARY.md` §0 (see the clarification under the goal). |
| 2026-10-12 | **Write a full paper for IEEE Sensors Journal** on the best family (RN + laws, RN + statistics). Held-out confirmation on 8 new Arena datasets: PAMAP2, UP-Fall, UMA-Fall, USC-HAD, MotionSense, WEAR, UCA-EHAR, W-HAR. Paper baselines are taken from the paper's Figure 6 (no re-runs). Measurements on the NUCLEO-L4R5ZI come after the held-out results. Plan in §5p. |
| 2026-10-11 | **Direction 3, aimed at a Nature-family journal: a capability claim, not another accuracy point.** The paper centres on sampling-schedule-agnostic inference (HysteresisNet's rate-independent play memory): one model for any sampling rate, dropped or irregular samples and event-driven sampling. Hardware then: NUCLEO-L4R5ZI only, no IMU and no power meter. Plan in §5k (stopped). Hardware update 2026-10-13: the lab has STM32 L4, H7 and N6 boards and a Fermion 10-DOF IMU module (ADXL345 / ITG-3205 / VCM5883L / BMP280, I2C); still no power meter (§5q). |

House rules for this project:
- Chat in Vietnamese; files, code comments and commits in English (`RESULTS_SUMMARY.md` is the one Vietnamese file).
- **Never choose a variant by test scores.** Use protocol B (§6) or validation, and say which.
- Report negative results; every claim needs a paired test over the same splits.

## 2. Where things stand

**Update after round 6 (2026-10-10): RecurrenceNet has the best mean rank over the 10 P5 datasets** (2.00; S4D 2.50,
HysteresisNet sel+clock 2.60; mean macro-F1 72.0 vs 71.3; 5 first places vs S4D's 1). RecurrenceNet − S4D +0.71
(95 % CI −0.83 … +2.26, 7/3, p = 0.19): non-inferior with the 1-point margin, not significantly better. On the 6 held-out
datasets that played no part in choosing it: +1.35 (CI −0.93 … +3.64, 4/6). Better than Mamba (+3.10, CI +1.46 … +4.74,
9/1). Caveat: its novelty is only partial.
- Its layer pattern (similarity to lagged copies of itself used as features, with a residual) is in SELFY (ICCV 2021).
- Its learned-radius sigmoid threshold is in Jeong & Yoo 2026.
- Re-checked on the primary papers on 2026-10-12, §5l. Details in §5c, `results/p5/summary_recurrencenet.md`.

**Verdict after P5 (10 WHAR datasets × 3 seeds, pre-registered; round 5, 2026-10-09): no new family beats the old ones.**
- **Mean rank** (1 = best): S4D 1.80, HysteresisNet sel+clock 2.00, QuantileNet 3.00, Mamba 3.70, no memory 4.50
  (Friedman p = 0.0004; Nemenyi CD 1.93). Mean macro-F1: 71.3 / 70.7 / 69.7 / 68.9 / 66.1.
- **HysteresisNet sel+clock is the strongest new family:** level with S4D (5 first places each, 5/5 paired, −0.6,
  95 % CI −2.1 … +0.9; non-inferiority within 1 point not shown), above Mamba on 9 of 10 datasets (+1.8, p = 0.065),
  and rate-independent where S4D is not (P2).
- **QuantileNet's UTD lead did not generalize:** third, below S4D on 8 of 10 (−1.6, CI −3.5 … +0.3), above Mamba
  (+0.8, CI +0.04 … +1.5, 7/3).
- **Larger WHAR baselines are stronger where they exist** (ActRecTut, UniMiB, UCI-HAR: best WHAR model > all five of
  ours). A fair claim against old families needs CNN/attention baselines at equal size (P6, needs approval).
- Details: `NEW_ARCHITECTURES.md` round 5, `results/p5/summary.md`, `results/p5/cd.svg`.

**UTD-MHAD detail (8 subject splits × 5 training seeds, WHAR Arena v1 protocol; round 4, 2026-10-09):**
- **QuantileNet has the best accuracy:** significantly above Mamba (+2.1, 7/1, p = 0.023), level with S4D
  (+1.45, 95 % CI −1.5 … +4.4). Non-inferiority to S4D with a 1-point margin is **not** shown; 8 splits cannot
  narrow the interval further, so this needs more datasets (P5).
- **HysteresisNet's play memory is rate-independent in a measurable way:** invariant to the device's sampling
  rate (25–100 Hz) and to dropped samples, and best on speed-varied shapes (P10). S4D collapses there.
- **Correction:** the large tempo gains of round 1 (+29 to +40 over S4D) came from warps that stretch only the
  time axis. With physically consistent speed changes (larger angular rates and accelerations) every model
  collapses; HysteresisNet is still best, by +9 over S4D and +4 over Mamba.
- **QuantileNet is not outlier-robust** at network level: below S4D on spikes (P3) and on the synthetic outlier task (P10).

| Model | Params | Macro-F1, 5 seeds (seed 0 only) | Physical tempo, uniform | Device rate 25–100 Hz | Spikes | Status |
|---|---|---|---|---|---|---|
| **QuantileNet** (new) | 20.8k | **73.99** (74.29) | 48.8 | 68.1 | 56.8 | lead candidate (accuracy, novelty) |
| **HysteresisNet selective + clock** (new) | 23.6k | 73.23 (73.36) | **55.2** | **73.3** | 40.2 | lead candidate (rate independence) |
| DualMemNet (round-3 upgrade) | 26.7k | — (73.93) | — | — | — | no gain; dropped |
| S4D (old) | 27.7k | 72.55 (73.79) | 45.9 | 52.2 | **59.2** | main bar |
| Mamba (old) | 21.1k | 71.89 (70.69) | 51.4 | 69.7 | 57.1 | |
| Same network, no memory (control) | 20.0k | 67.12 (67.61) | 46.1 | 67.2 | 54.4 | bar for any new memory |
| best WHAR model (Attend+Discriminate) | 374k | — (67.48) | — | — | — | |
| Edge-SSM v2 (original idea) | 33.6k | — (51.61) | — | — | — | failed |

Key paired results (8 splits, seeds averaged within a split, Wilcoxon):
- QuantileNet − S4D: +1.45 (4/4, p = 0.46). A tie. QuantileNet − Mamba: +2.10 (7/1, p = 0.023).
- QuantileNet − no memory: +6.87, HysteresisNet sel+clock − no memory: +6.10 (both 8/0, p = 0.008).
- HysteresisNet sel+clock − S4D: device rate +21.1, physical tempo +9.2, spikes −19.0 (all 8 of 8 splits one way).
- QuantileNet with the hard comparator, as on an MCU: split 0 83.5 → 84.9, split 4 76.8 → 76.0.
- Full tables: `results/seeds/summary_utd_mhad.md`, `results/robust/summary_utd_mhad.md`, `results/synthetic/summary.md`,
  `NEW_ARCHITECTURES.md` round 4.

Novelty (web literature check by sub-agents; key equations verified on the PDFs):
- **QuantileNet:** not found as a deep-network memory. Closest: Behrouz et al. 2025, arXiv:2504.13173, Eq. 12 (sign update, symmetric, no τ); Frugal Streaming 2014.
- **HysteresisNet:** partly new. Frydrych 2026, arXiv:2605.23603, proposes hysteretic memory as a sequence primitive, in theory only, with relays and no scan.
- **TropicalNet:** prior art. Peng et al., EMNLP 2018, App. D, Eq. 24c. Dropped.

Where the details are:
- `RESULTS_SUMMARY.md`: one-page summary with figures and tables.
- `NEW_ARCHITECTURES.md`: write-up of rounds 1–3.
- `report/index.html`: charts and paired tests.
- `RESULTS.md`: full tables.
- `EDGE_SSM_SUMMARY.md` §6 and `Edge-SSM atlas.html`: Edge-SSM history.

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

## 3. Environment and how to run

- **Project:** `D:\Projects\Edge_SSM`, uv env (Python 3.12, torch 2.14.0+cu126; GTX 1080 Ti, Pascal: float32 only, no AMP).
  `uv.lock` is copied from `D:\Projects\HAR\paper-era\env`, so packages match the WHAR baselines exactly.
- **WHAR code:** the paper-era checkouts `D:\Projects\HAR\paper-era\whar-{datasets,models,arena}` (144e006 / 9b65004 / 19b3019).
  Do not edit anything under `D:\Projects\HAR`.
  - `run_whar.py` stops with a `RuntimeError` if a checkout is at another commit or has local changes, if the
    protocol hash is not `e8739ef2c5b36c99`, or if `WHAR_DATASETS_DIR` points at `D:\Projects\HAR\datasets`.
  - Do not work around that error; it means the numbers would not be comparable with the paper.
  - WHAR model ids (e.g. `cnn_har`) also run through `run_whar.py`. Use this only to check the runner, and
    point `EDGE_SSM_RESULTS_DIR` to a scratch folder.
- **Data:** `D:\Projects\HAR\datasets_paper_era` holds only 4 datasets: utd_mhad, uci_har, unimib_shar, actrectut_gestures.
  `D:\Projects\HAR\datasets` holds 30, but those were preprocessed with the *latest* whar code (see P5).
- **Baselines:** results of the 17 WHAR models (paper-era, same splits) are in `D:\Projects\HAR\paper-era\env\results\whar_arena_v1`.
- **Use PowerShell, not Git Bash, for `uv`.** Logs written with `*>` are UTF-8 with a BOM in this setup (checked 2026-10-09; plain `grep` works). PowerShell `.ps1` scripts are blocked by execution policy, so do not change it.
- The GPU is often shared with `D:\Projects\AutoTinyAI_2` jobs, so expect 2–5× slower steps.
- Publishing Artifacts to claude.ai failed with "OAuth token revoked". The HTML report is local; open it in the built-in browser via `file:///`.

```
uv run python tests/test_edge_ssm.py                         # all models: parallel == streaming == reference
uv run python run_whar.py --models quantilenet --datasets utd_mhad [--splits 0 1]
uv run python warp_test.py --models quantilenet --dataset utd_mhad [--splits ...]   # trains + tempo warps, also writes run_whar results
uv run python warp_test.py --summarize                       # results/warp/summary_utd_mhad.md
uv run python select_variants.py --warp-splits=0-3 m1 m2 ... # protocol B on the choosing splits
uv run python profile_models.py                              # params, FLOPs, CPU latency -> results/profile.json
uv run python build_arch_figures.py; uv run python build_summary.py; uv run python build_report.py; uv run python compare_whar.py

uv run python run_whar.py --models quantilenet --datasets utd_mhad --seed 1      # P4: training seed 1 -> .../utd_mhad/seed-1/
uv run python run_whar.py --models s4d --datasets utd_mhad --need-checkpoint     # retrain finished splits that lack a checkpoint
uv run python seeds_summary.py                               # P4 -> results/seeds/summary_utd_mhad.md
uv run python robust_test.py --models quantilenet s4d        # P1-P3 on every saved checkpoint -> results/robust/
uv run python robust_test.py --summarize                     # results/robust/summary_utd_mhad.md
uv run python synthetic_test.py --models s4d --seeds 0 1 2   # P10 synthetic tasks -> results/synthetic/
uv run python synthetic_test.py --summarize                  # results/synthetic/summary.md

uv run python cloud/kaggle_submit.py push  cloud/jobs/kaggle_p4.json edge-ssm-p4   # Kaggle batch run (2x T4), private kernel
uv run python cloud/kaggle_submit.py fetch cloud/jobs/kaggle_p4.json edge-ssm-p4   # waits, downloads, merges into results/
uv run python cloud/colab_supervise.py cloud/jobs/colab_p4.json colab-p4          # Colab VM: run, pull every 3 min, merge, stop
```

**Training seeds (`--seed`).** The protocol's `cfg.seed = 0` still fixes the data side (subject splits, validation
windows, normalization). A training seed s ≠ 0 changes only the model initialization and the batch order. Seed 0
is unchanged: same calls, fingerprint and path as before (14 retrained splits reproduced to the last digit on
2026-10-09). Results of seed s ≥ 1 go to `<model>/<dataset>/seed-<s>/split-XX.json`, which the `split-*.json` globs of
the older scripts do not see. Every training run also saves its weights to `results/checkpoints/`.

**Cloud runs.** Logins are in place (Colab CLI token in `~/.config/colab-cli`, Kaggle token in `~/.kaggle/access_token`,
user `kebitmatf`). `cloud/remote_job.py` runs on the VM: it clones the WHAR code at the paper commits, pins the
data-side packages to the local `uv.lock` versions (pandas 3.0.6, numpy 2.5.3, ...), downloads UTD-MHAD from its
public source and builds the cache once before the parallel jobs start. No data is uploaded, only code and finished
result files. Give each machine a disjoint set of (model, seed, split); assign whole seeds to one machine so every
model of a seed runs on the same hardware. Colab VMs also have a `/kaggle` directory, so Kaggle is detected by
`KAGGLE_KERNEL_RUN_TYPE`. The train/val/test sizes built on the Colab VM matched the local ones for all 8 UTD-MHAD splits.

**Fast training paths (`EDGE_SSM_FAST=1`, added for P5).** The sequential loops of QuantileNet's trackers, Mamba's
selective scan and HysteresisNet's play operator run as TorchScript loops with hand-written backward passes, and the
memoryless control skips its identity scan. Same math; values and gradients equal the reference up to float rounding
(play operator: exact values, gradient differs only at exact ties, 2 of 1.2 M elements in a float32 check). Measured on
the GTX 1080 Ti per training step (batch 64, L = 150): QuantileNet 246 → 30 ms, Mamba 107 → 40 ms, HysteresisNet
sel+clock 56 → 32 ms, no memory 54 → 4 ms (S4D has no loop and is unchanged, ~9 ms). Retraining UTD-MHAD seed 0 with it (`results/fastcheck/`)
gave identical macro-F1 on all 8 splits for QuantileNet, S4D, Mamba and no memory, and 3 of 8 for HysteresisNet sel+clock
(mean 73.17 vs 73.36, inside its seed range 71.2–76.5). Off by default so stored results reproduce; every result file
records `fast_paths`.

Typical time per UTD-MHAD split (shared GPU): HysteresisNet 30–90 s; S4D 15 s; GRU 10 s;
Mamba about 2 min; **QuantileNet 3–7 min** (sequential loop over 150 steps).

## 4. Code map

| Path | What |
|---|---|
| `edge_ssm/quantile.py` | QuantileNet (`quantile_track`, `quantile_step`; `path=True` = PathQuantileNet) |
| `edge_ssm/hysteresis.py` | HysteresisNet (`play_scan` parallel clamp scan; `selective`, `clock`, `pooling`, `memoryless`, `learn_widths`) |
| `edge_ssm/dualmem.py` | DualMemNet (both memories in one block) |
| `edge_ssm/vortex.py`, `edge_ssm/tropical.py` | VortexNet; TropicalNet (prior art) |
| `edge_ssm/baselines.py` | GRU, S4D, Mamba (plain PyTorch, exact parallel scan) |
| `edge_ssm/model.py`, `edge_ssm/layers.py` | Edge-SSM v2 |
| `edge_ssm/registry.py` | every model id and its config |
| `run_whar.py` | trains with the paper-era harness; result json has `fingerprint`, `val_macro_f1`, device, params |
| `warp_test.py` | tempo robustness: uniform (×0.5–×2) and random within-window warps |
| `select_variants.py`, `check_quantile_hard.py` | variant selection; soft vs hard comparator |
| `seeds_summary.py` | P4: seed-averaged per split, paired over splits, 95 % CI and non-inferiority vs S4D |
| `robust_test.py` | P1–P3 from checkpoints: physical tempo, device rate, irregular sampling, noise, spikes, bias |
| `synthetic_test.py` | P10: four synthetic tasks with a stated winner for each memory |
| `early_test.py` | P9: checkpoints scored on the first 25/50/75 % of each window (`results/early/`) |
| `p5_summary.py` | P5: per-dataset scores, mean ranks, Friedman, Wilcoxon-Holm, CD diagram (`results/p5/`) |
| `compare_arena.py` | our finished results placed in the WHAR Arena paper's Figure 6 (17 models), no training (`results/arena/`) |
| `build_*.py`, `compare_whar.py`, `profile_models.py` | figures, summary, HTML report, tables, edge cost |
| `cloud/` | `remote_job.py` (runs on the VM: setup, per-dataset cache, bounded job queue), `colab_supervise.py`, `kaggle_submit.py`, `local_queue.py` (same job files on this PC), `make_p5_jobs.py`, `common.py`, `jobs/*.json` |
| `colab/` | old Colab supervisor, superseded by `cloud/` (never run) |

## 5. Experiment plan and status

Status values: `DONE` · `PARTIAL` · `TODO` · `NEEDS APPROVAL` (outside the one-dataset rule, or uploads data).
"Primary" marks the experiments that decide the claim.

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

### 5c. Round 6 (started 2026-10-10): memories from comparisons and counting

User request: a more innovative new family. Three candidates whose core computation is comparisons, a data
structure or counting, not a register updated by arithmetic (`edge_ssm/round6.py`; configs fixed in `registry.py`
before any result, ~19–25k params, same skeleton as the earlier families):
- **RainflowNet** (`rainflownet`): streaming rainflow counting (ASTM four-point rule) on each mixed channel. The state is
  the stack of unresolved turning points (the Preisach staircase); the output is the cycles that close at t, soft-binned
  by amplitude, plus the open excursions. Training uses an exact O(log L)-step parallel form of the closing rule
  (nearest-greater elements + range minima; checked equal to the streaming algorithm on 512 sequences, ties broken by a
  1e-9 ramp). Literature check: partly new. The extremum-stack memory in stacked residual layers is already in Frydrych,
  "Preisach Attention", arXiv:2605.23603 (theory only, relay readout, no experiments); new are the closed-cycle readout,
  the parallel training form and any experiment. Other neighbours: Topological Attention (Zeng et al., NeurIPS 2021,
  arXiv:2107.09031, 0-dim persistence of sliding windows, no gradient through it); a differentiable rainflow loss
  (arXiv:2609.12968); hand-crafted 1-D persistence descriptors for HAR on a Cortex-M33 (EWSN 2026). Rainflow cycles are
  not the full 0-dim persistence of an open window (0,5,1,4,2,3 closes nothing but has pairs).
- **OrdinalNet** (`ordinalnet`, "ordinal-pattern net" in reports; name clash with an R package): soft Bandt–Pompe order
  patterns of (u_{t−2τ}, u_{t−τ}, u_t) at τ = 1, 4, 16, scored per channel. Literature check: partly new (closest:
  Zanin 2023 continuous ordinal patterns; Roy et al. 2024, arXiv:2407.07524; LBP Networks, arXiv:1803.07125). Exact
  monotone invariance only with hard comparisons at inference, and on learned mixtures only offset/shared-gain invariance.
- **RecurrenceNet** (`recurrencenet`): soft recurrence-plot rows r_t(τ) = σ((ε − |z_t − z_{t−τ}|)/T) over 10 lags plus
  2-step diagonal persistence. Literature check: partly new (closest: learnable recurrence plot, Jeong & Yoo 2026,
  doi:10.3390/electronics15091798; RepNet arXiv:2006.15418; SELFY arXiv:2102.07092); r_t(τ) is a local
  attention-like score without softmax, so its novelty is the weakest of the three.

Pre-registration (written before any round-6 result):
- *Stage 1, mechanism (P10 tasks, 3 seeds).* Predictions: RainflowNet solves `freq` (cycle counts per window), unlike the
  pure play memory; OrdinalNet solves `freq` and `step_dir` (order of samples) and degrades least when spikes are added;
  RecurrenceNet solves `freq` but is direction-blind on `loop_dir` (a recurrence plot does not change when a path is
  traversed backwards).
- *Stage 2, accuracy screening on 4 dev datasets* (utd_mhad, uci_har, unimib_shar, actrectut_gestures; seeds 0–2; each
  (dataset, seed) on the same hardware type as the P5 runs of the old models, so pairs share hardware). A candidate
  advances if its mean macro-F1 over the 4 dev datasets is at least S4D's minus 1 point. If none does, round 6 is negative.
- *Stage 3, held-out:* the 6 other P5 datasets × seeds 0–2 for the candidates that advance; final verdict on all 10
  datasets with the P5 analysis (mean rank, Friedman, Wilcoxon-Holm, non-inferiority margin 1 vs S4D).
- No variant changes inside round 6; a change becomes a new id and starts at stage 1.

Stage 1 results (P10, 3 seeds; `results/synthetic/summary.md`):

| Model | freq clean / noisy | loop_dir under speed changes (slow / fast / jerky) | step_dir, 30 % spikes |
|---|---|---|---|
| RainflowNet | **97.5** / 52.2 | **95.4 / 99.6 / 97.6** | 62.3 |
| OrdinalNet | 100 / 95.4 | 85.1 / 49.0 / 81.3 | 40.7 |
| RecurrenceNet | 99.7 / 98.6 | 54.4 / 53.3 / 53.1 | 69.4 |
| HysteresisNet (base) | 24.4 / 24.9 | 91.1 / 99.8 / 98.6 | 45.9 |
| S4D | 100 / 99.9 | 73.3 / 34.6 / 77.9 | 52.6 |
| Mamba | 99.4 / 76.8 | 89.1 / 96.1 / 92.0 | 71.2 |

- Confirmed: all three solve `freq`, which the pure play memory cannot. RainflowNet is the first model here that is both
  speed-robust on `loop_dir` (like the play memory) and frequency-sensitive (like S4D), through cycle counts; noise
  breaks its counts (52.2 with noise).
- Refuted: OrdinalNet is the *least* spike-robust (40.7), not the most. RecurrenceNet is not direction-blind on `loop_dir`
  (99.9 at the training speed; the u path and channel mixing carry the direction), but it collapses under speed changes
  (lag-locked).

Stage 2 results (4 dev datasets × seeds 0–2; `stage2_summary.py` → `results/stage2_summary.md`). Seed-1 runs that the slow,
flaky Colab T4 could not finish were completed on Kaggle T4 (same GPU type); the first copy of each result was kept.

| Model | utd_mhad | uci_har | unimib_shar | actrectut | Mean | − S4D | Advances |
|---|---|---|---|---|---|---|---|
| **RecurrenceNet** | 73.32 | **96.62** | **60.78** | 40.37 | 67.77 | −0.24 | **yes** (3/4 above S4D) |
| RainflowNet | 71.19 | 95.47 | 58.22 | 38.36 | 65.81 | −2.21 | no |
| OrdinalNet | 70.89 | 96.52 | 56.96 | 39.24 | 65.90 | −2.11 | no |
| S4D | 73.27 | 94.86 | 60.50 | 43.43 | 68.02 | — | |
| HysteresisNet sel+clock | 73.94 | 94.49 | 55.50 | 43.57 | 66.87 | −1.14 | |

Stage 3 (RecurrenceNet on opportunity, sad, hapt, dsads, wisdm, daphnet × seeds 0–2; seeds 0–1 on Kaggle T4, seed 2 and
opportunity on Colab L4, as the old models). Done 2026-10-10. All 10 datasets (`p5_summary.py recurrencenet`):

| Model | Mean rank | Mean macro-F1 | First places |
|---|---|---|---|
| **RecurrenceNet** | **2.00** | **72.01** | **5** |
| S4D | 2.50 | 71.29 | 1 |
| HysteresisNet sel+clock | 2.60 | 70.69 | 4 |
| QuantileNet | 3.80 | 69.70 | 0 |
| Mamba | 4.60 | 68.91 | 0 |
| No memory | 5.50 | 66.06 | 0 |

Friedman p = 6e-5, Nemenyi CD 2.38 (`results/p5/cd_recurrencenet.svg`). Pre-registered claims: RecurrenceNet − S4D +0.71
[−0.83, +2.26] → **non-inferior** (margin 1), not better; − Mamba +3.10 [+1.46, +4.74] → better. Held-out 6 only (no part in
the choice): − S4D +1.35 [−0.93, +3.64], 4/6 (opportunity +0.41, sad +5.61, hapt −0.25, dsads +1.11, wisdm −0.07,
daphnet +1.30); − Mamba +3.48 [+0.41, +6.56], 5/6. It is the first new family that is non-inferior to S4D across datasets,
and the first with a better mean rank. Not significantly more accurate.
RecurrenceNet robustness and early decisions on UTD-MHAD (its 24 checkpoints, 3 seeds; the others 5 seeds;
`robust_test.py`, `early_test.py`): channel offset **39.1, the best of all models** (offset-invariant differences, as
designed; S4D 27.4); early decisions at 50 % of the window 44.4 (Mamba 52.6, S4D 35.7) and at 75 % **68.9, the best**;
but Gaussian noise 44.8 (S4D 66.9), spikes 40.3 (S4D 59.2), device rate 54.4 (HysteresisNet 73.3; lag-locked) and physical
tempo 44.0. Accuracy leader, fragile to noise and to sampling-rate changes.

### 5d. Round 7 (started 2026-10-10): learned Poincaré sections and other ideas from the family review

Ideas reviewed after round 6 (chat of 2026-10-10): learned Poincaré sections, an event-driven cycle-of-cycles hierarchy,
phase-locked template oscillators, elastic templates (streaming DTW), counter-only networks, a sandpile memory, and two
shared modules (gravity-frame inputs, a robust readout). Literature checks (web, sub-agents):
- **ReturnMapNet** (`returnmapnet`, `edge_ssm/round7.py`): partly new. Neighbours: learned event functions (Chen et al.,
  ICLR 2021, arXiv:2011.03902), trainable thresholds held between events (EGRU, ICLR 2023, arXiv:2206.06178), crossing-like
  pooling (MultiRocket, arXiv:2102.00457), hand-crafted zero-crossing rates and gait Poincaré sections. New: the learned
  section with a hysteresis band plus the return-map readout (where the trajectory pierced the section) and return time,
  inside an end-to-end classifier. Renamed from "PoincaréNet" (taken by hyperbolic networks).
- **Phase-locked template bank**: partly new. Single-unit dynamics are prior art (Righetti et al. 2006; Arora & Sethares
  2007 adaptive wavetable oscillators; Petrič et al. 2011); new would be learning templates, gains and initial frequencies
  by BPTT and stacking the units. Reviewers will read it as a test-time-training layer with a periodic inner model.
  Implemented as **PhaseLockNet** (`phaselocknet`, 17.5k params): 16 oscillators per block with 3-harmonic learned templates,
  tanh-limited phase/frequency drive, hand-written reverse pass (equal to autograd to ~1e-16 in float64), TorchScript
  loops (~0.25–0.5 s per training step, the slowest family). Added to round 7 on 2026-10-10 with the same stage-2 rule;
  its stage-1 runs were already under way and are reported as they are. Stage 1: `freq` 99.4 clean / **96.6 noisy** (the locked
  frequency survives noise, unlike RainflowNet and the play memory); `loop_dir` only 77.5 at the training speed and 71–74
  under speed changes; `step_dir` 47.8 at 30 % spikes; `long_order` 99.0 / 97.2.
  Stage 2 (stopped early, 2026-10-10): on seeds 1–2 it scores utd_mhad 64.86, uci_har 93.22, unimib_shar 51.34 against S4D's
  73.02, 94.89, 60.72 on the same seeds (−8.2, −1.7, −9.4). No seed-0 result can bring the dev mean within 1 point of S4D,
  so the slow local seed-0 runs (about 45 min per UCI split on the shared 1080 Ti) were stopped. Does not advance.
- Shared modules (gravity frame, robust readout) are not families; to be tested as ablations on every family.

ReturnMapNet design: per block 16 sections s_k = a_k·x + b_k with a learned Schmitt band; at each upward crossing the
features are a soft event weight, the 2-D pierce point P_k·x, the time since the last crossing and the last return time
(relative to the window), and the relay state; all computed in parallel (cumulative maxima), 21.7k params, 13 ms/step.

Pre-registration (before any result), same stages and rule as round 6:
- Stage 1 predictions: solves `freq` with and without noise (crossing rate and return time; the band rejects small noise);
  speed-robust on `loop_dir` (pierce points do not depend on speed); solves `step_dir` (relay state) but spikes cause
  spurious crossings.
- Stage 2: the 4 dev datasets × seeds 0–2, same hardware types as the old models (T4 work on Kaggle T4); advance if the mean
  over the 4 dev datasets is at least S4D's minus 1 point.

ReturnMapNet results (negative):
- Stage 1: `freq` 55.9 ± 22.6 clean, 29.3 noisy (prediction refuted: some seeds learn the rate, others do not);
  `loop_dir` 90.7 at the training speed, 84–91 under speed changes (speed-robust but weak); `step_dir` 46.1 at 30 % spikes
  (spurious crossings, as predicted); `long_order` 99.2 / 96.6.
- Stage 2: utd_mhad 69.02, uci_har 92.68, unimib_shar 50.37, actrectut 39.92; mean 63.00, −5.02 vs S4D, 0/4 datasets above
  it, close to the memoryless control (61.71). Does not advance.

### 5e. Round 8 (started 2026-10-10): laws instead of memories

Motivation (`results/arena/compare.md` §5-6, `NEW_ARCHITECTURES.md` round-8 backlog): models differ on unseen people,
and the Random Forest's margin over deep models grows with the channel count. Check without training (2026-10-10,
30 paper datasets): the rank correlation stays when sampling rate, subjects, activities and windows are controlled
(partial Spearman 0.73, p < 1e-4; raw 0.71), and none of those four correlates with the margin by itself
(|rho| <= 0.24). Sensor placement and modality (e.g. magnetometers) cannot be separated from the channel count.
The val-test gap of compare.md §6 is not evidence: validation F1 is nearly equal across models on each dataset. Our skeleton squeezes all channels through one linear layer (133 -> 32 on Opportunity).
The user chose LawNet first, no user-memory track, RecurrenceNet and S4D as the bar, and fast screening on the
10 P5 datasets before any new held-out data.

Models (`edge_ssm/round8.py`, configs fixed in `registry.py` before any result; tests `tests/test_round8.py`):
- **LawNet** (`lawnet`, d_model 48, ~19k params on UTD-MHAD): memoryless skeleton (per-sample blocks, mean over time)
  plus a window-level "law" path joined at the head. Per 3-axis sensor: closed-form ridge operators
  x_t ~ A_tau x_{t-tau} (tau = 1, 2, 4, 8, 16) read only through their characteristic polynomial; canonical
  correlations of every sensor pair at lags -4, 0, +4 (elementary symmetric functions); covariance shape and
  mean length. Invariant to rotating or re-scaling a sensor (checked exactly in the tests). Streaming form: running
  sums and 3x3 solves once per window.
- Sensors are found by one rule for every dataset: three consecutive channels whose names differ in exactly one
  character reading x, y, z. Triples found: UTD 2, UCI 3, UniMiB 1, ActRecTut 3 (its 2-axis gyros are left out),
  Opportunity 37 (quaternions, compass out), SAD 20, HAPT 2, DSADS 15, WISDM 1, Daphnet 3. The runner passes the
  channel names to models that ask for them; WHAR Arena excluded models with dataset-specific sensor maps, and
  this rule is not one, but it does read metadata.
- Controls: `lawnet_rand` (same number of triples from a fixed random permutation: not physical sensors),
  `statnet` (per-channel mean/std/min/max, the kind of feature behind the Random Forest, instead of the laws).
- Hybrids with RecurrenceNet's blocks (d_model 32): `lawnet_rec`, `recstatnet`.

Literature check (web, sub-agent, 2026-10-10; gate 2 passed, novelty partial): every building block exists.
- Closed-form per-window operators inside networks: Koopa (NeurIPS 2023, arXiv:2305.18803, forecasting only) and
  Neural Koopman Pooling (CVPR 2023, skeleton actions, one learned matrix per class, no invariant readout).
- Dynamical-model parameters or eigenvalues as recognition features: Bissacco et al. CVPR 2001; Chaudhry et al.
  CVPR 2009; Turaga et al. TPAMI 2011; Hankelets (CVPR 2012, view-invariant dynamics); Gersch 1977 (ARMA eigenvalues
  for EEG); raw VAR coefficients for wrist HAR (CBIC 2017).
- Rotation-invariant IMU features: Kobayashi et al. ICASSP 2011 (Gram matrices); Yurtman & Barshan 2017; TRI-HAR
  (arXiv:2608.15621, invariance to independent rotation per IMU, equivariant backbone, 13 MiB).
- Also covariance (SPD) descriptors for gait, and a differentiable CCA layer (Dorfer et al. 2018).
- Not found: characteristic-polynomial invariants of per-window multi-lag operators plus symmetric functions of
  lagged inter-sensor canonical correlations as end-to-end features for wearable HAR at MCU size. The defensible
  claim is a cheap, provably invariant parameterization, not a new principle.

Data: the 4 cached datasets from `HAR\datasets_paper_era` as before; the 6 others (Opportunity, SAD, HAPT, DSADS,
WISDM, Daphnet) from `datasets_paper_era_local/`, built by `prepare_local_datasets.py` (hard links to HAR's raw
files, paper-era parsing and windowing, HAR manifest unchanged before/after). Accepted only if every split's
train/val/test sizes equal the P5 runs, which the paper-era code built on the cloud from a fresh download.
Result (2026-10-10): SAD, Opportunity, HAPT, DSADS and WISDM match on every split, and HAR's files were unchanged.
**Daphnet does not**: the same 10 split sizes appear in another order, because the paper-era parser numbers
subjects in `os.listdir` order (`cfg_daphnet.py`: unsorted listdir, then `factorize`), which differs between
Windows (alphabetical) and the Linux VMs. The fold order also changes the validation draws. The local Daphnet cache
was renamed to `_daphnet_split_order_differs` so `run_whar.py` cannot pick it up; Daphnet runs stay on the cloud.
**DSADS does not either** (found 2026-10-10 by `error_analysis.py`): its split sizes match only because every subject
has the same number of windows. `cfg_dsads.py` numbers activities and subjects by unsorted `os.listdir` order (then
`factorize`), so the labels differ between Windows and Linux. Cloud-trained checkpoints score 3.5 % on the local
cache. No local DSADS run was made. The cache was renamed to `_dsads_order_differs`. SAD, HAPT, WISDM and Opportunity
re-score the cloud checkpoints to the stored F1 (to GPU rounding), so those four local caches are equivalent.
The size check alone is not enough: re-scoring a cloud checkpoint is the test to use.

Pre-registration (written before any round-8 result):
- *Stage A, screen:* the 5 models, seed 0, all splits of the 10 dev datasets, local GTX 1080 Ti (`EDGE_SSM_FAST=1`).
  The stored seed-0 results are the reference: RecurrenceNet 72.25, S4D 71.25, no-memory control 65.63 (mean of 10).
  On UTD, UCI, UniMiB and ActRecTut the references also ran on the 1080 Ti; on the other 6 they ran on T4/L4, an
  extra source of run-to-run noise (about 1 point per cell, compare.md §0).
- *Advance to stage B* (seeds 1-2): a model whose seed-0 mean over the 10 is at least RecurrenceNet's minus 1 point
  (71.25), plus the controls needed to interpret it. To save time, seeds 1-2 of the two hybrids on the 5 local dev
  datasets started while the cloud part of stage A was still running (`cloud/jobs/local_r8b.json`); they count only
  for a model that advances by the seed-0 rule.
- *Predictions:* (1) `lawnet` > `statnet` (laws beat plain statistics at equal skeleton); (2) `lawnet` > `lawnet_rand`
  on the 8 datasets with at least 2 physical triples; (3) `lawnet_rec` − `recurrencenet` is larger on the three
  datasets with >= 45 channels (SAD, DSADS, Opportunity) than on the other seven.
- *Held-out (later, needs no new approval beyond the user's 2026-10-10 go-ahead):* a candidate whose 3-seed dev mean
  is at least RecurrenceNet's (72.01) goes to new datasets not yet run with our models (PAMAP2, UP-Fall, UMA-Fall,
  GOTOV first), against RecurrenceNet and S4D.
- No variant changes inside round 8; a change becomes a new id.

**Outcome: stopped by the user after 5 of 10 dev datasets (seed 0; `r8_summary.py`).** UTD / ActRecTut / Opportunity /
UniMiB / UCI, mean of 5:

| Model | UTD | ActRecTut | Opp. | UniMiB | UCI | Mean 5 | Mean without ActRecTut | Splits won vs RecurrenceNet |
|---|---|---|---|---|---|---|---|---|
| LawNet | 70.90 | 44.51 | 59.33 | 58.40 | 94.94 | 65.62 | 70.89 | 8/34 |
| LawNet, random sensors | 68.46 | 45.49 | 63.11 | 58.40 | 95.61 | 66.22 | 71.40 | 8/34 |
| Channel statistics | 69.61 | 47.18 | 62.45 | 53.95 | 92.14 | 65.07 | 69.54 | 5/34 |
| LawNet + recurrence | 73.75 | 43.77 | 62.02 | 63.67 | 96.40 | 67.92 | 73.96 | 15/34 |
| Channel statistics + recurrence | 71.55 | 63.64 | 62.84 | 62.05 | 96.42 | 71.30 | 73.22 | 13/34 |
| RecurrenceNet | 74.59 | 35.08 | 64.91 | 61.96 | 96.74 | 66.65 | 74.55 | — |
| S4D | 73.79 | 43.76 | 60.99 | 60.06 | 94.80 | 66.68 | 72.41 | 8/34 |
| No memory | 67.61 | 37.89 | 57.80 | 49.58 | 92.74 | 61.12 | 66.93 | — |

- LawNet adds +4.5 over the no-memory control on every dataset but trails RecurrenceNet by 3.7 without ActRecTut.
- Prediction 2 failed so far: physical sensors beat random triples on 1 of 4 datasets (Opportunity −3.8). UniMiB has one
  sensor, and both give 58.40 exactly, as the invariance implies. Likely cause: rotation invariance discards the
  gravity direction (posture), which Opportunity's locomotion labels need.
- Prediction 3 failed so far: LawNet + recurrence trails RecurrenceNet most on Opportunity (−2.9).
- The one strong signal is window statistics fused at the head on ActRecTut (+28.6 for the statistics hybrid, above
  the paper's best 61.5). It rests on 2 subjects at one seed; without ActRecTut both hybrids trail RecurrenceNet.
- On Opportunity the side paths raise the size to 42-49k parameters (RecurrenceNet 28k).
- The 5 cloud datasets ran on to completion on Kaggle (no CLI cancel). Stage B was stopped.

**Correction (2026-10-11): with all 10 dev datasets (seed 0, Kaggle results merged), round 8 reads differently.**

| Model | Mean of 10 | − RecurrenceNet | SAD | DSADS | WISDM | Daphnet | Advances by the rule |
|---|---|---|---|---|---|---|---|
| Channel statistics + recurrence | **74.69** | **+2.44** (5/5) | 96.62 | 86.33 | 82.23 | 41.24 | yes |
| LawNet + recurrence | 73.77 | +1.52 (5/5) | **99.35** | **88.53** | 83.10 | 45.86 | yes |
| LawNet, random sensors | 72.14 | −0.12 | 98.69 | 87.55 | 77.87 | 44.27 | yes |
| LawNet | 71.73 | −0.52 (3/7) | 99.04 | 87.97 | 77.87 | 43.15 | yes |
| Channel statistics | 69.64 | −2.61 | 93.05 | 83.88 | 71.60 | 42.03 | no |
| RecurrenceNet | 72.25 | — | 91.97 | 86.24 | 84.30 | 43.53 | |

- By the pre-registered rule (at least RecurrenceNet − 1), four of the five would have advanced; the stop after 5 of 10
  datasets came too early. The local 5 were the least favourable.
- LawNet reaches 99.0 on SAD, above the paper's Random Forest (98.2).
- Predictions:
  - (1) laws > plain statistics: +2.09 (8/2, p = 0.084), supported.
  - (2) physical > random sensors: −0.51 (3/5), refuted.
  - (3) gain over RecurrenceNet larger at >= 45 channels: +2.26 vs +1.20, in the predicted direction but weak.
- Lesson for the early-stop rule: stop on the full dev set, or at least on a kill set that mixes local and cloud datasets.
  The 5 local datasets were systematically the hardest for window-level side paths.

### 5f. Round 9 (started 2026-10-10): models that scale with the number of sensors

Same target as round 8 (our skeleton squeezes all channels through one linear layer; the Random Forest wins as
sensors are added), without invariance (round 8 suggests it discards posture). `edge_ssm/round9.py`, configs fixed
in `registry.py` before any result, tests `tests/test_round9.py`:
- **ParliamentNet** (`parliamentnet`, d 48, ~21.6k params on UTD-MHAD, about the same on 60 or 133 channels): every
  3-axis sensor (round 8's name rule; leftover channels packed three by three) is an agent. All agents share one network:
  patch tokens of their own 3 channels (5 samples), a learned agent embedding, 2 RecurrenceNet blocks. After the window
  each agent posts 4 bits (sign, straight-through) and reads the mean of the others' bits, then casts class scores and a
  confidence; output = confidence-weighted vote. Parameters do not grow with the sensor count. Faster than
  RecurrenceNet per step (26-30 vs 36-46 ms on the 1080 Ti).
- **ForestNet** (`forestnet`, ~29k on UTD-MHAD, ~46k on Opportunity): RecurrenceNet trunk + batch-normalized
  per-channel mean/std/min/max -> 8 soft oblivious trees of depth 4 (one softly selected feature per split, as a
  Random Forest splits). Control: `recstatnet` (same features, linear head).

Literature check for ParliamentNet (web, sub-agent, 2026-10-10): a composition of known parts, no new mechanism.
- Weight-tied per-channel/per-sensor encoders: PatchTST, TinyHAR, DeepSense, UniMTS.
- "Read the mean of the other agents" is CommNet (NIPS 2016, Eq. 2); binary learned messages are DIAL, Havrylov & Titov
  2017, Gupta et al. 2020.
- Few-bit per-sensor messages for body-worn HAR under a bit budget: Hussien et al. 2022 (arXiv:2203.09631, WARD dataset,
  star topology to a fusion centre).
- Per-node classifiers with confidence-weighted votes and missing nodes: Origin (DATE 2021), DDNN (ICDCS 2017), Yang et al.
  2009.
- Not found: peer-to-peer learned bits between body nodes with a per-agent vote and no fusion centre. Risk noted: plain
  straight-through bits sometimes fail to train (Vanneste et al. 2022); ST-Gumbel or ST-DRU are the fallbacks.

Fast screen with early stop (user's rule, §1). Seed 0, local GTX 1080 Ti:
- *Kill set:* UTD-MHAD (6 channels), SAD (60), Opportunity (133). RecurrenceNet's seed-0 mean there is 77.16 (74.59, 91.97,
  64.91). A model whose mean is below 76.16 stops.
- A survivor runs the other 7 dev datasets at seed 0 and stops if its 10-dataset mean falls below RecurrenceNet's minus 1
  (71.25), then seeds 1-2, then new held-out datasets.

ParliamentNet on the kill set (seed 0): UTD 58.60, SAD **96.87**, Opportunity **70.67**; mean 75.38 < 76.16, so it **stops**.
It is still the strongest signal so far, and in the predicted direction: +4.9 on SAD (above every deep model in the paper,
CNN-HAR 95.6; RF 98.2) and +5.8 on Opportunity (paper best MLP-HAR 74.7), but −16.0 on UTD-MHAD, where the "parliament"
is two agents (accelerometer and gyroscope of one board).

ForestNet **stopped as an implementation failure**, not a test of the idea: UTD-MHAD 4.37 (chance level), validation F1
0-9 % on every split. The leaf scores are averaged over 8 trees and weighted by leaf probabilities (~1/16), so the logits
start near 0.01 and do not grow within 100 epochs. Its SAD and Opportunity runs were killed. A fixed version (leaf scale)
would be a new id; not pursued for now (`recstatnet` already covers "window statistics at the head").

**Round 9b** (`parliamentbody`, written after that screen, before any result of its own): the same parliament (d 32) plus a
whole-body agent, a RecurrenceNet over all channels at full rate (3 blocks, width 24), which posts, listens and votes like
the sensor agents. ~29-30k params on UTD-MHAD (RecurrenceNet 24.7k), ~33k on Opportunity. Because UTD, SAD and Opportunity
shaped this variant, they are not a clean test of it: it runs the same kill set (mean >= 76.16) only as a stop rule, and the
other 7 dev datasets decide.
Result: UTD 60.05, SAD 94.93, Opportunity 65.48 (mean 73.49): **stopped**. The whole-body agent did not rescue UTD-MHAD
and cost 2-5 points on the many-sensor datasets. Both parliaments are compositions of known parts (literature check
above), so the user's 2026-10-10 novelty rule retires the line; the per-sensor effect stays a lesson for later designs.

### 5g. Round 10 (started 2026-10-10): bold, literature-checked memories

User (2026-10-10): candidates must be completely new, with no overlap with earlier ones, bold and creative; train on the
PC and Colab together. Four ideas were checked on the web by sub-agents before any code:
- **EcoNet**: partly new, kept. Lotka-Volterra neural dynamics (Fukai & Tanaka 1997; winnerless competition, Rabinovich 2001)
  and untrained ecological reservoirs (Ushio et al. 2023) exist; a trained, input-driven Lotka-Volterra cell for sequence
  classification was not found.
- **RaceNet**: dropped as prior art. A race to a threshold readout is in first-spike SNNs (Liu et al. 2023); evidence
  accumulation to a threshold for early classification is in SPRT-TANDEM (ICLR 2021) and CBGT-Net (2024); learned
  boundaries are in FIRMBOUND (ICLR 2025).
- **GrammarNet**: partly new, kept. Grammars for activities exist (Ivanov & Bobick 2000), as do differentiable CKY for text
  classification (Maillard et al. 2017) and differentiable regular grammars for video (Piergiovanni et al. 2020). A
  context-free grammar trained end to end on raw IMU windows was not found.
- **BraidNet** (braid-group memory of crossing order): partly new, kept in reserve. Its learned-unitary form collapses to
  path development (Lou et al., TMLR 2024).

Models (`edge_ssm/round10.py`, configs fixed in `registry.py` before any result, tests `tests/test_round10.py`):
- **EcoNet** (`econet`, 18.4k params on UTD-MHAD). Shared skeleton; each block's memory is 4 communities of 8 species,
  updated by log n += 0.2 (r(u) − A n), r = 2 tanh(W u + b). A has softplus self-limitation on its diagonal and learned
  competition/facilitation off it. The community advances once per 3 samples. Readout: log-abundances, relative
  abundances, Shannon diversity.
  - The hand-written reverse pass equals autograd to 1e-15.
  - A test shows the memory is order-dependent: the same final input leaves different winners.
- **GrammarNet** (`grammarnet`, 17.8k on UTD-MHAD). A per-sample MLP embeds the signal, and 5-sample patches become tokens.
  16 nonterminals score each token, and a shared binary PCFG (learned rule probabilities) parses the window with the
  inside algorithm, checked against brute-force enumeration. The head reads the root posterior, the mean constituent
  posteriors at widths 2/4/8, and log Z per token.

- **Braid-holonomy net** (`bhnet`, 20.2k on UTD-MHAD; "BraidNet" is taken, arXiv 2104.10010). Each block's memory, per head:
  - 4 learned points in the plane ("strands") and the projective holonomy of a flat Knizhnik-Zamolodchikov-type connection,
    Psi_t = exp(kappa Σ dlog(z_i − z_j) τ_ij) Psi_{t−1}, with τ_ij the transposition matrices.
  - Transpositions satisfy the infinitesimal pure braid relations, so the holonomy depends only on how the strands wound
    around each other. A test checks this: homotopic paths give the same holonomy, the other side of a strand another one.
  - Computed by a normalized parallel prefix product. Readout: traces of Psi, Psi², Psi³ and the abelian (winding) part.
  - Literature check: partly new. No flat connection on a configuration space in an ML model was found; formally a special
    case of path development (Lou et al. 2024) restricted to a flat (curvature-free) family of generators.

Fast screen, same kill set and rule as round 9: UTD-MHAD, SAD, Opportunity at seed 0; stop if the mean is below 76.16
(RecurrenceNet 77.16 − 1). PC: UTD-MHAD and Opportunity (`cloud/jobs/local_r10a.json`, `local_r10b.json`); Colab: SAD
(`cloud/jobs/colab_r10a.json`, `colab_r10b.json`, one VM each).

**Outcome (seed 0): all three stopped early.**

| Model | UTD-MHAD | SAD | Opportunity | Verdict |
|---|---|---|---|---|
| EcoNet | 66.77 (6 of 8 splits) | 84.13 | 61.84 | mean ~70.9 < 76.16: stopped |
| GrammarNet | 61.34 (3 of 8) | 81.45 | 59.51 | cannot reach 76.16: stopped |
| Braid-holonomy net | 65.00 | stopped on Colab | 73.66 on splits 0-1 (RecurrenceNet 76.6 there) | stopped |
| RecurrenceNet | 74.59 | 91.97 | 64.91 | bar |
| No memory | 67.61 | 80.80 | 57.80 | |

All three land between the no-memory control and RecurrenceNet; GrammarNet and the braid holonomy are at or below the
control on UTD-MHAD. One Colab upload hung for 15 min and the supervisor retried on a new VM; the stopped VM was
terminated by hand (`colab stop`), and `colab sessions` showed none left.

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

### 5j. Round 12 (started 2026-10-11): a family defined by structure, computing in intrinsic time

User (2026-10-11): plan and run the search for a new architecture family; it must be a real breakthrough, innovative
and unusual, aimed at a Nature-family journal. Known layers are allowed; the structure must be new (§1).

Why this direction (from the evidence of rounds 6-11, not from test scores of the new model):
- The largest addressable errors are the gait type (WISDM +13.8, SAD +6.0) and person-specific style on gestures (§5h).
  Both are about time: the shape of one gait cycle, and gestures performed at different tempi.
- Tempo-free matching wins on gestures (1-NN DTW + standardization: ActRecTut 69.0 vs RecurrenceNet 40.4), and
  rate independence paid off where it was exact (HysteresisNet, device rate +21 over S4D), but the play memory threw
  away shape.
- Every earlier family kept one skeleton in sample time and changed only the memory cell.

**IsochronNet** (`edge_ssm/round12.py`, tests `tests/test_round12.py`, configs fixed in `registry.py` before any result).
The network computes in the movement's intrinsic time:
- K = 6 oscillators entrained by the input. Each is a learned channel projection with a learned FIR filter (Hann
  low-pass at init); the analytic signal (FFT Hilbert transform, reflect-padded) gives a unit phasor and an amplitude.
- Phase-indexed ring memory: per-sample features are written into 16 slots addressed by the phase (soft von Mises
  binning from the phasor). The writes are weighted by relative amplitude and by the phase step, so the slots integrate
  over phase, not samples. This is a differentiable epoch folding. Slot occupancy (time spent per phase) is a separate
  channel and keeps the timing.
- Progress line: an arc-length clock over a learned embedding addresses 16 slots, each sample weighted by its
  arc-length step, so the line is unchanged by monotone time warps (continuous limit; test with 2000 samples < 0.02).
- Readouts: circular convolution over the slots (mean and std over slots), a convolution over the line, amplitude-weighted
  relative phases of oscillator pairs at 1:1 and 1:2 locking, a symmetry spectrum (ring vs a learned channel mirror of
  itself shifted by 1/4 and 1/2 cycle; gait glide symmetry, Golubitsky et al. 1999), and clock statistics.
- A tonic path in sample time covers posture and static activities, and adds the per-channel window statistics
  (round 8's lever).
  - `isochron`: the tonic path is the per-sample encoder only, so all temporal structure comes from intrinsic time
    (16.4k params on UTD-MHAD).
  - `isochron_rec`: recstatnet's trunk (RecurrenceNet blocks + statistics) plus the ring, filled from the trunk's
    last hidden states (36.1k on UTD-MHAD; recstatnet 26.5k).
- Controls: `isochron_uniform` and `isochron_rec_uniform`. They have the same parameters, but the phases are replaced by
  sample time (1-4 fixed cycles per window, unit amplitude) and the progress clock by t/L, so ring and line become plain
  time bins.
- Tests (all pass): Hilbert phase of sinusoids; the ring is the same at two tempi; the line is the same under three
  monotone warps through the model's own clock; the uniform control has equal parameters and input-independent phases.
  - Limit found by a test: on one channel a speed change inside a cycle cannot be told apart from a different waveform,
    so the ring undoes tempo changes between cycles, not within one.

Literature checks (web, three sub-agents, 2026-10-11):
- **IsochronNet: partly new.**
  - Every part has a predecessor:
    - arc-length clock: TWINN, NIPS 1992; TWIESN 2006 ("intrinsic time warping invariance");
    - learned analytic phase and differentiable PLV: EEGminer 2024, GREEN 2025 (EEG);
    - phase as an input channel: PhASER (TMLR 2025);
    - folding by a fixed period in sample time: TimesNet, CycleNet (ring of slots indexed by t mod W), PhaseFormer;
    - phase-indexed weights with an external phase: PFNN 2017;
    - learned motion phases for synthesis: DeepPhase 2022;
    - learned warping to canonical time: TTN 2019, DTAN 2019;
    - gait symmetry groups: Golubitsky 1999; frieze groups for gait from video: Liu et al. 2002.
  - Not found: a slot memory addressed by a phase that the network extracts from the input itself and that varies
    within the window. Its writes are soft and amplitude-weighted (differentiable epoch folding), and it adds an
    occupancy (duty-factor) readout. Also not found: learned symmetry scores and 1:1/1:2 locking features in a
    wearable-HAR network.
  - Reviewer risks:
    - DFT pooling over slots reduces to lock-in demodulation (so the readout here is a circular convolution);
    - the FIR band works in sample time;
    - static postures add arc length through noise;
    - the biological framing is loose. The direct precedents are temporal scaling (Wang et al., Nat Neurosci 2018)
      and time-warp-invariant neurons (Gütig & Sompolinsky 2009).
  - Name: PhaseRing / RingNet / PhaseNet are taken; IsochronNet is free in web search.
- **Dendritic plateau neuron network (atlas proposal A): dropped on novelty**, before any training.
  - The unit is a combination of known parts:
    - DendSN (arXiv 2412.06355): leaky compartments plus a Mexican-hat nonlinearity;
    - ADA (Applied Intelligence 2023): the dCaAP bump;
    - Tugsbayar et al. (eLife 2026): nearly the same soma rule, feedforward × (1 + ReLU(apical)).
  - The apical gate is an SE/FiLM scale. The biology also mixes two cell types: dCaAPs were recorded in apical
    dendrites and act as anti-coincidence detectors, while BAC firing is L5 coincidence.
  - The code stays: `dendriticnet`, `dendritic_relu`, `dendritic_nogate`.
- **Reference-frame columns (Thousand Brains for time series): partly new, not built yet.**
  - Monty (arXiv 2507.04494) does location-consistent voting in 3-D space.
  - The Thousand Brains Project docs propose voting on a behaviour's "state" (unpublished, not trained).
  - Coupled HMMs (1997) link limb chains per class.
  - It needs several IMUs (it is trivial on UTD-MHAD's single board). Kept as round 12b: per-sensor isochron clocks
    with a class-specific relative-phase consistency vote, only if IsochronNet's phase machinery works.

**Pre-registration (written 2026-10-11, before any IsochronNet result).**
- *Kill set, seed 0, local GTX 1080 Ti, `EDGE_SSM_FAST=1`:*
  - datasets: UTD-MHAD, ActRecTut, SAD, Opportunity, WISDM. WISDM is added because the gait-type headroom is largest
    there; all five local caches re-score the cloud checkpoints (§5e).
  - references at seed 0:
    - recstatnet: 71.55, 63.64, 96.62, 62.84, 82.23, mean **75.38**;
    - RecurrenceNet: 74.59, 35.08, 91.97, 64.91, 84.30, mean 70.17;
    - statnet (memoryless trunk + statistics, i.e. `isochron` without its phasic path): 69.61, 47.18, 93.05, 62.45,
      71.60, mean 68.78.
  - SAD and WISDM references ran on T4 (about 1 point of hardware noise per cell).
- *Rules:*
  - `isochron` passes if its kill-set mean is at least **74.38** (recstatnet − 1): intrinsic time replaces sample-time
    memory.
  - `isochron_rec` passes if its mean is at least **76.38** (recstatnet + 1): it adds to the best model.
  - Early stop: a candidate more than 5 points below recstatnet on the mean of UTD-MHAD, ActRecTut and Opportunity
    (fast, run first) stops before SAD and WISDM.
  - The uniform controls run on UTD-MHAD, ActRecTut and Opportunity alongside, and on the whole kill set only for a
    candidate that passes.
  - A candidate that passes then runs:
    1. the other dev datasets at seed 0 (UCI-HAR, UniMiB, HAPT on the PC; DSADS, Daphnet on the cloud);
    2. seeds 1-2;
    3. new held-out datasets (PAMAP2, UP-Fall, UMA-Fall, GOTOV; §5e).
- *Predictions:*
  1. The gains are largest on gait data: `isochron_rec` − recstatnet > 0 on WISDM and SAD, largest on WISDM.
  2. Intrinsic time, not time binning, carries the gain: candidate − uniform control ≥ +1 on the mean of the datasets
     both ran.
  3. The phasic path adds temporal information: `isochron` − statnet ≥ +3 on UTD-MHAD.
  4. Later, from checkpoints (`robust_test.py`): under device-rate changes (25-100 Hz) `isochron` drops less than
     RecurrenceNet.
- No variant changes inside round 12; a change becomes a new id.

**Outcome (seed 0, PC; `r12_summary.py` -> `results/r12/killset.md`): both candidates stop.**

| Model | UTD-MHAD | ActRecTut | SAD | Opportunity | WISDM | Mean | Bar |
|---|---|---|---|---|---|---|---|
| `isochron` | 69.48 | 54.68 | 95.36 | 60.53 | 83.48 | 72.71 | 74.38 |
| `isochron_rec` | 73.58 | 53.78 | 96.48 | 57.13 | 84.60 | 73.12 | 76.38 |
| `isochron_uniform` (control) | 70.11 | 58.90 | — | 58.83 | 78.33 | | |
| `isochron_rec_uniform` (control) | 71.93 | 52.51 | — | 60.64 | — | | |
| recstatnet | 71.55 | 63.64 | 96.62 | 62.84 | 82.23 | 75.38 | |
| statnet | 69.61 | 47.18 | 93.05 | 62.45 | 71.60 | 68.78 | |
| RecurrenceNet | 74.59 | 35.08 | 91.97 | 64.91 | 84.30 | 70.17 | |

- Several jobs died of RAM exhaustion around 11:51-11:55: SAD halves and Opportunity jobs, and a `robust_test.py` run was
  launched alongside 5 training processes.
  - The missing splits were completed with 3 processes (`cloud/jobs/local_r12b.json`). The table above is complete for
    both candidates: −2.67 and −2.26 against recstatnet.
- Predictions:
  1. Gains on gait data:
     - `isochron_rec` − recstatnet on WISDM: +2.38 (5/5, p = 0.92), not significant.
     - The phasic path alone carries the gait information: `isochron` − statnet on WISDM **+11.88 (10/0, p = 0.002)**,
       level with RecurrenceNet (−0.82). This is the round's one strong signal.
     - Attribution (post hoc, `isochron_uniform` on WISDM): time bins alone give +6.73 over statnet (8/2, p = 0.027);
       the entrained phase adds **+5.15** on top (8/2, p = 0.049). So about half of the gait gain is due to intrinsic
       time. This is one dataset, one seed, and a borderline p.
     - SAD: `isochron_rec` − recstatnet −0.14.
  2. Intrinsic time vs time bins, refuted on the mean:
     - `isochron` − `isochron_uniform`: +0.50 over 4 datasets (UTD −0.63, ActRecTut −4.23, Opportunity +1.70,
       WISDM +5.15).
     - `isochron_rec` − `isochron_rec_uniform`: −0.20 over 3 (UTD +1.66, ActRecTut +1.28, Opportunity −3.52).
  3. Refuted: `isochron` − statnet on UTD-MHAD is −0.13, so the phasic path adds nothing on gestures.
  4. Device rate (UTD-MHAD checkpoints, seed 0, `robust_test.py`):
     - `isochron` keeps 69.4 at 25-100 Hz vs clean 69.5 (RecurrenceNet 55.3): met.
     - But statnet (memoryless) is equally rate-free (69.0), and so is the uniform control (70.0).
- Robustness on UTD-MHAD (seed 0, 8 splits; clean / physical tempo / Gaussian noise / sensor offset):

  | Model | Clean | Physical tempo | Gaussian noise | Sensor offset |
  |---|---|---|---|---|
  | `isochron` | 69.5 | **58.7** | **69.1** | **47.6** |
  | `isochron_uniform` | 70.1 | 56.0 | 68.6 | 35.8 |
  | statnet | 69.6 | 43.4 | 61.8 | 27.8 |
  | recstatnet | 71.6 | 42.6 | 54.8 | 37.8 |
  | HysteresisNet sel+clock | 73.4 | 52.4 | 54.4 | 35.2 |

  - The tempo and noise robustness comes from the structure without sample-time memory plus coarse time bins: the
    uniform control has it too (tempo −2.7, p = 0.25; noise −0.5).
  - Only the offset robustness is due to the intrinsic phase: +11.8 vs the uniform control (8/0, p = 0.008). A per-window
    mean removal would give the same, so it is not a unique capability.
- Reading:
  - Intrinsic-time folding does not beat sample-time memory on accuracy. It does not help gestures.
  - On gait it supplies the temporal information a memoryless network lacks (WISDM +11.9), and about half of it needs
    the entrained phase (+5.15, p = 0.049).
  - The phase helps where the movement is periodic and hurts or does nothing on gestures. Round 12b (per-sensor clocks with
    a relative-phase vote) would target exactly the periodic, multi-IMU case. Not started: needs the user's decision.
- Lesson: never start extra processes (e.g. `robust_test.py`) while 5 training jobs run; 16 GB RAM is the limit.

### 5k. Direction 3 (started 2026-10-11): sampling-schedule-agnostic inference, a capability paper

User (2026-10-11): after round 12, aim at a Nature-family journal with a capability the old families lack, not another
accuracy point. The user chose the claim, *sampling-schedule-agnostic inference*: one trained model that keeps its
accuracy at any sampling rate, with dropped, jittered or irregular samples, and under event-driven (send-on-delta)
sampling, so on-device compute scales with motion. Hardware: NUCLEO-L4R5ZI, no IMU, no power meter.

**Basis.** HysteresisNet's play operator y_t = clamp(y_{t−1}, u_t − r, u_t + r) is rate-independent (classical:
Krasnosel'skii-Pokrovskii, Visintin, Brokate-Sprekels). The clock channel keeps physical time inside the window, so the
model is invariant to the sampling *schedule* but still sees tempo (walking vs jogging). UTD-MHAD (P2, 5 seeds):
- 73.1-73.5 at 25-100 Hz (clean 73.2); 72.0 with half the samples dropped;
- S4D 43.9 at 25 Hz.
- P5 accuracy: 70.69 over the 10 dev datasets, S4D 71.29.

**Literature check (web, sub-agent, 2026-10-11): partly new.**
- The principle is prior art:
  - TWINN 1992 (trajectory-length normalization, variable rates);
  - path signatures / Logsig-RNN (reparametrization-invariant, tested with dropped frames);
  - Wei & Sun 2000 (rate-independent memory cell trained by backprop);
  - Frydrych 2026 (arXiv 2605.23603 and two follow-ups): Preisach-relay sequence layer, theory only, rate independence
    as "the separating property". It announces companion work with a learned version, **the main scoop risk**.
- Play/Preisach neurons exist only for regressing actuator or magnetic hysteresis.
- Not found: a deep, end-to-end trained stack of play operators as a sequence classifier on real sensor data, with a
  clock channel, measured schedule robustness against SSMs, and event-driven MCU execution.
- Theoretical differentiator to formalize: the play operator is 1-Lipschitz in the sup norm. So a stacked bound gives an
  error that depends only on the sup-norm reconstruction error of the sampled signal, never on timing. Signatures and
  NCDEs are continuous only in p-variation.
- Wording: say "sampling-schedule-agnostic" or "no fixed sample clock", not "clock-free" (the model uses timestamps).
  Do not claim "first rate-independent network" or "energy proportional to motion" without a measurement.
- Baselines a reviewer will demand:
  1. **S4D/S5/Mamba with Δ rescaled to the true timestamps.** The S4 family handles a new rate this way, so "S4D falls to
     44 at 25 Hz" is a strawman until this is run.
  2. Log-NCDE / log-signature models with a time channel at equal size.
  3. CNN/S4D trained with multi-rate + drop augmentation.
  4. CfC/LTC (Δt-aware).
  5. For energy: delta-GRU/EGRU on the same events; duty-cycled and wake-on-motion pipelines.
- Evidence a Nature-family venue expects, and where the project stands:
  - several datasets incl. real multi-device data (HHAR) and the irregular PAMAP2 benchmark (PAM, Raindrop); needs the
    user's approval;
  - real event-driven hardware data: **missing (no IMU)**;
  - measured system power: **missing (no power meter)**;
  - a theorem with proof;
  - tuned baselines, multiple seeds, paired statistics;
  - ablations: no clock, fixed vs input-dependent r, pooling.

**Plan, phase 1 (no new hardware).**
1. *Decisive check first, no training:* on existing checkpoints, rate changes ×0.5-×2 with S4D and Mamba run with Δ
   rescaled to the new rate, against HysteresisNet sel+clock. If Δ-rescaled S4D is as rate-robust, the claim narrows to
   irregular and event-driven sampling plus "no rate knowledge needed".
2. Schedule robustness on all dev datasets from checkpoints (3 seeds):
   - relative rates;
   - drop 10-50 %;
   - jitter;
   - send-on-delta events at several thresholds.
   Baselines get the realistic pipeline (interpolation back to their grid; Δ rescale where it exists). HysteresisNet
   gets the raw irregular samples with timestamp clock and time-weighted pooling. Report accuracy against the fraction
   of samples kept.
3. The theorem: invariance class (re-timing exact; resampling and drops bounded by sup-norm reconstruction error) with
   proof and a numerical check.
4. Accuracy within the principle: HysteresisNet + schedule-invariant per-channel window statistics (time-weighted
   mean/std, min/max; the round-11b lever). Pre-registered screen against recstatnet.
5. Trained baselines: multi-rate augmentation (S4D, recstatnet); a log-signature model; CfC if time allows.

**Phase 2 (NUCLEO-L4R5ZI, hardware-in-the-loop).**
- C kernels for event-driven HysteresisNet inference.
- Test windows and event streams replayed over UART.
- On-device accuracy checked equal to the PC.
- DWT cycles per event and per window, against CNN-HAR / TinyHAR / S4D via stedgeai at a fixed rate.
- Energy reported as cycles × datasheet current and labelled as an estimate.

**Hardware gap for the paper.** A power profiler (e.g. Nordic PPK2 or STLINK-V3PWR) and an IMU shield (X-NUCLEO-IKS4A1)
would close the two missing items. Both are inexpensive; the user decides.

**Step 1 result (2026-10-11): the decisive check is negative on UTD-MHAD.** Setup: `schedule_test.py`, checkpoints
only, seed 0, 8 splits; tests `tests/test_schedule.py`. The zero-order-hold form of the native pipeline equals
event-driven processing to 1e-10.
- HysteresisNet stays at 73-74 under every schedule.
- The baselines do too once they get the standard pipeline:
  - interpolation back to their grid, or Δ rescaled (S4D);
  - S4D with Δ halved at ×0.5 rate: 73.4 (clean 73.8);
  - S4D with 50 % of samples dropped, then interpolated: 73.3;
  - S4D on send-on-delta streams interpolated to the grid: 74.0 with 33 % of the samples, 70.5 with 17 %.
    HysteresisNet's native event-driven pipeline gives 72.9 and 63.8.
- The earlier "S4D falls to 44" holds only for the naive pipeline (resampled data fed as if at the training rate).
- RecurrenceNet and recstatnet recover the same way with interpolation. Mamba's Δ rescale helps less, because its
  4-tap conv stays in samples.
- What is left of the claim: HysteresisNet needs no rate or timestamp knowledge, and its event-driven compute scales
  with the number of events. But S4D at a halved fixed rate with Δ rescaled gets 73.4 at half the compute, so there is no
  clear efficiency edge either.
- **WISDM (20 Hz) and SAD (50 Hz), seed 0, same check: the negative result holds** (`results/schedule/summary_*.md`).
  - SAD: every model is flat under every schedule with the standard pipeline. HysteresisNet's clean accuracy is far
    lower (87.1 vs recstatnet 96.6).
  - WISDM at 10 Hz (×0.5): all models lose information, and the baselines lose less: S4D 78.0, RecurrenceNet 78.6,
    HysteresisNet native 72.4.
  - One narrow exception: heavy random sample loss at a low native rate. On WISDM, drop 50 %: HysteresisNet 72.4 vs the
    best baseline 67.0 (recstatnet); drop 70 %: 61.3 vs 56.4. Interpolation cannot rebuild a 20 Hz gait cycle from half
    its samples, while the play memory needs only the extremes that survive. One dataset, one seed; drop-augmented
    training or a better reconstruction may close it.
- **Verdict on the direction-3 claim:** sampling-schedule agnosticism is not a capability the old families lack. With
  timestamps and the standard pipeline, S4D, RecurrenceNet and recstatnet keep their accuracy across rates, jitter,
  moderate drops and send-on-delta streams. The remaining advantages (no rate knowledge needed; graceful under severe
  loss at low rates) are too narrow for a Nature-family claim. Stopped here; the user decides the next direction.

### 5l. Round 13 (2026-10-11): a family built on the round-11b statistics mechanism (evaluation, no deep training)

Separate from §5k (another session's "direction 3"). User, in the round-11b session: propose a new architecture family
from the mechanism found in §5i (per-channel statistics taken before channel mixing), with a literature check first.
Evaluation only so far; nothing below trained a deep model.

**1. The mechanism is narrower than the kill set suggested** (3-seed means, `compare_arena.py`):
- recstatnet − RecurrenceNet per dataset: ActRecTut **+21.4**; the other 9 average **+0.6** (UTD-MHAD −1.1,
  WISDM −1.0, SAD +3.0, Opportunity +2.3).
  - ActRecTut has 2 subjects and 2 splits, and it dominated the 4-dataset kill set of the ablation and the a/b test.
- lawnet_rec − RecurrenceNet is broader: +1.8 on the 9 without ActRecTut (SAD +5.8, DSADS +4.2, UniMiB +3.6).
- Physical sensor structure is not needed. `lawnet_rand` (random channel triples) scores 72.14 at seed 0 against
  `lawnet`'s 71.73 (10 datasets). The lever is window-level second-order statistics of small channel groups, not
  rotation invariance of real sensors.
- The hybrid pattern is published. Corrected by the re-check below:
  - the closest paper is Ravì et al. (IEEE JBHI 2017);
  - DDNN (Qian et al., IJCAI 2019) matches only the pattern: a learned distribution stream fused with temporal and
    spatial streams;
  - Plötz et al. (IJCAI 2011) used ECDF features for HAR.

**Novelty re-check of the best family (2026-10-12; three sub-agents read the primary papers and quoted them).**
- **Statistics side path + fusion (`recstatnet`): prior art.**
  - Ravì, Wong, Lo, Yang, IEEE JBHI 21(1) 2017. §III.D computes 17 fixed shallow features per axis, including mean,
    std, min and max. §III.E concatenates them with deep features before a fully connected softmax layer. §III.F
    trains both together. The paper targets on-node, low-power computation.
  - Ignatov (Applied Soft Computing 2018) and Peppas et al. (2020) do the same with other statistics.
  - DDNN (IJCAI 2019) matches only the pattern. Its "statistics" stream is learned (Eq. 2 in §4.2: an encoder, then a
    mean, with reconstruction and MMD losses), and it has a third, spatial stream. It computes no raw per-channel
    moments.
  - So only the trunk differs from Ravì 2017.
- **RecurrenceNet: partly new.**
  - Closest layer pattern: SELFY (ICCV 2021). It turns similarity between a feature map and its own copies at fixed
    temporal offsets into features and adds them back as a residual block.
  - The soft threshold σ((ε − d)/0.1) with a learned radius ε is nearly identical to Jeong & Yoo 2026 (Electronics
    15:1798, Eq. 2, γ = 10). Their recurrence plot is a fixed-embedding front-end image for a CNN, not a layer.
  - RepNet (CVPR 2020) uses a self-similarity matrix of learned embeddings as a bottleneck, with softmax rows.
  - LogSparse attention (2019) uses exponentially spaced lags with softmax over values.
  - Not found anywhere:
    - a soft recurrence plot computed inside a sequence model and stacked over blocks;
    - a causal, banded recurrence plot on a sparse fixed lag set (O(T·K), suits an MCU);
    - several learned 2-D projections, each with its own radius;
    - a differentiable diagonal-line (RQA) feature;
    - any of this in wearable HAR.
  - "Local attention without softmax" is accurate for the score but misleading for the layer. There is no value
    pathway: the scores themselves are the features.
- **Laws side path (`lawnet_rec`): partly new.**
  - Closest paper: Dong et al. 2021 (J. Ambient Intell. Humaniz. Comput.). It concatenates DMD operator features with
    deep features before a linear classifier. Its data is skeletons from video, it fits one operator per sequence,
    and it is explicitly not end to end.
  - Neural Koopman Pooling (CVPR 2023) uses learnable per-class operators on deep features.
  - Koopa (NeurIPS 2023) fits per-window operators for forecasting.
  - VAR coefficients for wrist HAR (CBIC 2017) feed shallow classifiers.
  - Rotation-invariant IMU features (Yurtman & Barshan 2017; Kobayashi 2011) are SO(3)-invariant only.
  - Not found:
    - closed-form lagged operators fitted per window on IMU data and fused end to end with a deep trunk;
    - characteristic-polynomial readout, invariant to any invertible 3×3 mixing (GL(3): gain and cross-axis, not only
      rotation);
    - a multi-lag operator bank;
    - signed-lag canonical correlations between sensor triples as features.
  - Caveat: random triples work as well as physical sensors, so the claim cannot rest on sensor-rotation robustness.
- **Defensible framing:** an RQA-style recurrence layer, optionally with a per-window GL(3)-invariant operator path,
  for MCU-sized HAR. Neither "a new architecture family" nor "the first statistics + deep fusion" holds.

**2. Premise probe: how much do segment statistics carry without a deep network?** (`tree_probe.py` ->
`results/tree_probe/`).
- Features: per channel mean, std, min, max and RMS of first differences, over:
  - the whole window (`root`);
  - the 16 leaves of a depth-4 dyadic tree;
  - all 31 tree segments, absolute (`tree_abs`) or each coded relative to its parent (`tree_rel`).
- Classifiers, fixed in advance and not tuned: logistic regression (LR) and ExtraTrees (ET, 300 trees).
- Splits and normalization are the harness's own. This is analysis, not model selection: no variant is chosen from it.

  | Dataset | root LR / ET | leaves LR / ET | tree_abs LR / ET | tree_rel LR / ET | RecurrenceNet (3 seeds) | recstatnet (3 seeds) | Paper best | Paper RF |
  |---|---|---|---|---|---|---|---|---|
  | UTD-MHAD | 63.1 / 61.8 | 64.8 / 68.6 | 66.6 / 72.2 | 62.7 / 70.6 | 73.3 | 72.2 | 68.2 | 51.2 |
  | ActRecTut | 64.0 / 65.6 | 56.1 / 49.6 | 56.8 / 49.4 | 64.0 / 61.4 | 40.4 | 61.7 | 61.5 | 41.9 |
  | SAD | 98.0 / 98.7 | 98.1 / 96.2 | 98.2 / 97.4 | 99.0 / 98.1 | 92.9 | 95.9 | 98.2 | 98.3 |
  | Opportunity | 61.4 / 72.0 | 63.2 / 72.4 | 62.9 / 72.5 | 73.3 / 71.0 | 61.6 | 63.9 | 74.7 | 70.6 |
  | WISDM | 72.3 / 76.4 | 66.6 / 71.1 | 77.6 / 75.8 | 75.1 / 73.7 | 82.9 | 81.9 | 85.5 | 71.6 |
  | UCI-HAR | 90.4 / 90.6 | 83.7 / 90.7 | 92.4 / 89.8 | 89.7 / 92.0 | 96.6 | 96.3 | 96.1 | 84.7 |
  | UniMiB | 43.4 / 50.8 | 30.9 / 52.1 | 46.9 / 57.3 | 44.1 / 53.0 | 60.8 | 61.8 | 63.3 | 44.5 |
  | HAPT | 82.1 / 80.9 | 78.3 / 78.2 | 80.0 / 80.8 | 79.6 / 76.6 | 83.3 | 84.5 | 84.8 | 67.1 |
  | **Mean of 8** | 71.8 / 74.6 | 67.7 / 72.4 | 72.7 / 74.4 | 73.5 / 74.6 | 74.0 | **77.3** | 79.0 | 66.2 |

  DSADS and Daphnet are left out, because their local caches number subjects differently (§5e). Means of 8 for
  other models: lawnet_rec 76.3, S4D 73.4, no memory 67.5.
- Reading:
  - Simple segment statistics with a classical classifier match or beat every deep model of ours on 3 of the 8
    datasets: ActRecTut, SAD and Opportunity (Opportunity ET about 72 against 62-64).
  - Deep temporal models lead on the other 5:
    - WISDM about +6, UCI-HAR about +4, UniMiB about +4;
    - HAPT about +1, UTD-MHAD about +1 over `tree_abs` ET.
  - Over 8 datasets the ET variants (74.4-74.6) are level with RecurrenceNet (74.0) and S4D (73.4), and below
    recstatnet (77.3).
  - The two kinds of evidence are complementary. That is why the hybrids, which have both, are best.
  - The paper's Random Forest is a weak version of this family. It uses tsfresh's minimal whole-window statistics,
    100 trees with `max_depth` 8 and `min_samples_leaf` 10 (paper-era `whar_models/random_forest/model.py`).
  - So the paper table understates the interval-statistics family (QUANT, Time Series Forest). A claim that a model
    "beats the old families" needs that baseline (P6).
  - MCU size: ET with unlimited depth is far too large for a microcontroller (hundreds of thousands of nodes). LR on
    `tree_abs` has about 25k weights on UTD-MHAD, the size of RecurrenceNet.

**3. Candidate structures and literature (web, three sub-agents, 2026-10-11). All three are partly new.**
- **C2F-Net** (coarse-to-fine statistic tree).
  - Structure:
    - node descriptors are the probe's statistics on a dyadic tree, coded relative to the parent;
    - a per-channel embedding with shared weights comes first, and channels are mixed only within a node;
    - a top-down GRU runs from the root to the leaves;
    - there is a readout at every level. The network never sees individual samples.
  - Closest prior work:
    - QUANT (Dempster et al. 2023): dyadic intervals with extra-trees;
    - Time Series Forest (2013);
    - TPRNN (2023): a top-down pyramidal RNN for forecasting;
    - top-down tree-LSTM (Teng & Zhang, TACL 2017), and fixed balanced trees as well as parse trees (Shi et al. 2018);
    - mWDN (2018): per-level heads;
    - PRISM (arXiv Dec 2025): a learnable tree partition for forecasting; only the abstract was read.
  - Reviewer risks:
    - relative coding is a Haar/Laplacian pyramid of statistics;
    - the recurrence is known;
    - "anytime" is weak for a fixed window.
  - The claim would have to rest on measured accuracy per byte and per operation against ET/LR on the same
    descriptors.
- **Law cascade.** Closed-form autoregressive fits per window, invariants read out, the innovation residual passed to
  the next stage.
  - Closest prior work: Koopa (NeurIPS 2023). It does closed-form per-window operator fits with residual stacking, for
    forecasting.
  - Also: N-BEATS; LDS features for action recognition (Bissacco 2001 to Turaga 2011); Koopman pooling (CVPR 2023);
    DMD eigenvalues fused with a CNN for action recognition (2021).
  - Reviewer risks: "Koopa for classification", and the multichannel Levinson lattice.
- **Sklar network.** A quantile-function branch plus a branch on within-window ranks, fused at the head.
  - Closest prior work: DDNN (2019), ECDF features (2011), QUANT, FSPool, Markov transition fields.
  - Reviewer risks: z-score plus re-injected statistics may give the same; the copula framing is decorative.

**4. Verdict of the evaluation (before any build).**
- The mechanism points back to known patterns:
  - order-free segment statistics, i.e. the classical feature + classifier family;
  - late fusion with a temporal network (DDNN-type hybrids).
  - It does not point to a new structure.
- C2F-Net sees only segment statistics, so its expected accuracy is about the level of ET on those statistics: 74.5
  over 8 datasets, against recstatnet's 77.3. Its only defensible edge is size and compute per window (31 nodes
  instead of every sample), and its structure is only partly new.
- Law cascade and Sklar network: partly new, with close prior art (Koopa, DDNN). Neither is expected to beat
  recstatnet.
- Recommendation: do not build them as candidate new families.
- The finding worth keeping concerns the old families. A tuned interval-statistics baseline (QUANT-like) belongs in
  every comparison (P6). The paper's Random Forest understates this family by about 8 points over these 8 datasets.
- Decision pending with the user.

### 5m. Round 14 (started 2026-10-12, the direction-3 session): new families, designed from the evidence

Separate from §5l, which is the round-11b session's evaluation and reached its own conclusions on statistics-based
families. User, in this session (2026-10-12): after direction 3 stopped, focus on developing a new type of architecture.

**Design principle, from the strongest repeated evidence: do not mix channels early.**
- Statistics taken before mixing helped. MLP-HAR (every channel × time position) is the paper's best on Opportunity.
- Per-sensor agents added +5-6 on SAD and Opportunity.
- The Random Forest's margin grows with the channel count.
- Every family of ours mixes all channels in its first linear layer.
- §5l's factor analysis: within a dataset the test person carries 61 % of the variance and the model 6 %. A new family
  has to change how the network meets an unseen person, not how it remembers.

**Candidates and literature (web, three sub-agents, 2026-10-12). All partly new.**
- **LevelNet, a value-domain network.** Per channel (raw, difference, a few learned projections), at M levels: soft
  occupation above the level (empirical CDF) and soft level-crossing rate (Rice). A convolution along the level axis,
  shared over channels, gives one token per channel, and the channel tokens are mixed late.
  - Closest prior work: Hammerla et al. ECDF (ISWC 2013), QUANT, MiniRocket PPV, histogram layers.
  - Also DECT (ICLR 2024). On a path graph its Euler characteristic of {x ≥ u} is occupation minus pairs above u, i.e.
    up-crossings, so LevelNet's two features are DECT's vertex and edge terms kept apart.
  - Not found:
    - a differentiable level-crossing layer trained end to end for time-series classification;
    - level-axis convolution as a per-channel tokenizer;
    - DECT on any time series.
  - Main reviewer risk: "end-to-end learning adds nothing over the same features in a Random Forest". It needs that
    baseline.
  - Name free in ML; alternative "ExcursionNet".
- **Group self-similarity network.** RecurrenceNet's self-comparison is extended to shifts, reversal, time scaling and
  learned channel mirrors.
  - The combination as stacked memory is new.
  - But a full self-similarity matrix already holds reversal (anti-diagonals) and scaling (line slope), and is read by
    RepNet/SELFY-style CNNs.
  - Recurrence-based symmetry detection exists in physics (2024, 2026).
  - Expected effect is small: memory type moves results by about seed noise (§5l).
- **StrokeNet (kinematic theory, sigma-lognormal strokes).**
  - Closest prior work:
    - Bio-PM (arXiv 2603.10961, 2026): submovement tokens of wrist accelerometry for cross-subject HAR, no parametric
      fit;
    - SSSUMO: an amortized minimum-jerk parser, no IMU and no classifier;
    - sigma-lognormal fits to IMU data exist only for clinical analysis.
  - High technical risk:
    - gravity in the accelerometer;
    - integration drift;
    - a 150 ms stroke is about 7 samples at 50 Hz;
    - non-unique decompositions;
    - lognormality does not cover cyclic or pendular motion.
  - The name is taken.

**Pre-test for LevelNet (no network; `value_probe.py` -> `results/value_probe/summary.md`).**
- Method: a linear probe, i.e. class-balanced logistic regression on standardized per-channel features fitted on the
  training subjects. It compares value-domain features (occupation and crossings of the signal and its difference at 12
  fixed levels; `level3` adds occupation per third) with the four statistics.
- The four statistics are §5l's `root` features minus the RMS of differences, so the two probes are comparable.
- Results over the 8 locally valid dev datasets (mean of splits):

  | Feature set (linear probe) | UTD | ActRecTut | UCI | UniMiB | Opp. | SAD | HAPT | WISDM | Mean of 8 |
  |---|---|---|---|---|---|---|---|---|---|
  | four statistics | 63.1 | 65.7 | 85.8 | 35.7 | 62.4 | 96.7 | 78.2 | 71.8 | 69.9 |
  | value domain (`level`) | 63.4 | 58.8 | 92.9 | 51.8 | 67.8 | 99.2 | 79.0 | 73.0 | 73.2 |
  | value domain + thirds (`level3`) | 71.0 | 62.0 | 92.3 | 52.5 | 65.5 | 99.2 | 83.5 | 71.8 | **74.7** |
  | `level3` + statistics | 70.5 | 62.9 | 92.3 | 55.4 | 65.3 | 99.3 | 83.3 | 73.4 | **75.3** |
  | seed-0 deep models: recstatnet / RecurrenceNet / statnet | 71.6 / 74.6 / 69.6 | 63.6 / 35.1 / 47.2 | 96.4 / 96.7 / 92.1 | 62.1 / 62.0 / 53.9 | 62.8 / 64.9 / 62.4 | 96.6 / 92.0 / 93.0 | 84.0 / 83.2 / 80.5 | 82.2 / 84.3 / 71.6 | 77.4 / 74.1 / 71.3 |

- Reading:
  - A linear probe on value-domain features reaches RecurrenceNet's level (74.7 vs 74.1) and S4D's (73.4, 3 seeds).
    It is also above §5l's best segment-statistics probe (ExtraTrees 74.4-74.6), but below recstatnet (77.4).
  - Value domain over statistics: UniMiB +16, UTD-MHAD +8, UCI-HAR +7, Opportunity +5, HAPT +5. ActRecTut −4.
  - Gait type (WISDM) stays about 10 points below the temporal models with either feature set.
  - **ExtraTrees on the same fixed features** (300 trees, class-balanced, not tuned; `value_probe.py --et`):

    | Probe | UTD | ActRecTut | UCI | UniMiB | Opp. | SAD | HAPT | WISDM | Mean of 8 |
    |---|---|---|---|---|---|---|---|---|---|
    | ET on `level3` | 70.7 | 63.3 | 92.0 | 56.9 | 72.4 | 99.2 | 82.7 | 76.6 | 76.7 |
    | ET on `level3` + statistics | 72.9 | 68.3 | 91.9 | 57.8 | 72.3 | 99.2 | 83.0 | 76.7 | **77.8** |

    - Non-learned value-domain features + ExtraTrees (77.8) are above every deep model of ours on the mean of these 8
      (recstatnet 77.4 at seed 0), and above §5l's segment-statistics ExtraTrees (74.4-74.6).
    - They win on ActRecTut, Opportunity, SAD and UTD-MHAD. They lose on UCI-HAR (−4.5), UniMiB (−4.3), WISDM (−5.5)
      and HAPT (−1.0), the gait and fine-shape datasets.
    - Consequence for LevelNet: to show that end-to-end learning adds anything, it must beat 77.8 on these datasets, at
      MCU size (unbounded ExtraTrees is far too large for a microcontroller). Otherwise the finding is about the old
      classical family, which the WHAR table understates (its Random Forest: 66.2 here).
- Implementation (user: "install and run LevelNet", 2026-10-12):
  - Code: `edge_ssm/round14.py`; tests `tests/test_round14.py`. All pass: sharp-step level maps of a sinusoid give
    occupation 1/2 and 8 crossings for 4 cycles; the phase plane sums to 1; train steps are finite for every id.
  - Ids, configs fixed in `registry.py` before any result:
    - `levelnet`: 16 levels, 8 virtual channels, d_model 32, thirds, crossings, level-axis conv, one transformer layer
      over channel tokens. 20.5k params on UTD-MHAD, 24.1k on Opportunity.
    - `levelnet_plane`: adds the phase-plane density, 29.9k / 33.5k.
    - Controls: `levelnet_early` (channels mixed first), `levelnet_nocross`, `levelnet_mlp`.
  - Speed: 13-60 ms per training step on the 1080 Ti.

**Pre-registration (written 2026-10-12, before any LevelNet result).**
- *Kill set:* seed 0, PC, the 5 datasets of round 12 (UTD-MHAD, ActRecTut, SAD, Opportunity, WISDM). References on the
  same splits:

  | Reference | UTD | ActRecTut | SAD | Opp. | WISDM | Mean |
  |---|---|---|---|---|---|---|
  | recstatnet (seed 0) | 71.55 | 63.64 | 96.62 | 62.84 | 82.23 | 75.38 |
  | ExtraTrees on `level3` + statistics | 72.9 | 68.3 | 99.2 | 72.3 | 76.7 | 77.88 |
  | linear probe on `level3` | 71.0 | 62.0 | 99.2 | 65.5 | 71.8 | 73.90 |

- *Stage 1:* a candidate continues if its kill-set mean is at least **75.38**. Early stop: more than 5 points below
  recstatnet on the mean of UTD-MHAD, ActRecTut and Opportunity (66.01), which run first.
- *Stage 2*, for a candidate that continues:
  - UCI-HAR, UniMiB and HAPT on the PC, then the mean of 8;
  - "learning adds over fixed features" is claimed only if that mean reaches **77.8** (ExtraTrees on the same
    features);
  - then DSADS and Daphnet on the cloud, then seeds 1-2.
- The controls run on the kill set only for a candidate that continues.
- *Predictions:*
  1. `levelnet` beats the linear probe on `level3` by at least 2 on the kill-set mean.
  2. `levelnet_plane` > `levelnet` on WISDM.
  3. LevelNet > recstatnet on Opportunity and SAD; < recstatnet on WISDM.
- No variant changes inside round 14; a change becomes a new id.

**Outcome so far (seed 0, PC; `r14_summary.py` -> `results/r14/killset.md`).**
- `levelnet` **stopped early by the rule**: UTD-MHAD 58.67, ActRecTut 48.28, Opportunity 68.04, so its fast-3 mean is
  58.33 against the threshold of 61.01.
  - Its SAD run had already finished when Opportunity did: 90.62 on 10 splits (recstatnet 96.62).
  - Its WISDM jobs were stopped as they started (`logs/r14_early_stop.log`).
- Opportunity is the one clear win: 75.5 / 75.1 / 73.5 / 48.0 against recstatnet's 74.8 / 71.3 / 61.9 / 43.3, i.e.
  +5.2 on the mean. This is in the predicted direction (prediction 3, many channels with late mixing).
- On UTD-MHAD and ActRecTut the end-to-end network is far *below* a linear probe on the same kind of fixed features
  (71.0 and 62.0). The flexible tokenizer plus channel attention fits the training subjects and generalizes worse.
  This is the factor-analysis lesson again (§5l): the test person dominates, and capacity does not help.
- `levelnet_plane` **also stopped early**: UTD-MHAD 65.61, ActRecTut 48.06, Opportunity 64.38 (76.0 / 71.0 / 70.9 /
  39.6), so fast-3 is 59.35 < 61.01.
  - SAD had already finished: 94.80.
  - WISDM: 2 of 10 splits done (82.19) when its jobs were stopped by the rule.
- Predictions:
  1. Refuted. LevelNet does not beat the linear probe on fixed features; it is far below. On the 4 datasets with both,
     `levelnet` 66.40 against the probe's 74.42.
  2. Not measured (WISDM stopped).
  3. Half met. Opportunity is above recstatnet (+5.2 `levelnet`, +1.5 `levelnet_plane`), SAD is below (−6.0, −1.8).
- **Verdict: LevelNet stops.**
  - The value-domain *representation* is strong: a linear probe gives 74.7 and ExtraTrees 77.8 on 8 datasets.
  - The *learned network* built on it is worse than both. It loses 12 points to the linear probe on UTD-MHAD and 14
    on ActRecTut, while it wins on the 133-channel Opportunity.
  - Reading: with 1-7 training subjects, the extra capacity (per-channel tokenizer, channel attention, learned
    projections) is spent on the training people. A low-capacity classifier on fixed, well-chosen features generalizes
    better to a new person.
  - Same direction as §5l's factor analysis. It is also the strongest argument so far that the next family should
    restrict what the network can learn about a person, not add new computations.
- The controls (`levelnet_early`, `levelnet_nocross`, `levelnet_mlp`) were not run (candidates stopped).

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

### 5o. Review of the Nature-architecture atlas against the project (2026-10-12, direction-3 session)

User: the other session understood "new structure" as a new neural cell inside the same network. In this session,
look for new architecture types whose *architectural mechanism* is new, built from modern layers if useful. Review the
families in `Atlas of novel AI architectures in Nature journals.html` and propose new families.

**The atlas's own definition.** A novel architecture changes at least one of:
1. how neurons compute;
2. what the network learns;
3. the symmetry built in;
4. how the system is organized;
5. the physical substrate.

It marks 8 of its 53 papers as genuinely new: NCP, CfC, dANN, NG-RC, DeepONet, NeuralGCM, PNN and QCNN. The rest are
variants, systems or hardware.

| Atlas family | Mechanism | In this project so far | What transfers to few-subject wearable HAR |
|---|---|---|---|
| A1 spiking / neuromorphic | event-driven, sparse | event-like memories (hysteresis, rainflow, return maps); the schedule claim failed fairly (§5k) | needs neuromorphic hardware for its energy claim |
| A2 dynamic neurons (CfC, NCP, dANN) | input-dependent time constants; designed sparse wiring; dendrites with restricted input sampling | the cell-level rounds 1-12; dendritic unit dropped on novelty | dANN: restricted input sampling against overfitting (variant at best) |
| A3 reservoir / NG-RC / HDC | fixed rich features, only the readout is trained; no backprop on device | not as a family, but our probes are exactly this (fixed value-domain features + linear or tree readout: 74.7 / 77.8) | strongest match to the evidence, but a known family |
| B1 operators / KAN / PINN | DeepONet: branch(input function) × trunk(query); operators instead of fixed maps | no | the classifier as an operator of the wearer (candidate P2) |
| B2 equivariant | symmetry built in | LawNet's rotation invariance lost posture | partial |
| C1 hybrid physics (NeuralGCM) | known physics solver + learned residual | no | differentiable body/sensor physics; single IMUs lack the body model |
| C2 foundation models for new data types (TabPFN) | a Transformer pre-trained on a synthetic prior performs learning in its forward pass on a new small dataset | no | small data where trees beat deep nets is exactly TabPFN's setting (candidate P1) |
| D1 neuro-symbolic / propose-verify (FunSearch) | the system around the model: an LLM proposes programs, an evaluator checks | no | evolve MCU-sized recognizer programs (candidate P3); the user has ShinkaEvolve |
| D2 world models | learn a model, act in imagination | analysis-by-synthesis judged partly new (§5h) | weak for classification |
| E1/E2/F1 physical, unconventional, quantum | substrate | no | outside scope (no hardware) |

- Reading: every round so far changed only the first item, the neuron or memory cell. The evidence points at items 2
  and 4: what is learned (a learner for small person-varied data) and how the system is organized (the dataset and the
  wearer as inputs).

**Candidates (literature checks by three sub-agents, running):**
- **P1. Movement-PFN, an in-context activity recognizer** (after C2 / TabPFN).
  - Input: a dataset (labelled windows of known people, optionally windows of the wearer) plus the query. One forward
    pass, no gradient training on the target dataset.
  - Pre-trained only on a synthetic prior of person-varied movement tasks; context and query come from different
    synthetic subjects, so cross-person generalization is meta-learned.
  - Personalization = adding the wearer's windows to the context. On an MCU the support set is compressed into a few
    memory tokens per class.
- **P2. Person-operator network** (after B1 / DeepONet).
  - logits(x | wearer) = Σ_k B_k(wearer's unlabelled windows) · T_k(x).
  - Trained episodically over the training subjects.
  - Personalizes from unlabelled windows with a forward pass.
- **P3. LLM-evolved recognizer programs** (after D1 / FunSearch).
  - An LLM proposes and mutates small feature + decision programs within an MCU budget.
  - An evaluator scores nested leave-subject-out F1 plus cost; held-out datasets give the verdict.
  - Output is interpretable C.

**Literature verdicts (web, three sub-agents, 2026-10-12).**
- **P2: prior art in substance.**
  - ARM-CML (NeurIPS 2021) already has a context net over a domain's unlabelled inputs, averaged and fed to the
    predictor, trained episodically per domain (FEMNIST users).
  - The branch × trunk product readout is Kumagai & Iwata (arXiv 1807.02927); the theory is Blanchard et al. (NeurIPS
    2011, JMLR 2021).
  - Speech did it a decade ago: i-vector and summary-network speaker-aware training, and context-adaptive DNNs.
  - Only "HAR + MCU" would be new. Kept as a baseline (ARM-CML, AdaBN, per-person standardization), not as a family.
- **P3: the search method is prior art and it is not a family.**
  - FunSearch / AlphaEvolve / ShinkaEvolve; LLM-evolved feature programs (LLM-FE, ELLM-FT, ...).
  - GP-evolved HAR programs (Xie et al. 2013-14); cost-constrained CGP wearable classifiers (2022-23).
  - New would be only the application and protocol. The main risk is overfitting the evaluator with few subjects, and
    the result could land in the same ±1-3 band. Dropped as a family.
- **P1: partly new; the only candidate that is a new organization for HAR and aims at the dominant factor.**
  - Not found: an in-context classifier with a *person-structured* prior (nuisance shared by all windows of a
    synthetic subject; context and query from different subjects), evaluated leave-one-subject-out on HAR,
    personalized by adding the wearer's windows to the context, and deployed on an MCU with per-class memory tokens.
  - Fast-moving neighbours:
    - synthetic-prior in-context TSC: TiCT (2511.19694), TimEE (2607.07500, Hutter/Purucker group), ChorusTIC
      (2608.24033); generic priors, UCR/UEA splits, no subject structure;
    - TabPFN on TSC features: RocketPFN, MASHT, TS2TabPFN;
    - compressed ICL memory: TIC-FM; MotherNet;
    - **BayaHAR** (ISWC 2026, 2606.04798): gradient-free few-shot personalization on WHAR, LOSO, 1-shot +2.8 to
      +33.4 macro-F1. It overlaps our person-headroom finding and must be cited and beaten.
  - Risks:
    - multichannel synthetic priors transfer poorly (TimEE, ChorusTIC);
    - 27 classes on UTD-MHAD;
    - context length;
    - MCU tokens may reduce to prototypes;
    - scoop risk from the TabPFN group.
  - The decisive reviewer ablation: synthetic prior vs meta-training on the ~29 other real WHAR datasets.
  - Quick probe available: TabPFN on our fixed features, LOSO, with personalization by context. It needs `tabpfn` in a
    separate environment, and **the user must accept the Prior Labs weight licence personally**.

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

### 5q. A lifelong-plastic architecture (2026-10-12, direction-3 session; proposal, nothing trained)

User (2026-10-12): Nature is the target, so the architecture must be our own and truly innovative; P1 (in-context,
TabPFN-like) is set aside.

**Argument.**
- Forward-architecture changes move cross-person accuracy by ±1-3 points (§5n), however novel they are.
- The only large measured effect is adapting to the wearer (+6 to +11 on few-subject data, §5n).
- Brains handle this with local, forward-only, multi-timescale plasticity. Today's architectures lack it, and the user's
  atlas lists it as an unbuilt gap (proposals A-E: one-shot writes, three-factor rules, spaced consolidation, glial
  gain control).
- Proposal: an architecture defined by *what may change after deployment and by which local rule*:
  - a frozen population core;
  - a glia-like label-free self-calibration of gains;
  - a plastic readout with one-shot (BTSP-like) writes for user-confirmed moments and a confidence-gated three-factor
    person weight;
  - spaced consolidation from fast to person weights;
  - all rates, gates and thresholds meta-learned in person-held-out episodes, then run forward-only on the MCU.

**Literature (web, sub-agent): partly new.**
- Closest:
  - ANML (neuromodulated meta-learning, SGD inner loop, new classes);
  - differentiable plasticity / Backpropamine;
  - meta-learned rules (Najarro & Risi, Confavreux, Tyulmankov 2022, Shervani-Tabar 2023);
  - Stewart et al. 2024 (meta-trained three-factor rule on Loihi, one-shot classes);
  - AGMP 2026 (astrocyte state gating plasticity);
  - recall-gated consolidation (Lindsey & Litwin-Kumar, eLife 2024);
  - per-user BN, ARM, OFTTA, T3A/LAME/FOA, TinyOL / GAP9 last-layer updates;
  - BayaHAR (ISWC 2026).
- Not found:
  - plasticity meta-learned across held-out *persons*;
  - one forward-only, dataset-free MCU model with both a label-free and a few-shot path plus consolidation;
  - glial gain control for person calibration;
  - a spacing counter gating consolidation;
  - BTSP-style writes for user-confirmed labels on a wearable.
- WHAR Arena (arXiv 2606.13194) already notes that architectures differ little and that inter-subject variability
  matters. Claim only our quantification (61 % vs 6 %).
- Risks:
  - every part has a plain ML twin (AdaBN/LHUC, imprinting/prototypes, pseudo-label delta rule, EMA + threshold), and
    each must be beaten;
  - cheap baselines may already take the gain (BayaHAR, last-layer SGD, OFTTA);
  - meta-overfitting with 8 people;
  - self-training drift;
  - "scale instead": Kaifosh et al., Nature 2025, generic EMG model from thousands of people.
- Evidence a Nature-family venue expects (Moin et al., Nat Electron 2021):
  - a real MCU wearable with measured energy;
  - new users followed over days;
  - full ablations;
  - ideally a mechanistic finding, e.g. the meta-learned rules resemble BTSP or spacing effects.
- What the project lacks for that: an IMU, a power meter, and a human-participant study (ethics approval).

**Gate 2 (forward-only rules vs retraining, on fixed features; no network) was written as `plasticity_probe.py`, but
the write was interrupted. The partial file was removed. It needs the user's go-ahead to be written again.**

**Lab hardware (user, 2026-10-13).**
- Boards: STM32 L4 (NUCLEO-L4R5ZI, low-power Cortex-M4F), H7 (Cortex-M7) and N6 (Cortex-M55 with the Neural-ART NPU).
- IMU module: DFRobot Fermion 10-DOF, I2C, 3-8 V, 26 × 18 mm:
  - ADXL345 accelerometer;
  - ITG-3205 gyroscope;
  - VCM5883L magnetometer;
  - BMP280 barometer.
  The page gives no ranges or rates. From the datasheets: ADXL345 up to ±16 g and 3200 Hz; ITG-3205 ±2000 °/s; BMP280
  about 1 Pa (≈ 8 cm of height).
- Fit:
  - 6-axis accelerometer + gyroscope at 50 Hz matches UTD-MHAD's channel set.
  - The N6 can run the frozen core on its NPU, with the plastic readout on the CPU, the same split as the architecture.
  - The L4 is the low-power target.
  - The barometer would separate stairs from level walking, our largest error class. That is a sensor fix, not
    architecture, so keep it out of the architecture claim.
- Still missing:
  - an energy measurement (no power meter; Nucleo IDD jumper + ammeter gives average current only; a profiler such as
    STLINK-V3PWR or Nordic PPK2 gives per-inference energy);
  - ethics approval for a user study.

**Family specification (user, 2026-10-13: "what matters now is finding the new model-architecture family"). Re-added
2026-10-14: the first copy was lost when the file was overwritten by another session.**
- Every family in `RESULTS_SUMMARY.md` §0 differs only in its within-window state. All of them freeze their weights
  after training.
- The proposal adds an *across-window synaptic state*: which synapses keep changing during deployment, at which
  timescale, and gated by what.
  - Forward pass: logits = (W + P + F) · (G ⊙ φ(x)).
    - φ: the frozen population core. Proposed core: per-channel value-domain maps + a temporal memory, channels mixed
      late.
    - W: the population readout.
    - G: glia-like gains, updated without labels from the wearer's feature statistics (slow).
    - F: engrams written in one shot (feature × label outer product) when the user confirms an activity; they decay.
    - P: person weights that absorb F only after repeated, spaced confirmations (a spacing counter).
  - Offline, φ, W and every plasticity parameter are meta-learned in episodes that hold out a person. The parameters
    are the gain rate, write strength, decay, and consolidation threshold and spacing.
  - Deployment is forward-only: outer products and running averages.
  - Mapping to the user's atlas: G = astro layer (D), F = one-shot plateau writes (B), P + counter = three-speed
    synapse (C).
  - Evaluation: per-person streams on the cross-person benchmarks (label-free, then 1-3 confirmations per class).
    This extends the WHAR protocol.
  - Comparators: adapted baselines, plus the plain ML twin of each part (AdaBN/LHUC for G, imprinting/prototypes for F,
    last-layer SGD and BayaHAR for P).

**Duplication check of this exact specification (web, sub-agent, 2026-10-14): no paper has this formulation, but every
element has an equivalent, so it is a new combination, not a new family.**
- W + F with a decaying, meta-learned Hebbian trace: equivalent to differentiable plasticity (Miconi et al., ICML 2018).
- F, a one-shot feature × label write: the same mechanism as Hebbian fast weights (Munkhdalai & Trischler 2018; Ba et
  al. 2016) and weight imprinting.
- F → P gate: functionally equivalent to recall-gated consolidation (Lindsey & Litwin-Kumar, eLife 2024), which
  reproduces spacing effects but is theory only, with a hand-set threshold.
  - Related: the count-gated switch of Hebbian Softmax (ICML 2018); the two-store tag-and-capture model of Leimer et
    al. (2019).
- G, label-free gains meta-trained on per-user statistics: equivalent in principle to ARM-BN (NeurIPS 2021).
- Meta-learned plasticity on the edge: Stewart et al. 2024 (Loihi; new classes, not held-out persons).
- Framing: "an architecture defined by which parts update and how fast" overlaps Nested Learning / HOPE (NeurIPS 2025),
  fast-weight programmers, TTT and Titans.
  - **"Unlike all families, weights freeze" does not hold.** It must be restricted to the baselines compared.
- Defensible:
  - the combination for activity recognition (three plastic stores on a frozen population model);
  - all plasticity hyperparameters meta-learned in person-held-out episodes;
  - forward-only updates on an MCU with no stored data.
  - Reviewers will likely call it "a meta-learned plastic readout on a frozen encoder".
- Name: "EngramNet" is taken. Taken uses: the Engram Network blockchain; the Engram Neural Network (arXiv 2507.21474);
  DeepSeek Engram (arXiv 2601.07372). Free in a quick search: TriPlast, PlastiHAR, GliaTrace.

**Execution plan for direction 1 (working name TriPlast): `docs/lifelong_plastic_plan.md`** (kept outside this file to
avoid overwrites by the other session). It covers phases 0-4 with stop rules, the session-disjoint protocol, baselines,
datasets and decisions. Main new caveat: the person headroom in §5n may be inflated, because shots and scored windows were
not session-disjoint. Step 0.2 re-measures it.

### 5r. MCU measurements for the IEEE Sensors paper (started 2026-10-13, NUCLEO-L4R5ZI)

User (2026-10-13): the board is plugged in; deploy. Follows §5p (MCU after the held-out results).

**Set-up (fixed before any board number).**
- Board: NUCLEO-L4R5ZI, Cortex-M4F at 120 MHz (5 Flash wait states, ART prefetch + I/D caches on), 2 MB Flash,
  640 KB SRAM (the linker script spans SRAM1-3). DWT cycle counter; one window per UART request (921600 baud).
- Firmware `firmware/nucleo_l4r5zi`, copied from AutoTinyAI_2 (HAL, clock, UART protocol, DWT, stedgeai runner;
  AutoTinyAI_2 untouched). Host tools in `mcu/`, run in a separate venv (`C:/Users/MinhHuy/.venvs/edge_mcu`: torch,
  onnx, onnxruntime, pyserial; the project venv and its lock stay as they are). Test windows come from the harness
  (`mcu_dump.py`, main venv): `results/mcu/data/<dataset>/split-XX.npz` (test, validation, 512 training windows).
- Shape: UCI-HAR (9 channels x 150 samples at 50 Hz, 6 classes; 3 sensor triples, 3 pairs), split 0, seed-0
  checkpoints. Other shapes later if needed.
- Two deployments of our models:
  1. **ST Edge AI 4.0** (`stedgeai generate`, st-ai C API), the same compiler as the baselines; FP32 and INT8
     (static PTQ, QDQ, per-channel weights, 512 training windows for calibration; the variant (first layer in
     float, MinMax or percentile) chosen per model by INT8-FP32 agreement on validation windows, never on test).
     RN + laws: the closed-form law solves run as C on the MCU (`law_frontend.c`), the learned layers in the network.
  2. **Hand-written C** (`rn_kernel.c`, float32, -O3): RecurrenceNet is causal, so it runs one sample at a time
     (ring buffers); the statistics path keeps running sums; the law path stores the window and solves at the end.
- Baselines: WHAR Arena models through stedgeai with random weights (cost only; their accuracy is the paper's).
- Reported per model: Flash and RAM (stedgeai report and linker map), latency per window (median DWT), for the
  hand-written kernel also cost per sample and latency after the last sample; parity with ONNX Runtime / PyTorch on
  the split's test windows and on-board macro-F1 for ours.

**Findings so far (2026-10-13).**
- stedgeai 4.0 needed export forms (`mcu/deploy_models.py`); every fault was found with a minimal graph checked by
  `stedgeai validate --mode host` or profiled per node on the board:
  - it reads an ONNX tensor (1, A, B) as A channels by B width. Sequences kept as (1, L, D) put time on the channel
    axis: a 1.9k-element Sub took 1.7 M cycles. Export forms keep sequences channel-first, per-step Linear as Conv1d
    k=1, per-lag features concatenated on channels;
  - a Transpose of the graph input feeding a MatMul gives NaN; broadcasting and reductions over 4-D tensors give wrong
    values; Pad along time, a Slice ending at -1, a traced x.shape and a Concat feeding a MatMul do not compile;
  - FP32 1x1 convolutions run at 12-17 cycles/MAC on the M4 whether the weights sit in Flash or are copied to SRAM;
    INT8 convolutions at ~5 cycles/MAC, but the int8<->float conversions around the float elementwise work cost
    ~150-300 cycles per element, and erf (GELU) ~140.
- Export forms match the originals (UCI-HAR split 0: FP32 macro-F1 identical, max |logit difference| < 1e-5; RN +
  laws 3e-6 with the law input). INT8 host F1: RN + laws 99.66 (FP32 99.77), RN 99.65 (99.53), RN + statistics
  99.43 (99.66), S4D 96.23 (96.45). Quantizing the first layer cost 2-3 points (input outliers), so it stays float.
- First board numbers (stedgeai, 20 windows): RN + laws FP32 655 ms (law front-end 5.1 ms), INT8 430 ms;
  TinyHAR INT8 494 ms. TinierHAR does not compile (its GRU needs a Transpose over the batch axis); MLP-HAR does not
  export (FFT).
- Hand-written C (`rn_kernel.c`) matches PyTorch: FP32 on every test window of all 10 UCI-HAR splits (argmax 100 %,
  F1 identical: RN 96.74, RN + statistics 96.42, RN + laws 96.40); INT8 (int8 weights per row, each input vector
  quantized on the fly, SMLAD dot products) 99.9 % argmax agreement, F1 -0.02 to -0.03. Checked on the PC with the
  same C built by MinGW (`mcu/rn_c.py`), then on the board. Footprint: 98-115 KB Flash (FP32 weights), 10-21 KB RAM.
- Kernel speed, RecurrenceNet FP32 per 3-s window, found by per-kind DWT counters on the board: plain loops 360 ms
  (matvec 262 ms, ~9 cycles/MAC); weights read in order in 8-row blocks 320; + weights in SRAM 255; + no FMA
  (VFMA takes 3 cycles on the M4, VMUL + VADD one each) and VSQRT 240; + short exp/erf approximations (errors ~1e-7)
  204 (SRAM) / 269 (Flash); + compact loops (the unrolled copies overflowed the ART's 1 KB instruction cache)
  195 (SRAM). INT8 183 (SRAM): the matvec still costs ~4 cycles/MAC, so INT8 gains little here.
- 3.5 M MACs per window for the RN trunk (UCI-HAR shape) against 11.1 M (TinyHAR) and 17.6 M (Triple-Cross-Attn).
- Not deployable through stedgeai 4.0: TinierHAR (its BiGRU needs a Transpose before the GRU, refused even in an
  export form with permuted weights), DynamicWHAR (internal error), DeepConvLSTM (float LSTM weights 2.0 MB > 2 MB
  Flash, activations 617 KB), MLP-HAR (FFT, no ONNX export), SA-HAR (its forward changes a buffer; no trace).
  CNN-HAR compiles once its LocalResponseNorm is exported as the ONNX LRN operator (`mcu/deploy_baselines.py`).

**Results (UCI-HAR split 0, finished 2026-10-14; `results/mcu/summary_uci_har.md`, `docs/factors/mcu_uci_har.svg`).**
The board left the USB bus once mid-batch (2026-10-13 16:45); those runs were redone after it was plugged back in.

| Model | Build | ms / 3-s window | Flash KB | RAM KB | Notes |
|---|---|---|---|---|---|
| RN + laws | C kernel INT8 | 210.6 | 44.9 | 21.3 | 1.37 ms per sample (6.8 % CPU at 50 Hz), 6.2 ms after the last sample; 868 windows, argmax = PyTorch 100 % |
| RN + statistics | C kernel INT8 | 206.8 | 36.1 | 10.9 | argmax 99.7 %; F1 99.54 vs 99.66 (split 0) |
| RecurrenceNet | C kernel INT8 | 205.5 | 33.3 | 10.8 | |
| RN + laws | C kernel FP32 | 271.2 | 115.4 | 20.9 | 1.77 ms per sample; weights in SRAM: 201 ms, 126 KB RAM |
| RN + laws / RN + stats / RN | ST Edge AI INT8 | 429 / 432 / 420 | 343 / 326 / 310 | 192 / 181 / 180 | FP32 648-658 ms |
| S4D | ST Edge AI INT8 | 555.6 | 86.0 | 145.1 | FP32 1258 ms |
| TinyHAR | ST Edge AI INT8 | 494.4 | 617.8 | 216.3 | FP32 881 ms, 1064 KB Flash |
| CNN-HAR | ST Edge AI INT8 | 778.7 | 110.8 | 264.5 | export form (LRN); FP32 1380 ms |
| Triple-Cross-Attn | ST Edge AI INT8 | 900.2 | 437.7 | 471.0 | FP32 1893 ms |

- Every build that runs reproduces the PC's predictions on the windows sent (argmax agreement 99.7-100 %), except
  MLP-Mixer INT8 (42 %; random weights, near-tied logits or a kernel fault; not counted as deployed).
- Of the paper's 14 deep models, 3 deploy faithfully through ST Edge AI 4.0 on this board (TinyHAR, Triple-Cross-Attn,
  and CNN-HAR with an export form). Not compiled: TinierHAR, DynamicWHAR, Attend+Discriminate (input order),
  DeepConvLSTM-Attn (RandomNormalLike in the graph), DeepConvShallowLSTM (Transpose before the LSTM). Do not fit:
  DANA (RAM 125 %, Flash 104 % in INT8), DeepConvLSTM (RAM 108 %), Guan-LSTM (Flash 154 %), MLP-Mixer FP32 (Flash 125 %).
  Not exported: MLP-HAR (FFT), SA-HAR (forward changes a buffer).
- Same compiler, INT8: our three models are the fastest deployed (420-433 ms against 494-900 ms) with the least RAM
  but one (180-192 KB against 145 KB for S4D, 216-471 KB for the paper's models). Our hand-written kernel needs
  11-21 KB RAM and 33-45 KB Flash (INT8) and spreads the work over the window: ~7 % of the CPU at 50 Hz.
- Accuracy of the INT8 kernel on all 10 splits (PC build of the same C): F1 -0.02 / -0.03 / -0.03 (RN, RN + stats,
  RN + laws), 99.9 % argmax agreement; ST Edge AI INT8 -0.11 to -0.28.
- Energy is estimated only (time x 3.3 V x 110 uA/MHz x 120 MHz, DS12023 run-mode figure): ~9 mJ per window for the
  INT8 kernel, 18-19 mJ for our ST Edge AI builds, 21-39 mJ for the paper's deployable models. A power profiler is
  needed for measured energy.
- Why the C kernel is small (RN + laws INT8, linker map): RAM 21 KB against 192 KB. ST Edge AI runs layer by layer
  over the whole window, so it holds window-sized intermediate tensors (102 KB activations) plus a descriptor struct
  per node and tensor in RAM (89 KB for ~500 nodes); the kernel runs one sample at a time (the model is causal) and
  keeps per block a 32-value state and the last 49 embeddings (9.4 KB for 3 blocks, independent of the window
  length), plus the stored window for the law solves (5.4 + 5.4 KB). Flash 45 KB against 343 KB: one set of loops
  reused for every block and step instead of generated code per node (224 KB), layer-name strings (33 KB) and the
  runtime library (35 KB); weights ~27 KB int8 + ~3 KB float scales against 56 KB (first layer kept float, int32
  biases, and the zero-padded constant matrices our export form needed).
- Caveats for the paper: the baselines are compiled by the vendor tool without hand tuning, while our kernel is hand
  written; the like-for-like comparison is the ST Edge AI rows. Baseline weights are random (cost does not depend on
  them). One dataset shape (9 x 150).

### 5s. STM32N6 (NUCLEO-N657X0-Q) with ST's Neural-ART flow (started 2026-10-14)

User (2026-10-14): a NUCLEO-N657X0-Q is plugged in; evaluate on it. STM32N657 (Rev B), Cortex-M55 at 800 MHz,
Neural-ART NPU at 1 GHz, 4.2 MB SRAM, no internal Flash (weights in the 64 MB octo-SPI Flash).

**Flow (ST's own tools, `mcu/n6.py`).** `stedgeai generate --target stm32n6 --st-neural-art` on the same INT8 QDQ
graphs as the L4R5 (a Nucleo memory pool: no hyperRAM, 48 MB of Flash for weights; `C:/stm32build/n6/cfg`); ST's
`n6_loader.py` builds the NPU_Validation project (copied to `C:/stm32build/n6`, Middlewares and Drivers linked;
build config N6-Nucleo), writes the weights to the external Flash and loads the firmware into RAM; `stedgeai validate
--mode target` runs test windows (evenly spaced over split 0's test set) and reports time per epoch (NPU / CPU) and
agreement with the ONNX reference. Board in development boot (BOOT1 at 1-3, set by the user); the L4R5 stays on its
own ST-LINK (serial numbers). Fixes to run ST's scripts here: GCC_PATH as an environment variable (the loader passes
it shlex-quoted, which make on Windows ignores), Git's sh on PATH, mingw32-make, short 8.3 GCC path,
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python for AiRunner. Host validation is not available for the NPU target.

**Findings so far.**
- ST's MNIST sample: 0.168 ms per inference, 78 % on the NPU. The flow works on this board.
- The Neural-ART compiler's software epochs compute some layouts wrongly (found by validating sub-graphs on the board,
  `C:/stm32build/n6/probe/rn_stages.py`): a reduction over the channel axis with a broadcast (an encoder +
  ChannelNorm graph: L2-relative error 7), and Slice / Concat along the time axis (a delayed copy of z: L2r 2.5; a
  padded conv followed by a slice: 1.15). Pad does not compile. As exported, S4D (4 % argmax agreement), MLP-Mixer
  (0 %) and our recurrence models (5-16 %) ran but gave wrong outputs; CNN-HAR was right (100 %, 68 ms) and
  Triple-Cross-Attn failed validation; TinyHAR, DANA, DeepConvLSTM and Guan-LSTM do not compile (LSTM/GRU shapes).
- N6 export forms (`deployable(..., masked=True)`, same function, INT8 accuracy unchanged on the host): norms as
  1/D-matrix convolutions, time delays as depthwise convolutions of length 2 lag + 1 with symmetric padding (output
  length L, nothing sliced), full-shape masks instead of broadcasts, the law input split by 0/1 selection products,
  S4D's causal convolution of length 2L - 1 and its GLU as two convolutions. These extra convolutions stay float
  (they replace float work). Board results pending.
- On this flow the HAR graphs are dominated by software epochs (98 % of the time for our models and CNN-HAR): the
  NPU accelerates convolutions and products, but LayerNorm, LRN, sigmoid, sqrt and the float islands run on the CPU.

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

## 7. Open questions for the user

1. **P5 is done** (approved 2026-10-09, option A, 10 datasets): no new family beats S4D. How should the search go on?
2. **Approve P6?** Stronger old baselines at equal size (TCN, minGRU, LinOSS in plain PyTorch; MiniRocket/QUANT need
   `uv add aeon`, which changes the locked environment).
3. Which MCU board for P8 (an STM32 is available via the `stm32-*` skills)? What target window rate?
4. Is the target claim "more accurate", or "non-inferior accuracy plus better robustness and cost"? The second is reachable with the current evidence plus P4 and P8.

## 8. Session log

| Date | Session work |
|---|---|
| 2026-10-08 | Built Edge-SSM and baselines; WHAR benchmark on 3 datasets; Edge-SSM failed. Rounds 1–2 of new families; tempo robustness test. |
| 2026-10-09 | Round 2 finished (QuantileNet best); round 3 upgrades (no gain); literature checks verified; HTML report, `RESULTS_SUMMARY.md`, structure figures; HAR ATLAS change reverted; this plan written. |
| 2026-10-09 | Added §2a (data provenance and WHAR Arena protocol); re-verified that all 8 UTD-MHAD split sizes match the WHAR baselines. |
| 2026-10-09 | Compared `run_whar.py` line by line with the HAR paper-era runner (table in §2a); added the `check_paper_era()` guard. No new runs. |
| 2026-10-09 | Documented how the splits are generated and the UTD-MHAD facts behind them (§2a). Split-01's test subject has 14/27 activities, so its F1 ceiling is 51.9. Recorded the paper-era-only and code-reading decisions in §1, the guard in §3, and the seed caveat in P4. |
| 2026-10-09 | Round 4 on PC + Colab + Kaggle: `--seed` and checkpoints in `run_whar.py`; `cloud/` runner; P4 (5 seeds × 8 splits × 5 models, 200 runs), P1–P3 (`robust_test.py` on all 200 checkpoints), P10 (`synthetic_test.py`, 4 tasks × 6 models × 3 seeds). QuantileNet best (74.0), ties S4D, beats Mamba; play memory rate-independent; round-1 tempo gains overstated; quantile memory not outlier-robust. Reports rebuilt. |
| 2026-10-09 | P5 approved and run (option A): 11 datasets pre-registered, mhealth unprocessable by the paper-era code, 10 × 5 models × 3 seeds on PC + 4 Colab sessions + 2 Kaggle kernels, with `EDGE_SSM_FAST` training paths (3.7× faster, checked on UTD). S4D best mean rank (1.80), HysteresisNet sel+clock 2.00, QuantileNet 3.00, Mamba 3.70. P9 early classification: new families worse than Mamba at half a window. |
| 2026-10-10 | Rounds 6–7: family review; 7 new candidates with web literature checks (all partly new; counter networks mostly prior art). Round 6: RainflowNet (exact O(log L) parallel rainflow), OrdinalNet, RecurrenceNet; round 7: ReturnMapNet, PhaseLockNet (hand-written BPTT). Pre-registered 3-stage screening on PC + Colab + Kaggle. RecurrenceNet passed all stages: best mean rank over the 10 P5 datasets (2.00 vs S4D 2.50), non-inferior to S4D, +3.1 over Mamba; the other four did not advance. |
| 2026-10-10 | Compared finished results with the WHAR Arena paper table (Figure 6, 17 models), no new training (`compare_arena.py`, `results/arena/compare.md`, summary §11). Our harness reproduces Figure 6 on the 4 dev datasets (Spearman 0.90–0.98, mean abs delta 0.5–2.6 points). Over the 10 P5 datasets at seed 0, RecurrenceNet (24.7k params) has mean rank 6.20 of 23, level with CNN-HAR (72k) and behind TinyHAR 5.60 (185k). Its mean F1 is 72.25 against 73.1–73.5 for the best four, and no paired difference is significant. Most of the gap is ActRecTut (35 vs 61.5). S4D ranks 9th, HysteresisNet 11th, QuantileNet 12th, Mamba 17th, the no-memory control 20th. Weak spots: SAD and DSADS (Random Forest wins), Opportunity, ActRecTut. |
| 2026-10-10 | Round 8 backlog (ideas only, no runs; `NEW_ARCHITECTURES.md`). Two analyses added to `compare_arena.py` (§5–6): in the paper table the Random Forest's margin over the best deep model rises with the channel count (Spearman 0.71, p = 1.3e-5, 30 datasets); every model fits the training subjects almost perfectly and RecurrenceNet has the smallest seen-minus-unseen gap (14.5). Candidates: LawNet (operator invariants), plant–command deconvolution, sensor parliament, NeuroForest, level-crossing input, and a user-memory track outside the protocol. |
| 2026-10-10 | Round 8 (§5e). Plan review: the val-test gap argument corrected; the RF-vs-channels correlation survives controls (partial 0.73). Local paper-era caches built from HAR's raw files via hard links (`prepare_local_datasets.py`): 5 of 6 match P5 split for split, Daphnet does not (listdir order). LawNet + 4 controls/hybrids implemented and tested; literature check: partly new. Stage A on 5 of 10 dev datasets: LawNet −3.7 vs RecurrenceNet without ActRecTut; stopped by the user (new rule: stop early, try another model). |
| 2026-10-10 | Rounds 9-10 (§5f-5g), fast screen on UTD-MHAD / SAD / Opportunity with a kill rule, PC + Colab + Kaggle. ParliamentNet: SAD 96.87 and Opportunity 70.67 (+4.9, +5.8 over RecurrenceNet) but UTD 58.60, stopped; a composition of known parts. ParliamentBody, ForestNet (training failure) stopped. User: candidates must be completely new and bold. Seven bold ideas literature-checked; RaceNet dropped as prior art; EcoNet (Lotka-Volterra community), GrammarNet (PCFG inside algorithm) and a braid-holonomy net (flat KZ connection, homotopy-invariant) implemented with tests and all stopped early, between the no-memory control and RecurrenceNet. |
| 2026-10-11 | Round 11 evaluation without training (§5h): error analysis with oracle headroom, style probe (1-NN DTW), literature for DrawNet / arrow of time / analysis-by-synthesis. Found the local DSADS cache numbers labels differently from the cloud (listdir order), now disabled. 1-NN DTW + per-window standardization beats every deep model on ActRecTut (69.0 vs 40.4) and matches RecurrenceNet on UTD-MHAD, but loses on locomotion. |
| 2026-10-11 | Round 12 (§5j), first under the clarified goal (known layers allowed, structure must be new). IsochronNet (computes in intrinsic time: entrained Hilbert-phase oscillators, phase-indexed ring memory = differentiable epoch folding, arc-length progress line, relative-phase and symmetry readouts) with tests; literature: partly new (core mechanism not found); dendritic plateau network dropped on novelty, reference-frame columns kept as 12b. Kill-set screen on 5 datasets: both candidates stop (isochron 72.81 vs bar 74.38; isochron_rec far from 76.38). One strong signal: on WISDM the phasic path adds +11.9 over its memoryless twin (10/0). UTD robustness: best on tempo, noise and offset, but only the offset gain is due to the intrinsic phase. Some jobs died of RAM exhaustion (completed later). Final: isochron 72.71, isochron_rec 73.12 vs recstatnet 75.38. Post hoc WISDM control: half of the gait gain (+5.15 of +11.9, p = 0.049) is due to the entrained phase. |
| 2026-10-11 | Round 11b (§5i) and C finished. Seeds 1-2 of the round-8 models (Colab + Kaggle): 3-seed means recstatnet 74.70, lawnet_rec 74.42, RecurrenceNet 72.01, S4D 71.29; lawnet_rec − S4D +3.12 [+0.58, +5.67], Holm p 0.049, the only gain over S4D that survives Holm (dev datasets, so optimistic). C on all 10 datasets: 1-NN DTW + standardization 65.3 mean, #1 on UTD-MHAD and ActRecTut, #14-18 on locomotion. `benchmark11.py` now also tests against S4D, with Holm over models. a/b mechanism test: the first VM vanished and the T4 fallback ran out of RAM building the Opportunity cache; the remainder (Opportunity, part of SAD) reran on Colab L4 (`cloud/jobs/colab_r11d_ab2.json`). a/b result: (a) 68.83, (b) 66.97, both below 69, so H2 is favoured by the pre-registered rule; (a)+(b) 69.72. Per-channel statistics before mixing carry the gain. |
| 2026-10-11 | Direction 3 (§5k, user: capability paper for a Nature-family journal; claim chosen: sampling-schedule-agnostic inference with HysteresisNet; hardware NUCLEO-L4R5ZI only, no IMU, no power meter). Literature: partly new; principle is prior art (TWINN, signatures, Wei & Sun 2000, Frydrych 2026), scoop risk from Frydrych's announced follow-up; key reviewer demand: Δ-rescaled S4D. `schedule_test.py` (rates, drops, jitter, send-on-delta; naive / grid / hold / Δ / native pipelines) with tests. Decisive check negative on UTD-MHAD, WISDM and SAD: with interpolation or Δ rescaling the baselines are as schedule-robust as HysteresisNet; only severe loss at a low native rate (WISDM, drop 50-70 %) favours it. Claim stopped. |
| 2026-10-11 | Round 13 evaluation (§5l), the round-11b session's direction 3, no deep training. The 11b mechanism is mostly ActRecTut (recstatnet − RecurrenceNet +0.6 on the other 9). Random channel triples work as well as physical sensors for LawNet. `tree_probe.py`: dyadic segment statistics + ExtraTrees / logistic regression on 8 dev datasets reach 74.4-74.6 (RecurrenceNet 74.0, S4D 73.4, recstatnet 77.3, paper RF 66.2); they win on ActRecTut, SAD and Opportunity and lose on gait/fine-shape datasets. Literature for C2F-Net, law cascade and Sklar network (3 sub-agents): all partly new (QUANT, Koopa, DDNN). Recommendation: do not build them; add a QUANT-like baseline (P6). Awaiting the user. |
| 2026-10-12 | Synthesis of rounds 1-13, no training: `synthesis.py` -> `results/synthesis/summary.md`, written up in Vietnamese as `RESULTS_SUMMARY.md` §12 (via `build_summary.py`). Factor reading: test person and training-subject count dominate (person spread 8 points, seen-unseen gap 14.5); having a temporal memory ~6; late window statistics ~2.5 (few subjects / many channels); memory type ~1-3, the size of seed noise. Classical-minus-deep gap tracks fewer subjects (Spearman −0.93, n = 8) and more channels (+0.81), which are confounded here. |
| 2026-10-12 | Factor analysis and model atlas, no training. `factor_analysis.py` (`results/factors/`): variance decomposition of 2,460 runs (dataset 76.6 % of all variance; within a dataset test person 61 %, seed 14 %, model 6 %); data factors over the paper's 30 datasets (channel count predicts the Random Forest's margin, +7.2 per sd, p < 0.001; fewer subjects make the model choice matter more, −1.5 per sd, p 0.005; no property predicts difficulty, R² 0.03); paired component contrasts with t-intervals (memory +5.95 [+2.79, +9.10], 10/10; laws on RN +2.41 [+0.35, +4.47]; Mamba − S4D −2.39, p 0.010). Fixed a key clash that let the kill-set recstatnet contrast overwrite the 10-dataset one. Figures `build_factor_figures.py` -> `docs/factors/*.svg`, embedded in `RESULTS_SUMMARY.md` §12; new page `Model atlas.html` (`build_model_atlas.py`): scoreboard, one card per family, every run, and the factor figures. |
| 2026-10-12 | Pareto fronts, no training (`pareto.py` -> `results/pareto/`, `docs/factors/pareto.svg`, in the atlas and `RESULTS_SUMMARY.md` §12.7). Cost measured per dataset on the real input shape and averaged geometrically over the 10 datasets; accuracy = seed-0 mean (the paper ran once). Parameter front: TinierHAR 14.5k (71.4) → LawNet random groups 21.4k (72.1) → RecurrenceNet 24.9k (72.3) → RN + statistics 28.7k (74.7); every paper model above 15k parameters is dominated. Latency front (PyTorch eager CPU): statistics-memoryless 0.57 ms → Triple-Cross-Attn 3.5 ms → RN + statistics 5.8 ms. FLOPs left out of the fronts (the counter misses LSTM/FFT/scan work: Guan-LSTM counted 0.57 MFLOPs). Correction: the WHAR Arena v1 protocol uses 3-s windows for every dataset (whar_arena/training.py overrides cfg.window_time); the dataset-property tables and factor texts that used per-dataset config window lengths were fixed (memory gain vs samples per window is ρ −0.36, not +0.46). |
| 2026-10-12 | Round 14 (§5m, direction-3 session; user: focus on a new type of architecture). Design principle from the evidence: do not mix channels early. Three candidates literature-checked by sub-agents, all partly new: LevelNet (value-domain network; closest Hammerla ECDF, QUANT, DECT), group self-similarity network (closest RepNet/SELFY; small expected effect), StrokeNet (sigma-lognormal; closest Bio-PM 2026; high technical risk, name taken). `value_probe.py`: a linear probe on per-channel value-domain features (occupation + level crossings of signal and difference, + thirds) reaches 74.7 over 8 dev datasets (RecurrenceNet 74.1, recstatnet 77.4, four statistics 69.9). ExtraTrees-on-same-features baseline running. The first write of the LevelNet module was interrupted; the partial file was removed, so LevelNet is not implemented yet. |
| 2026-10-12 | `value_probe.py --et`: ExtraTrees on the fixed value-domain features (+ statistics) reaches 77.8 over the 8 dev datasets, above recstatnet (77.4, seed 0) and every deep model of ours; it wins on ActRecTut, Opportunity, SAD, UTD-MHAD and loses on UCI-HAR, UniMiB, WISDM, HAPT. Bar for LevelNet set at 77.8 (learning must add over fixed features), at MCU size. |
| 2026-10-12 | Round 14 screen (§5m): LevelNet implemented (`edge_ssm/round14.py`, tests pass) and screened on the kill set, seed 0, PC. Both candidates stopped early by the pre-registered rule: `levelnet` fast-3 58.33, `levelnet_plane` 59.35 (threshold 61.01). Their WISDM jobs were stopped as they started (`logs/r14_early_stop.log`). Opportunity is above recstatnet (+5.2 / +1.5), but UTD-MHAD and ActRecTut are 12-14 points below a linear probe on the same kind of fixed features. Lesson: with few training subjects the learned tokenizer and channel attention fit the training people; low-capacity classifiers on fixed value-domain features generalize better (probe 74.7, ExtraTrees 77.8 over 8 datasets). |
| 2026-10-12 | Best family drawn (`docs/factors/best_family.svg`) and compared model by model with the benchmark (`vs_benchmark.py` -> `results/vs_benchmark/`, `docs/factors/vs_benchmark.svg`; in the atlas and `RESULTS_SUMMARY.md` §12). Seed 0, paired over the 10 datasets, Holm over the 17 paper models: RN + laws and RN + statistics are ahead of every paper model on the mean (+0.3 to +1.6 over the top five), but no difference against the eight strongest is significant (all 95 % intervals include 0). Significant after Holm only against MLP-Mixer, SA-HAR and k-NN. Rank among the 17 paper models: RN + laws first on 5 datasets (mean rank 4.6), RN + statistics on 3 (4.7); both lose on Opportunity (#11-13) and WISDM (#9-11). |
| 2026-10-12 | Novelty re-check of the best family on primary papers (three sub-agents, quoted sections; §5l). Corrections: the closest prior art for the statistics side path + head fusion is Ravì et al. (IEEE JBHI 2017), not DDNN (which has a learned statistics stream and three streams); RecurrenceNet's layer pattern is SELFY (ICCV 2021) and its learned-radius sigmoid is Jeong & Yoo 2026; the laws path's closest is Dong et al. 2021 (DMD on video skeletons, not end to end). Not found: a stacked causal banded soft recurrence plot with RQA line features inside a sequence model, and a per-window GL(3)-invariant multi-lag operator path on IMUs trained end to end. Texts in the atlas, `RESULTS_SUMMARY.md` §12, `NEW_ARCHITECTURES.md` and §2 updated. |
| 2026-10-12 | Strategic evaluation (§5n), no network trained. Under the fixed protocol a new architecture can be expected to move the mean by ~1-3 points (model RMS 2.4 vs seed 3.6 and person 7.5; combining our two best families perfectly gains ~+2). `person_probe.py`: adapting a linear probe on fixed features to the test person gains +3.1 (1 shot/class) to +4.9 (3 shots), and +6 to +11 on the few-subject gesture and many-channel datasets; unlabelled per-person standardization +1.5 (Opportunity +7). Recommendation: an adaptation-native family (B) behind three cheap gates (headroom done; literature; standard adapters on existing checkpoints), with a write-up of the findings in parallel. B changes the protocol and needs the user's approval. |
| 2026-10-12 | Atlas review (§5o, direction-3 session; user: new architectural mechanisms, modern layers allowed). The atlas's genuinely new entries change the neuron, what is learned, symmetry, system organization or substrate; every project round changed only the neuron/memory. Three candidates literature-checked by sub-agents: P1 Movement-PFN (in-context recognizer with a person-structured synthetic prior) partly new, the only one aimed at the dominant factor, fast-moving neighbours (TiCT, TimEE, ChorusTIC, BayaHAR); P2 person-operator network prior art (ARM-CML, Kumagai & Iwata, speaker-aware training); P3 LLM-evolved programs a search method, not a family. No training. |
| 2026-10-12 | Paper for IEEE Sensors Journal (user). Held-out confirmation pre-registered in §5p before any run: 8 datasets (PAMAP2, UP-Fall, UMA-Fall, USC-HAD, MotionSense, WEAR, UCA-EHAR, W-HAR) × lawnet_rec, recstatnet, RecurrenceNet, S4D × seeds 0-2; paper baselines from Figure 6. Kaggle's weekly GPU quota was spent, so the runs are on Colab A (PAMAP2, UMA-Fall; L4), B (USC-HAD, W-HAR; L4), C (UP-Fall, MotionSense; T4 fallback) and the PC (UCA-EHAR, WEAR, local paper-era caches built from HAR raw files via `prepare_local_datasets.py`; HAR unchanged). Analysis ready: `heldout_analysis.py`. Paper plan: `paper/PLAN.md`. |
| 2026-10-13 | Held-out confirmation finished (§5p; 8 datasets × 4 models × 3 seeds on Colab A/B/C/D and the PC; PAMAP2's second VM lost the UCI download and 6 splits were rerun; Colab C ran out of VM cycles and was continued; UP-Fall ran on the PC because its parser fails under Colab's Python 3.13). Pre-registered results: RN + laws − S4D +4.75 [+1.85, +7.65], 8/8, p 0.008 (confirmed); non-inferiority to TinyHAR / CNN-HAR / Triple-Cross-Attn with a 1-point margin not shown (point estimates −0.02 / +0.76 / +0.49); RN + laws on the parameter front at 29.5k (TinyHAR 296k for +0.02); RN + laws − RN +3.15, 8/8. Losses on UP-Fall and PAMAP2 (many channels; Random Forest best). |
| 2026-10-13 | MCU deployment on the NUCLEO-L4R5ZI started (§5r; user: board plugged in). Firmware copied from AutoTinyAI_2 into `firmware/nucleo_l4r5zi` (RAM spans 640 KB, PROFILE command, law front-end, hand-written backend); host tools in `mcu/` (separate venv `edge_mcu`). stedgeai 4.0 needed channel-first conv-style export forms (`mcu/deploy_models.py`; faults found with host validation and per-node board profiles). Results so far, UCI-HAR split 0: our hand-written C kernel matches PyTorch exactly on all 10 splits (INT8: 99.9 % argmax, F1 -0.02); FP32 on the board 266-271 ms per 3-s window, 1.77 ms per sample (8.9 % CPU at 50 Hz), 98-118 KB Flash, 11-21 KB RAM; RN INT8 206 ms, 34 KB Flash. stedgeai: RN + laws INT8 430 ms (20 windows), TinyHAR INT8 494 ms / 216 KB RAM, Triple-Cross-Attn INT8 900 ms / 471 KB; TinierHAR, DynamicWHAR, DeepConvLSTM, MLP-HAR, SA-HAR not deployable. The board left the USB bus at 16:45 mid-batch; the rest is in `cloud/jobs/mcu_uci_har_resume.ps1`. |
| 2026-10-14 | Duplication check of the lifelong-plastic family specification (§5q; spec re-added after it was lost to an overwrite): no identical formulation, but each element has an equivalent (differentiable plasticity, Hebbian fast weights, recall-gated consolidation eLife 2024, ARM-BN; framing overlaps Nested Learning/TTT/Titans). A new combination, not a new family. Name EngramNet taken. |
| 2026-10-14 | Direction 1 checked and planned (`docs/lifelong_plastic_plan.md`). BayaHAR verified as the main competitor (closed-form Bayesian prototypes, gradient-free, one-shot, +2.8 to +33.4 F1, no MCU measurement). New caveat: the §5n headroom was measured without session-disjoint shots, so it may be inflated. Phase 0 (session structure, session-disjoint headroom, forward-only rules vs retraining, vs BayaHAR-style update) comes first with stop rules. Two phase-0 script writes were interrupted by an automated filter and are pending. |
| 2026-10-14 | Full survey of wearer adaptation (`docs/personalization_survey.md`; 4 sub-agents: few-shot, label-free UDA/TTA/DG, on-device/continual, protocols/leakage/WHAR numbers). WHAR Arena (KIT TECO) has no personalization protocol and lists it as future work; BayaHAR is by the same group (code exists, same-recording support/query, class filtering inflates HAPT/HARTH). Realistic leakage-free gains: label-free +2-5, few-shot ~+10 on UTD-MHAD (trial-disjoint), small where many subjects. Open: a leakage-free WHAR personalization protocol + head-to-head; an integrated forward-only learner on Cortex-M. User constraints recorded: WHAR data only; STM32 boards only, no energy meter. |
| 2026-10-14 | MCU measurements finished for UCI-HAR (§5r results; board reconnected after dropping off USB). Our C kernel INT8: 206-211 ms per 3-s window, 1.37 ms per sample (6.8 % CPU at 50 Hz), 33-45 KB Flash, 11-21 KB RAM, board = PyTorch on 868 windows (RN + stats 99.7 %). Same compiler (ST Edge AI 4.0 INT8): ours 420-433 ms / 180-192 KB RAM vs TinyHAR 494 / 216, CNN-HAR 779 / 265, Triple-Cross-Attn 900 / 471, S4D 556 / 145. Only 3 of the paper's 14 deep models deploy faithfully on the L4R5 (TinyHAR, Triple-Cross-Attn, CNN-HAR via an export form); the rest do not compile, export or fit (MLP-Mixer INT8 runs but disagrees with the PC). Table `results/mcu/summary_uci_har.md` (`python -m mcu.summary`), figure `docs/factors/mcu_uci_har.svg` (`mcu_figure.py`). Energy only estimated (no power meter). |
| 2026-10-14 | Direction 1 deployed and split into a new project, `D:\Projects\HAR_Lifelong` (code, docs, survey, evidence copied; reads this project's `datasets_paper_era_local`). Phase 0 here: segment structure (`session_check.py`), session-disjoint person headroom +8.6 to +13.2 F1 on few-subject datasets (gate 0.2 passed), but BayaHAR-style prototypes and a local delta rule already take it (gate 0.4 stop rule met: best forward-only margin +1.9). Continue in HAR_Lifelong. |
| 2026-10-14 | One results atlas: `Edge_SSM Results Atlas.html` (`build_results_atlas.py`, main env) gathers every result: taxonomy of ~40 models in 6 classes (memory cell, side path, structure, ablation, no training, old family) with paired Δ vs S4D on the same datasets, architectures (ours and the paper's 17), dev and held-out accuracy with tests, accuracy Pareto, L4R5 results with Pareto fronts on latency/RAM/Flash (`mcu_figure.py` now draws them: every build, and ST Edge AI builds only), N6 status, deployability of the 14 deep models, factors, 13 graded findings, and a 3-paper split: P1 IEEE Sensors Journal (architecture + 18-dataset evaluation + MCU summary), P2 IEEE Sensors Letters (what drives HAR accuracy: person 61 % vs model 6 %, data factors over 30 datasets), P3 IEEE Embedded Systems Letters (deploying published HAR nets on Cortex-M: 3/14 faithful, compiler faults, streaming C kernel). Held-out note added: DynamicWHAR (2.3M params) 68.45 and the paper's RF 68.02 are above RN + laws 67.02 (differences not significant). |
