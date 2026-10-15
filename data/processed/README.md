# data/processed: the paper-era caches (shared) and this project's derived data

**Shared caches.** The windows every run trains and tests on are built by the paper-era whar-datasets code
(144e006) and live in `D:\Projects\HAR\datasets_paper_era\<id>` (`run_whar.DATASETS_DIR`; override with
`WHAR_DATASETS_DIR`). 13 datasets are cached on this PC:

| Dataset | Built by | Note |
|---|---|---|
| utd_mhad, uci_har, unimib_shar, actrectut_gestures | the HAR reproduction | split sizes equal the WHAR authors' |
| hapt, opportunity, sad, wisdm, wear | Edge_SSM `prepare_local_datasets.py` from HAR's raw files | split sizes checked equal to the cloud runs |
| up_fall, uca_ehar | the same, for the held-out runs | the stored held-out results ran on these |
| dsads, daphnet | the same | parser numbers subjects in listing order: per-split indices differ from the stored (Linux) runs; compare dataset means only |

Not cached here (built on cloud VMs only): pamap2, uma_fall, usc_had, motion_sense, w_har. Build one with
`uv run python scripts/prepare_data.py <id>`: it adds `datasets_paper_era/<id>` (hard links to the raw files, the
paper-era hash files, then the paper-era pipeline) and changes nothing that exists in HAR.

**Splits.** The protocol ships a rule, not split files: `run_whar.prepare_dataset` regenerates the subject splits
(LKSOSplitter, k = 10 subject groups, validation 20 % of the training windows drawn with `cfg.seed = 0`) identically
on every run. Never write a split by hand and never change `cfg.seed`. `splits.json` and `manifest.json` (agents/
03_data.md) are to be exported from the harness at stage 03 (gate G2); they record, not define, the splits.
