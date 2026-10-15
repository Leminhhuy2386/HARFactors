# Running this project on the HPC

Cluster login: `ssh hpc-gw` (10.20.8.11, user `huy.leminh`; `~/.ssh/config` on the PC). Slurm.

## Layout on the cluster (uploaded 2026-10-15)

```
/work/huy.leminh/code/Dr.Huy/HAR/          = $HAR_DIR, mirrors D:\Projects\HAR on the PC
├── datasets_paper_era/<id>/               the 13 paper-era caches (read-only; never edit by hand)
├── datasets/<id>/data/                    raw files of all 30 WHAR datasets (no processed folders)
├── HARFactors/                            this project (a copy of D:\Projects\HARFactors)
└── CortexHAR/                             the deployment project (its board work stays on the PC)
```

Everything WHAR in `D:\Projects\HAR` that the paper-era protocol uses is here (65.5 GiB, 33,532 files, every file
matched to the PC by count and size on 2026-10-15):

- **Paper-era caches** (`datasets_paper_era/`, 28.5 GiB), copies of the 13 PC caches: actrectut_gestures, daphnet,
  dsads, hapt, opportunity, sad, uca_ehar, uci_har, unimib_shar, up_fall, utd_mhad, wear, wisdm. The 10 development
  datasets are among them. The same split sizes and windows apply. dsads and daphnet carry the PC's (Windows)
  subject numbering.
- **Raw files** (`datasets/<id>/data/`, all 30 datasets). For the 13 cached datasets they are hard links to the
  cache's `data/`; the other 17 were sent from `D:\Projects\HAR\datasets`.
- **Not uploaded, on purpose:** HAR's latest-code processed folders (`datasets/<id>/sessions`, `windows`, ...) and the
  latest-code step-hash files at the top of each raw `data/` (`download_hash.txt`, `extracting_hash.txt`). Those hashes
  differ from the paper-era ones for all 13 cached datasets, so they would not let the paper-era pipeline skip its
  download step anyway.
- **A new cache** (one of the 17 uncached datasets) is built only with the paper-era code, into
  `datasets_paper_era/<id>`, and only with the user's approval, the same rule as on the PC. Build it on the PC and copy
  it here, so both sides hold the same cache.

## Environment (not installed yet)

Status 2026-10-15: only the project and the data were uploaded. The environment is not installed and `setup.sh` has not completed a run yet (`RUN_REPORT.md` § HPC copy says what is staged). The plan below is the intended setup, still untested.

### First time: `bash scripts/hpc/setup.sh` on the gateway (internet; no computation there)

The script:
- installs uv into `~/.local/bin` and a managed Python 3.12 into `/work/huy.leminh/.local/share/uv/python`;
- builds `.venv` from `uv.lock` (`uv sync --frozen`);
- resets `third_party/whar-*` to their pinned commits (files copied from Windows have CRLF line endings, which git
  would report as local changes);
- checks the paper-era guard and the data folder.

**Site quirk (checked 2026-10-15):** the Sophos firewall blocks `download-r2.pytorch.org`, where `uv.lock`
gets `torch 2.14.0+cu126`. It serves a "Blocked site" page instead of the wheel; PyPI and GitHub are fine. So:
- the wheel is downloaded on the PC and checked against the sha256 in `uv.lock`;
- it is copied to `/work/huy.leminh/.cache/whl/`;
- `setup.sh` checks the hash again and installs it next to the `uv sync --frozen --no-install-package torch`
  environment.

`env.sh` sets `UV_NO_SYNC=1`, so `uv run` never tries to re-fetch torch. Use `uv run --no-sync` in commands.

After setup, check a compute node end to end: `scripts/hpc/submit.sh smoke normal 00:30:00 -- bash scripts/hpc/smoke.sh` (use `gpu` instead of `normal` for the GPU path). It prints `SMOKE OK`, and the analyses it re-runs must leave `results/metrics/` unchanged.

## Site firewall (probed from hpc-gw, 2026-10-15)

| Host | Needed for | Status from hpc-gw |
|---|---|---|
| `download.pytorch.org` | PyTorch wheel index (`uv.lock`: torch 2.14.0+cu126) | **blocked**: Sophos "Blocked site" (HTTP 403), TLS re-signed by Sophos |
| `download-r2.pytorch.org` | the torch wheel files themselves | **blocked**: same |
| `pypi.nvidia.com` | NVIDIA CUDA wheels (fast mirror of the `nvidia-*` packages) | **no connection** |
| `files.pythonhosted.org` | every PyPI package file | reachable but slow (about 0.46 MB/s; the CUDA wheels total about 2.4 GB) |
| `pypi.org` | PyPI index | OK (2.4 MB/s) |
| `github.com`, `release-assets.githubusercontent.com`, `objects.githubusercontent.com` | uv-managed Python downloads | OK |

To ask the admin (user, 2026-10-15): allow the first three hosts, exempt them from HTTPS inspection, and lift the throttling on
`files.pythonhosted.org`. Until then, install torch and the `nvidia-*`/`triton` wheels from the hash-checked
copies in `/work/huy.leminh/.cache/whl` (downloaded on the PC).

## Every session

```bash
cd /work/huy.leminh/code/Dr.Huy/HAR/HARFactors
source scripts/hpc/env.sh            # HAR_DIR, uv paths, UV_FROZEN
scripts/hpc/submit.sh <name> <partition> <walltime> -- uv run --no-sync python <script> [args]
squeue -u $USER; tail -f experiments/logs/hpc/<name>_<jobid>.out
```

- Never run training or analyses on the login node. Use `submit.sh`, which calls sbatch, writes one log per job
  under `experiments/logs/hpc/`, and cancels a stale job of the same name. The site's skill is in `docs/hpcskill.md`.
- Partitions seen on 2026-10-15:
  - `gpu`: hpc21-24, 1 GPU each, usually all busy, so jobs queue;
  - `normal` / `small` / `tiny`: hpc01-20, CPU only, 32 cores and 368 GB each;
  - `dgx-*` (A100): down.
- This project's analyses are CPU work, so use `normal`. Use `gpu` only for training or for regenerating
  predictions from checkpoints.

## Keeping the PC and the cluster in step

- The PC copy is the reference.
- Results produced on the cluster come back to the PC (`experiments/`, `results/`), with their ledger lines.
- Never edit the same file on both sides.
- Write `docs/*.md` and the agent prompts on the PC.
