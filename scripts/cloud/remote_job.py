"""Runs ON a Colab or Kaggle VM: set up the paper-era WHAR code, then run run_whar.py jobs in parallel.

Inputs next to this file (Colab: uploaded to /content/edge; Kaggle: embedded by kaggle_run.py and
recreated in /kaggle/job_src):
  code.tar.gz           scripts/train.py and src/{utils,training,models} (scripts/cloud/common.py: pack_code)
  job.json              {"jobs": [{"models": [...], "dataset": "utd_mhad", "seed": 1,
                                   "splits": [0, 1] (optional), "gpu": 0 (optional)}, ...],
                         "max_parallel": 4 (optional), "env": {"EDGE_SSM_FAST": "1"} (optional)}
  prior_results.tar.gz  optional: results already finished elsewhere, so the runner skips them

The WHAR libraries are cloned at the paper commits and installed without touching the VM's CUDA
torch; the data-side packages are pinned to the versions of the local uv.lock, so the windows and
the normalization match the local cache. The dataset is downloaded and preprocessed here by
whar-datasets itself (no data is uploaded). Each dataset's cache is built by one process before any
job of that dataset starts, so jobs never write a cache concurrently. At most max_parallel jobs run
at once, taken in the order of the job file; jobs without a "gpu" are spread over the VM's GPUs.

Outputs go to OUT/results (same layout as the local results/, checkpoints included) and
OUT/train_<i>.log. On Colab OUT is packed into results.tar.gz every PACK_EVERY seconds for the
supervisor to pull; on Kaggle OUT is /kaggle/working, which Kaggle returns when the run ends.
"""

import json
import os
import subprocess
import sys
import tarfile
import time

KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or not os.path.isdir("/content")  # Colab VMs also have /kaggle
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
WORK = "/kaggle/temp_edge" if KAGGLE else "/content/edge"  # clones, datasets, code
OUT = "/kaggle/working" if KAGGLE else "/content/edge/out"
INPUTS = HERE if KAGGLE else "/content/edge"
COMMITS = {"whar-datasets": "144e006", "whar-models": "9b65004", "whar-arena": "19b3019"}
PINNED = ("pandas==3.0.6 numpy==2.5.3 dask==2026.8.0 pyarrow==25.0.1 scipy==1.18.1 pywavelets==1.10.0 "
          "scikit-learn==1.8.0 tsfresh==0.21.2 einops==0.8.2 pydantic==2.13.5 jsonschema==4.26.0 "
          "pyyaml==6.0.3 short-unique-id shortuuid python-dotenv kaggle tqdm requests")
PACK_EVERY = 120
DEADLINE = 11 * 3600  # Kaggle stops runs at 12 h; stop cleanly so finished splits are returned


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout.strip()


def pack() -> None:
    if KAGGLE:
        return
    tmp = f"{WORK}/results.tar.gz.tmp"
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(OUT, arcname=".")
    os.replace(tmp, f"{WORK}/results.tar.gz")


def count_results() -> int:
    return sum(1 for _, _, fs in os.walk(f"{OUT}/results") for f in fs
               if f.startswith("split-") and f.endswith(".json"))


t_start = time.time()
os.makedirs(WORK, exist_ok=True)
os.makedirs(f"{OUT}/results", exist_ok=True)
os.chdir(WORK)
open(f"{OUT}/STARTED", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
pack()  # lets the supervisor see the VM is alive during the setup
for repo, commit in COMMITS.items():
    if not os.path.isdir(repo):
        sh(f"git clone -q https://github.com/teco-kit/{repo}.git")
    sh(f"git -C {repo} checkout -q {commit}")
    print(repo, sh(f"git -C {repo} log --oneline -1"), flush=True)
sh(f"{sys.executable} -m pip install -q --no-deps -e whar-datasets -e whar-models")
sh(f"{sys.executable} -m pip install -q {PINNED}")
print(sh(f"{sys.executable} -c \"import torch, pandas, numpy; print('torch', torch.__version__, "
         f"torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0), '| pandas', pandas.__version__, "
         f"'| numpy', numpy.__version__)\""), flush=True)

cfg = json.load(open(f"{INPUTS}/job.json"))
if cfg.get("playwright"):  # whar-datasets declares playwright; its opportunity downloader drives a browser
    sh(f"{sys.executable} -m pip install -q playwright")
    sh(f"{sys.executable} -m playwright install --with-deps chromium")
    print("playwright + chromium installed", flush=True)

with tarfile.open(f"{INPUTS}/code.tar.gz") as tf:
    tf.extractall(WORK, filter="data")
if os.path.exists(f"{INPUTS}/prior_results.tar.gz"):  # finished elsewhere -> skipped by the runner
    with tarfile.open(f"{INPUTS}/prior_results.tar.gz") as tf:
        tf.extractall(f"{OUT}/results", filter="data")

env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
       "WHAR_ARENA_SRC": f"{WORK}/whar-arena/src", "WHAR_DATASETS_DIR": f"{WORK}/datasets",
       "EDGE_SSM_RESULTS_DIR": f"{OUT}/results", "RESULTS_DIR": f"{OUT}/results", **cfg.get("env", {})}
n_gpus = int(sh(f"{sys.executable} -c \"import torch; print(torch.cuda.device_count())\"") or 1)
max_parallel = int(cfg.get("max_parallel", len(cfg["jobs"])))

# Dataset caches (download, windows, per-split normalization) are built by one process per dataset
# (up to warm_parallel at once) before any job of that dataset starts, so no two processes write one cache.
WARM = ("import sys\nsys.path.insert(0, '.')\nfrom src.training import run_whar as R\n"
        "ds = R.prepare_dataset(sys.argv[1])\n"
        "for i, s in enumerate(ds['splits']):\n"
        "    R.prepare_split(ds, s)\n"
        "    print('cache ready', sys.argv[1], i, len(s.train_indices), len(s.val_indices), len(s.test_indices), flush=True)\n")
to_warm = list(dict.fromkeys(j["dataset"] for j in cfg["jobs"] if "dataset" in j))
ready, failed, warming = set(), set(), {}
warm_parallel = int(cfg.get("warm_parallel", 3))  # different datasets have separate caches


def step_warmup() -> None:
    for name, proc in list(warming.items()):
        if proc.poll() is not None:
            (ready if proc.returncode == 0 else failed).add(name)
            print(f"[{(time.time() - t_start) / 60:.0f} min] cache {name}: "
                  f"{'ready' if proc.returncode == 0 else 'FAILED, its jobs are skipped'}", flush=True)
            del warming[name]
    while to_warm and len(warming) < warm_parallel:
        name = to_warm.pop(0)
        warming[name] = subprocess.Popen([sys.executable, "-c", WARM, name], cwd=WORK, env=env,
                                         stdout=open(f"{OUT}/warmup_{name}.log", "w"), stderr=subprocess.STDOUT)


def command(job: dict) -> list[str]:
    if "module" in job:  # a project module, e.g. {"module": "src.models.classical.elastic_baseline", "args": [...]}
        return [sys.executable, "-m", job["module"], *job.get("args", [])]  # cwd is WORK, the project root
    if "script" in job:  # a file relative to the project root, e.g. {"script": "scripts/train.py", "args": [...]}
        return [sys.executable, f"{WORK}/{job['script']}", *job.get("args", [])]
    cmd = [sys.executable, f"{WORK}/scripts/train.py", "--models", *job["models"], "--datasets", job["dataset"],
           "--seed", str(job.get("seed", 0))]
    if job.get("splits") is not None:
        cmd += ["--splits", *map(str, job["splits"])]
    return cmd


pending = list(enumerate(cfg["jobs"]))  # in the given order (the job builder puts long jobs first)
running: dict[int, subprocess.Popen] = {}
codes: dict[int, int] = {}
last_pack, slot = 0.0, 0
while pending or running or warming:
    step_warmup()
    for i, p in list(running.items()):
        if p.poll() is not None:
            codes[i] = p.returncode
            del running[i]
    for i, job in list(pending):
        if len(running) >= max_parallel:
            break
        if "dataset" in job and job["dataset"] in failed:
            pending.remove((i, job))
            codes[i] = -1
            continue
        if "dataset" in job and job["dataset"] not in ready:
            continue
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = str(job["gpu"] if job.get("gpu") is not None else slot % max(n_gpus, 1))
        slot += 1
        cmd = command(job)
        running[i] = subprocess.Popen(cmd, cwd=WORK, stdout=open(f"{OUT}/train_{i}.log", "a"),
                                      stderr=subprocess.STDOUT, env=e)
        pending.remove((i, job))
        print(f"[{(time.time() - t_start) / 60:.0f} min] job {i} on gpu {e['CUDA_VISIBLE_DEVICES']}: "
              f"{' '.join(cmd[2:])}", flush=True)
    if time.time() - last_pack > PACK_EVERY:
        pack()
        last_pack = time.time()
        print(f"[{(time.time() - t_start) / 60:.0f} min] result files: {count_results()}, running: {sorted(running)}, "
              f"pending: {len(pending)}", flush=True)
    if KAGGLE and time.time() - t_start > DEADLINE:
        for p in running.values():
            p.terminate()
        print(f"deadline reached; stopped {sorted(running)}, skipped {len(pending)} pending jobs "
              "(finished splits are kept)", flush=True)
        break
    time.sleep(10)
open(f"{OUT}/DONE", "w").write(json.dumps({"codes": codes, "failed_datasets": sorted(failed)}))
pack()
print("ALL JOBS FINISHED; exit codes:", codes, "| failed datasets:", sorted(failed), flush=True)
