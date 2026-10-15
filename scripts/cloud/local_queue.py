"""Run a job file on this PC with at most max_parallel processes (same format as remote_job.py).

Datasets must already be cached in WHAR_DATASETS_DIR (default: the paper-era cache); each dataset's
splits are prepared once, in order, before its jobs start. Logs go to logs/<job file stem>_<i>.log.
The PC has 16 GB RAM: keep max_parallel at 5 or less (more processes died with exit code 116).

    uv run python scripts/cloud/local_queue.py configs/cloud_jobs/local_p5.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    job_file = Path(sys.argv[1])
    cfg = json.loads(job_file.read_text(encoding="utf-8"))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **cfg.get("env", {})}
    max_parallel = int(cfg.get("max_parallel", 4))
    logs = PROJECT / "experiments" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for d in dict.fromkeys(j["dataset"] for j in cfg["jobs"] if "dataset" in j):
        warm = ("import sys\nsys.path.insert(0, '.')\nfrom src.training import run_whar as R\nds = R.prepare_dataset(sys.argv[1])\n"
                "[R.prepare_split(ds, s) for s in ds['splits']]\nprint('cache ready', sys.argv[1], len(ds['splits']))")
        r = subprocess.run([sys.executable, "-c", warm, d], cwd=PROJECT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")  # the child writes UTF-8 (progress bars)
        print(time.strftime("%H:%M:%S"), r.stdout.strip().splitlines()[-1] if r.returncode == 0 else f"cache {d} FAILED", flush=True)
    pending = list(enumerate(cfg["jobs"]))
    running: dict[int, subprocess.Popen] = {}
    while pending or running:
        for i, p in list(running.items()):
            if p.poll() is not None:
                print(time.strftime("%H:%M:%S"), f"job {i} exited {p.returncode}", flush=True)
                del running[i]
        while pending and len(running) < max_parallel:
            i, job = pending.pop(0)
            cmd = [sys.executable, str(PROJECT / "scripts" / "train.py"), "--models", *job["models"], "--datasets",
                   job["dataset"], "--seed", str(job.get("seed", 0))]
            if job.get("splits") is not None:
                cmd += ["--splits", *map(str, job["splits"])]
            running[i] = subprocess.Popen(cmd, cwd=PROJECT, env=env, stderr=subprocess.STDOUT,
                                          stdout=open(logs / f"{job_file.stem}_{i}.log", "a", encoding="utf-8"))
            print(time.strftime("%H:%M:%S"), f"job {i}: {' '.join(cmd[2:])}", flush=True)
        time.sleep(10)
    print(time.strftime("%H:%M:%S"), "ALL LOCAL JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()
