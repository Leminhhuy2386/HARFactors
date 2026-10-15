"""Run a job file on a Colab GPU VM and merge its results into ./results as they appear
(replaces colab/supervise.py, adapted from D:\\Projects\\HAR\\paper-era\\colab\\supervise_colab.py).

Per cycle: create a VM (first GPU in GPUS that is granted), upload the code, the job and the
results already finished locally for the job (the runner skips them), run remote_job.py in the
foreground (a busy kernel is not culled), pull results.tar.gz every POLL seconds and copy new files
into ./results. If the VM disappears, start a new cycle with whatever is finished by then. The VM
is always stopped on exit.

    uv run python scripts/cloud/colab_supervise.py configs/cloud_jobs/colab_p4.json [prefix]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import job_summary, load_job, merge, pack_code, pack_prior  # noqa: E402

HERE = Path(__file__).resolve().parent
POLL = 180
MAX_CYCLES = 6
# the models are small and launch-bound: an A100 would not be faster. COLAB_GPUS=T4 pins the hardware,
# e.g. to finish a seed on the GPU type it started on.
GPUS = os.environ.get("COLAB_GPUS", "L4,T4").split(",")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
JOB_FILE = Path(sys.argv[1])
PREFIX = sys.argv[2] if len(sys.argv) > 2 else JOB_FILE.stem.replace("_", "-")
WORK = HERE / "work" / PREFIX


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def colab(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["colab", *args], env=ENV, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", "timeout")


def pull(session: str) -> tuple[bool, bool, str]:
    """Download and merge. Returns (download ok, job done, summary)."""
    tar_path = WORK / "results.tar.gz"
    if colab("download", "-s", session, "/content/edge/results.tar.gz", str(tar_path)).returncode != 0:
        return False, False, ""
    extract = WORK / "extracted"
    shutil.rmtree(extract, ignore_errors=True)
    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract, filter="data")
    except (tarfile.TarError, EOFError):
        return True, False, "tar not readable yet"
    new = merge(extract, PREFIX, log)
    return True, (extract / "DONE").exists(), f"{new} new | {job_summary(extract)}"


def new_session(name: str) -> str | None:
    for gpu in GPUS:
        r = colab("new", "-s", name, "--gpu", gpu)
        if r.returncode == 0:
            log(f"session {name}: got {gpu}")
            return gpu
        log(f"colab new --gpu {gpu} failed: {(r.stdout + r.stderr).strip()[-300:]}")
    return None


def run_cycle(cycle: int, job: dict) -> bool:
    """True when the job completed (DONE seen)."""
    session = f"{PREFIX}{cycle}"
    if new_session(session) is None:
        return False
    exec_proc = None
    try:
        mk = WORK / "_mk.py"
        mk.write_text("import os; os.makedirs('/content/edge', exist_ok=True)\n")
        colab("exec", "-s", session, "--timeout", "120", "-f", str(mk))
        uploads = [(pack_code(WORK / "code.tar.gz"), "/content/edge/code.tar.gz"),
                   (JOB_FILE, "/content/edge/job.json"),
                   (pack_prior(job, WORK / "prior_results.tar.gz"), "/content/edge/prior_results.tar.gz")]
        for local, remote in uploads:
            if colab("upload", "-s", session, str(local), remote).returncode != 0:
                log(f"upload of {local.name} failed")
                return False
        exec_proc = subprocess.Popen(
            ["colab", "exec", "-s", session, "--timeout", "43200", "-f", str(HERE / "remote_job.py")],
            env=ENV, stdin=subprocess.DEVNULL, stdout=open(WORK / f"exec_{session}.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        log(f"cycle {cycle}: session {session} running the job")
        failures = 0
        while True:
            time.sleep(POLL)
            ok, done, summary = pull(session)
            if ok:
                failures = 0
                log(f"cycle {cycle}: {summary}")
                if done:
                    return True
            else:
                failures += 1
                log(f"cycle {cycle}: download failed ({failures})")
                if failures >= 3:
                    log(f"cycle {cycle}: VM appears gone; starting a new cycle")
                    return False
            if exec_proc.poll() is not None and not done:
                tail = (WORK / f"exec_{session}.log").read_text(encoding="utf-8", errors="replace")[-3000:]
                fatal = ("SystemExit", "RuntimeError:", "ModuleNotFoundError", "SyntaxError", "ImportError")
                if any(f in tail for f in fatal):  # the job itself failed: a new VM would fail the same way
                    # (network errors of `colab exec`, e.g. ReadTimeout, only start a new cycle)
                    raise SystemExit(f"cycle {cycle}: remote job failed, see exec_{session}.log:\n{tail[-800:]}")
                log(f"cycle {cycle}: exec ended without DONE; starting a new cycle")
                return False
    finally:
        if exec_proc is not None:
            exec_proc.kill()
        colab("stop", "-s", session)
        log(f"cycle {cycle}: stopped {session}")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    job = load_job(JOB_FILE)
    for cycle in range(1, MAX_CYCLES + 1):
        if run_cycle(cycle, job):
            log("job finished")
            return
        time.sleep(30)
    log("giving up after max cycles")


if __name__ == "__main__":
    main()
