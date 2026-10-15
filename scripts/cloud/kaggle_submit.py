"""Run a job file as a Kaggle Notebooks batch run (2x T4) and merge its outputs into ./results.

Kaggle takes one code file per kernel. The kaggle-gpu skill's driver (kaggle_run.py) embeds
remote_job.py together with code.tar.gz, job.json and prior_results.tar.gz, and runs it on the VM;
results come back from /kaggle/working when the run ends (or at the 11 h stop in remote_job.py).
Kernels are private.

    uv run python scripts/cloud/kaggle_submit.py push  configs/cloud_jobs/kaggle_p4.json edge-ssm-p4
    uv run python scripts/cloud/kaggle_submit.py fetch configs/cloud_jobs/kaggle_p4.json edge-ssm-p4 [--wait]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import job_summary, load_job, merge, pack_code, pack_prior  # noqa: E402

HERE = Path(__file__).resolve().parent
DRIVER = Path(os.environ.get("KAGGLE_RUN", Path.home() / ".claude/skills/kaggle-gpu/scripts/kaggle_run.py"))


def main() -> None:
    action, job_file, slug = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    work = HERE / "work" / slug
    work.mkdir(parents=True, exist_ok=True)
    if action == "push":
        job = load_job(job_file)
        shutil.copy2(job_file, work / "job.json")
        files = [pack_code(work / "code.tar.gz"), work / "job.json", pack_prior(job, work / "prior_results.tar.gz")]
        subprocess.run([sys.executable, str(DRIVER), "--slug", slug, "--script", str(HERE / "remote_job.py"),
                        "--embed", *map(str, files), "--accelerator", "NvidiaTeslaT4"], check=True)
    elif action == "fetch":
        out = work / "output"
        shutil.rmtree(out, ignore_errors=True)
        extra = ["--wait"] if "--wait" in sys.argv else []
        subprocess.run([sys.executable, str(DRIVER), "--slug", slug, "--fetch-only", "--out", str(out),
                        "--poll", "300", *extra], check=True)
        print(f"{merge(out, slug)} new result files | {job_summary(out)}")
    else:
        raise SystemExit(f"unknown action {action}")


if __name__ == "__main__":
    main()
