"""Local side of the cloud runs: pack the code and the finished results for a job, merge pulled results.

A job file (configs/cloud_jobs/*.json) lists run_whar.py invocations, or other project scripts:
  {"jobs": [{"models": [...], "dataset": "utd_mhad", "seed": 1, "splits": [0, 1], "gpu": 0},
            {"script": "synthetic_test.py", "args": ["--models", "quantilenet"], "gpu": 1}, ...]}
Give each machine a disjoint set of (model, dataset, seed, split): the runner only skips results it
can see, so overlapping assignments would train the same thing twice.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
LOCAL_RESULTS = PROJECT / "experiments"  # run results and checkpoints
PROTOCOL_ID = "whar_arena_v1"


def pack_code(out: Path) -> Path:
    with tarfile.open(out, "w:gz") as tf:
        # the training harness, the models, the classical baselines and the train entry point
        files = [PROJECT / "scripts" / "train.py", *sorted((PROJECT / "src").glob("*.py"))]
        for area in ("utils", "training", "models"):
            files += sorted((PROJECT / "src" / area).rglob("*.py"))
        for f in files:
            tf.add(f, arcname=f.relative_to(PROJECT).as_posix())
    return out


def pack_prior(job: dict, out: Path) -> Path:
    """Result files already finished locally for the job's (model, dataset, seed)."""
    with tarfile.open(out, "w:gz") as tf:
        for j in job["jobs"]:
            if "script" in j:  # other scripts skip their own finished runs; give them disjoint work
                continue
            seed = j.get("seed", 0)
            for m in j["models"]:
                d = LOCAL_RESULTS / PROTOCOL_ID / m / j["dataset"]
                d = d / f"seed-{seed}" if seed else d
                for f in sorted(d.glob("split-*.json")) if d.exists() else []:
                    tf.add(f, arcname=str(f.relative_to(LOCAL_RESULTS)))
    return out


def merge(extract: Path, tag: str, log=print) -> int:
    """Copy new result and checkpoint files from an extracted output dir into ./results; keep logs."""
    new = 0
    src_root = extract / "results"
    for src in sorted(src_root.rglob("*")) if src_root.exists() else []:
        if src.suffix not in (".json", ".pt") or src.name.startswith("summary"):
            continue
        dst = LOCAL_RESULTS / src.relative_to(src_root)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if src.suffix == ".json":
            new += 1
            log(f"  merged {dst.relative_to(LOCAL_RESULTS)}")
    logs = PROJECT / "experiments" / "logs" / "cloud" / tag
    logs.mkdir(parents=True, exist_ok=True)
    for lg in list(extract.glob("train_*.log")) + list(extract.glob("warmup.log")):
        shutil.copy2(lg, logs / lg.name)
    return new


def job_summary(extract: Path) -> str:
    heads = []
    for lg in sorted(extract.glob("train_*.log")):
        text = lg.read_text(errors="replace")
        lines = [l for l in text.splitlines() if l.startswith("=== ")]
        heads.append(lines[-1][4:90] if lines else "-")
        if "Traceback" in text:
            heads.append(f"TRACEBACK in {lg.name}")
    return " | ".join(heads)


def load_job(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
