"""Provenance records (agents/01_integrity.md, § Provenance record) and the run ledger lines for analysis runs.

Every computed result file gets a sibling <name>.provenance.json written by `write`. Analysis runs append START and END
lines to experiments/LEDGER.md through `ledger_start` and `ledger_end`.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from src.utils import paths as PATHS

LEDGER = PATHS.EXPERIMENTS / "LEDGER.md"
MANIFEST = PATHS.DATA / "processed" / "manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(config: dict | None) -> str | None:
    return None if config is None else hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def git_commit() -> str:
    head = subprocess.run(["git", "-C", str(PATHS.ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
    if head.returncode:
        return "none"
    dirty = subprocess.run(["git", "-C", str(PATHS.ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return head.stdout.strip() + ("-dirty" if dirty else "")


def data_manifest_sha256() -> str:
    return json.loads(MANIFEST.read_text())["data_manifest_sha256"]


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(PATHS.ROOT))


def write(output: Path, *, run_id: str, started: float, inputs: list[Path], config: dict | None = None,
          produced_by: str | None = None, synthetic: bool = False) -> Path:
    """Write <output>.provenance.json next to `output` (after `output` itself is written)."""
    output = Path(output)
    rec = {
        "produced_by": produced_by or (rel(Path(sys.argv[0])) if Path(sys.argv[0]).exists() else sys.argv[0]),
        "command": " ".join(shlex.quote(a) for a in [Path(sys.executable).name, *sys.argv]),
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_sec": round(time.time() - started, 2),
        "git_commit": git_commit(),
        "config_sha256": config_sha256(config),
        "config": config,
        "data_manifest_sha256": data_manifest_sha256(),
        "input_files": {rel(p): "sha256:" + sha256(p) for p in sorted(set(map(Path, inputs)))},
        "output_sha256": sha256(output),
        "synthetic": synthetic,
    }
    out = output.with_name(output.name + ".provenance.json")
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return out


def new_run_id(shortname: str) -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + shortname


def _append(line: str) -> None:  # the ledger was written on Windows: keep its CRLF line endings
    with open(LEDGER, "a", encoding="utf-8", newline="") as f:
        f.write(line + "\r\n")


def ledger_start(run_id: str, config: dict | None, synthetic: bool = False) -> None:
    sha = config_sha256(config) or "-"
    _append(f"{run_id} | START | config {sha[:12]} | data {data_manifest_sha256()[:12]} | synthetic={str(synthetic).lower()}")


def ledger_end(run_id: str, summary: str, started: float, status: str = "completed") -> None:
    dur = time.strftime("%H:%M:%S", time.gmtime(time.time() - started))
    _append(f"{run_id} | END   | {summary} | {dur} | status={status}")
