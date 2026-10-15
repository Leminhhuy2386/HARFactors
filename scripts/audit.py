"""The user's single entry point for checking this project without trusting its author (agents/01_integrity.md).

    uv run --no-sync python scripts/audit.py --all              # every check implemented so far
    uv run --no-sync python scripts/audit.py --provenance --ledger --data --synthetic

Each check prints PASS or FAIL with its reasons. Exit code 1 if any check fails. Checks that later stages add are listed
as SKIPPED until they exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_DIRS = ["results/metrics", "results/predictions", "results/figure_data", "results/table_data"]
IMPORTED_DIRS = {"checkpoints", "elastic", "error_analysis", "external_har_paper_era", "logs", "person_probe",
                 "style_probe", "tree_probe", "value_probe", "whar_arena_v1"}  # experiments/: IMPORT block of the ledger
RUN_ID = re.compile(r"^\d{8}_\d{6}_[A-Za-z0-9_.-]+$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_files() -> list[Path]:
    files = [p for d in RESULT_DIRS for p in (ROOT / d).rglob("*") if p.is_file()]
    for run in (ROOT / "experiments").iterdir():
        if run.is_dir() and RUN_ID.match(run.name):
            files += [p for sub in ("metrics", "predictions") for p in (run / sub).rglob("*") if p.is_file()]
    return sorted(p for p in files if not p.name.endswith(".provenance.json") and p.name != ".gitkeep")


def g2_manifest() -> str | None:
    g = ROOT / "results/gates/G2_data.json"
    return json.loads(g.read_text())["observed_value"]["data_manifest_sha256"] if g.exists() else None


# ------------------------------------------------------------------------------------------------- checks
def check_provenance() -> list[str]:
    """Every computed result has a provenance record whose output and input hashes match the files on disk."""
    errs = []
    for f in result_files():
        pv = f.with_name(f.name + ".provenance.json")
        if not pv.exists():
            errs.append(f"no provenance: {f.relative_to(ROOT)}")
            continue
        rec = json.loads(pv.read_text())
        if rec.get("output_sha256") != sha256(f):
            errs.append(f"output hash mismatch: {f.relative_to(ROOT)}")
        for inp, h in rec.get("input_files", {}).items():
            p = ROOT / inp
            if not p.exists():
                errs.append(f"input missing: {inp} (for {f.relative_to(ROOT)})")
            elif "sha256:" + sha256(p) != h:
                errs.append(f"input changed since: {inp} (for {f.relative_to(ROOT)})")
    return errs


def check_ledger() -> list[str]:
    """Ledger, run files and run directories agree exactly; every START has an END."""
    errs = []
    lines = (ROOT / "experiments/LEDGER.md").read_text(encoding="utf-8").splitlines()
    imported = {l.split(" | ")[0].strip() for l in lines if " | IMPORT | " in l}
    on_disk = {str(p.relative_to(ROOT / "experiments").with_suffix("")).replace("\\", "/")
               for p in (ROOT / "experiments/whar_arena_v1").rglob("split-*.json")}
    errs += [f"run file not in ledger: {r}" for r in sorted(on_disk - imported)]
    errs += [f"ledger entry without run file: {r}" for r in sorted(imported - on_disk)]
    first = lambda l: l.split(" | ")[0].strip()  # noqa: E731
    starts = [first(l) for l in lines if " | START | " in l and RUN_ID.match(first(l))]
    ends = {first(l) for l in lines if " | END " in l and RUN_ID.match(first(l))}
    errs += [f"START without END: {r}" for r in starts if r not in ends]
    errs += [f"run listed twice: {r}" for r in {r for r in starts if starts.count(r) > 1}]
    for d in sorted((ROOT / "experiments").iterdir()):
        if not d.is_dir() or d.name in IMPORTED_DIRS:
            continue
        if not RUN_ID.match(d.name):
            errs.append(f"directory is neither imported nor a run_id: experiments/{d.name}")
        elif d.name not in starts:
            errs.append(f"run directory not in ledger: experiments/{d.name}")
    for r in starts:
        if not (ROOT / "experiments" / r).is_dir():
            errs.append(f"ledger run without directory: experiments/{r}")
    return errs


def check_data() -> list[str]:
    """The manifest re-hashes to G2's value, and every provenance record carries that value."""
    errs = []
    want = g2_manifest()
    if want is None:
        return ["results/gates/G2_data.json missing"]
    man = json.loads((ROOT / "data/processed/manifest.json").read_text())
    for path, h in man["entries"].items():
        p = ROOT / path
        if not p.exists():
            errs.append(f"data file missing: {path}")
        elif sha256(p) != h:
            errs.append(f"data file changed: {path}")
    listing = "".join(f"{k}  {v}\n" for k, v in sorted(man["entries"].items()))
    if hashlib.sha256(listing.encode()).hexdigest() != want or man["data_manifest_sha256"] != want:
        errs.append("manifest does not hash to G2's data_manifest_sha256")
    for f in result_files():
        pv = f.with_name(f.name + ".provenance.json")
        if pv.exists() and json.loads(pv.read_text()).get("data_manifest_sha256") != want:
            errs.append(f"provenance with another data manifest: {f.relative_to(ROOT)}")
    return errs


def check_synthetic() -> list[str]:
    """No reported result descends from data/synthetic/."""
    errs = []
    for f in result_files():
        pv = f.with_name(f.name + ".provenance.json")
        if not pv.exists():
            continue
        rec = json.loads(pv.read_text())
        if rec.get("synthetic"):
            errs.append(f"synthetic result: {f.relative_to(ROOT)}")
        errs += [f"input from data/synthetic: {f.relative_to(ROOT)}" for i in rec.get("input_files", {})
                 if i.startswith("data/synthetic")]
    return errs


CHECKS = {"provenance": check_provenance, "ledger": check_ledger, "data": check_data, "synthetic": check_synthetic}
LATER = ["recompute", "reproduce", "seeds", "authenticity", "traceability", "manuscript"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    for name in [*CHECKS, *LATER]:
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--max-errors", type=int, default=20)
    a = ap.parse_args()
    chosen = [n for n in CHECKS if a.all or getattr(a, n)]
    failed = False
    for n in chosen:
        errs = CHECKS[n]()
        failed |= bool(errs)
        print(f"{'FAIL' if errs else 'PASS'}  --{n}" + (f"  ({len(errs)} problems)" if errs else ""))
        for e in errs[:a.max_errors]:
            print("      ", e)
        if len(errs) > a.max_errors:
            print(f"       ... and {len(errs) - a.max_errors} more")
    for n in LATER:
        if a.all or getattr(a, n):
            print(f"SKIPPED --{n}  (not implemented yet: added by a later stage)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
