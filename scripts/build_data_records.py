"""Data records of agents/03_data.md, Step 2, for the stored runs and the quoted table this project analyses.

- data/processed/splits.json: per dataset and split, the identifier and train/val/test window counts every stored run
  carries (S1, S2, S3 and every other run in experiments/), whether all runs agree, and the test subjects and counts
  the harness regenerates from the caches on this machine (data/processed/harness_splits.json).
  It records the harness's splits; LKSOSplitter defines them.
- data/processed/manifest.json: the sha256 of every data file the analyses read (run records, the quoted table, the
  authors' snapshot and result files), and `data_manifest_sha256` over the sorted list. Every provenance record carries
  that value.

Reads no metric. Run from the project root:  uv run --no-sync python scripts/build_data_records.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import S1, S2, S3, run_files  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
# every data file an analysis reads, as globs relative to the project root
DATA_GLOBS = [
    "experiments/whar_arena_v1/**/*.json",
    "experiments/external_har_paper_era/whar_arena_v1/**/*.json",
    "docs/sources/arena_extracted.json",
    "third_party/whar-arena/apps/web/src/generated/paper-benchmark.json",
    "third_party/whar-arena/data/results/whar_arena_v1/**/split-*.json",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest() -> dict:
    files = sorted({p for g in DATA_GLOBS for p in ROOT.glob(g) if p.is_file()})
    entries = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    listing = "".join(f"{k}  {v}\n" for k, v in sorted(entries.items()))
    return {"globs": DATA_GLOBS, "files": len(entries),
            "data_manifest_sha256": hashlib.sha256(listing.encode()).hexdigest(),
            "how": "sha256 over the lines '<path>  <sha256>\\n' sorted by path", "entries": entries}


def splits() -> dict:
    sizes = defaultdict(lambda: defaultdict(set))  # dataset -> split -> {(identifier, train, val, test)}
    runs = defaultdict(lambda: defaultdict(int))
    for f in sorted(ROOT.glob("experiments/whar_arena_v1/**/split-*.json")) + \
            sorted(ROOT.glob("experiments/external_har_paper_era/whar_arena_v1/**/split-*.json")):
        r = json.loads(f.read_text())
        s = r["source"]
        key = (s.get("split_identifier"), s["train_size"], s["val_size"], s["test_size"])
        sizes[r["dataset_id"]][r["split_id"]].add(key)
        runs[r["dataset_id"]][r["split_id"]] += 1
    harness_file = PROCESSED / "harness_splits.json"
    harness = json.loads(harness_file.read_text())["datasets"] if harness_file.exists() else {}
    out = {}
    for d in sorted(sizes):
        per = {}
        for sp in sorted(sizes[d]):
            stored = sorted(sizes[d][sp], key=str)
            h = harness.get(d, {}).get("splits", {}).get(sp)
            entry = {"stored_runs": runs[d][sp],
                     "stored_sizes": [dict(zip(["identifier", "train_size", "val_size", "test_size"], k)) for k in stored],
                     "stored_runs_agree": len(stored) == 1}
            if h:
                entry["harness"] = h
                entry["harness_matches_stored"] = len(stored) == 1 and stored[0] == (
                    h["identifier"], h["train_size"], h["val_size"], h["test_size"])
            per[sp] = entry
        out[d] = {"splits": per, "subjects": harness.get(d, {}).get("subjects"),
                  "cache_on_this_machine": d in harness}
    sets = {name: {"description": desc, "runs": len(run_files(spec))} for name, spec, desc in
            [("S1", S1, "development population"), ("S2", S2, "held-out (confirmatory)"),
             ("S3", S3, "independent architectures")]}
    return {"grouping_key": "subject_id: leave-subjects-out (LKSOSplitter, k = 10 subject groups); the validation "
                            "set is 20 % of the training subjects' windows drawn with cfg.seed = 0",
            "defined_by": "third_party/whar-datasets (144e006) LKSOSplitter via src/training/run_whar.prepare_dataset",
            "sets": sets, "datasets": out}


def main() -> None:
    m = manifest()
    s = splits()
    (PROCESSED / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    (PROCESSED / "splits.json").write_text(json.dumps(s, indent=1) + "\n")
    print("manifest files", m["files"], "data_manifest_sha256", m["data_manifest_sha256"])
    print("splits datasets", len(s["datasets"]), "sets", {k: v["runs"] for k, v in s["sets"].items()})


if __name__ == "__main__":
    main()
