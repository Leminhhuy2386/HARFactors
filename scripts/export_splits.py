"""Record the harness's subject splits (agents/03_data.md, Step 2). The splits are defined by the paper-era
LKSOSplitter, not here: for every dataset whose paper-era cache is on this machine, this regenerates them exactly as
src/training/run_whar.prepare_dataset does and writes, per split, its identifier, the test subjects and the
train/val/test window counts. It also asserts that the shared cache is unchanged (size and mtime of every file).

    uv run --no-sync python scripts/export_splits.py        (on a compute node)  ->  data/processed/harness_splits.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import run_whar as R  # noqa: E402

OUT = ROOT / "data" / "processed" / "harness_splits.json"


def tree_manifest(d: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


def export(dataset_id: str) -> dict:
    cache = R.datasets_dir_for(dataset_id) / dataset_id
    before = tree_manifest(cache)
    ds = R.prepare_dataset(dataset_id)
    session_df, window_df = ds["session_df"], ds["window_df"]
    subject_of_session = dict(zip(session_df["session_id"], session_df["subject_id"]))
    splits = {}
    for i, split in enumerate(ds["splits"]):
        test_sessions = window_df.loc[split.test_indices, "session_id"].unique()
        splits[f"split-{i:02d}"] = {
            "identifier": split.identifier,
            "test_subjects": sorted({int(subject_of_session[s]) for s in test_sessions}),
            "train_size": len(split.train_indices), "val_size": len(split.val_indices),
            "test_size": len(split.test_indices),
        }
        # group-disjointness by construction, checked: no test subject's window is in train or val
        trainval = set(split.train_indices) | set(split.val_indices)
        assert not trainval & set(split.test_indices), (dataset_id, i)
        tv_subjects = {subject_of_session[s] for s in window_df.loc[list(trainval), "session_id"].unique()}
        assert not tv_subjects & set(splits[f"split-{i:02d}"]["test_subjects"]), (dataset_id, i)
    after = tree_manifest(cache)
    assert before == after, f"{dataset_id}: the harness changed the shared cache"
    return {"cache": str(cache), "subjects": int(session_df["subject_id"].nunique()), "windows": int(len(window_df)),
            "sessions": int(len(session_df)), "num_folds_cfg": int(ds["cfg"].num_folds), "seed": int(ds["cfg"].seed),
            "splits": splits}


def main() -> None:
    t0 = time.time()
    ids = sorted(p.name for p in R.DATASETS_DIR.iterdir() if (p / "windows").is_dir())
    out = {"produced_by": "scripts/export_splits.py", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "datasets_dir": str(R.DATASETS_DIR), "whar_code": R.check_paper_era(R.load_protocol()),
           "grouping_key": "subject_id (LKSOSplitter: subjects sorted, subject i -> fold i mod min(k, #subjects))",
           "datasets": {}}
    for d in ids:
        print("==", d, flush=True)
        out["datasets"][d] = export(d)
    out["duration_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), "datasets", len(ids))


if __name__ == "__main__":
    main()
