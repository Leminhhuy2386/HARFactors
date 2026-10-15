"""Build a paper-era cache for a dataset that HAR_DIR/datasets_paper_era does not hold yet, from the raw files the
HAR reproduction downloaded into HAR_DIR/datasets/<id>/data. Copied from Edge_SSM's prepare_local_datasets.py; only
the target changed (the shared datasets_paper_era instead of a project folder).

What it writes, and what it never touches:
- datasets_paper_era/<id>/data/ is a new directory of hard links to HAR's raw files (no copy, no extra disk space).
  HAR's own step-hash files are not linked. Nothing that exists under HAR is changed; a new folder is added.
- HAR's processed cache in datasets/ (sessions/windows, built by the latest code) is never read.
- The paper-era download and extraction steps key their hash on {dataset_id, download_url} only, so this script
  writes the paper-era hashes into data/; the paper-era pipeline then skips downloading and extracting, and parses
  and windows the raw files itself.
- A manifest (path, size, mtime) of HAR's raw files is compared before and after: hard links share bytes, so a write
  through one would change the raw file too.

Check of equivalence: the train/val/test sizes of every split must equal those of the stored recurrencenet results
(built on cloud VMs from a fresh download, or on this PC). Parsers that list files unsorted (dsads, daphnet, usc_had,
motion_sense) can number subjects differently on Windows and Linux; then only dataset means compare.

Usage: uv run python scripts/prepare_data.py pamap2 uma_fall
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from src.training import run_whar
from src.training.run_whar import EXTRA_DATASETS_DIR, HERE
from whar_arena import training
from whar_arena.protocol import load_protocol
from whar_datasets import PreProcessingPipeline, WHARDatasetID
from src.utils import paths as PATHS  # noqa: E402

HAR_RAW = run_whar.RAW_DIR
HASH_FILES = {"download_hash.txt", "extracting_hash.txt"}


def manifest(root: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def link_raw(dataset_id: str) -> int:
    src, dst = HAR_RAW / dataset_id / "data", EXTRA_DATASETS_DIR / dataset_id / "data"
    n = 0
    for dirpath, _, files in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            if rel == Path(".") and name in HASH_FILES:
                continue
            target = dst / rel / name
            if not target.exists():
                os.link(Path(dirpath) / name, target)
            n += 1
    return n


def build(dataset_id: str) -> None:
    protocol = load_protocol()
    run_whar.check_paper_era(protocol)
    cfg = training._configure_dataset_cfg(protocol=protocol, dataset_id=WHARDatasetID(dataset_id),
                                          datasets_dir=EXTRA_DATASETS_DIR, max_epochs=None)
    pre = PreProcessingPipeline(cfg)
    for step in (pre.downloading_step, pre.extracting_step):  # raw files are in place: mark them as such
        step._save_hash(step._compute_hash())
    run_whar.seed_everything(cfg.seed)
    pre.run()


def check_splits(dataset_id: str) -> bool:
    ds = run_whar.prepare_dataset(dataset_id)
    assert ds["datasets_dir"] == str(EXTRA_DATASETS_DIR), ds["datasets_dir"]
    ok = True
    for i, split in enumerate(ds["splits"]):
        ref = PATHS.RUNS / "recurrencenet" / dataset_id / f"split-{i:02d}.json"
        got = (len(split.train_indices), len(split.val_indices), len(split.test_indices))
        if not ref.exists():
            print(f"  split-{i:02d}: {got} (no P5 reference)")
            continue
        src = json.loads(ref.read_text())["source"]
        want = (src["train_size"], src["val_size"], src["test_size"])
        ok &= got == want
        print(f"  split-{i:02d}: local {got}  P5 {want}  {'ok' if got == want else 'DIFFERENT'}")
    return ok


def main() -> None:
    for dataset_id in sys.argv[1:]:
        print(f"##### {dataset_id}", flush=True)
        before = manifest(HAR_RAW / dataset_id)
        print(f"  linked {link_raw(dataset_id)} raw files", flush=True)
        build(dataset_id)
        after = manifest(HAR_RAW / dataset_id)
        if before != after:
            changed = sorted(set(before.items()) ^ set(after.items()))[:5]
            raise RuntimeError(f"HAR files changed for {dataset_id}: {changed}")
        print("  HAR raw files unchanged", flush=True)
        print(f"  splits match P5: {check_splits(dataset_id)}", flush=True)


if __name__ == "__main__":
    main()
