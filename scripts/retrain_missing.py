"""Retrain the stored S1 runs that have no checkpoint anywhere, so that every run has one (the user's request of
2026-10-16). Same harness, model config, data cache, split and seed as the stored run; the GPU differs (the stored
runs trained on a GTX 1080 Ti or a cloud GPU), so the new weights are not bit-identical.

Nothing stored is touched: the harness writes the retrained run record and checkpoint under this run's own directory
(RESULTS_DIR = experiments/<run_id>), and results/metrics/reproducibility/retrained.json compares each retrained run
with the stored record. The stored records stay the ones every analysis reads.

    WHAR_DATASETS_DIR=<private cache copy> uv run --no-sync python scripts/retrain_missing.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import S1, cells, run_path  # noqa: E402
from src.utils import paths as PATHS  # noqa: E402
from src.utils import provenance as PV  # noqa: E402


def main() -> None:
    missing = [(m, d, s, i) for m, d, s, i in cells(S1)
               if not (PATHS.CHECKPOINTS / m / d / f"seed-{s}" / f"split-{i:02d}.pt").exists()]
    run_id = PV.new_run_id("retrain_missing_checkpoints")
    run_dir = PATHS.EXPERIMENTS / run_id
    run_dir.mkdir()
    os.environ["RESULTS_DIR"] = str(run_dir)  # read by run_whar at import: the outputs stay in this run's directory
    from src.training import run_whar as R
    import torch
    config = {"cells": [list(c) for c in missing], "datasets_dir": str(R.DATASETS_DIR), "results_dir": str(run_dir),
              "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    started = time.time()
    PV.ledger_start(run_id, config)
    try:
        groups = {}
        for m, d, s, i in missing:
            groups.setdefault((m, d, s), []).append(i)
        for (m, d, s), splits in groups.items():
            R.run(m, d, max_epochs=None, patience=None, split_filter=set(splits), seed=s, need_checkpoint=True)
        rows = []
        for m, d, s, i in missing:
            new_rec = run_dir / "whar_arena_v1" / m / d / (f"seed-{s}" if s else "") / f"split-{i:02d}.json"
            new_ck = run_dir / "checkpoints" / m / d / f"seed-{s}" / f"split-{i:02d}.pt"
            old = json.loads(run_path(S1, m, d, s, i).read_text())
            new = json.loads(new_rec.read_text())
            rows.append({"model": m, "dataset": d, "seed": s, "split": i,
                         "retrained_record": str(new_rec.relative_to(ROOT)), "checkpoint": str(new_ck.relative_to(ROOT)),
                         "checkpoint_exists": new_ck.exists(), "fingerprint_equal": new["fingerprint"] == old["fingerprint"],
                         "config_equal": new["model"]["config"] == old["model"]["config"],
                         "test_size_equal": new["source"]["test_size"] == old["source"]["test_size"],
                         "stored_device": old.get("device"), "retrained_device": new.get("device"),
                         "macro_f1": new["metrics"]["macro_f1"], "stored_macro_f1": old["metrics"]["macro_f1"],
                         "delta_macro_f1_points": 100 * (new["metrics"]["macro_f1"] - old["metrics"]["macro_f1"]),
                         "best_epoch": new["source"]["best_epoch"], "stored_best_epoch": old["source"]["best_epoch"]})
        out = PATHS.METRICS / "reproducibility" / "retrained.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"run_id": run_id, "note": "retrained to give every stored run a checkpoint; the "
                                   "stored records remain the analysed ones", "rows": rows}, indent=1) + "\n")
        PV.write(out, run_id=run_id, started=started, inputs=[run_path(S1, *c) for c in missing], config=config,
                 produced_by="scripts/retrain_missing.py")
        for r in rows:
            print(f"{r['model']}/{r['dataset']} seed {r['seed']} split {r['split']}: retrained {100 * r['macro_f1']:.2f} "
                  f"vs stored {100 * r['stored_macro_f1']:.2f} ({r['delta_macro_f1_points']:+.2f}); "
                  f"config {r['config_equal']}, fingerprint {r['fingerprint_equal']}", flush=True)
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    PV.ledger_end(run_id, f"{len(missing)} runs retrained for their checkpoints", started)


if __name__ == "__main__":
    main()
