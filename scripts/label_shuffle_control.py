"""Label-shuffle control (agents/05_verify.md, Level 3.5.3), approved by the user on 2026-10-16.

The harness trains S4D exactly as for its stored runs (paper-era protocol, seed 0), except that the labels of the
training windows are permuted (a fixed seeded permutation of the training rows: the multiset of labels, and so the
class weights of the loss, are unchanged). Validation and test keep their true labels. If information about the labels
reached the model through anything other than the training labels (an overlapping split, a leaked feature, a
normalisation fitted on everything), performance would stay above chance.

Chance level of each split: the mean macro-F1 of 100 random permutations of its true labels used as predictions
(same class frequencies, the harness metric).

Tolerance, fixed here before running: on every run, shuffled-label macro-F1 <= chance + 5 points on validation and on
test; and the stored true-label run of the same split is at least 20 points above the shuffled one on validation.

The runs use manipulated labels, so they are recorded as synthetic in the ledger and their outputs stay in their run
directory (never in results/). The verdict enters gate G5.

    WHAR_DATASETS_DIR=<private cache copy> uv run --no-sync python scripts/label_shuffle_control.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.training import run_whar as R  # noqa: E402
from src.utils import provenance as PV  # noqa: E402
from whar_arena import training as T  # noqa: E402

MODEL = "s4d"
CELLS = [("up_fall", 0), ("up_fall", 1), ("up_fall", 2), ("uca_ehar", 0), ("uca_ehar", 1), ("uca_ehar", 2)]
PERM_SEED = 20261016
CHANCE_PERMUTATIONS = 100
MARGIN_CHANCE = 5.0   # points above chance still counted as "collapsed"
MARGIN_REAL = 20.0    # the true-label run must be this far above, on validation
TOLERANCE = (f"every run: shuffled-label macro-F1 <= chance + {MARGIN_CHANCE} points on validation and on test; the "
             f"stored true-label run >= shuffled + {MARGIN_REAL} points on validation")


def chance(labels: torch.Tensor, k: int, seed: int) -> float:
    y = T._target_labels(labels).tolist()
    rng = np.random.default_rng(seed)
    return 100 * st.mean(T._classification_metrics(y, rng.permutation(y).tolist(), k)["macro_f1"]
                         for _ in range(CHANCE_PERMUTATIONS))


def main() -> None:
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    run_id = PV.new_run_id("label_shuffle_control")
    run_dir = ROOT / "experiments" / run_id
    run_dir.mkdir()
    config = {"model": MODEL, "cells": CELLS, "perm_seed": PERM_SEED, "chance_permutations": CHANCE_PERMUTATIONS,
              "tolerance": TOLERANCE, "datasets_dir": str(R.DATASETS_DIR), "train_seed": 0}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    started = time.time()
    PV.ledger_start(run_id, config, synthetic=True)
    rows = []
    try:
        for dataset in sorted({d for d, _ in CELLS}):
            ds = R.prepare_dataset(dataset)
            for d, i in CELLS:
                if d != dataset:
                    continue
                t0 = time.time()
                split = ds["splits"][i]
                loader, dl, shape = R.prepare_split(ds, split)
                y, x = dl["train"].dataset.tensors
                perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(PERM_SEED + i))
                shuffled = DataLoader(TensorDataset(y[perm], x), dl["train"].batch_size, True,
                                      generator=dl["train"].generator)
                model, m = R.fit(MODEL, ds, split, f"split-{i:02d}", loader,
                                 {"train": shuffled, "val": dl["val"], "test": dl["test"]}, shape)
                val = T._evaluate(model, dl["val"], device=next(model.parameters()).device, cfg=ds["cfg"],
                                  criterion=None)["macro_f1"]
                k = shape[2]
                stored = json.loads((ROOT / "experiments/whar_arena_v1" / MODEL / dataset / f"split-{i:02d}.json")
                                    .read_text())["metrics"]
                row = {"dataset": dataset, "split": i, "train_windows": len(y),
                       "shuffled_val_macro_f1": 100 * val, "shuffled_test_macro_f1": 100 * m.macro_f1,
                       "chance_val_macro_f1": chance(dl["val"].dataset.tensors[0], k, PERM_SEED),
                       "chance_test_macro_f1": chance(dl["test"].dataset.tensors[0], k, PERM_SEED),
                       "stored_val_macro_f1": 100 * stored["val_macro_f1"], "stored_test_macro_f1": 100 * stored["macro_f1"],
                       "best_epoch": m.best_epoch, "epochs_ran": m.epochs_ran, "seconds": round(time.time() - t0, 1)}
                row["collapsed"] = (row["shuffled_val_macro_f1"] <= row["chance_val_macro_f1"] + MARGIN_CHANCE
                                    and row["shuffled_test_macro_f1"] <= row["chance_test_macro_f1"] + MARGIN_CHANCE)
                row["real_far_above"] = row["stored_val_macro_f1"] >= row["shuffled_val_macro_f1"] + MARGIN_REAL
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = {"run_id": run_id, "model": MODEL, "runs": len(rows), "tolerance": TOLERANCE,
                   "all_collapsed": all(r["collapsed"] for r in rows),
                   "all_real_far_above": all(r["real_far_above"] for r in rows),
                   "mean_shuffled_test_macro_f1": st.mean(r["shuffled_test_macro_f1"] for r in rows),
                   "mean_chance_test_macro_f1": st.mean(r["chance_test_macro_f1"] for r in rows),
                   "mean_stored_test_macro_f1": st.mean(r["stored_test_macro_f1"] for r in rows),
                   "mean_shuffled_val_macro_f1": st.mean(r["shuffled_val_macro_f1"] for r in rows),
                   "mean_chance_val_macro_f1": st.mean(r["chance_val_macro_f1"] for r in rows),
                   "mean_stored_val_macro_f1": st.mean(r["stored_val_macro_f1"] for r in rows),
                   "rows": rows}
        summary["caught"] = summary["all_collapsed"] and summary["all_real_far_above"]
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        PV.ledger_end(run_id, f"collapsed={summary['all_collapsed']} shuffled test F1 "
                              f"{summary['mean_shuffled_test_macro_f1']:.2f} vs chance "
                              f"{summary['mean_chance_test_macro_f1']:.2f}", started)
        print("summary", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise


if __name__ == "__main__":
    main()
