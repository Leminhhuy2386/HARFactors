"""Regenerate test predictions from stored checkpoints (agents/04_research.md, scripts/evaluate.py; used by
05_verify). No training.

For each requested (model, dataset, seed, split): the harness rebuilds the split exactly as for the stored run
(src/training/run_whar.prepare_dataset / prepare_split: the train-only normalisation included), the checkpoint's
weights are loaded into the registry model, and the test windows are classified in eval mode with the harness's own
input conversion. The predictions are written to a CSV; the metrics are computed from that CSV only, with the harness's
metric function, and compared with the stored run record.

Each dataset's test reads are logged to results/test_access.log. Point WHAR_DATASETS_DIR at a private copy of the
caches: the harness may write normalised samples into the cache it reads.

    WHAR_DATASETS_DIR=<copy> uv run --no-sync python scripts/evaluate.py --cells cells.json --run-id <run_id>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.models.registry import build  # noqa: E402
from src.training import run_whar as R  # noqa: E402
from src.utils import paths as PATHS  # noqa: E402
from src.utils import provenance as PV  # noqa: E402
from whar_arena import training as T  # noqa: E402

ACCESS_LOG = ROOT / "results" / "test_access.log"


def checkpoint_path(model: str, dataset: str, seed: int, split: int) -> Path:
    return PATHS.CHECKPOINTS / model / dataset / f"seed-{seed}" / f"split-{split:02d}.pt"


def stored_run(model: str, dataset: str, seed: int, split: int) -> Path:
    base = PATHS.RUNS / model / dataset
    return (base / f"seed-{seed}" if seed else base) / f"split-{split:02d}.json"


def metrics_from_csv(path: Path, num_classes: int) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    y_true = [int(r["y_true"]) for r in rows]
    y_pred = [int(r["y_pred"]) for r in rows]
    return T._classification_metrics(y_true, y_pred, num_classes)


@torch.no_grad()
def predict(model, loader, cfg) -> tuple[list[int], list[int]]:
    y_true, y_pred = [], []
    for labels, inputs in loader:
        y_true += T._target_labels(labels).tolist()
        y_pred += model(T._to_model_inputs(inputs, cfg)).argmax(dim=1).tolist()
    return y_true, y_pred


def evaluate(cells: list[tuple[str, str, int, int]], out_dir: Path, run_id: str, started: float) -> list[dict]:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    by_ds = defaultdict(list)
    for c in cells:
        by_ds[c[1]].append(c)
    results = []
    for dataset, cs in sorted(by_ds.items()):
        with open(ACCESS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | scripts/evaluate.py | {run_id} | verification (G5): "
                    f"regenerate {len(cs)} stored test predictions on {dataset} from checkpoints\n")
        ds = R.prepare_dataset(dataset)
        cfg = ds["cfg"]
        for split in sorted({c[3] for c in cs}):
            loader, dl, shape = R.prepare_split(ds, ds["splits"][split])
            L, C, K = shape
            names = R.channel_names(cfg, C)
            test_indices = list(ds["splits"][split].test_indices)
            for model_id, _, seed, _ in [c for c in cs if c[3] == split]:
                ck_path = checkpoint_path(model_id, dataset, seed, split)
                rec = {"model": model_id, "dataset": dataset, "seed": seed, "split": split,
                       "checkpoint": str(ck_path.relative_to(ROOT)) if ck_path.is_relative_to(ROOT) else str(ck_path)}
                if not ck_path.exists():
                    results.append(rec | {"status": "no checkpoint"})
                    continue
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                torch.manual_seed(seed)
                net = build(model_id, input_channels=C, window_length=L, num_classes=K, channel_names=names)
                init = {k: v.clone() for k, v in net.state_dict().items()}
                net.load_state_dict(ck["state_dict"], strict=True)
                net.eval()
                y_true, y_pred = predict(net, dl["test"], cfg)
                sub = Path(model_id) / dataset / f"seed-{seed}"
                pred_f = out_dir / "predictions" / sub / f"split-{split:02d}.csv"
                pred_f.parent.mkdir(parents=True, exist_ok=True)
                with open(pred_f, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["window_index", "y_true", "y_pred"])
                    w.writerows(zip(test_indices, y_true, y_pred))
                m = metrics_from_csv(pred_f, K)
                stored_f = stored_run(model_id, dataset, seed, split)
                stored = json.loads(stored_f.read_text())
                sm = stored["metrics"]
                ss = stored["source"]
                state = ck["state_dict"]
                rec |= {"status": "evaluated", "num_classes": K, "test_size": len(y_true), "stored_test_size": ss["test_size"],
                        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                        "stored_macro_f1": sm["macro_f1"], "stored_accuracy": sm["accuracy"],
                        "stored_balanced_accuracy": sm["balanced_accuracy"], "checkpoint_macro_f1": ck.get("macro_f1"),
                        "checkpoint_fingerprint_matches": ck.get("fingerprint") == stored.get("fingerprint"),
                        "checkpoint_seed": ck.get("seed"),
                        "params": sum(q.numel() for q in net.parameters()),
                        "stored_params": stored["model"]["params"],
                        "finite": all(torch.isfinite(v).all().item() for v in state.values() if v.dtype.is_floating_point),
                        "differs_from_init": any(not torch.equal(init[k], state[k]) for k in init
                                                 if init[k].dtype.is_floating_point),
                        "predicted_classes": len(set(y_pred)), "equal_to_truth": y_pred == y_true,
                        "train_norm_mean_abs": float(dl["train"].dataset.tensors[1].mean(dim=(0, 1)).abs().max()),
                        "train_norm_std_dev": float((dl["train"].dataset.tensors[1].std(dim=(0, 1)) - 1).abs().max())}
                met_f = out_dir / "metrics" / sub / f"split-{split:02d}.json"
                met_f.parent.mkdir(parents=True, exist_ok=True)
                met_f.write_text(json.dumps(rec, indent=1) + "\n")
                for f in (pred_f, met_f):
                    PV.write(f, run_id=run_id, started=started, inputs=[ck_path, stored_f],
                             config={"cells": len(cells)}, produced_by="scripts/evaluate.py")
                results.append(rec)
            print(f"{dataset} split {split}: {sum(1 for c in cs if c[3] == split)} runs", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="JSON list of [model, dataset, seed, split]")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    cells = [tuple(c) for c in json.loads(Path(a.cells).read_text())]
    out_dir = ROOT / "experiments" / a.run_id
    out_dir.mkdir()
    started = time.time()
    config = {"cells": [list(c) for c in cells], "datasets_dir": str(R.DATASETS_DIR)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    PV.ledger_start(a.run_id, config)
    try:
        res = evaluate(cells, out_dir, a.run_id, started)
    except BaseException as e:
        PV.ledger_end(a.run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    (out_dir / "evaluated.json").write_text(json.dumps(res, indent=1) + "\n")
    n = sum(r["status"] == "evaluated" for r in res)
    PV.ledger_end(a.run_id, f"{n} of {len(res)} evaluated", started)
    print("evaluated", n, "of", len(res))


if __name__ == "__main__":
    main()
