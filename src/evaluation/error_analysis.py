"""Where do the current models fail? Confusion matrices from saved checkpoints (inference only, no
training), to size the headroom of round-11 ideas before building them (RESEARCH_PLAN.md §5h).

For every checkpoint of the given models (all seeds, all splits; Daphnet is skipped because its local
splits are ordered differently from the cloud runs that trained the checkpoints), the test windows are
scored and the confusions summed per dataset. Errors are attributed to categories defined from the class
names: time-reversal pairs (a ~ b played backwards), stairs up/down, static postures. For each category
the "oracle headroom" is the mean macro-F1 gain per split if every error inside that category were moved
to the diagonal: an upper bound on what an idea that solves exactly that confusion can add.

    uv run python -m src.evaluation.error_analysis --models recurrencenet s4d hystnet_nomem
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.training import run_whar as R
from src.evaluation.robust_test import test_tensors

CKPT = R.RESULTS_DIR / "checkpoints"
OUT = R.RESULTS_DIR / "error_analysis"
# dsads and daphnet are left out: their local caches number labels/subjects in another order than the cloud
# runs that trained the checkpoints (RESEARCH_PLAN.md §5e)
DATASETS = ["utd_mhad", "actrectut_gestures", "uci_har", "unimib_shar", "opportunity", "sad", "hapt", "wisdm"]

TIME_REVERSAL = {
    "utd_mhad": [("Swipe Left", "Swipe Right"), ("Draw Circle Clockwise", "Draw Circle Counterclockwise"),
                 ("Sit To Stand", "Stand To Sit")],
    "hapt": [("Stand To Sit", "Sit To Stand"), ("Sit To Lie", "Lie To Sit"), ("Stand To Lie", "Lie To Stand")],
    "actrectut_gestures": [("Open Window", "Close Window")],
    "unimib_shar": [("Standing Up Fs", "Sitting Down"), ("Standing Up Fl", "Lying Down Fs")],
}
STAIRS = {
    "uci_har": [("Walking Upstairs", "Walking Downstairs")], "hapt": [("Walking Upstairs", "Walking Downstairs")],
    "sad": [("Upstairs", "Downstairs")], "wisdm": [("Upstairs", "Downstairs")],
    "dsads": [("Ascending Stairs", "Descending Stairs")], "unimib_shar": [("Going Up S", "Going Down S")],
}
POSTURES = {
    "uci_har": ["Sitting", "Standing", "Laying"], "hapt": ["Sitting", "Standing", "Laying"],
    "sad": ["Standing", "Sitting"], "wisdm": ["Sitting", "Standing"], "opportunity": ["Stand", "Sit", "Lie"],
    "dsads": ["Sitting", "Standing", "Lying On Back", "Lying On Right Side", "Standing In An Elevator Still"],
}
DRAWINGS = {"utd_mhad": ["Draw X", "Draw Circle Clockwise", "Draw Circle Counterclockwise", "Draw Triangle"]}
# locomotion variants that differ in gait pattern only (level / stairs / jogging)
GAITS = {
    "uci_har": ["Walking", "Walking Upstairs", "Walking Downstairs"],
    "hapt": ["Walking", "Walking Upstairs", "Walking Downstairs"],
    "sad": ["Walking", "Jogging", "Upstairs", "Downstairs"],
    "wisdm": ["Walking", "Jogging", "Upstairs", "Downstairs"],
    "unimib_shar": ["Walking", "Running", "Going Up S", "Going Down S"],
    "utd_mhad": ["Jog", "Walk"],
}
NULL_CLASS = {"opportunity": "Unknown", "actrectut_gestures": "Null"}


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Harness definition: mean over all classes, absent classes count as 0."""
    tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0), cm.sum(1)
    f1 = np.where(pred + true > 0, 2 * tp / np.maximum(pred + true, 1), 0.0)
    return float(f1.mean())


def category_cells(names: list[str], dataset: str) -> dict[str, set[tuple[int, int]]]:
    idx = {n: i for i, n in enumerate(names)}
    cats: dict[str, set[tuple[int, int]]] = {}
    for cat, table in (("time_reversal", TIME_REVERSAL), ("stairs_up_down", STAIRS)):
        cells = set()
        for a, b in table.get(dataset, []):
            if a in idx and b in idx:
                cells |= {(idx[a], idx[b]), (idx[b], idx[a])}
        cats[cat] = cells
    for cat, table in (("static_postures", POSTURES), ("drawings", DRAWINGS), ("gait_types", GAITS)):
        group = [idx[n] for n in table.get(dataset, []) if n in idx]
        cats[cat] = {(i, j) for i in group for j in group if i != j}
    null = NULL_CLASS.get(dataset)
    cats["null_class"] = ({(idx[null], j) for j in range(len(names)) if j != idx[null]}
                          | {(i, idx[null]) for i in range(len(names)) if i != idx[null]}) if null in idx else set()
    return cats


def fix(cm: np.ndarray, cells: set[tuple[int, int]]) -> np.ndarray:
    out = cm.copy()
    for i, j in cells:
        out[i, i] += out[i, j]
        out[i, j] = 0
    return out


@torch.no_grad()
def predict(model, x: torch.Tensor, batch: int = 256) -> list[int]:
    device = next(model.parameters()).device
    return [p for i in range(0, len(x), batch) for p in model(x[i:i + batch].to(device)).argmax(1).cpu().tolist()]


def run(models: list[str], dataset_id: str) -> None:
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    act = pd.read_csv(Path(ds["datasets_dir"]) / dataset_id / "metadata" / "activity_df.csv")
    names = list(act.sort_values("activity_id")["activity_name"])
    nc = len(names)
    device = R.training._get_device()
    per_model: dict[str, list[np.ndarray]] = {m: [] for m in models}
    split_f1: dict[str, list[float]] = {m: [] for m in models}
    for split_index, split in enumerate(ds["splits"]):
        split_id = f"split-{split_index:02d}"
        cks = {m: sorted((CKPT / m / dataset_id).glob(f"seed-*/{split_id}.pt")) for m in models}
        if not any(cks.values()):
            continue
        _, dataloaders, shape = R.prepare_split(ds, split)
        x, y = test_tensors(dataloaders, cfg)
        for m in models:
            for ck in cks[m]:
                state = torch.load(ck, map_location="cpu", weights_only=False)
                model = R.build(m, input_channels=shape[1], window_length=shape[0], num_classes=shape[2],
                                channel_names=R.channel_names(cfg, shape[1]))
                model.load_state_dict(state["state_dict"])
                model.to(device).eval()
                cm = np.zeros((nc, nc), dtype=np.int64)
                for t, p in zip(y, predict(model, x)):
                    cm[t, p] += 1
                f1 = macro_f1_from_confusion(cm)
                if abs(f1 - state["macro_f1"]) > 1e-6:
                    print(f"  note: {m} {split_id} {ck.parent.name} rescored {f1:.4f} vs stored {state['macro_f1']:.4f}")
                per_model[m].append(cm)
                split_f1[m].append(f1)
    cats = category_cells(names, dataset_id)
    report = {"dataset": dataset_id, "classes": names, "models": {}}
    for m in models:
        cms = per_model[m]
        if not cms:
            continue
        total = sum(cms)
        errors = int(total.sum() - np.trace(total))
        entry = {"runs": len(cms), "macro_f1": 100 * st.mean(split_f1[m]), "errors": errors, "categories": {}}
        for cat, cells in cats.items():
            if not cells:
                continue
            in_cat = int(sum(total[i, j] for i, j in cells))
            gain = 100 * st.mean(macro_f1_from_confusion(fix(cm, cells)) - macro_f1_from_confusion(cm) for cm in cms)
            entry["categories"][cat] = {"error_share": in_cat / max(errors, 1), "oracle_gain": gain}
        off = [(int(total[i, j]), names[i], names[j]) for i in range(nc) for j in range(nc) if i != j and total[i, j]]
        entry["top_confusions"] = [{"count": c, "true": a, "pred": b, "share": c / max(errors, 1)}
                                   for c, a, b in sorted(off, reverse=True)[:8]]
        recall = np.diag(total) / np.maximum(total.sum(1), 1)
        entry["worst_recall"] = sorted(((float(r), names[i]) for i, r in enumerate(recall) if total[i].sum()))[:5]
        entry["confusion_total"] = total.tolist()
        report["models"][m] = entry
        cat_txt = ", ".join(f"{c} {v['error_share']:.0%} (+{v['oracle_gain']:.1f})" for c, v in entry["categories"].items())
        print(f"=== {dataset_id} {m}: F1 {entry['macro_f1']:.2f} over {len(cms)} runs; errors {errors}; {cat_txt}", flush=True)
        for t in entry["top_confusions"][:4]:
            print(f"      {t['true']} -> {t['pred']}: {t['count']} ({t['share']:.0%})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{dataset_id}.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["recurrencenet", "s4d", "hystnet_nomem"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    for d in args.datasets:
        run(args.models, d)


if __name__ == "__main__":
    main()
