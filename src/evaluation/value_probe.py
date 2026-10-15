"""Round 13 pre-test (RESEARCH_PLAN.md §5l): do value-domain features carry more cross-subject information than the
per-channel window statistics behind recstatnet? No network is trained.

Per split, a linear probe (multinomial logistic regression on standardized features, class-balanced, fitted on the
training subjects' train + validation windows) is scored on the test subject(s) with the harness macro-F1.
Feature sets, all per channel, before any channel mixing:
- stats: mean, std, min, max (recstatnet's side path);
- level: at M fixed levels (z units), the fraction of the window above the level (soft-free empirical CDF) and the
  level-crossing rate, for the signal and for its first difference (scaled by its training std);
- level3: the occupation part of `level` computed separately on the three thirds of the window (coarse order);
- level+stats, level3+stats.
With --et, ExtraTrees (300 trees, class-balanced, fixed in advance, not tuned) is also fitted on level3 and level3+stats:
the "same features, strong non-learned classifier" baseline for LevelNet (results in macro_f1_et).

    uv run python -m src.evaluation.value_probe --datasets utd_mhad actrectut_gestures ...
    uv run python -m src.evaluation.value_probe --summarize
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training import run_whar as R

OUT = R.RESULTS_DIR / "value_probe"
LEVELS = np.linspace(-2.5, 2.5, 12)
SETS = ["stats", "level", "level3", "level+stats", "level3+stats"]
LOCAL_OK = ["utd_mhad", "actrectut_gestures", "uci_har", "unimib_shar", "opportunity", "sad", "hapt", "wisdm"]


def tensors(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for labels, inputs in loader:
        xs.append(R.training._to_model_inputs(inputs, cfg))
        ys.append(R.training._target_labels(labels))
    return torch.cat(xs).double().numpy(), torch.cat(ys).numpy()


def occupation(x: np.ndarray) -> np.ndarray:
    """x (N, C, L) -> fraction of samples above each level (N, C, M)."""
    return (x[..., None] > LEVELS).mean(axis=2)


def crossings(x: np.ndarray) -> np.ndarray:
    """x (N, C, L) -> level-crossing rate (N, C, M)."""
    above = x[..., None] > LEVELS
    return (above[:, :, 1:] != above[:, :, :-1]).mean(axis=2)


def features(x: np.ndarray, dx_std: np.ndarray) -> dict[str, np.ndarray]:
    n = len(x)
    dx = np.diff(x, axis=-1) / dx_std[None, :, None]
    stats = np.concatenate([x.mean(-1), x.std(-1), x.min(-1), x.max(-1)], axis=1)
    level = np.concatenate([occupation(x), crossings(x), occupation(dx), crossings(dx)], axis=-1).reshape(n, -1)
    thirds = np.array_split(np.arange(x.shape[-1]), 3)
    occ3 = np.concatenate([occupation(x[..., idx]) for idx in thirds], axis=-1).reshape(n, -1)
    level3 = np.concatenate([level, occ3], axis=1)
    return {"stats": stats, "level": level, "level3": level3,
            "level+stats": np.concatenate([level, stats], axis=1),
            "level3+stats": np.concatenate([level3, stats], axis=1)}


def probe(ftr: np.ndarray, ytr: np.ndarray, fte: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(ftr)
    clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(ftr), ytr)
    return clf.predict(scaler.transform(fte))


ET_SETS = ["level3", "level3+stats"]


def et_probe(ftr: np.ndarray, ytr: np.ndarray, fte: np.ndarray) -> np.ndarray:
    clf = ExtraTreesClassifier(n_estimators=300, n_jobs=4, class_weight="balanced", random_state=0)
    return clf.fit(ftr, ytr).predict(fte)


def run_et(datasets: list[str]) -> None:
    for d in datasets:
        ds = R.prepare_dataset(d)
        cfg = ds["cfg"]
        nc = int(cfg.num_of_activities)
        for i, split in enumerate(ds["splits"]):
            out = OUT / d / f"split-{i:02d}.json"
            res = json.loads(out.read_text())
            if "macro_f1_et" in res:
                continue
            t0 = time.time()
            _, dl, _ = R.prepare_split(ds, split)
            xtr, ytr = tensors(dl["train"], cfg)
            xva, yva = tensors(dl["val"], cfg)
            xte, yte = tensors(dl["test"], cfg)
            xfit, yfit = np.concatenate([xtr, xva]), np.concatenate([ytr, yva])
            dx_std = np.diff(xfit, axis=-1).std(axis=(0, 2)) + 1e-8
            ffit, fte = features(xfit, dx_std), features(xte, dx_std)
            res["macro_f1_et"] = {s: R.training._classification_metrics(
                yte.tolist(), et_probe(ffit[s], yfit, fte[s]).tolist(), nc)["macro_f1"] for s in ET_SETS}
            out.write_text(json.dumps(res, indent=2) + "\n")
            print(f"{d} split {i} ET: " + " ".join(f"{s} {100 * v:.1f}" for s, v in res["macro_f1_et"].items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)


def run(datasets: list[str]) -> None:
    for d in datasets:
        ds = R.prepare_dataset(d)
        cfg = ds["cfg"]
        nc = int(cfg.num_of_activities)
        for i, split in enumerate(ds["splits"]):
            out = OUT / d / f"split-{i:02d}.json"
            if out.exists():
                continue
            t0 = time.time()
            _, dl, _ = R.prepare_split(ds, split)
            xtr, ytr = tensors(dl["train"], cfg)
            xva, yva = tensors(dl["val"], cfg)
            xte, yte = tensors(dl["test"], cfg)
            xfit, yfit = np.concatenate([xtr, xva]), np.concatenate([ytr, yva])
            dx_std = np.diff(xfit, axis=-1).std(axis=(0, 2)) + 1e-8
            ffit, fte = features(xfit, dx_std), features(xte, dx_std)
            res = {"dataset": d, "split": i, "n_fit": len(yfit), "n_test": len(yte), "macro_f1": {}}
            for s in SETS:
                pred = probe(ffit[s], yfit, fte[s])
                res["macro_f1"][s] = R.training._classification_metrics(yte.tolist(), pred.tolist(), nc)["macro_f1"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=2) + "\n")
            print(f"{d} split {i}: " + " ".join(f"{s} {100 * v:.1f}" for s, v in res["macro_f1"].items())
                  + f" ({time.time() - t0:.0f}s)", flush=True)


def summarize() -> None:
    from src.evaluation import p5_summary as P
    lines = ["# Round 13 pre-test: value-domain features vs per-channel statistics (linear probe, no network)", "",
             "Macro-F1 %, mean over splits. Probe: logistic regression on standardized per-channel features, fitted on "
             "the training subjects. Deep models: seed 0 of the stored runs, for context.", "",
             "| Dataset | " + " | ".join(SETS) + " | " + " | ".join(f"ET {s}" for s in ET_SETS)
             + " | recstatnet | RecurrenceNet | statnet |", "|---|" + "---|" * (len(SETS) + len(ET_SETS) + 3)]
    means = {s: [] for s in SETS + [f"ET {s}" for s in ET_SETS]}
    for d in LOCAL_OK:
        rs = [json.loads(f.read_text()) for f in sorted((OUT / d).glob("split-*.json"))]
        if len(rs) < P.N_SPLITS[d]:
            continue
        vals = {s: 100 * st.mean(r["macro_f1"][s] for r in rs) for s in SETS}
        if all("macro_f1_et" in r for r in rs):
            vals.update({f"ET {s}": 100 * st.mean(r["macro_f1_et"][s] for r in rs) for s in ET_SETS})
        for s in means:
            if s in vals:
                means[s].append(vals[s])
        refs = []
        for m in ("recstatnet", "recurrencenet", "statnet"):
            r = P.runs(m, d)[0]
            refs.append(f"{st.mean(r.values()):.1f}" if len(r) == P.N_SPLITS[d] else "…")
        lines.append(f"| {d} | " + " | ".join(f"{vals[s]:.1f}" if s in vals else "…" for s in means) + " | "
                     + " | ".join(refs) + " |")
    if means["stats"]:
        lines.append(f"| mean ({len(means['stats'])}) | " + " | ".join(
            f"{st.mean(v):.1f}" if len(v) == len(means["stats"]) else "…" for v in means.values()) + " | | | |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=LOCAL_OK)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--et", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    elif a.et:
        run_et(a.datasets)
    else:
        run(a.datasets)


if __name__ == "__main__":
    main()
