"""How much do segment statistics alone carry? A no-deep-training probe for the coarse-to-fine statistic tree
(RESEARCH_PLAN.md §5k). Protocol splits and global z-score as in the harness; classifiers with fixed settings.

Features per window (C channels, depth-4 dyadic tree = 31 segments):
  root      per-channel mean, std, min, max, RMS of first differences over the whole window (statnet's kind + 1);
  leaves    the same 5 statistics on the 16 leaf segments only (a piecewise description, absolute);
  tree_abs  all 31 segments, absolute;
  tree_rel  all 31 segments, each coded relative to its parent segment (root absolute).
Classifiers: multinomial logistic regression on standardized features (C = 1) and ExtraTrees (300 trees).
This is a premise check, not model selection: nothing here is tuned, and no variant is chosen from it.

    uv run python -m src.models.classical.tree_probe --datasets utd_mhad actrectut_gestures
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training import run_whar as R
from src.evaluation.robust_test import test_tensors
from src.models.classical.style_probe import train_tensors

OUT = R.RESULTS_DIR / "tree_probe"
DEPTH = 4
EPS = 1e-3


def segments(length: int, depth: int) -> list[list[tuple[int, int]]]:
    """Dyadic segments per level: levels[l] = [(start, end), ...] with 2**l entries."""
    levels = [[(0, length)]]
    for _ in range(depth):
        nxt = []
        for a, b in levels[-1]:
            m = (a + b) // 2
            nxt += [(a, m), (m, b)]
        levels.append(nxt)
    return levels


def seg_stats(x: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """x (N, C, L) -> (N, C, 5): mean, std, min, max, RMS of first differences on [a, b)."""
    s = x[..., a:b]
    d = s.diff(dim=-1) if b - a > 1 else torch.zeros_like(s)
    return torch.stack([s.mean(-1), s.std(-1, unbiased=False), s.amin(-1), s.amax(-1),
                        d.square().mean(-1).sqrt()], dim=-1)


def features(x: torch.Tensor) -> dict[str, np.ndarray]:
    levels = segments(x.shape[-1], DEPTH)
    stats = [[seg_stats(x, a, b) for a, b in lvl] for lvl in levels]
    root = stats[0][0]
    leaves = torch.cat(stats[-1], dim=-1)
    tree_abs = torch.cat([s for lvl in stats for s in lvl], dim=-1)
    rel = [root]
    for li in range(1, len(levels)):
        for k, s in enumerate(stats[li]):
            p = stats[li - 1][k // 2]
            mu_p, sd_p, dr_p = p[..., 0:1], p[..., 1:2] + EPS, p[..., 4:5] + EPS
            rel.append(torch.cat([(s[..., 0:1] - mu_p) / sd_p, torch.log((s[..., 1:2] + EPS) / sd_p),
                                  (s[..., 2:3] - mu_p) / sd_p, (s[..., 3:4] - mu_p) / sd_p,
                                  torch.log((s[..., 4:5] + EPS) / dr_p)], dim=-1))
    tree_rel = torch.cat(rel, dim=-1)
    return {k: v.flatten(1).numpy() for k, v in
            (("root", root), ("leaves", leaves), ("tree_abs", tree_abs), ("tree_rel", tree_rel))}


def run(dataset_id: str, splits: set[int] | None) -> dict:
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    nc = int(cfg.num_of_activities)
    scores: dict[str, list[float]] = {}
    for split_index, split in enumerate(ds["splits"]):
        if splits is not None and split_index not in splits:
            continue
        _, dataloaders, _ = R.prepare_split(ds, split)
        xte, yte = test_tensors(dataloaders, cfg)
        xtr, ytr = train_tensors(dataloaders, cfg)
        ftr, fte = features(xtr), features(xte)
        for kind in ftr:
            sc = StandardScaler().fit(ftr[kind])
            a, b = np.nan_to_num(sc.transform(ftr[kind])), np.nan_to_num(sc.transform(fte[kind]))
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(a, ytr)
            et = ExtraTreesClassifier(n_estimators=300, n_jobs=4, random_state=0).fit(a, ytr)
            for clf, name in ((lr, "lr"), (et, "et")):
                pred = clf.predict(b).tolist()
                scores.setdefault(f"{kind}/{name}", []).append(
                    100 * R.training._classification_metrics(yte, pred, nc)["macro_f1"])
        print(f"  {dataset_id} split-{split_index:02d}: " + ", ".join(f"{k} {v[-1]:.1f}" for k, v in scores.items()),
              flush=True)
    result = {"dataset": dataset_id, "depth": DEPTH, "splits": sorted(splits) if splits else "all",
              "per_split": scores, "mean": {k: st.mean(v) for k, v in scores.items()}}
    print(f"=== {dataset_id}: " + ", ".join(f"{k} {v:.2f}" for k, v in result["mean"].items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{dataset_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["utd_mhad"])
    ap.add_argument("--splits", nargs="+", type=int)
    args = ap.parse_args()
    torch.set_num_threads(4)
    for d in args.datasets:
        run(d, set(args.splits) if args.splits else None)


if __name__ == "__main__":
    main()
