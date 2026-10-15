"""Elastic 1-nearest-neighbour baselines on the WHAR Arena v1 splits (no training): how large is the gap
between deep models and plain elastic matching? (RESEARCH_PLAN.md §5h, option C.)

Each test window is matched to every training window of its split (train + validation windows, i.e. all
windows of the other subjects; global z-score fitted on train, as the harness does) with four distances:
  euclid      Euclidean;
  euclid_std  Euclidean after standardizing each window per channel (removes per-window offset and gain);
  dtw         dynamic time warping, Sakoe-Chiba band of 10 % of the window;
  dtw_std     DTW on per-window standardized windows (removes offset, gain and tempo).
Macro-F1 with the harness's own metric (all classes; absent classes count as 0).

DTW is computed row by row inside the band on the GPU. Within a row, D[i,j] = min(e_j, c_j + D[i,j-1])
with e_j = c_j + min(D[i-1,j], D[i-1,j-1]) is solved in parallel as D = S + cummin(e - S), S = cumsum(c).

DSADS and Daphnet: their local caches number subjects/labels in listdir order, unlike the cloud runs
(RESEARCH_PLAN.md §5e). 1-NN does not depend on that order (the references of a test subject are all other
subjects' windows, and macro-F1 is unchanged by relabelling), so they are read from
the shared cache's PC builds; their per-split indices do not match the P5 split indices, only their means.

    uv run python -m src.models.classical.elastic_baseline --datasets utd_mhad actrectut_gestures
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.training import run_whar as R

OUT = R.RESULTS_DIR / "elastic"
REORDERED = R.DATASETS_DIR  # the PC builds of dsads and daphnet are in HAR's datasets_paper_era
REORDERED_IDS = {"dsads", "daphnet"}
KINDS = ("euclid", "euclid_std", "dtw", "dtw_std")


def dtw_band(a: torch.Tensor, b: torch.Tensor, w: int) -> torch.Tensor:
    """a (Na, L, C), b (Nb, L, C) -> (Na, Nb) DTW cost (squared Euclidean local cost, band |i - j| <= w)."""
    na, length, _ = a.shape
    offs = torch.arange(-w, w + 1, device=a.device)
    a2, b2 = (a * a).sum(-1), (b * b).sum(-1)
    inf = torch.tensor(float("inf"), device=a.device, dtype=a.dtype)
    prev = None
    for i in range(length):
        j = i + offs
        valid = (j >= 0) & (j < length)
        jc = j.clamp(0, length - 1)
        cost = (a2[:, i, None, None] + b2[:, jc][None] - 2 * torch.einsum("nc,mkc->nmk", a[:, i], b[:, jc])).clamp_min(0)
        if prev is None:  # D[-1, -1] = 0: the path starts at (0, 0)
            e = torch.where(j == 0, cost, inf)
        else:
            up = torch.cat([prev[..., 1:], torch.full_like(prev[..., :1], float("inf"))], dim=-1)  # D[i-1, j]
            e = cost + torch.minimum(up, prev)  # prev[..., k] is D[i-1, j-1]
        c0 = torch.where(valid, cost, torch.zeros_like(cost))
        e = torch.where(valid, e, inf)
        s = torch.cumsum(c0, dim=-1)
        prev = torch.where(valid, s + torch.cummin(e - s, dim=-1).values, inf)
    return prev[..., w]


def dtw_band_paired(a: torch.Tensor, b: torch.Tensor, w: int) -> torch.Tensor:
    """Banded DTW of aligned pairs: a, b (N, L, C) -> (N,), the same recursion as dtw_band."""
    length = a.shape[1]
    offs = torch.arange(-w, w + 1, device=a.device)
    inf = torch.tensor(float("inf"), device=a.device, dtype=a.dtype)
    prev = None
    for i in range(length):
        j = i + offs
        valid = (j >= 0) & (j < length)
        cost = ((a[:, i, None, :] - b[:, j.clamp(0, length - 1), :]) ** 2).sum(-1)  # (N, K)
        if prev is None:
            e = torch.where(j == 0, cost, inf)
        else:
            up = torch.cat([prev[..., 1:], torch.full_like(prev[..., :1], float("inf"))], dim=-1)
            e = cost + torch.minimum(up, prev)
        c0 = torch.where(valid, cost, torch.zeros_like(cost))
        e = torch.where(valid, e, inf)
        s = torch.cumsum(c0, dim=-1)
        prev = torch.where(valid, s + torch.cummin(e - s, dim=-1).values, inf)
    return prev[..., w]


def standardize(x: torch.Tensor) -> torch.Tensor:  # (N, C, L)
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-3)


def split_tensors(dataloaders, cfg, parts: tuple[str, ...]) -> tuple[torch.Tensor, list[int]]:
    xs, ys = [], []
    for part in parts:
        for labels, inputs in dataloaders[part]:
            xs.append(R.training._to_model_inputs(inputs, cfg))
            ys += R.training._target_labels(labels).tolist()
    return torch.cat(xs).float(), ys


def lb_keogh(test: torch.Tensor, upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    """Lower bound of the banded DTW cost for every (test, ref) pair: test (Na, C, L), envelopes (Nb, C, L)
    -> (Na, Nb). Each test sample must be matched to some reference sample inside the band, whose value lies
    between the envelopes, so its cost is at least the distance to the envelope (summed over channels)."""
    out = []
    flat_u, flat_l = upper.flatten(1), lower.flatten(1)
    for k in range(0, len(test), 64):
        a = test[k:k + 64].flatten(1)[:, None, :]  # (na, 1, C*L)
        rows = []
        for m in range(0, len(flat_u), 1024):
            u, lo = flat_u[m:m + 1024][None], flat_l[m:m + 1024][None]
            rows.append((torch.relu(a - u) ** 2 + torch.relu(lo - a) ** 2).sum(-1))
        out.append(torch.cat(rows, dim=1))
    return torch.cat(out)


def nearest_dtw_pruned(test: torch.Tensor, refs: torch.Tensor, band: int, batch: int = 256) -> torch.Tensor:
    """Exact 1-NN under banded DTW with LB_Keogh pruning. test, refs (N, C, L) on the device.

    References are visited per test window in increasing lower-bound order, `batch` at a time; a test
    window stops once its next lower bound is not below its best DTW cost so far (exact, since LB <= DTW)."""
    upper = F.max_pool1d(refs, 2 * band + 1, stride=1, padding=band)
    lower = -F.max_pool1d(-refs, 2 * band + 1, stride=1, padding=band)
    lb = lb_keogh(test, upper, lower)  # (Na, Nb)
    order = lb.argsort(dim=1)
    lb_sorted = lb.gather(1, order)
    n_test, n_ref = lb.shape
    best = torch.full((n_test,), float("inf"), device=test.device)
    best_idx = torch.zeros(n_test, dtype=torch.long, device=test.device)
    a_all, b_all = test.transpose(1, 2).contiguous(), refs.transpose(1, 2).contiguous()  # (N, L, C)
    for start in range(0, n_ref, batch):
        active = (lb_sorted[:, start] < best).nonzero().squeeze(1)
        if len(active) == 0:
            break
        cand = order[active, start:start + batch]  # (n_active, <=batch)
        for k in range(0, len(active), 64):  # 64 test windows x their own candidates per DTW call
            rows, cidx = active[k:k + 64], cand[k:k + 64]
            a = a_all[rows].repeat_interleave(cidx.shape[1], dim=0)  # (r * c, L, C)
            d = dtw_band_paired(a, b_all[cidx.flatten()], band).view(len(rows), -1)  # (r, c)
            dmin, arg = d.min(dim=1)
            better = dmin < best[rows]
            best[rows] = torch.where(better, dmin, best[rows])
            best_idx[rows] = torch.where(better, cidx[torch.arange(len(rows)), arg], best_idx[rows])
    return best_idx


def nearest(test: torch.Tensor, refs: torch.Tensor, kind: str, band: int, budget: int = 16_000_000) -> torch.Tensor:
    """Index of the nearest reference for every test window. test, refs (N, C, L) on the device."""
    out = []
    if kind == "euclid":
        flat = refs.flatten(1)
        for k in range(0, len(test), 256):
            out.append(torch.cdist(test[k:k + 256].flatten(1), flat).argmin(1))
        return torch.cat(out)
    return nearest_dtw_pruned(test, refs, band)


def test_subjects(ds: dict, split) -> list[int]:
    win = ds["window_df"].iloc[list(split.test_indices)]
    sub = ds["session_df"].set_index("session_id").loc[win["session_id"], "subject_id"]
    return sorted(int(s) for s in set(sub.tolist()))


def run(dataset_id: str, splits: set[int] | None) -> None:
    if dataset_id in REORDERED_IDS:
        R.EXTRA_DATASETS_DIR = REORDERED
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    nc = int(cfg.num_of_activities)
    device = R.training._get_device()
    for split_index, split in enumerate(ds["splits"]):
        if splits is not None and split_index not in splits:
            continue
        out = OUT / dataset_id / f"split-{split_index:02d}.json"
        if out.exists():
            continue
        t0 = time.time()
        _, dataloaders, shape = R.prepare_split(ds, split)
        xte, yte = split_tensors(dataloaders, cfg, ("test",))
        xref, yref = split_tensors(dataloaders, cfg, ("train", "val"))
        yref_t = torch.tensor(yref)
        band = max(1, int(round(0.1 * shape[0])))
        scores = {}
        for kind in KINDS:
            prep = standardize if kind.endswith("_std") else (lambda v: v)
            idx = nearest(prep(xte).to(device), prep(xref).to(device), kind.split("_")[0], band).cpu()
            pred = yref_t[idx].tolist()
            scores[kind] = R.training._classification_metrics(yte, pred, nc)["macro_f1"]
        result = {"dataset_id": dataset_id, "split_id": f"split-{split_index:02d}", "macro_f1": scores,
                  "n_test": len(yte), "n_ref": len(yref), "band": band, "window_length": shape[0],
                  "test_subjects": test_subjects(ds, split), "datasets_dir": ds["datasets_dir"],
                  "seconds": round(time.time() - t0, 1)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"=== {dataset_id} split-{split_index:02d}: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in scores.items())
              + f" ({result['seconds']:.0f}s, {len(yte)} test x {len(yref)} refs)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", type=int)
    args = ap.parse_args()
    for d in args.datasets:
        run(d, set(args.splits) if args.splits else None)


if __name__ == "__main__":
    main()
