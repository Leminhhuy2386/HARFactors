"""Does fitting a window's style (amplitude, offset, tempo) make unseen people match seen ones?
A no-training probe for the analysis-by-synthesis idea (RESEARCH_PLAN.md §5h).

1-nearest-neighbour classification of each test window against the training windows of the same
split (protocol splits, global z-score as in the harness), with four distances:
  euclid      Euclidean on the windows as given;
  euclid_std  Euclidean after standardizing each window per channel (removes per-window offset and gain);
  dtw         dynamic time warping (Sakoe-Chiba band, 10 % of the window) on the windows as given;
  dtw_std     DTW on the per-window standardized windows (removes offset, gain and tempo).
If dtw_std is far above euclid, per-window style explains much of the cross-subject error.
DTW runs on the GPU, anti-diagonal by anti-diagonal, for all (test, train) pairs of a chunk.

    uv run python -m src.models.classical.style_probe --datasets utd_mhad
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import torch

from src.training import run_whar as R
from src.evaluation.robust_test import test_tensors

OUT = R.RESULTS_DIR / "style_probe"


def train_tensors(dataloaders, cfg) -> tuple[torch.Tensor, list[int]]:
    xs, ys = [], []
    for labels, inputs in dataloaders["train"]:
        xs.append(R.training._to_model_inputs(inputs, cfg))
        ys += R.training._target_labels(labels).tolist()
    for labels, inputs in dataloaders["val"]:  # validation windows are training subjects too
        xs.append(R.training._to_model_inputs(inputs, cfg))
        ys += R.training._target_labels(labels).tolist()
    return torch.cat(xs).float(), ys


def standardize(x: torch.Tensor) -> torch.Tensor:  # (N, C, L)
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-3)


def dtw_distances(a: torch.Tensor, b: torch.Tensor, band: int) -> torch.Tensor:
    """a (Na, C, L), b (Nb, C, L) -> (Na, Nb) DTW cost with a Sakoe-Chiba band; squared Euclidean local cost."""
    na, nb, length = a.shape[0], b.shape[0], a.shape[-1]
    at, bt = a.transpose(1, 2), b.transpose(1, 2)  # (N, L, C)
    cost = torch.cdist(at.reshape(-1, at.shape[-1]), bt.reshape(-1, bt.shape[-1])) ** 2  # (Na*L, Nb*L)
    cost = cost.view(na, length, nb, length).permute(0, 2, 1, 3)  # (Na, Nb, L, L)
    inf = torch.tensor(float("inf"), device=a.device, dtype=a.dtype)
    d = torch.full((na, nb, length + 1, length + 1), float("inf"), device=a.device, dtype=a.dtype)
    d[:, :, 0, 0] = 0.0
    for s in range(2, 2 * length + 1):  # anti-diagonal i + j = s (1-based)
        i = torch.arange(max(1, s - length), min(length, s - 1) + 1, device=a.device)
        j = s - i
        keep = (i - j).abs() <= band
        i, j = i[keep], j[keep]
        if len(i) == 0:
            continue
        best = torch.minimum(torch.minimum(d[:, :, i - 1, j], d[:, :, i, j - 1]), d[:, :, i - 1, j - 1])
        d[:, :, i, j] = torch.where(torch.isinf(best), inf, best + cost[:, :, i - 1, j - 1])
    return d[:, :, length, length]


def nn_predict(test: torch.Tensor, train: torch.Tensor, ytrain: torch.Tensor, kind: str, band: int,
               chunk: int) -> list[int]:
    preds = []
    for k in range(0, len(test), chunk):
        a = test[k:k + chunk]
        if kind == "euclid":
            dist = torch.cdist(a.flatten(1), train.flatten(1))
        else:
            dist = torch.cat([dtw_distances(a, train[m:m + 64], band) for m in range(0, len(train), 64)], dim=1)
        preds += ytrain[dist.argmin(1).cpu()].tolist()
    return preds


def run(dataset_id: str, max_train: int, chunk: int, splits: set[int] | None = None) -> dict:
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    nc = int(cfg.num_of_activities)
    device = R.training._get_device()
    scores: dict[str, list[float]] = {k: [] for k in ("euclid", "euclid_std", "dtw", "dtw_std")}
    gen = torch.Generator().manual_seed(0)
    for split_index, split in enumerate(ds["splits"]):
        if splits is not None and split_index not in splits:
            continue
        _, dataloaders, shape = R.prepare_split(ds, split)
        xte, yte = test_tensors(dataloaders, cfg)
        xtr, ytr = train_tensors(dataloaders, cfg)
        if len(xtr) > max_train:  # fixed random subset of the training windows (same for every distance)
            keep = torch.randperm(len(xtr), generator=gen)[:max_train]
            xtr, ytr = xtr[keep], [ytr[i] for i in keep.tolist()]
        ytr_t = torch.tensor(ytr)
        band = max(1, int(0.1 * shape[0]))
        for kind, prep in (("euclid", lambda v: v), ("euclid_std", standardize), ("dtw", lambda v: v),
                           ("dtw_std", standardize)):
            a, b = prep(xte).to(device), prep(xtr).to(device)
            pred = nn_predict(a, b, ytr_t, "euclid" if kind.startswith("euclid") else "dtw", band, chunk)
            scores[kind].append(100 * R.training._classification_metrics(yte, pred, nc)["macro_f1"])
        print(f"  {dataset_id} split-{split_index:02d}: " + ", ".join(f"{k} {v[-1]:.1f}" for k, v in scores.items()),
              flush=True)
    result = {"dataset": dataset_id, "max_train": max_train, "splits": sorted(splits) if splits else "all",
              "per_split": scores,
              "mean": {k: st.mean(v) for k, v in scores.items()}}
    print(f"=== {dataset_id}: " + ", ".join(f"{k} {v:.2f}" for k, v in result["mean"].items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{dataset_id}{'' if splits is None else '_splits' + '-'.join(map(str, sorted(splits)))}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["utd_mhad"])
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--splits", nargs="+", type=int)
    args = ap.parse_args()
    for d in args.datasets:
        run(d, args.max_train, args.chunk, set(args.splits) if args.splits else None)


if __name__ == "__main__":
    main()
