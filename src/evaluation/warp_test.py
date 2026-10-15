"""Tempo robustness: train each model on each split exactly as run_whar.py does, then score it on
time-warped copies of the test windows (same warps for every model).

Warps act on the normalized test windows by linear interpolation:
  uniform s : the whole window played s times faster (s > 1 gives fewer samples, s < 1 more)
  random sig: same length, local speed exp(sig * z(t)) with z a smooth random curve of unit std,
              one warp per window (people speed up and slow down within one gesture)
A pointwise feature averaged over time is nearly unchanged by a uniform warp but not by a random
one; a rate-independent memory sees the same path under both. Only the amplitude-preserving time
axis is changed: a faster real gesture would also have larger angular rates.

Only models that accept any window length are tested (the sequence models of edge_ssm.registry).

Usage:
  uv run python -m src.evaluation.warp_test --models hystnet s4d --dataset utd_mhad
  uv run python -m src.evaluation.warp_test --summarize            -> results/warp/summary.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import torch

from src.training import run_whar as R
from src.utils import paths as PATHS  # noqa: E402

HERE = Path(__file__).resolve().parents[2]  # project root
OUT = PATHS.EXPERIMENTS / "warp"
UNIFORM = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]
RANDOM = [0.25, 0.5, 0.75]
REPEATS = 3


def resample(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x (B, C, L) at fractional sample positions pos, (L',) or (B, L') -> (B, C, L')."""
    length = x.shape[-1]
    pos = pos.to(x.device, x.dtype).clamp(0, length - 1)
    i0 = pos.floor().long().clamp(max=length - 2)
    w = pos - i0
    if pos.dim() == 1:
        return x[..., i0] * (1 - w) + x[..., i0 + 1] * w
    idx = i0.unsqueeze(1).expand(-1, x.shape[1], -1)
    w = w.unsqueeze(1)
    return torch.gather(x, 2, idx) * (1 - w) + torch.gather(x, 2, idx + 1) * w


def uniform_positions(length: int, speed: float) -> torch.Tensor:
    return torch.linspace(0, length - 1, max(2, round(length / speed)))


def random_positions(batch: int, length: int, sigma: float, gen: torch.Generator) -> torch.Tensor:
    t = torch.linspace(0, 1, length)
    z = torch.zeros(batch, length)
    for k in (1, 2, 3):  # a few slow components: speed changes over ~1/3 of the window or more
        amp = torch.randn(batch, 1, generator=gen) / k
        phase = 2 * math.pi * torch.rand(batch, 1, generator=gen)
        z = z + amp * torch.sin(2 * math.pi * k * t + phase)
    z = z / z.std(dim=1, keepdim=True).clamp_min(1e-6)
    pos = torch.cumsum(torch.exp(sigma * z), dim=1)
    return (pos - pos[:, :1]) / (pos[:, -1:] - pos[:, :1]) * (length - 1)


@torch.no_grad()
def score(model, test_loader, cfg, warp) -> float:
    model.eval()
    device = next(model.parameters()).device
    y_true, y_pred = [], []
    for labels, inputs in test_loader:
        x = warp(R.training._to_model_inputs(inputs, cfg).to(device))
        y_pred += model(x).argmax(dim=1).cpu().tolist()
        y_true += R.training._target_labels(labels).tolist()
    return R.training._classification_metrics(y_true, y_pred, int(cfg.num_of_activities))["macro_f1"]


def run(models: list[str], dataset_id: str, split_filter) -> None:
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    for split_index, split in enumerate(ds["splits"]):
        if split_filter is not None and split_index not in split_filter:
            continue
        split_id = f"split-{split_index:02d}"
        todo = [m for m in models if not (OUT / m / dataset_id / f"{split_id}.json").exists()]
        if not todo:
            continue
        loader, dataloaders, shape = R.prepare_split(ds, split)
        for model_id in todo:
            t0 = time.time()
            model, metrics = R.fit(model_id, ds, split, split_id, loader, dataloaders, shape)
            if not R.result_path(ds, model_id, split_id).exists():  # same training as run_whar.py
                R.write_result(model_id, ds, split_id, model, metrics, dataloaders, shape, time.time() - t0)
            test = dataloaders["test"]
            result = {"model_id": model_id, "dataset_id": dataset_id, "split_id": split_id,
                      "clean_macro_f1_harness": metrics.macro_f1, "uniform": {}, "random": {}}
            for s in UNIFORM:
                pos = uniform_positions(shape[0], s)
                result["uniform"][str(s)] = score(model, test, cfg, lambda x, p=pos: resample(x, p))
            for level, sigma in enumerate(RANDOM):
                reps = []
                for rep in range(REPEATS):
                    gen = torch.Generator().manual_seed(10_000 * split_index + 100 * level + rep)
                    reps.append(score(model, test, cfg, lambda x, g=gen, sg=sigma: resample(
                        x, random_positions(x.shape[0], x.shape[-1], sg, g))))
                result["random"][str(sigma)] = reps
            path = OUT / model_id / dataset_id / f"{split_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"=== {model_id}/{dataset_id}/{split_id}: clean {metrics.macro_f1:.4f} | uniform "
                  + " ".join(f"{k}:{v:.3f}" for k, v in result["uniform"].items()) + " | random "
                  + " ".join(f"{k}:{st.mean(v):.3f}" for k, v in result["random"].items())
                  + f" | {time.time() - t0:.0f}s", flush=True)


def summarize(dataset_id: str) -> None:
    from scipy.stats import wilcoxon

    rows = {}
    for d in sorted(OUT.glob(f"*/{dataset_id}")):
        runs = [json.loads(p.read_text()) for p in sorted(d.glob("split-*.json"))]
        if runs:
            rows[d.parent.name] = runs
    conds = [("uniform", str(s)) for s in UNIFORM] + [("random", str(s)) for s in RANDOM]

    def value(run, kind, key):
        v = run[kind][key]
        return 100 * (st.mean(v) if isinstance(v, list) else v)

    head = [f"x{k}" if kind == "uniform" else f"rand {k}" for kind, k in conds]
    lines = [f"# Tempo robustness on {dataset_id}", "",
             "Test macro-F1 (%) under time warps of the test windows, mean over splits (n in brackets). "
             "`xS` = played S times faster (uniform); `rand σ` = random within-window speed changes, "
             "std of log speed σ, mean of 3 warps. Same warps for every model.", "",
             "| Model | " + " | ".join(head) + " |", "|---|" + "---|" * len(conds)]
    for m, runs in sorted(rows.items(), key=lambda kv: -st.mean(value(r, "uniform", "1.0") for r in kv[1])):
        cells = [f"{st.mean(value(r, kind, k) for r in runs):.1f}" for kind, k in conds]
        lines.append(f"| {m} ({len(runs)}) | " + " | ".join(cells) + " |")
    lines += ["", "Drop from the unwarped score (points), mean over splits:", "",
              "| Model | " + " | ".join(head) + " |", "|---|" + "---|" * len(conds)]
    for m, runs in rows.items():
        cells = [f"{st.mean(value(r, kind, k) - value(r, 'uniform', '1.0') for r in runs):+.1f}" for kind, k in conds]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if "hystnet" in rows:
        lines += ["", "HysteresisNet − model, paired over splits (mean Δ, wins/losses, Wilcoxon p):", "",
                  "| Model | " + " | ".join(head) + " |", "|---|" + "---|" * len(conds)]
        ref = {r["split_id"]: r for r in rows["hystnet"]}
        for m, runs in rows.items():
            if m == "hystnet":
                continue
            cells = []
            for kind, k in conds:
                ds_ = [value(ref[r["split_id"]], kind, k) - value(r, kind, k) for r in runs if r["split_id"] in ref]
                p = wilcoxon(ds_).pvalue if len(ds_) > 1 and any(ds_) else float("nan")
                cells.append(f"{st.mean(ds_):+.1f} ({sum(x > 0 for x in ds_)}/{sum(x < 0 for x in ds_)}, p={p:.2g})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    out = OUT / f"summary_{dataset_id}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(R.CONFIGS))
    ap.add_argument("--dataset", default="utd_mhad")
    ap.add_argument("--splits", nargs="+", type=int)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.dataset)
    else:
        run(args.models, args.dataset, set(args.splits) if args.splits else None)


if __name__ == "__main__":
    main()
