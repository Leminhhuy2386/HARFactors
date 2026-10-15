"""Robustness tests P1-P3 (RESEARCH_PLAN.md §5b) on saved checkpoints: no retraining.

Every checkpoint under experiments/checkpoints/<model>/<dataset>/seed-<s>/split-XX.pt (written by
run_whar.py) is rebuilt and scored on perturbed copies of its split's test windows. The
perturbations are drawn from generators seeded by (split, condition, repeat), so every model and
every training seed sees exactly the same perturbed windows.

P1 physical tempo. A gesture done s times faster has its time axis compressed by s, angular rates
   scaled by s and dynamic acceleration scaled by s^2, while gravity stays. Gravity is split off
   the accelerometer channels with a 0.3 Hz zero-phase low-pass. Inputs are globally z-scored, so
   in z units: accel z' = lp(z) + s^2 (z - lp(z)) (exact, the low-pass has unit DC gain), and
   gyro z' = s z + (s - 1) mu / sigma with the split's own train mean and std. Two approximations:
   wrist rotations above 0.3 Hz also move gravity between axes, and those are scaled as if
   dynamic; within-window warps ignore the term in the derivative of the speed. Conditions:
   phys_xS (uniform, 7 speeds) and phys_rand_SIG (within-window speed changes, the same warps as
   warp_test.py, local speed v(t) applied per output sample).
P2 sampling rate and irregular sampling. rate_F: the window resampled to F Hz with an
   anti-aliasing polyphase filter (same signal, different device rate); drop_P: a fraction P of
   the samples removed at random (order kept); jitter: each sample read at a random offset of up
   to +-0.45 samples.
P3 noise and sensor faults (z units). noise_S: Gaussian noise of std S; spike_P: a fraction P of
   the samples of each channel hit by a +-5 spike; bias_B: a constant offset of +-B per channel.

Usage:
  uv run python -m src.evaluation.robust_test --models quantilenet s4d --dataset utd_mhad   # all seeds found
  uv run python -m src.evaluation.robust_test --summarize                                     # results/robust/summary_utd_mhad.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample_poly

from src.training import run_whar as R
from src.evaluation.warp_test import random_positions, resample, uniform_positions

HERE = Path(__file__).resolve().parents[2]  # project root
CKPT = R.RESULTS_DIR / "checkpoints"
OUT = R.RESULTS_DIR / "robust"
SPEEDS = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]
SIGMAS = [0.25, 0.5, 0.75]
RATES = [25, 33.3, 75, 100]  # Hz, trained at 50
DROPS = [0.1, 0.3, 0.5]
NOISE = [0.1, 0.25, 0.5]
SPIKES = [0.01, 0.03, 0.1]
BIASES = [0.25, 0.5, 1.0]
REPEATS = 3
GRAVITY_HZ = 0.3


def channel_kinds(cfg) -> list[str]:
    names = list(cfg.selected_channels or cfg.available_channels)
    return ["gyro" if "gyro" in n.lower() else "acc" if n.lower().startswith("a") else "other" for n in names]


def norm_stats(ds: dict, split, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Train mean and std per channel, as the harness's global z-score computed them."""
    from whar_datasets.processing.utils.normalization import get_norm_params
    from whar_datasets.utils.loading import load_windows

    if "windows" not in ds:
        ds["windows"] = load_windows(ds["pre"].windows_dir)
    mean, std = get_norm_params(cfg, split.train_indices, ds["window_df"], ds["windows"])
    names = list(cfg.selected_channels or cfg.available_channels)
    return np.array([mean[n] for n in names]), np.array([std[n] for n in names])


class Perturber:
    """Builds every perturbed copy of one split's test windows (B, C, L), model-independent."""

    def __init__(self, x: torch.Tensor, kinds: list[str], mean: np.ndarray, std: np.ndarray, fs: float,
                 split_index: int):
        self.x, self.fs, self.split_index = x, fs, split_index
        self.acc = torch.tensor([k == "acc" for k in kinds])
        self.gyro = torch.tensor([k == "gyro" for k in kinds])
        self.offset = torch.tensor(mean / std, dtype=x.dtype)[:, None]  # mu / sigma per channel
        b, a = butter(2, GRAVITY_HZ / (fs / 2))
        self.lp = torch.tensor(filtfilt(b, a, x.numpy(), axis=-1).copy(), dtype=x.dtype)

    def gen(self, name: str, rep: int) -> torch.Generator:
        h = sum(ord(c) * 31 ** i for i, c in enumerate(name)) % 1_000_003
        return torch.Generator().manual_seed(1_000_000 * self.split_index + 1000 * h % 999_983 + rep)

    def physical(self, pos: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Resample at positions pos and scale by the local speed (B, L') or scalar."""
        z, lp = resample(self.x, pos), resample(self.lp, pos)
        v = speed if speed.dim() == 0 else speed[:, None, :]
        out = z.clone()
        out[:, self.acc] = (lp + v ** 2 * (z - lp))[:, self.acc]
        out[:, self.gyro] = (v * z + (v - 1) * self.offset)[:, self.gyro]
        return out

    def conditions(self):
        """Yields (name, list of perturbed tensors, one per repeat)."""
        x, (n, c, length) = self.x, self.x.shape
        for s in SPEEDS:
            yield f"phys_x{s}", [self.physical(uniform_positions(length, s), torch.tensor(s))]
        for level, sigma in enumerate(SIGMAS):
            reps = []
            for rep in range(REPEATS):
                g = torch.Generator().manual_seed(10_000 * self.split_index + 100 * level + rep)  # = warp_test
                pos = random_positions(n, length, sigma, g)
                v = torch.gradient(pos, dim=1)[0] * 1.0  # input samples per output sample
                reps.append(self.physical(pos, v))
            yield f"phys_rand_{sigma}", reps
        for f in RATES:
            up, down = {25: (1, 2), 33.3: (2, 3), 75: (3, 2), 100: (2, 1)}[f]
            yield f"rate_{f}", [torch.tensor(resample_poly(x.numpy(), up, down, axis=-1).copy(), dtype=x.dtype)]
        for p in DROPS:
            reps = []
            for rep in range(REPEATS):
                g = self.gen(f"drop{p}", rep)
                keep = round((1 - p) * length)
                idx = torch.argsort(torch.rand(n, length, generator=g), dim=1)[:, :keep].sort(dim=1).values
                reps.append(torch.gather(x, 2, idx[:, None, :].expand(-1, c, -1)))
            yield f"drop_{p}", reps
        reps = []
        for rep in range(REPEATS):
            g = self.gen("jitter", rep)
            pos = torch.arange(length, dtype=x.dtype) + (torch.rand(n, length, generator=g) - 0.5) * 0.9
            reps.append(resample(x, pos))
        yield "jitter_0.45", reps
        for s in NOISE:
            yield f"noise_{s}", [x + s * torch.randn(x.shape, generator=self.gen(f"noise{s}", r)) for r in range(REPEATS)]
        for p in SPIKES:
            reps = []
            for rep in range(REPEATS):
                g = self.gen(f"spike{p}", rep)
                hit = torch.rand(x.shape, generator=g) < p
                sign = torch.where(torch.rand(x.shape, generator=g) < 0.5, -1.0, 1.0)
                reps.append(x + hit * sign * 5.0)
            yield f"spike_{p}", reps
        for b in BIASES:
            reps = []
            for rep in range(REPEATS):
                g = self.gen(f"bias{b}", rep)
                sign = torch.where(torch.rand(n, c, 1, generator=g) < 0.5, -1.0, 1.0)
                reps.append(x + b * sign)
            yield f"bias_{b}", reps


@torch.no_grad()
def macro_f1(model, x: torch.Tensor, y: list[int], num_classes: int, batch: int = 256) -> float:
    device = next(model.parameters()).device
    pred = []
    for i in range(0, len(x), batch):
        pred += model(x[i:i + batch].to(device)).argmax(dim=1).cpu().tolist()
    return R.training._classification_metrics(y, pred, num_classes)["macro_f1"]


def test_tensors(dataloaders, cfg) -> tuple[torch.Tensor, list[int]]:
    xs, ys = [], []
    for labels, inputs in dataloaders["test"]:
        xs.append(R.training._to_model_inputs(inputs, cfg))
        ys += R.training._target_labels(labels).tolist()
    return torch.cat(xs).float(), ys


def run(models: list[str], dataset_id: str, seeds: set[int] | None, split_filter: set[int] | None) -> None:
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    device = R.training._get_device()
    for split_index, split in enumerate(ds["splits"]):
        if split_filter is not None and split_index not in split_filter:
            continue
        split_id = f"split-{split_index:02d}"
        todo = []
        for m in models:
            for ck in sorted((CKPT / m / dataset_id).glob(f"seed-*/{split_id}.pt")):
                seed = int(ck.parent.name.split("-")[1])
                out = OUT / m / dataset_id / f"seed-{seed}" / f"{split_id}.json"
                if (seeds is None or seed in seeds) and not out.exists():
                    todo.append((m, seed, ck, out))
        if not todo:
            continue
        _, dataloaders, shape = R.prepare_split(ds, split)
        x, y = test_tensors(dataloaders, cfg)
        mean, std = norm_stats(ds, split, cfg)
        conds = list(Perturber(x, channel_kinds(cfg), mean, std, cfg.sampling_freq, split_index).conditions())
        for m, seed, ck, out in todo:
            t0 = time.time()
            state = torch.load(ck, map_location="cpu", weights_only=False)
            model = R.build(m, input_channels=shape[1], window_length=shape[0], num_classes=shape[2])
            model.load_state_dict(state["state_dict"])
            model.to(device).eval()
            nc = int(cfg.num_of_activities)
            result = {"model_id": m, "dataset_id": dataset_id, "split_id": split_id, "train_seed": seed,
                      "trained_on": state.get("device"), "scored_on": str(device),
                      "checkpoint_macro_f1": state["macro_f1"], "clean": macro_f1(model, x, y, nc),
                      "conditions": {name: [macro_f1(model, xi, y, nc) for xi in reps] for name, reps in conds}}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n")
            c = result["conditions"]
            print(f"=== {m}/{dataset_id}/{split_id} seed {seed}: clean {result['clean']:.4f} "
                  f"(ckpt {state['macro_f1']:.4f}) | phys x0.5 {c['phys_x0.5'][0]:.3f} x2 {c['phys_x2.0'][0]:.3f} "
                  f"| rate25 {c['rate_25'][0]:.3f} | spike.03 {st.mean(c['spike_0.03']):.3f} "
                  f"| {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- summary

GROUPS = {
    "P1 physical tempo, uniform (mean of 7 speeds)": [f"phys_x{s}" for s in SPEEDS],
    "P1 physical tempo, within-window (mean of 3 σ)": [f"phys_rand_{s}" for s in SIGMAS],
    "P2 device rate (mean of 4 rates)": [f"rate_{f}" for f in RATES],
    "P2 irregular sampling (drops + jitter)": [f"drop_{p}" for p in DROPS] + ["jitter_0.45"],
    "P3 Gaussian noise (mean of 3)": [f"noise_{s}" for s in NOISE],
    "P3 spikes / outliers (mean of 3)": [f"spike_{p}" for p in SPIKES],
    "P3 channel bias (mean of 3)": [f"bias_{b}" for b in BIASES],
}


def load_runs(dataset_id: str) -> dict[str, dict[str, dict[int, dict]]]:
    """model -> split -> seed -> result."""
    runs: dict = {}
    for p in sorted(OUT.glob(f"*/{dataset_id}/seed-*/split-*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["model_id"], {}).setdefault(r["split_id"], {})[r["train_seed"]] = r
    return runs


def split_means(runs_m: dict, key) -> dict[str, float]:
    """Per split: the mean over training seeds of key(result) (in %)."""
    return {s: 100 * st.mean(key(r) for r in by_seed.values()) for s, by_seed in runs_m.items()}


def cond_value(r: dict, name: str) -> float:
    return st.mean(r["conditions"][name])


def summarize(dataset_id: str, refs=("s4d", "mamba", "hystnet_nomem")) -> None:
    from scipy.stats import wilcoxon

    runs = load_runs(dataset_id)
    models = sorted(runs, key=lambda m: -st.mean(split_means(runs[m], lambda r: r["clean"]).values()))
    all_conds = list(next(iter(next(iter(runs.values())).values())).values())[0]["conditions"].keys()

    def group_key(names):
        return lambda r: st.mean(cond_value(r, n) for n in names)

    lines = [f"# Robustness tests P1-P3 on {dataset_id}", "",
             "Built by `robust_test.py --summarize` from the checkpoints in `results/checkpoints/`. Macro-F1 (%) "
             "per split is the mean over training seeds, then the mean over splits is shown. Every model and "
             "seed sees the same perturbed test windows. Definitions are in the docstring of `robust_test.py`.", ""]
    lines += ["## Group means", "", "| Model | splits × seeds | clean | " + " | ".join(GROUPS) + " |",
              "|---|---|---|" + "---|" * len(GROUPS)]
    for m in models:
        n_seeds = sorted({len(v) for v in runs[m].values()})
        cells = [f"{st.mean(split_means(runs[m], group_key(names)).values()):.1f}" for names in GROUPS.values()]
        lines.append(f"| {m} | {len(runs[m])} × {'/'.join(map(str, n_seeds))} | "
                     f"{st.mean(split_means(runs[m], lambda r: r['clean']).values()):.1f} | " + " | ".join(cells) + " |")

    lines += ["", "## Every condition", "", "| Condition | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name in ["clean", *all_conds]:
        key = (lambda r: r["clean"]) if name == "clean" else (lambda r, n=name: cond_value(r, n))
        lines.append(f"| {name} | " + " | ".join(f"{st.mean(split_means(runs[m], key).values()):.1f}" for m in models) + " |")

    for ref in refs:
        if ref not in runs:
            continue
        lines += ["", f"## Model − {ref}, paired over splits (mean Δ, wins/losses, Wilcoxon p)", "",
                  "| Model | clean | " + " | ".join(GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
        for m in models:
            if m == ref:
                continue
            cells = []
            for key in [lambda r: r["clean"], *[group_key(n) for n in GROUPS.values()]]:
                a, b = split_means(runs[m], key), split_means(runs[ref], key)
                d = [a[s] - b[s] for s in a if s in b]
                p = wilcoxon(d).pvalue if len(d) > 1 and any(d) else float("nan")
                cells.append(f"{st.mean(d):+.1f} ({sum(v > 0 for v in d)}/{sum(v < 0 for v in d)}, p={p:.2g})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    out = OUT / f"summary_{dataset_id}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(R.CONFIGS))
    ap.add_argument("--dataset", default="utd_mhad")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--splits", nargs="+", type=int)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.dataset)
    else:
        run(args.models, args.dataset, set(args.seeds) if args.seeds else None,
            set(args.splits) if args.splits is not None else None)


if __name__ == "__main__":
    main()
