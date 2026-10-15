"""Sampling-schedule robustness (RESEARCH_PLAN.md §5k) on saved checkpoints: no retraining.

Each checkpoint under experiments/checkpoints/<model>/<dataset>/seed-<s>/split-XX.pt is scored on its split's test windows
received under another sampling schedule. Conditions, relative to the dataset's native rate:
- rate_xF: anti-aliased polyphase resampling to F times the native rate (another device, same signal);
- drop_P: a fraction P of the samples lost at random (timestamps known);
- jitter: each sample read at a random time within +-0.45 sample periods (timestamps known);
- sod_D: send-on-delta events: a sample is sent when any channel differs from the last sent value by more than D
  (z units); the first sample is always sent. The fraction of samples sent is recorded.

Pipelines (each model gets every one that applies):
- naive: the received samples taken as uniformly sampled (rate_xF, drop_P, jitter);
- grid: linear interpolation back to the native-rate grid (for rate_xF: polyphase resampling back), the standard fix
  when timestamps or the rate are known;
- hold: zero-order hold on the native grid (the reconstruction send-on-delta guarantees to within D);
- dt: S4D and Mamba on the received uniform sequence with their step size Δ scaled by 1/F (rate_xF only);
- native: HysteresisNet on the received samples with its clock channel at their true times and each sample weighted
  by its share of the window's time. For drop_P and sod_D this is computed as a zero-order hold of both the signal and
  the clock on the native grid, which is exactly the event-driven computation: held inputs leave the play operators
  and every pointwise layer unchanged, so the plain mean over the grid equals the time-weighted mean over the events.

    uv run python -m src.evaluation.schedule_test --models hystnet_sel_clock s4d mamba recstatnet --dataset utd_mhad
    uv run python -m src.evaluation.schedule_test --summarize --dataset utd_mhad
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from src.training import run_whar as R
from src.evaluation.robust_test import macro_f1, test_tensors

CKPT = R.RESULTS_DIR / "checkpoints"
OUT = R.RESULTS_DIR / "schedule"
RATES = {0.5: (1, 2), 0.67: (2, 3), 1.5: (3, 2), 2.0: (2, 1)}
DROPS = [0.3, 0.5, 0.7]
SOD = [0.1, 0.25, 0.5, 1.0]
REPEATS = 2
DT_MODELS = {"s4d", "mamba"}


def set_dt_scale(model: torch.nn.Module, scale: float) -> None:
    for m in model.modules():
        if hasattr(m, "dt_scale"):
            m.dt_scale = scale


def ramp(t: np.ndarray, length: int) -> np.ndarray:
    """Clock value at time t (in native sample periods): the native grid maps onto linspace(-1, 1, length)."""
    return 2.0 * t / (length - 1) - 1.0


def reconstruct(x: np.ndarray, t: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window received at times t (sorted, native sample periods) with values x (C, K) -> on the native grid:
    linear interpolation (C, L), zero-order hold (C, L), and the held clock (L,) (the clock of the last sample
    received at or before each grid point; before the first sample, the first one)."""
    grid = np.arange(length, dtype=np.float64)
    lin = np.stack([np.interp(grid, t, xc) for xc in x])
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    return lin, x[:, idx], ramp(t[idx], length)


def send_on_delta(x: np.ndarray, delta: float) -> np.ndarray:
    """x (C, L) -> indices of the samples sent: the first, then any that moves a channel more than delta from the
    last sent value."""
    sent, last = [0], x[:, 0]
    for i in range(1, x.shape[1]):
        if np.abs(x[:, i] - last).max() > delta:
            sent.append(i)
            last = x[:, i]
    return np.asarray(sent)


class Schedules:
    """Received versions of the test windows x (N, C, L), drawn from generators seeded by (split, condition, repeat)
    so every model and seed sees the same ones."""

    def __init__(self, x: torch.Tensor, split_index: int) -> None:
        self.x, self.split_index = x.double().numpy(), split_index

    def gen(self, name: str, rep: int) -> np.random.Generator:
        return np.random.default_rng([self.split_index, sum(map(ord, name)), rep])

    def conditions(self):
        """Yields (name, list over repeats of dict pipeline-input -> arrays, fraction of samples received)."""
        x = self.x
        n, c, length = x.shape
        for f, (up, down) in RATES.items():
            xr = resample_poly(x, up, down, axis=-1)
            back = resample_poly(xr, down, up, axis=-1)[..., :length]
            if back.shape[-1] < length:
                back = np.concatenate([back, np.repeat(back[..., -1:], length - back.shape[-1], axis=-1)], axis=-1)
            yield f"rate_x{f}", [{"uniform": xr, "grid": back, "dt_scale": 1.0 / f}], xr.shape[-1] / length
        for p in DROPS:
            reps = []
            for rep in range(REPEATS):
                g = self.gen(f"drop{p}", rep)
                keep = round((1 - p) * length)
                lin, hold, clk = (np.empty_like(x), np.empty_like(x), np.empty((n, length)))
                naive = np.empty((n, c, keep))
                for i in range(n):
                    t = np.sort(g.choice(length, keep, replace=False)).astype(np.float64)
                    xi = x[i][:, t.astype(int)]
                    naive[i] = xi
                    lin[i], hold[i], clk[i] = reconstruct(xi, t, length)
                reps.append({"uniform": naive, "grid": lin, "hold": hold, "native_x": hold, "native_clock": clk})
            yield f"drop_{p}", reps, 1 - p
        reps = []
        for rep in range(REPEATS):
            g = self.gen("jitter", rep)
            pos = np.arange(length) + (g.random((n, length)) - 0.5) * 0.9
            pos = np.clip(pos, 0, length - 1)
            grid = np.arange(length, dtype=np.float64)
            read = np.stack([np.stack([np.interp(pos[i], grid, xc) for xc in x[i]]) for i in range(n)])
            lin = np.stack([np.stack([np.interp(grid, pos[i], rc) for rc in read[i]]) for i in range(n)])
            dur = np.diff(np.concatenate([pos, np.full((n, 1), float(length))], axis=1), axis=1)
            w = dur / dur.sum(axis=1, keepdims=True)
            reps.append({"uniform": read, "grid": lin, "native_x": read, "native_clock": ramp(pos, length),
                         "native_weights": w})
        yield "jitter", reps, 1.0
        for d in SOD:
            lin, hold, clk = np.empty_like(x), np.empty_like(x), np.empty((n, length))
            kept = 0
            for i in range(n):
                idx = send_on_delta(x[i], d)
                kept += len(idx)
                lin[i], hold[i], clk[i] = reconstruct(x[i][:, idx], idx.astype(np.float64), length)
            yield f"sod_{d}", [{"grid": lin, "hold": hold, "native_x": hold, "native_clock": clk}], kept / (n * length)


@torch.no_grad()
def score(model, inputs: dict, model_id: str, y: list[int], nc: int, batch: int = 256) -> dict[str, float]:
    """Macro-F1 of every pipeline that applies to this model and condition."""
    device = next(model.parameters()).device
    native_ok = model_id.startswith("hystnet")

    def f1(x, clock=None, weights=None):
        pred = []
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
            if clock is None and weights is None:
                out = model(xb)
            else:
                cb = None if clock is None else torch.as_tensor(clock[i:i + batch], dtype=torch.float32, device=device)
                wb = None if weights is None else torch.as_tensor(weights[i:i + batch], dtype=torch.float32, device=device)
                out = model(xb, clock=cb, weights=wb)
            pred += out.argmax(dim=1).cpu().tolist()
        return R.training._classification_metrics(y, pred, nc)["macro_f1"]

    res = {}
    if "uniform" in inputs:
        res["naive"] = f1(inputs["uniform"])
    for key in ("grid", "hold"):
        if key in inputs:
            res[key] = f1(inputs[key])
    if "dt_scale" in inputs and model_id in DT_MODELS:
        set_dt_scale(model, inputs["dt_scale"])
        res["dt"] = f1(inputs["uniform"])
        set_dt_scale(model, 1.0)
    if native_ok:
        if "native_x" in inputs:
            res["native"] = f1(inputs["native_x"], inputs.get("native_clock"), inputs.get("native_weights"))
        elif "dt_scale" in inputs:
            res["native"] = res["naive"]  # a uniform resampling: the clock spans the window whatever the rate
    return res


def run(models: list[str], dataset_id: str, seeds: set[int] | None) -> None:
    ds = R.prepare_dataset(dataset_id)
    cfg = ds["cfg"]
    device = R.training._get_device()
    nc = int(cfg.num_of_activities)
    for split_index, split in enumerate(ds["splits"]):
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
        t0 = time.time()
        conds = list(Schedules(x, split_index).conditions())
        print(f"{dataset_id} {split_id}: schedules built in {time.time() - t0:.0f}s", flush=True)
        for m, seed, ck, out in todo:
            state = torch.load(ck, map_location="cpu", weights_only=False)
            model = R.build(m, input_channels=shape[1], window_length=shape[0], num_classes=shape[2])
            model.load_state_dict(state["state_dict"])
            model.to(device).eval()
            result = {"model_id": m, "dataset_id": dataset_id, "split_id": split_id, "train_seed": seed,
                      "checkpoint_macro_f1": state["macro_f1"], "clean": macro_f1(model, x, y, nc),
                      "received": {}, "conditions": {}}
            for name, reps, frac in conds:
                result["received"][name] = frac
                scored = [score(model, inp, m, y, nc) for inp in reps]
                result["conditions"][name] = {k: [s[k] for s in scored] for k in scored[0]}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n")
            c = result["conditions"]
            print(f"=== {m}/{dataset_id}/{split_id} seed {seed}: clean {result['clean']:.3f} | x0.5 "
                  + " ".join(f"{k} {v[0]:.3f}" for k, v in c["rate_x0.5"].items())
                  + f" | sod_0.25 ({result['received']['sod_0.25']:.2f}) "
                  + " ".join(f"{k} {v[0]:.3f}" for k, v in c["sod_0.25"].items()), flush=True)


def summarize(dataset_id: str) -> None:
    rows = {}
    for f in sorted(OUT.glob(f"*/{dataset_id}/seed-*/split-*.json")):
        r = json.loads(f.read_text())
        rows.setdefault(r["model_id"], []).append(r)
    if not rows:
        print("no results")
        return
    conds = list(next(iter(rows.values()))[0]["conditions"])
    lines = [f"# Sampling-schedule robustness on {dataset_id} (checkpoints, no retraining)", "",
             "Macro-F1 %, mean over splits and training seeds (repeats averaged). Each cell: the model's best pipeline "
             "(naive / grid / hold / dt), and HysteresisNet's native event-driven pipeline in **bold**. Received: "
             "fraction of the native samples that reach the model.", "",
             "| Condition | Received | " + " | ".join(rows) + " |", "|---|---|" + "---|" * len(rows)]
    first = next(iter(rows.values()))
    lines.append("| clean | 1.00 | " + " | ".join(f"{100 * st.mean(r['clean'] for r in rs):.1f}" for rs in rows.values()) + " |")
    for cond in conds:
        frac = st.mean(r["received"][cond] for r in first)
        cells = []
        for m, rs in rows.items():
            pipes = rs[0]["conditions"][cond].keys()
            means = {p: 100 * st.mean(st.mean(r["conditions"][cond][p]) for r in rs) for p in pipes}
            best = max((p for p in means if p != "native"), key=means.get)
            cell = f"{means[best]:.1f} ({best})"
            if "native" in means:
                cell += f", **{means['native']:.1f}**"
            cells.append(cell)
        lines.append(f"| {cond} | {frac:.2f} | " + " | ".join(cells) + " |")
    n = {m: len(rs) for m, rs in rows.items()}
    lines += ["", "Runs per model: " + ", ".join(f"{m} {k}" for m, k in n.items())]
    out = OUT / f"summary_{dataset_id}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(R.CONFIGS))
    ap.add_argument("--dataset", default="utd_mhad")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.dataset)
    else:
        run(a.models, a.dataset, set(a.seeds) if a.seeds else None)


if __name__ == "__main__":
    main()
