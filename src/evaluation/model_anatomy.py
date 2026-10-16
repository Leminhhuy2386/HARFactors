"""Descriptive: the structure of the ten population networks and what their memories learned (no data, no training).

1. Structure. Each network is built for the reference input of gate G1 (6 channels named acc_x ... gyr_z, 150 samples,
   27 classes, as UTD-MHAD) and its parameters are counted per part: encoder, residual blocks, window-level side path,
   final LayerNorm and head. The total must equal G1's count (checked here).
2. Size per dataset. The parameter count, input shape and sampling rate of every stored run (S1, S2), read from the
   run records (they vary with channels and classes only).
3. Memory operators on a toy input (illustration). A synthetic 3-s, 50-Hz gait-like signal with a posture step is fed
   through the code's own operators at their initial settings: the play operator bank (HysteresisNet), the quantile
   trackers (QuantileNet) and the soft recurrence rows (RecurrenceNet). The toy signal is not data.
4. Learned memory parameters. From every stored checkpoint of S1's eight re-executable datasets (all splits, seeds
   0-2): S4D's mode time constants (in fractions of the window) and decay rates, HysteresisNet's base play widths,
   QuantileNet's quantile levels and steps, RecurrenceNet's recurrence radii, each against the same network freshly
   initialised. State dictionaries are read; no input enters.

    uv run --no-sync python -m src.evaluation.model_anatomy   ->   results/metrics/model_anatomy/data.json
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data.sets import S1, S2
from src.models.hysteresis import play_scan
from src.models.quantile import quantile_track
from src.models.registry import build
from src.models.round6 import RecurrenceBlock
from src.utils import paths as PATHS
from src.utils import provenance as PV

OUT = PATHS.METRICS / "model_anatomy"
REF = {"input_channels": 6, "window_length": 150, "num_classes": 27,
       "channel_names": ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]}
LEARNED_DATASETS = ["utd_mhad", "uci_har", "unimib_shar", "actrectut_gestures", "opportunity", "sad", "hapt", "wisdm"]


def count(module) -> int:
    return int(sum(p.numel() for p in module.parameters())) if module is not None else 0


def structure() -> dict:
    g1 = json.loads((PATHS.RESULTS / "gates" / "G1_environment_rerun2.json").read_text())["observed_value"]["models"]
    out = {}
    for m in S1["models"]:
        torch.manual_seed(0)
        net = build(m, **REF)
        parts = {"encoder": count(net.encoder), "blocks": count(net.blocks), "side": count(getattr(net, "side", None)),
                 "norm": count(net.norm), "head": count(net.head)}
        total = count(net)
        out[m] = {"class": type(net).__name__, "block": type(net.blocks[0]).__name__, "layers": len(net.blocks),
                  "d_model": int(net.head.in_features - getattr(getattr(net, "side", None), "out", 0)),
                  "embedding_dim": int(net.head.in_features), "params": total, "params_by_part": parts,
                  "params_per_block": count(net.blocks[0]), "g1_params": g1[m]["params"],
                  "equal_to_g1": total == g1[m]["params"], "s2": m in S2["models"]}
        assert sum(parts.values()) == total, m
    return out


def sizes() -> dict:
    """Per dataset and model, from the seed-0 run records: parameters and input shape (split 00), and the number of
    windows (each window is in exactly one test group, so the test sizes over all splits add up to the total)."""
    out = {}
    for spec in (S1, S2):
        for d, n_splits in spec["datasets"].items():
            for m in spec["models"]:
                f = PATHS.RUNS / m / d / "split-00.json"
                if not f.exists():
                    continue
                r = json.loads(f.read_text())["model"]
                rec = {k: r[k] for k in ("params", "input_channels", "window_length", "num_classes", "sampling_freq")}
                srcs = [json.loads((PATHS.RUNS / m / d / f"split-{i:02d}.json").read_text())["source"]
                        for i in range(n_splits) if (PATHS.RUNS / m / d / f"split-{i:02d}.json").exists()]
                rec["splits"] = len(srcs)
                rec["windows"] = int(sum(s["test_size"] for s in srcs))
                out.setdefault(d, {})[m] = rec
    return out


def toy() -> dict:
    """The operators at their initial settings on a synthetic gait-like signal with a posture step at 1.5 s."""
    fs, length = 50, 150
    t = np.arange(length) / fs
    u = np.sin(2 * np.pi * 1.0 * t) + 0.4 * np.sin(2 * np.pi * 2.0 * t + 0.5) + np.where(t >= 1.5, 0.8, 0.0)
    ut = torch.tensor(u, dtype=torch.float32)
    widths = torch.tensor([0.05, 0.2, 0.6, 1.5])  # HysteresisBlock default widths
    play = play_scan(ut.expand(len(widths), -1), widths.unsqueeze(-1))  # (K, L)
    trackers = ((0.1, 0.05), (0.5, 0.02), (0.5, 0.1), (0.9, 0.05))  # QuantileBlock defaults
    tau = torch.tensor([[a for a, _ in trackers]])
    eta = torch.tensor([[b for _, b in trackers]])
    with torch.no_grad():
        q = quantile_track(ut.view(1, -1, 1), tau, eta, 0.1)[0, :, 0]  # (L, K)
        block = RecurrenceBlock(32, n_embed=1)
        lag8 = torch.cat([ut[:1].expand(8), ut[:-8]])
        z = torch.stack([ut, lag8], dim=-1).view(1, length, 1, 2)  # 2-D delay embedding (u_t, u_{t-8})
        rec = block.recurrence(z)[0, :, 0]  # (L, n_lags)
    return {"fs": fs, "t": t.tolist(), "u": u.tolist(), "play_widths": widths.tolist(), "play": play.tolist(),
            "trackers": [list(x) for x in trackers], "quantile": q.T.tolist(), "recurrence_lags": list(block.lags),
            "recurrence_radius": float(F.softplus(block.eps_raw).item()), "recurrence": rec.T.tolist(),
            "note": "synthetic input; operators at their initial settings (illustration, not data)"}


def extract(m: str, sd: dict, length: int) -> dict[str, np.ndarray]:
    """The memory parameters of one state dict: S4D mode time constants (fraction of the window) and decay rates,
    play base widths, quantile levels and steps, recurrence radii."""
    out: dict[str, list] = {}
    for k, v in sd.items():
        if m == "s4d" and k.endswith("kernel.log_dt"):
            a_real = torch.exp(sd[k.replace("log_dt", "log_a_real")])  # (H, N/2)
            out.setdefault("s4d", []).append((1.0 / (a_real * torch.exp(v).unsqueeze(-1)) / length).flatten())
            out.setdefault("s4d_a_real", []).append(a_real.flatten())
        elif m == "hystnet_sel_clock" and k.endswith(".rho"):
            out.setdefault(m, []).append(F.softplus(v).flatten())
        elif m == "quantilenet" and k.endswith("tau_logit"):
            out.setdefault("quantilenet_tau", []).append(torch.sigmoid(v).flatten())
        elif m == "quantilenet" and k.endswith("eta_raw"):
            out.setdefault("quantilenet_eta", []).append(F.softplus(v).flatten())
        elif m == "recurrencenet" and k.endswith("eps_raw"):
            out.setdefault(m, []).append(F.softplus(v).flatten())
    return {k: torch.cat(v).double().numpy() for k, v in out.items()}


def learned() -> tuple[dict, list]:
    """Learned memory parameters of every stored S1 checkpoint (seeds 0-2) of the eight datasets, against the same
    networks freshly initialised for the same input shape (one fresh network per checkpoint, torch seeds 0, 1, ...).
    Quantile levels, recurrence radii and S4D decay rates start at fixed values, so their change is exact per element;
    play widths, quantile steps and S4D step sizes start at random values and are compared as distributions."""
    keep: dict[str, list] = {}
    init: dict[str, list] = {}
    files = []
    n_init = 0
    for d in LEARNED_DATASETS:
        for m in ("s4d", "hystnet_sel_clock", "quantilenet", "recurrencenet"):
            shape = json.loads((PATHS.RUNS / m / d / "split-00.json").read_text())["model"]
            for s in S1["seeds"]:
                for f in sorted((PATHS.CHECKPOINTS / m / d / f"seed-{s}").glob("split-*.pt")):
                    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
                    files.append(f)
                    torch.manual_seed(n_init)
                    n_init += 1
                    fresh = build(m, input_channels=shape["input_channels"], window_length=shape["window_length"],
                                  num_classes=shape["num_classes"]).state_dict()
                    for k, v in extract(m, sd, shape["window_length"]).items():
                        keep.setdefault(k, []).append(v)
                    for k, v in extract(m, fresh, shape["window_length"]).items():
                        init.setdefault(k, []).append(v)
    keep = {k: np.concatenate(v) for k, v in keep.items()}
    init = {k: np.concatenate(v) for k, v in init.items()}
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    q = lambda x: dict(zip(("p05", "p25", "p50", "p75", "p95"), np.quantile(x, qs).tolist()))  # noqa: E731
    summary = {}
    for k in keep:
        lin = k == "quantilenet_tau"
        both = np.concatenate([keep[k], init[k]])
        edges = (np.linspace(0, 1, 41) if lin else
                 np.linspace(np.log10(both.min()), np.log10(both.max()), 41))
        tr = (lambda x: x) if lin else (lambda x: np.log10(x))
        summary[k] = {"n": int(len(keep[k])), "quantiles": q(keep[k]), "init_quantiles": q(init[k]),
                      "scale": "linear" if lin else "log10", "hist_edges": edges.tolist(),
                      "hist": np.histogram(tr(keep[k]), bins=edges)[0].tolist(),
                      "init_hist": np.histogram(tr(init[k]), bins=edges)[0].tolist()}
    for k, kind in (("quantilenet_tau", "abs"), ("recurrencenet", "abs"), ("s4d_a_real", "log10")):
        delta = np.abs(keep[k] - init[k]) if kind == "abs" else np.abs(np.log10(keep[k] / init[k]))
        summary[k]["exact_change"] = {"kind": "|learned - initial|" if kind == "abs" else "|log10(learned / initial)|",
                                      "quantiles": q(delta), "share_below_0.05": float(np.mean(delta < 0.05))}
    summary["checkpoints_read"] = len(files)
    summary["s4d_fraction_of_window_above_1"] = float(np.mean(keep["s4d"] > 1))
    summary["s4d_init_fraction_of_window_above_1"] = float(np.mean(init["s4d"] > 1))
    return summary, files


def main() -> int:
    run_id = PV.new_run_id("model_anatomy")
    config = {"reference_input": REF, "datasets": LEARNED_DATASETS}
    (PATHS.EXPERIMENTS / run_id).mkdir()
    (PATHS.EXPERIMENTS / run_id / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    started = time.time()
    PV.ledger_start(run_id, config)
    try:
        describe(run_id, started, config)
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    PV.ledger_end(run_id, "model anatomy: structure, sizes, toy operators, learned memory parameters", started)
    return 0


def describe(run_id: str, started: float, config: dict) -> None:
    learned_summary, files = learned()
    data = {"reference_input": REF, "structure": structure(), "sizes": sizes(), "toy": toy(), "learned": learned_summary}
    if not all(v["equal_to_g1"] for v in data["structure"].values()):
        raise SystemExit("parameter count differs from gate G1")
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "data.json"
    f.write_text(json.dumps(data, indent=1) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=files + [PATHS.RESULTS / "gates" / "G1_environment_rerun2.json"],
             config=config, produced_by="src/evaluation/model_anatomy.py")
    for m, v in data["structure"].items():
        print(f"{m:18s} {v['class']:14s} {v['block']:16s} L={v['layers']} D={v['d_model']} emb={v['embedding_dim']} "
              f"params={v['params']} {v['params_by_part']}")
    print("checkpoints read:", learned_summary["checkpoints_read"])
    for k in ("s4d", "s4d_a_real", "hystnet_sel_clock", "quantilenet_tau", "quantilenet_eta", "recurrencenet"):
        v = learned_summary[k]
        print(k, "learned", {kk: round(vv, 4) for kk, vv in v["quantiles"].items()})
        print(" " * len(k), "initial", {kk: round(vv, 4) for kk, vv in v["init_quantiles"].items()})
        if "exact_change" in v:
            print(" " * len(k), v["exact_change"]["kind"], {kk: round(vv, 4) for kk, vv in
                                                            v["exact_change"]["quantiles"].items()},
                  "share < 0.05:", round(v["exact_change"]["share_below_0.05"], 3))


if __name__ == "__main__":
    sys.exit(main())
