"""Accuracy-cost Pareto fronts of every model: ours and the WHAR Arena paper's 14 deep models.

Cost is measured here, on each dataset's real input shape (the shape recorded in our result files: 3-s windows at the
dataset's rate): trainable parameters, FLOPs per window (torch FlopCounterMode, which counts matmul / conv / bmm only),
and CPU latency per window (1 thread, batch 1, median). Each model's cost is the geometric mean over the 10 datasets.
Accuracy is the mean macro-F1 over the 10 datasets at seed 0, the paper's single run; 3-seed means are listed too.
No training.

    uv run python -m src.evaluation.pareto            -> results/metrics/pareto/costs.json, results/metrics/pareto/summary.md, figures/pareto.svg
    uv run python -m src.evaluation.pareto --plot     -> redraw from results/metrics/pareto/costs.json without re-measuring
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.evaluation import compare_arena as CA
from src.training import run_whar as R
from src.visualization.build_factor_figures import BORDER, C_DATA, C_MODEL, GRID, INK, INK2, MUTED, line, mix, rect, svg, t, title, write
from whar_arena import training
from whar_datasets import WHARDatasetID
from whar_models import WHARModelID, build_model

from src.models.registry import build
from src.utils import paths as PATHS  # noqa: E402

OUT = PATHS.METRICS / "pareto"
DS = CA.DATASETS
FIGURE = "pareto.svg"
SCOPE = "10 datasets"
HELDOUT_MODE = False
OURS = ["recstatnet", "lawnet_rec", "stylealignnet", "recurrencenet", "s4d", "hystnet_sel_clock", "lawnet", "lawnet_rand",
        "statnet", "quantilenet", "mamba", "hystnet_nomem"]
LABEL = {"recstatnet": "RN + statistics", "lawnet_rec": "RN + laws", "stylealignnet": "StyleAlignNet",
         "recurrencenet": "RecurrenceNet", "s4d": "S4D", "hystnet_sel_clock": "HysteresisNet", "lawnet": "LawNet",
         "lawnet_rand": "LawNet (random groups)", "statnet": "Statistics (memoryless)", "quantilenet": "QuantileNet",
         "mamba": "Mamba", "hystnet_nomem": "No memory"}
PAPER_DEEP = [a for a in CA.ARENA if a not in CA.CLASSICAL]


def shapes() -> dict[str, dict]:
    """Input shape and channel names per dataset, as the harness built them."""
    from whar_arena.protocol import load_protocol
    protocol = load_protocol()
    out = {}
    for d in DS:
        m = json.loads((CA.RES / "recurrencenet" / d / "split-00.json").read_text())["model"]
        cfg = training._configure_dataset_cfg(protocol=protocol, dataset_id=WHARDatasetID(d),
                                              datasets_dir=R.datasets_dir_for(d), max_epochs=None)
        out[d] = {"c": m["input_channels"], "l": m["window_length"], "k": m["num_classes"],
                  "names": R.channel_names(cfg, m["input_channels"])}
    return out


def make(model: str, sh: dict) -> torch.nn.Module:
    if model in CA.WHAR_IDS:
        return build_model(WHARModelID(CA.WHAR_IDS[model]), input_channels=sh["c"], window_length=sh["l"],
                           num_classes=sh["k"])
    return build(model, input_channels=sh["c"], window_length=sh["l"], num_classes=sh["k"], channel_names=sh["names"])


def measure(model: torch.nn.Module, x: torch.Tensor, repeats: int = 15) -> tuple[int, int, float]:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            model(x)
        for _ in range(3):
            model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return params, int(fc.get_total_flops()), 1e3 * st.median(times)


def run_measure() -> dict:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    sh = shapes()
    rows = {}
    for model in OURS + PAPER_DEEP:
        rows[model] = {}
        for d in DS:
            torch.manual_seed(0)
            net = make(model, sh[d]).eval()
            p, f, ms = measure(net, torch.randn(1, sh[d]["c"], sh[d]["l"]))
            rows[model][d] = {"params": p, "flops": f, "latency_ms": ms}
            if model in OURS:  # sanity: the parameter count must match the trained model's
                rec = json.loads((CA.RES / model / d / "split-00.json").read_text())["model"]["params"]
                if rec != p:
                    print(f"  WARNING {model}/{d}: params {p} vs trained {rec}", flush=True)
            print(f"{model:30s} {d:20s} {p / 1e3:8.1f}k {f / 1e6:9.2f} MFLOP {ms:8.2f} ms", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "costs.json").write_text(json.dumps({"shapes": {d: {k: v for k, v in s.items() if k != "names"} for d, s in sh.items()},
                                                "rows": rows}, indent=1) + "\n")
    return rows


def gmean(v: list[float]) -> float:
    return math.exp(st.mean(math.log(max(x, 1e-9)) for x in v))


def table(rows: dict) -> list[dict]:
    out = []
    for m in OURS + PAPER_DEEP:
        ours = m in OURS
        if HELDOUT_MODE and ours:  # held-out datasets are not in compare_arena.N_SPLITS
            from src.evaluation import heldout_analysis as HA
            f1 = {d: HA.mean_f1(m, d, (0,)) for d in DS}
            f3 = {d: HA.mean_f1(m, d) for d in DS}
        else:
            f1 = {d: (CA.ours(m, d, 0) if ours else CA.PAPER[d][m][0]) for d in DS}
            f3 = {d: CA.ours_seed_mean(m, d) for d in DS} if ours else None
        out.append({"model": m, "label": LABEL.get(m, m), "ours": ours,
                    "f1": st.mean(f1.values()),
                    "f1_3seed": st.mean(f3.values()) if f3 and None not in f3.values() else None,
                    "params": gmean([rows[m][d]["params"] for d in DS]),
                    "params_utd": rows[m][DS[0]]["params"],
                    "flops": gmean([rows[m][d]["flops"] for d in DS]),
                    "latency_ms": gmean([rows[m][d]["latency_ms"] for d in DS])})
    return out


def front(items: list[dict], key: str) -> list[dict]:
    """Pareto-optimal items: no other item is both cheaper (or equal) and more accurate."""
    best, out = -1.0, []
    for it in sorted(items, key=lambda r: (r[key], -r["f1"])):
        if it["f1"] > best:
            out.append(it)
            best = it["f1"]
    return out


def panel(body, x0, y0, w, h, items, key, xlab, fmt_tick, ticks, note) -> None:
    body.append(rect(x0, y0, w, h, "#FFFFFF", BORDER, 6))
    px0, px1, py0, py1 = x0 + 54, x0 + w - 18, y0 + h - 40, y0 + 34
    lo_x, hi_x = math.log10(min(ticks)), math.log10(max(ticks))
    lo_y = 2 * math.floor((min(i["f1"] for i in items if i["f1"] > 55) - 1) / 2)
    hi_y = 2 * math.ceil((max(i["f1"] for i in items) + 1) / 2)
    sx = lambda v: px0 + (math.log10(v) - lo_x) / (hi_x - lo_x) * (px1 - px0)  # noqa: E731
    sy = lambda v: py0 - (v - lo_y) / (hi_y - lo_y) * (py0 - py1)  # noqa: E731
    for yv in range(lo_y, hi_y + 1, 2):
        body.append(line(px0, sy(yv), px1, sy(yv), GRID))
        body.append(t(px0 - 8, sy(yv) + 4, f"{yv}", 10, MUTED, "end"))
    for tv in ticks:
        body.append(line(sx(tv), py1, sx(tv), py0, GRID))
        body.append(t(sx(tv), py0 + 16, fmt_tick(tv), 10, MUTED, "middle"))
    body.append(t((px0 + px1) / 2, y0 + h - 8, xlab, 10.5, INK2, "middle"))
    body.append(t(x0 + 12, y0 + 20, f"mean macro-F1, {SCOPE} (seed 0)", 11, INK, weight="600"))
    body.append(t(x0 + w - 14, y0 + 20, note, 10.5, INK2, "end"))
    # fronts: all models (ink), paper only (teal, dashed)
    for fr, col, dash, sw in ((front([i for i in items if not i["ours"]], key), C_DATA, True, 1.6),
                              (front(items, key), INK, False, 2.2)):
        pts = []
        for i, it in enumerate(fr):
            if i:
                pts.append((sx(it[key]), sy(fr[i - 1]["f1"])))
            pts.append((sx(it[key]), sy(it["f1"])))
        pts.append((px1, sy(fr[-1]["f1"])))
        d = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y) in enumerate(pts))
        da = ' stroke-dasharray="6 4"' if dash else ""
        body.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}"{da} stroke-linejoin="round"/>')
    on_front = {it["model"] for it in front(items, key)}
    # labels avoid every marker and every earlier label
    placed: list[tuple[float, float, float, float]] = [(sx(i[key]) - 6, sy(i["f1"]) - 6, sx(i[key]) + 6, sy(i["f1"]) + 6)
                                                       for i in items]
    for it in sorted(items, key=lambda r: -r["f1"]):
        x, y = sx(it[key]), sy(it["f1"])
        col = C_MODEL if it["ours"] else C_DATA
        if it["ours"]:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="#FFFFFF" stroke-width="1.5"/>')
        else:
            body.append(f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="1.5" fill="#FFFFFF" stroke="{col}" stroke-width="2"/>')
        # label: try right, then left, then above/below, skipping spots already used
        lab = it["label"]
        wtxt = 5.6 * len(lab)
        for dx, dy, anchor in ((9, 4, "start"), (-9, 4, "end"), (9, -9, "start"), (9, 15, "start"), (-9, -9, "end"),
                               (-9, 15, "end"), (0, -11, "middle"), (0, 19, "middle"), (14, -16, "start"), (-14, -16, "end"),
                               (14, 24, "start"), (-14, 24, "end")):
            lx = x + dx if anchor == "start" else x + dx - wtxt if anchor == "end" else x - wtxt / 2
            ly = y + dy
            box = (lx, ly - 9, lx + wtxt, ly + 2)
            if box[0] < px0 or box[2] > px1 + 4 or box[1] < py1 - 6 or box[3] > py0:
                continue
            if all(box[2] < a[0] or box[0] > a[2] or box[3] < a[1] or box[1] > a[3] for a in placed):
                placed.append(box)
                # white halo keeps the label readable where it crosses a front line
                body.append(t(x + dx, ly, lab, 10, INK if it["model"] in on_front else INK2, anchor,
                              "600" if it["model"] in on_front else "normal")
                            .replace("<text ", '<text stroke="#FFFFFF" stroke-width="3.5" paint-order="stroke" ', 1))
                break


def draw(rows: dict) -> str:
    items = table(rows)
    W, ph = 1120, 420
    H = 92 + 2 * (ph + 16) + 34
    body: list[str] = []
    title(body, "Accuracy against cost: Pareto fronts of our models and the WHAR Arena paper's 14 deep models",
          f"Cost measured on each dataset's real input (3-s windows), geometric mean over the {SCOPE}. Solid line: front of "
          "all models; dashed: front of the paper's models alone. Bold labels sit on the overall front.")
    lx = 24
    body.append(f'<circle cx="{lx + 6}" cy="72" r="5.5" fill="{C_MODEL}"/>')
    body.append(t(lx + 16, 76, "ours", 10.5, INK2))
    body.append(f'<rect x="{lx + 60}" y="67" width="10" height="10" rx="1.5" fill="#FFFFFF" stroke="{C_DATA}" stroke-width="2"/>')
    body.append(t(lx + 76, 76, "WHAR Arena paper (deep)", 10.5, INK2))
    body.append(line(lx + 230, 72, lx + 262, 72, INK, 2.2))
    body.append(t(lx + 268, 76, "front, all models", 10.5, INK2))
    body.append(f'<path d="M{lx + 380},72 L{lx + 412},72" stroke="{C_DATA}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    body.append(t(lx + 418, 76, "front, paper models only", 10.5, INK2))
    k = lambda v: f"{v / 1e3:g}k" if v < 1e6 else f"{v / 1e6:g}M"  # noqa: E731
    y = 92
    panel(body, 24, y, W - 48, ph, items, "params", "trainable parameters (log scale)", k,
          [5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 3e6], "fewer parameters → left")
    y += ph + 16
    panel(body, 24, y, W - 48, ph, items, "latency_ms", "CPU latency per window, PyTorch eager, 1 thread, batch 1 (ms, log scale)",
          lambda v: f"{v:g}", [0.3, 0.5, 1, 2, 5, 10, 20, 50], "faster → left")
    body.append(t(24, H - 14, "Latency depends on the implementation (unfused PyTorch on a desktop CPU), not only on the "
                  "architecture. FLOPs are left out: the counter misses LSTM, FFT and scan work, so they are not comparable.",
                  10.5, MUTED))
    return svg(W, H, "Pareto fronts of accuracy against parameters and latency", body)


def summary(rows: dict) -> None:
    items = table(rows)
    fr = {key: {it["model"] for it in front(items, key)} for key in ("params", "latency_ms")}
    L = ["# Accuracy against cost (Pareto)", "",
         f"Mean macro-F1 over the {SCOPE} at seed 0 (the paper ran once); cost = geometric mean over the {SCOPE} of "
         "the measured cost on each dataset's input. ★ = on the Pareto front of all models for that cost. FLOPs count "
         "matmul/conv only (LSTM, FFT and scan work is missed), so they are listed without a front.", "",
         f"| Model | Source | F1 seed 0 | F1 3 seeds | Params | Params ({DS[0]}) | MFLOPs / window | CPU ms / window |",
         "|---|---|---|---|---|---|---|---|"]
    for it in sorted(items, key=lambda r: -r["f1"]):
        star = lambda key: " ★" if it["model"] in fr[key] else ""  # noqa: E731
        f3 = "" if it["f1_3seed"] is None else f"{it['f1_3seed']:.2f}"
        L.append(f"| {it['label']} | {'ours' if it['ours'] else 'paper'} | {it['f1']:.2f} | {f3} | "
                 f"{it['params'] / 1e3:.1f}k{star('params')} | {it['params_utd'] / 1e3:.1f}k | {it['flops'] / 1e6:.2f} | "
                 f"{it['latency_ms']:.2f}{star('latency_ms')} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="redraw from results/pareto/costs.json")
    ap.add_argument("--heldout", action="store_true", help="the 8 held-out datasets of RESEARCH_PLAN.md §5p")
    args = ap.parse_args()
    if args.heldout:
        from src.evaluation import heldout_analysis as HA
        global OUT, DS, OURS, FIGURE, SCOPE, HELDOUT_MODE
        OUT, DS, OURS = PATHS.METRICS / "pareto_heldout", HA.HELDOUT, list(HA.OURS)
        FIGURE, SCOPE, HELDOUT_MODE = "pareto_heldout.svg", "8 held-out datasets", True
    rows = json.loads((OUT / "costs.json").read_text())["rows"] if args.plot else run_measure()
    summary(rows)
    write(FIGURE, draw(rows))


if __name__ == "__main__":
    sys.exit(main())
