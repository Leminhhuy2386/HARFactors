"""Figures for the synthesis (RESULTS_SUMMARY.md §12, Edge-SSM atlas "Synthesis"): standalone SVG files in
figures/, each with its own light background so it reads in any Markdown viewer, light or dark.

- factor_map.svg      : data, model and evaluation factors pointing at macro-F1, arrow width = measured effect;
- variance.svg        : where the variance of macro-F1 comes from (2,460 runs);
- data_factors.svg    : three dataset properties against three outcomes over the paper's 30 datasets;
- data_corr.svg       : Spearman correlations and the OLS regression over the 30 datasets;
- components_forest.svg : paired effect of each model component, 95 % interval over datasets;
- components_heatmap.svg: the same effects per dataset;
- eval_noise.svg      : person spread vs seed spread vs choice of memory, and the ActRecTut flip;
- ranking.svg         : all factors on one scale.

Reads results/metrics/factors/data.json (factor_analysis.py) and results/metrics/synthesis/data.json (synthesis.py); run those first.
    uv run python -m src.visualization.build_factor_figures
"""

from __future__ import annotations

import json
import math
import statistics as st
from html import escape
from pathlib import Path

from src.evaluation import compare_arena as CA
from src.evaluation import factor_analysis as FA
from src.utils import paths as PATHS  # noqa: E402

HERE = Path(__file__).resolve().parents[2]  # project root
OUT = PATHS.FIGURES
BG, BORDER, INK, INK2, MUTED, GRID, PANEL = "#FCFCFB", "#D9D8D2", "#1B2A33", "#4B5B63", "#8A8984", "#E4E3DD", "#FFFFFF"
C_DATA, C_MODEL, C_EVAL = "#1D6F8C", "#B85C1E", "#5B4AA8"
POS, NEG, MID = "#1D6F8C", "#A3336E", "#F0EFEC"
FONT = 'font-family="Segoe UI, system-ui, -apple-system, sans-serif"'


# ------------------------------------------------------------------ drawing helpers
def t(x, y, s, size=12, color=INK, anchor="start", weight="normal", italic=False) -> str:
    style = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{style}>{escape(s)}</text>')


def rect(x, y, w, h, fill=PANEL, stroke=BORDER, rx=6, sw=1, dash=False) -> str:
    d = ' stroke-dasharray="5 3"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def line(x1, y1, x2, y2, color=GRID, sw=1, dash=False) -> str:
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}/>'


def svg(w, h, label, body: list[str], defs: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{escape(label)}" {FONT}>{defs}' + rect(0.5, 0.5, w - 1, h - 1, BG, BORDER, 10) + "".join(body) + "</svg>")


def title(body: list[str], head: str, sub: str, x=24) -> None:
    body.append(t(x, 34, head, 15, INK, weight="600"))
    body.append(t(x, 54, sub, 11, INK2))


def mix(c1: str, c2: str, a: float) -> str:
    """Blend hex c1 toward c2 by a in [0, 1]."""
    p = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    q = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(u + (v - u) * a):02x}" for u, v in zip(p, q))


def diverging(v: float, cap: float) -> str:
    a = max(-1.0, min(1.0, v / cap))
    return mix(MID, POS, a) if a >= 0 else mix(MID, NEG, -a)


def fmt_p(pv: float) -> str:
    return "p<0.001" if pv < 0.001 else f"p {pv:.3f}"


def ink_on(fill: str) -> str:
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.5 else INK


def write(name: str, s: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(s, encoding="utf-8")
    print("wrote", OUT / name)


# ------------------------------------------------------------------ data
F = json.loads((PATHS.METRICS / "factors" / "data.json").read_text())
SYN = json.loads((PATHS.METRICS / "synthesis" / "data.json").read_text())
COMP = F["components"]
DS = CA.DATASETS
D30 = [d for d in CA.PAPER if d != "Mean"]
DEEP = [a for a in CA.ARENA if a not in CA.CLASSICAL]


def comp(a: str, b: str) -> dict:
    return COMP[f"{a}-{b}"]


# ------------------------------------------------------------------ 1. factor map
def factor_map() -> str:
    W, H = 1120, 900
    v = F["variance"]
    within = sum(x for k, x in v["ss"].items() if k != "dataset")
    share = {k: 100 * x / within for k, x in v["ss"].items() if k != "dataset"}
    tot = 100 * v["ss"]["dataset"] / v["total"]
    rms = {k: math.sqrt(x / v["n"]) for k, x in v["ss"].items()}
    ols = F["data"]["ols"]
    m = lambda a, b: comp(a, b)  # noqa: E731
    mem, laws, stats_, mamba = m("recurrencenet", "hystnet_nomem"), m("lawnet_rec", "recurrencenet"), \
        m("recstatnet", "recurrencenet"), m("mamba", "s4d")
    data = [  # (title, line1, line2, magnitude, robust)
        ("Which dataset", f"{tot:.0f} % of all variance (RMS {rms['dataset']:.0f} pts)",
         f"not predicted by its size (R² {ols['median deep F1 (level)']['r2']:.2f}, 30 sets)", rms["dataset"], True),
        ("Training subjects", "fewer people: models differ more", "(−1.5 pts per sd, p 0.005); trees beat deep", 6.0, True),
        ("Number of channels", "more channels: Random Forest beats deep", "(+7.2 pts per sd, p < 0.001, n 30)", 7.2, True),
        ("Sampling rate (3-s windows)", "memory gain not tied to samples/window", "(ρ −0.36, n 10); WISDM 60 samples: +15", 1.5, False),
        ("Number of classes", "memory gain larger with more classes", "(ρ +0.48 to +0.57, not significant)", 2.0, False),
        ("Activity type", "gesture: shape and style; gait: memory;", "many sensors: group statistics", 4.0, False),
    ]
    model = [
        ("Temporal memory vs none", f"+{mem['mean']:.1f} [{mem['lo']:+.1f}, {mem['hi']:+.1f}], {mem['wins']}/{mem['n']} datasets",
         "largest on gait (WISDM +15)", mem["mean"], True),
        ("Window laws (lagged 2nd order)", f"+{laws['mean']:.1f} [{laws['lo']:+.1f}, {laws['hi']:+.1f}] on RN, {laws['wins']}/{laws['n']}",
         "+5.5 on a memoryless net", laws["mean"], True),
        ("Window statistics (std, min/max)", f"+{stats_['mean']:.1f} [{stats_['lo']:+.1f}, {stats_['hi']:+.1f}] on RN",
         "+21 on ActRecTut, +0.6 on the other 9", stats_["mean"], False),
        ("Memory type", "vs S4D: RN +0.7, Hyst −0.6, Quant −1.6 (n.s.)",
         f"Mamba {mamba['mean']:+.1f} (p {mamba['p']:.2f})", 1.6, False),
        ("Per-sensor weight sharing", "+5 on SAD and Opportunity,", "−16 on single-board UTD-MHAD (seed 0)", 5.0, False),
        ("Normalization / pooling", "≤ ±2 points; dropping LayerNorm", "−2.2 outside ActRecTut", 1.0, False),
        ("Model size", "no gain: ρ(log params, F1) −0.43", "over 30 datasets (14 deep models)", 0.4, False),
    ]
    evals = [
        ("Test person", f"{share['test person (split)']:.0f} % of variance within a", f"dataset; split sd ≈ 8 pts", rms["test person (split)"], True),
        ("Training seed", f"{share['seed']:.0f} %; ±1.2–1.7 pts on a", "dataset mean (ActRecTut 6)", 1.5, True),
        ("Model choice", f"{share['model']:.0f} % (+{share['model x dataset']:.0f} % × dataset,", f"+{share['model x person']:.0f} % × person)", rms["model"], True),
        ("A 2-person dataset", "ActRecTut alone flips the", "ranking of the top models", 3.0, False),
        ("Weak baselines, reused", "paper RF 65 vs tuned trees ~74.5;", "dev datasets: p-values optimistic", 3.0, False),
    ]
    body: list[str] = []
    title(body, "What moves macro-F1 on unseen people: the factors and their measured effects",
          "Arrow width grows with the typical effect in F1 points; dashed = not significant or sign depends on the dataset. "
          "Sources: 30 paper datasets, our 10 × 3 seeds, 2,460 runs.")
    cx, cy, nw, nh = 560, 380, 250, 96
    node = (cx - nw / 2, cy - nh / 2)
    body.append(rect(node[0], node[1], nw, nh, "#FFFFFF", INK, 12, 2))
    body.append(t(cx, cy - 12, "Macro-F1", 18, INK, "middle", "700"))
    body.append(t(cx, cy + 10, "on people never seen in training", 11.5, INK2, "middle"))
    body.append(t(cx, cy + 28, "(WHAR Arena protocol)", 10.5, MUTED, "middle"))

    def width(mag: float) -> float:
        return min(14.0, 1.2 + 0.8 * mag)

    def column(items, x, color, side: str, y0=84, bh=78, gap=10) -> None:
        body.append(t(x, y0 - 8, {"l": "DATA", "r": "MODEL"}[side], 11, color, weight="700"))
        for i, (head, l1, l2, mag, robust) in enumerate(items):
            y = y0 + i * (bh + gap)
            bw = 300
            body.append(rect(x, y, bw, bh, "#FFFFFF", color, 7, 1.4, dash=not robust))
            body.append(rect(x, y, 6, bh, color, color, 2, 0))
            body.append(t(x + 16, y + 22, head, 12.5, INK, weight="600"))
            body.append(t(x + 16, y + 42, l1, 10.8, INK2))
            body.append(t(x + 16, y + 59, l2, 10.8, INK2))
            ym = y + bh / 2
            if side == "l":
                x1, x2 = x + bw, node[0]
            else:
                x1, x2 = x, node[0] + nw
            yt = cy + (i - (len(items) - 1) / 2) * 9
            c1 = x1 + (x2 - x1) * 0.5
            dash = "" if robust else ' stroke-dasharray="7 5"'
            body.append(f'<path d="M{x1:.1f},{ym:.1f} C{c1:.1f},{ym:.1f} {c1:.1f},{yt:.1f} {x2:.1f},{yt:.1f}" fill="none" '
                        f'stroke="{color}" stroke-opacity="0.55" stroke-width="{width(mag):.1f}" stroke-linecap="round"{dash}/>')

    column(data, 24, C_DATA, "l")
    column(model, W - 324, C_MODEL, "r", y0=84, bh=72, gap=8)
    # evaluation row
    ey, eh = 730, 92
    ew = (W - 48 - 4 * 12) / 5
    body.append(t(24, ey - 10, "EVALUATION (why a ranking can flip)", 11, C_EVAL, weight="700"))
    for i, (head, l1, l2, mag, robust) in enumerate(evals):
        x = 24 + i * (ew + 12)
        body.append(rect(x, ey, ew, eh, "#FFFFFF", C_EVAL, 7, 1.4, dash=not robust))
        body.append(rect(x, ey, ew, 6, C_EVAL, C_EVAL, 2, 0))
        body.append(t(x + 12, ey + 28, head, 12.5, INK, weight="600"))
        body.append(t(x + 12, ey + 50, l1, 10.8, INK2))
        body.append(t(x + 12, ey + 67, l2, 10.8, INK2))
        xm = x + ew / 2
        xt = cx + (i - 2) * 45
        dash = "" if robust else ' stroke-dasharray="7 5"'
        # rise a little, run sideways below the model column (which ends above y 660), then go straight up
        yb = ey - 52
        body.append(f'<path d="M{xm:.1f},{ey:.1f} C{xm:.1f},{ey - 30:.1f} {xt:.1f},{ey - 22:.1f} {xt:.1f},{yb:.1f} '
                    f'L{xt:.1f},{cy + nh / 2:.1f}" '
                    f'fill="none" stroke="{C_EVAL}" stroke-opacity="0.55" stroke-width="{width(mag):.1f}" stroke-linecap="round"{dash}/>')
    # width legend
    lx, ly = 24, H - 18
    body.append(t(lx, ly, "arrow width", 10.5, MUTED))
    for k, mag in enumerate((1, 3, 6, 12)):
        x = lx + 70 + k * 72
        body.append(line(x, ly - 4, x + 34, ly - 4, INK2, width(mag)))
        body.append(t(x + 40, ly, f"{mag}", 10.5, MUTED))
    body.append(t(lx + 70 + 4 * 72 - 6, ly, "pts", 10.5, MUTED))
    return svg(W, H, "Factor map: data, model and evaluation factors and their effects on macro-F1", body)


# ------------------------------------------------------------------ 2. variance
def variance() -> str:
    v = F["variance"]
    W, H = 1120, 380
    total = v["total"]
    within = {k: x for k, x in v["ss"].items() if k != "dataset"}
    tw = sum(within.values())
    body: list[str] = []
    title(body, f"Where the variance of macro-F1 comes from ({v['n']:,} runs: 10 of our models × 10 datasets × subject splits × 3 seeds)",
          "Sum-of-squares decomposition. Top: everything. Bottom: what is left once the dataset is fixed, i.e. what decides a ranking.")
    # stacked bar, total
    x0, y0, bw = 24, 84, W - 48
    order = ["dataset", "test person (split)", "seed", "model x person", "model x dataset", "model"]
    colors = {"dataset": "#BDBBB3", "test person (split)": C_EVAL, "seed": mix(C_EVAL, "#FFFFFF", 0.45),
              "model x person": mix(C_MODEL, "#FFFFFF", 0.45), "model x dataset": mix(C_MODEL, "#FFFFFF", 0.2),
              "model": C_MODEL}
    x = x0
    for k in order:
        w = bw * v["ss"][k] / total
        body.append(f'<rect x="{x:.1f}" y="{y0}" width="{max(w - 2, 0.5):.1f}" height="34" rx="3" fill="{colors[k]}"/>')
        if w > 90:
            body.append(t(x + 10, y0 + 22, f"{k}: {100 * v['ss'][k] / total:.1f} %", 12, ink_on(colors[k]), weight="600"))
        x += w
    body.append(t(W - 24, y0 + 56, "person 14.4 % · seed 3.3 % · model×person 2.6 % · model×dataset 1.7 % · model 1.5 %",
                  10.5, INK2, "end"))
    # within-dataset bars
    yb, lab_w = 172, 230
    body.append(t(24, yb - 14, "Within a dataset (100 % = everything except the dataset's own level)", 12, INK, weight="600"))
    rms = {k: math.sqrt(x / v["n"]) for k, x in v["ss"].items()}
    span = W - 48 - lab_w - 170
    for i, k in enumerate(order[1:]):
        y = yb + i * 36
        s = 100 * within[k] / tw
        body.append(t(24, y + 18, k, 12, INK))
        body.append(f'<rect x="{24 + lab_w}" y="{y + 4}" width="{span * s / 70:.1f}" height="22" rx="3" fill="{colors[k]}"/>')
        body.append(t(24 + lab_w + span * s / 70 + 8, y + 20, f"{s:.1f} %   (typical size {rms[k]:.1f} pts)", 11.5, INK2))
    return svg(W, H, "Variance decomposition of macro-F1", body)


# ------------------------------------------------------------------ 3. data factors (30 datasets)
def scatter_panel(body, x0, y0, w, h, xs, ys, labels, ours, xlab, ylab, xticks, note, highlight) -> None:
    body.append(rect(x0, y0, w, h, "#FFFFFF", BORDER, 6))
    px0, px1, py0, py1 = x0 + 46, x0 + w - 14, y0 + h - 38, y0 + 30
    lo_x, hi_x = min(min(xs), math.log2(min(xticks))) - 0.25, max(max(xs), math.log2(max(xticks))) + 0.25
    lo_y, hi_y = min(ys), max(ys)
    pad = 0.08 * (hi_y - lo_y)
    lo_y, hi_y = lo_y - pad, hi_y + pad
    step = 5 if hi_y - lo_y > 18 else 2 if hi_y - lo_y > 6 else 1
    sx = lambda v: px0 + (v - lo_x) / (hi_x - lo_x) * (px1 - px0)  # noqa: E731
    sy = lambda v: py0 - (v - lo_y) / (hi_y - lo_y) * (py0 - py1)  # noqa: E731
    yt = math.ceil(lo_y / step) * step
    while yt <= hi_y:
        body.append(line(px0, sy(yt), px1, sy(yt), INK2 if yt == 0 else GRID, 1.2 if yt == 0 else 1))
        body.append(t(px0 - 6, sy(yt) + 4, f"{yt:g}", 10, MUTED, "end"))
        yt += step
    for tv in xticks:  # real values; positions are log2
        xv = math.log2(tv)
        body.append(line(sx(xv), py0, sx(xv), py0 + 4, MUTED))
        body.append(t(sx(xv), py0 + 16, f"{tv:g}", 10, MUTED, "middle"))
    body.append(t((px0 + px1) / 2, y0 + h - 6, xlab, 10.5, INK2, "middle"))
    body.append(t(x0 + 12, y0 + 18, ylab, 11, INK, weight="600"))
    body.append(t(x0 + w - 12, y0 + 18, note, 10.5, INK2, "end"))
    # least-squares line in log space, for the eye
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    xa, xb = min(xs), max(xs)
    body.append(line(sx(xa), sy(my + b * (xa - mx)), sx(xb), sy(my + b * (xb - mx)), mix(C_DATA, "#FFFFFF", 0.35), 2, True))
    for xv, yv, lab, our in zip(xs, ys, labels, ours):
        if our:
            body.append(f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="4.6" fill="{C_DATA}" stroke="#FFFFFF" stroke-width="1.5"/>')
        else:
            body.append(f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="4.2" fill="#FFFFFF" stroke="{MUTED}" stroke-width="1.4"/>')
        if lab in highlight:
            dx, anchor = highlight[lab]
            body.append(t(sx(xv) + dx, sy(yv) - 7, lab, 9.5, INK2, anchor))


def data_factors() -> str:
    W, H = 1120, 470
    body: list[str] = []
    title(body, "Data factors over the paper's 30 datasets (17 models each, one run)",
          "Filled = our 10 datasets, open = the other 20. Dashed line: least squares, for the eye. ρ = Spearman; β = OLS "
          "change per sd of the property with the other three held fixed.")
    ch = [math.log2(CA.CHANNELS[d]) for d in D30]
    su = [math.log2(FA.CFG[d][1]) for d in D30]
    f = lambda d, a: CA.PAPER[d][a][0]  # noqa: E731
    rf = [f(d, "Random Forest") - max(f(d, a) for a in DEEP) for d in D30]
    spread = [max(f(d, a) for a in DEEP) - st.median(f(d, a) for a in DEEP) for d in D30]
    psd = [st.mean(CA.PAPER[d][a][1] for a in DEEP) for d in D30]
    ours = [d in DS for d in D30]
    short = {d: CA.SHORT.get(d, d) for d in D30}
    labels = [short[d] for d in D30]
    corr = F["data"]["corr"]
    ols = F["data"]["ols"]
    r = lambda o, p: corr[f"{o}|{p}"][0]  # noqa: E731
    pw = (W - 48 - 2 * 16) / 3
    scatter_panel(body, 24, 76, pw, 370, ch, rf, labels, ours, "channels (log scale)", "Random Forest − best deep model",
                  [3, 6, 12, 24, 48, 96],
                  f"ρ {r('RF − best deep (classical margin)', 'log2 channels'):+.2f}, β +7.2 (p<0.001)",
                  {"Opportunity": (-7, "end"), "SAD": (-7, "end"), "UTD-MHAD": (7, "start"), "WISDM": (7, "start")})
    scatter_panel(body, 24 + pw + 16, 76, pw, 370, su, spread, labels, ours, "training subjects (log scale)",
                  "best − median deep model", [2, 4, 8, 16, 32],
                  f"ρ {r('best − median deep (does the model matter)', 'log2 subjects'):+.2f}, β −1.5 (p 0.005)",
                  {"ActRecTut": (6, "start"), "Opportunity": (6, "start"), "UTD-MHAD": (6, "start"), "WISDM": (-6, "end")})
    scatter_panel(body, 24 + 2 * (pw + 16), 76, pw, 370, ch, psd, labels, ours, "channels (log scale)",
                  "spread over test people (split sd)", [3, 6, 12, 24, 48, 96],
                  f"ρ {r('mean split sd of deep models (person spread)', 'log2 channels'):+.2f}, β +2.5 (p 0.002)",
                  {"Opportunity": (-7, "end"), "UTD-MHAD": (7, "start"), "UCI-HAR": (7, "start")})
    _ = ols
    return svg(W, H, "Data factors over 30 datasets", body)


def data_corr() -> str:
    W = 1120
    corr, ols = F["data"]["corr"], F["data"]["ols"]
    outs = ["median deep F1 (level)", "best − median deep (does the model matter)", "RF − best deep (classical margin)",
            "mean split sd of deep models (person spread)"]
    olab = ["Level of the deep models (median F1)", "How much the model matters (best − median)",
            "Classical margin (RF − best deep)", "Spread over test people (split sd)"]
    preds = ["log2 subjects", "log2 channels", "log2 classes", "log2 samples/window"]
    plab = ["subjects", "channels", "classes", "samples per 3-s window"]
    use = ["log2 subjects", "log2 channels", "log2 classes", "log2 samples/window"]
    H = 120 + 4 * 40 + 70 + 4 * 34 + 40
    body: list[str] = []
    title(body, "Which dataset properties go with which outcome (30 datasets)",
          "Cell = Spearman ρ, colour scaled to ±0.8; bold = p < 0.05. Lower table: OLS regression with standardized "
          "properties (β = F1 points per sd), n = 30.")
    x0, lw, cw, y0 = 24, 330, 180, 84
    for j, p in enumerate(plab):
        body.append(t(x0 + lw + j * cw + cw / 2, y0, p, 11.5, INK2, "middle", "600"))
    for i, (o, ol) in enumerate(zip(outs, olab)):
        y = y0 + 12 + i * 40
        body.append(t(x0, y + 25, ol, 12, INK))
        for j, p in enumerate(preds):
            rho, pv = corr[f"{o}|{p}"]
            fill = diverging(rho, 0.8)
            body.append(rect(x0 + lw + j * cw + 2, y + 4, cw - 4, 32, fill, "none", 4, 0))
            body.append(t(x0 + lw + j * cw + cw / 2, y + 25, f"{rho:+.2f}" + (f"  {fmt_p(pv)}" if pv < 0.1 else ""), 11.5,
                          ink_on(fill), "middle", "700" if pv < 0.05 else "normal"))
    y1 = y0 + 12 + 4 * 40 + 40
    body.append(t(x0, y1, "OLS (all four properties together)", 12, INK, weight="600"))
    body.append(t(x0 + lw - 40, y1, "R²", 11.5, INK2, "middle", "600"))
    for j, p in enumerate(["subjects", "channels", "classes", "samples per 3-s window"]):
        body.append(t(x0 + lw + j * cw + cw / 2, y1, p, 11.5, INK2, "middle", "600"))
    for i, (o, ol) in enumerate(zip(outs, olab)):
        y = y1 + 10 + i * 34
        r = ols[o]
        body.append(t(x0, y + 20, ol, 12, INK))
        body.append(t(x0 + lw - 40, y + 20, f"{r['r2']:.2f}", 12, INK, "middle", "700" if r["r2"] > 0.3 else "normal"))
        for j in range(len(use)):
            b, pv = r["beta"][j + 1], r["p"][j + 1]
            body.append(t(x0 + lw + j * cw + cw / 2, y + 20, f"{b:+.2f}" + (f" ({fmt_p(pv)})" if pv < 0.1 else ""), 11.5,
                          POS if (pv < 0.05 and b > 0) else NEG if (pv < 0.05 and b < 0) else INK2, "middle",
                          "700" if pv < 0.05 else "normal"))
    return svg(W, H, "Correlation and regression of dataset properties against outcomes", body)


# ------------------------------------------------------------------ 4. components
ROWS = [  # (group, a, b)
    ("Memory vs none", "recurrencenet", "hystnet_nomem"), ("Memory vs none", "s4d", "hystnet_nomem"),
    ("Memory vs none", "hystnet_sel_clock", "hystnet_nomem"), ("Memory vs none", "quantilenet", "hystnet_nomem"),
    ("Memory vs none", "mamba", "hystnet_nomem"),
    ("Memory type (vs S4D)", "recurrencenet", "s4d"), ("Memory type (vs S4D)", "hystnet_sel_clock", "s4d"),
    ("Memory type (vs S4D)", "quantilenet", "s4d"), ("Memory type (vs S4D)", "mamba", "s4d"),
    ("Window statistics", "statnet", "hystnet_nomem"), ("Window statistics", "recstatnet", "recurrencenet"),
    ("Window laws", "lawnet", "hystnet_nomem"), ("Window laws", "lawnet_rec", "recurrencenet"),
    ("Window laws", "lawnet_rec", "recstatnet"),
    ("Sensor grouping (seed 0)", "lawnet", "lawnet_rand"),
]


def components_forest() -> str:
    W = 1120
    rh, gh = 26, 26
    groups = list(dict.fromkeys(g for g, _, _ in ROWS))
    H = 100 + len(ROWS) * rh + len(groups) * gh + 40
    body: list[str] = []
    title(body, "Model components: paired effect on macro-F1 (10 datasets, same splits, 3 seeds unless noted)",
          "Dot = mean over datasets of (a − b); bar = 95 % t-interval over datasets; filled = interval excludes 0. Right: "
          "datasets won and Wilcoxon p.")
    lx, px0, px1 = 24, 360, 820
    lo, hi = -5, 10
    sx = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)  # noqa: E731
    ytop, ybot = 84, H - 34
    for v in range(lo, hi + 1, 1 if False else 5):
        pass
    for v in (-5, -2.5, 0, 2.5, 5, 7.5, 10):
        body.append(line(sx(v), ytop, sx(v), ybot, INK2 if v == 0 else GRID, 1.3 if v == 0 else 1))
        body.append(t(sx(v), ybot + 16, f"{v:+g}" if v else "0", 10, MUTED, "middle"))
    body.append(t(px1 + 20, ytop + 2, "won", 10.5, MUTED))
    body.append(t(px1 + 80, ytop + 2, "mean [95 % CI]", 10.5, MUTED))
    body.append(t(px1 + 210, ytop + 2, "p", 10.5, MUTED))
    y = ytop + 8
    cur = None
    for g, a, b in ROWS:
        if g != cur:
            y += gh
            body.append(t(lx, y - 6, g, 12, C_MODEL, weight="700"))
            cur = g
        c = comp(a, b)
        yc = y + rh / 2
        body.append(t(lx + 14, yc + 4, f"{FA.NAME[a]} − {FA.NAME[b]}", 11.5, INK))
        sig = not math.isnan(c["lo"]) and (c["lo"] > 0 or c["hi"] < 0)
        if not math.isnan(c["lo"]):
            body.append(line(sx(max(lo, c["lo"])), yc, sx(min(hi, c["hi"])), yc, C_MODEL, 2.2))
        body.append(f'<circle cx="{sx(c["mean"]):.1f}" cy="{yc:.1f}" r="5.5" fill="{C_MODEL if sig else "#FFFFFF"}" '
                    f'stroke="{C_MODEL}" stroke-width="2"/>')
        body.append(t(px1 + 20, yc + 4, f"{c['wins']}/{c['n']}", 11, INK2))
        body.append(t(px1 + 80, yc + 4, f"{c['mean']:+.2f} [{c['lo']:+.1f}, {c['hi']:+.1f}]", 11, INK, weight="600" if sig else "normal"))
        body.append(t(px1 + 210, yc + 4, f"{c['p']:.3f}" if not math.isnan(c["p"]) else "–", 11, INK2))
        y += rh
    return svg(W, H, "Forest plot of model component effects", body)


def components_heatmap() -> str:
    W = 1120
    rh = 30
    H = 140 + len(ROWS) * rh + 60
    body: list[str] = []
    title(body, "The same component effects, dataset by dataset (F1 points, a − b)",
          "Colour: blue = a better, magenta = a worse, scaled to ±10 (stronger values saturate). ★ = 95 % interval over the "
          "dataset's subject splits excludes 0. Under each dataset: subjects / channels.")
    lx, x0, cw, y0 = 24, 330, 74.5, 104
    for j, d in enumerate(DS):
        body.append(t(x0 + j * cw + cw / 2, y0, CA.SHORT[d], 11, INK, "middle", "600"))
        body.append(t(x0 + j * cw + cw / 2, y0 + 15, f"{FA.CFG[d][1]} / {CA.CHANNELS[d]}", 10, MUTED, "middle"))
    for i, (g, a, b) in enumerate(ROWS):
        y = y0 + 26 + i * rh
        c = comp(a, b)
        body.append(t(lx, y + 19, f"{FA.NAME[a]} − {FA.NAME[b]}", 11.5, INK))
        for j, d in enumerate(DS):
            p = c["per_dataset"].get(d)
            if p is None:
                continue
            fill = diverging(p["effect"], 10)
            body.append(rect(x0 + j * cw + 1.5, y + 2, cw - 3, rh - 4, fill, "none", 3, 0))
            star = "★" if not math.isnan(p["lo"]) and (p["lo"] > 0 or p["hi"] < 0) else ""
            body.append(t(x0 + j * cw + cw / 2, y + 19, f"{p['effect']:+.1f}{star}", 11, ink_on(fill), "middle"))
    # legend
    ly = H - 30
    body.append(t(lx, ly + 4, "scale", 10.5, MUTED))
    for k, v in enumerate((-10, -5, -2, 0, 2, 5, 10)):
        x = lx + 50 + k * 54
        body.append(rect(x, ly - 10, 50, 18, diverging(v, 10), "none", 3, 0))
        body.append(t(x + 25, ly + 3, f"{v:+d}" if v else "0", 10, ink_on(diverging(v, 10)), "middle"))
    return svg(W, H, "Heatmap of component effects per dataset", body)


# ------------------------------------------------------------------ 5. evaluation noise
def eval_noise() -> str:
    W, H = 1120, 480
    body: list[str] = []
    title(body, "Evaluation factors: who is tested moves results more than the model does",
          "Left: spread over test people (split sd, RecurrenceNet seed 0), over seeds (sd of the dataset mean), and between "
          "5 memory types (max − min).")
    body.append(t(24, 70, "Right: removing the one 2-person dataset (ActRecTut) reorders the top models.", 11, INK2))
    mems = ["recurrencenet", "s4d", "hystnet_sel_clock", "quantilenet", "mamba"]
    person = {d: st.stdev(list(CA.split_f1(CA.RES / "recurrencenet" / d).values())) for d in DS}
    seed = {d: st.stdev([CA.ours("recurrencenet", d, s) for s in (0, 1, 2)]) for d in DS}
    memspread = {d: max(CA.ours_seed_mean(m, d) for m in mems) - min(CA.ours_seed_mean(m, d) for m in mems) for d in DS}
    x0, y0, w, h = 24, 84, 640, 370
    body.append(rect(x0, y0, w, h, "#FFFFFF", BORDER, 6))
    px0, px1, py0, py1 = x0 + 40, x0 + w - 12, y0 + h - 40, y0 + 44
    top = 16
    sy = lambda v: py0 - v / top * (py0 - py1)  # noqa: E731
    for v in range(0, top + 1, 4):
        body.append(line(px0, sy(v), px1, sy(v), GRID))
        body.append(t(px0 - 6, sy(v) + 4, f"{v}", 10, MUTED, "end"))
    gw = (px1 - px0) / len(DS)
    cols = [("test person", C_EVAL), ("seed", mix(C_EVAL, "#FFFFFF", 0.5)), ("memory type", C_MODEL)]
    for j, d in enumerate(DS):
        for k, (name, col) in enumerate(cols):
            val = (person, seed, memspread)[k][d]
            bx = px0 + j * gw + 6 + k * (gw - 12) / 3
            body.append(f'<rect x="{bx:.1f}" y="{sy(val):.1f}" width="{(gw - 12) / 3 - 2:.1f}" height="{py0 - sy(val):.1f}" rx="2" fill="{col}"/>')
        body.append(t(px0 + j * gw + gw / 2, py0 + 15, CA.SHORT[d][:8], 9.5, INK2, "middle"))
    body.append(t(x0 + 12, y0 + 20, "Spread in F1 points", 11, INK, weight="600"))
    for k, (name, col) in enumerate(cols):
        lx = x0 + 200 + k * 130
        body.append(rect(lx, y0 + 10, 12, 12, col, "none", 2, 0))
        body.append(t(lx + 18, y0 + 20, name, 10.5, INK2))
    body.append(t(x0 + w / 2, y0 + h - 8, f"means: person {st.mean(person.values()):.1f}, seed {st.mean(seed.values()):.1f}, "
                  f"memory type {st.mean(memspread.values()):.1f}", 10.5, INK2, "middle"))
    # slope chart: mean of 10 vs mean without ActRecTut
    tab = SYN["table"]
    models = [("recstatnet", "RN + stats"), ("lawnet_rec", "RN + laws"), ("recurrencenet", "RecurrenceNet"),
              ("s4d", "S4D"), ("Paper CNN-HAR", "CNN-HAR (paper)"), ("Paper TinyHAR", "TinyHAR (paper)")]
    sx0, sy0, sw, sh = 680, 84, 416, 370
    body.append(rect(sx0, sy0, sw, sh, "#FFFFFF", BORDER, 6))
    body.append(t(sx0 + 12, sy0 + 20, "Mean F1: all 10 datasets → without ActRecTut", 11, INK, weight="600"))
    lo, hi = 70.5, 78
    yy = lambda v: sy0 + sh - 40 - (v - lo) / (hi - lo) * (sh - 90)  # noqa: E731
    xa, xb = sx0 + 150, sx0 + sw - 150
    body.append(line(xa, sy0 + 44, xa, sy0 + sh - 36, GRID))
    body.append(line(xb, sy0 + 44, xb, sy0 + sh - 36, GRID))
    body.append(t(xa, sy0 + sh - 18, "10 datasets", 10.5, MUTED, "middle"))
    body.append(t(xb, sy0 + sh - 18, "without ActRecTut", 10.5, MUTED, "middle"))
    def spread_out(ys: list[float], gap: float = 13.0) -> list[float]:
        """Move label baselines apart by at least gap, keeping their order and mean position."""
        order = sorted(range(len(ys)), key=lambda i: ys[i])
        out = [ys[i] for i in order]
        for k in range(1, len(out)):
            out[k] = max(out[k], out[k - 1] + gap)
        shift = (sum(ys) - sum(out)) / len(out)
        res = [0.0] * len(ys)
        for k, i in enumerate(order):
            res[i] = out[k] + shift
        return res

    pts = []
    for key, lab in models:
        row = tab[key]
        a = st.mean(row[d] for d in DS)
        b = st.mean(row[d] for d in DS if d != "actrectut_gestures")
        pts.append((key, lab, a, b, C_MODEL if not key.startswith("Paper") else MUTED))
    la = spread_out([yy(a) for _, _, a, _, _ in pts])
    lb = spread_out([yy(b) for _, _, _, b, _ in pts])
    for (key, lab, a, b, col), ya, yb in zip(pts, la, lb):
        body.append(line(xa, yy(a), xb, yy(b), col, 2))
        body.append(f'<circle cx="{xa:.1f}" cy="{yy(a):.1f}" r="4" fill="{col}"/>')
        body.append(f'<circle cx="{xb:.1f}" cy="{yy(b):.1f}" r="4" fill="{col}"/>')
        body.append(t(xa - 8, ya + 4, f"{lab} {a:.1f}", 10, INK2, "end"))
        body.append(t(xb + 8, yb + 4, f"{b:.1f} {lab}", 10, INK2))
    return svg(W, H, "Evaluation noise and the ActRecTut ranking flip", body)


# ------------------------------------------------------------------ 6. ranking
def ranking() -> str:
    v = F["variance"]
    rms = {k: math.sqrt(x / v["n"]) for k, x in v["ss"].items()}
    mem, laws, stats_ = comp("recurrencenet", "hystnet_nomem"), comp("lawnet_rec", "recurrencenet"), comp("recstatnet", "recurrencenet")
    items = [  # (label, kind, low, typical, high, note)
        ("Which dataset", "data", 0, rms["dataset"], 30, f"RMS {rms['dataset']:.0f} pts between datasets"),
        ("Who is tested (subject split)", "eval", 2.4, rms["test person (split)"], 15.5, "split sd 2.4–15.5; seen−unseen gap 14.5"),
        ("Classical trees vs deep, by subjects/channels", "data", -7.1, 0.4, 10.9, "−7 (many people) … +11 (few people)"),
        ("Temporal memory vs none", "model", mem["lo"], mem["mean"], mem["hi"], f"+{mem['mean']:.1f}, 10/10 datasets"),
        ("Window laws added to RN", "model", laws["lo"], laws["mean"], laws["hi"], f"+{laws['mean']:.1f}, 8/10, p {laws['p']:.2f}"),
        ("Window statistics added to RN", "model", stats_["lo"], stats_["mean"], stats_["hi"], "+2.7, mostly ActRecTut"),
        ("Memory type (vs S4D)", "model", -2.4, -0.6, 0.7, "Mamba −2.4 … RN +0.7"),
        ("Training seed", "eval", 0.2, 1.5, 6.1, "sd of a dataset mean 0.2–6.1 pts"),
        ("Normalization / pooling", "model", -2.2, -0.3, 1.8, "kill set, 3 seeds"),
        ("Model size (10× more parameters)", "model", -1.0, 0.0, 0.5, "ρ −0.43 over 30 datasets"),
    ]
    W = 1120
    rh = 36
    H = 110 + len(items) * rh + 40
    body: list[str] = []
    title(body, "All factors on one scale: typical size of the effect on macro-F1 (points)",
          "Bar = range (95 % interval or observed range across datasets); dot = typical value. Colour = kind of factor.")
    lx, px0, px1 = 24, 380, 860
    lo, hi = -10, 30
    sx = lambda u: px0 + (u - lo) / (hi - lo) * (px1 - px0)  # noqa: E731
    ytop, ybot = 84, H - 36
    for u in (-10, -5, 0, 5, 10, 15, 20, 25, 30):
        body.append(line(sx(u), ytop, sx(u), ybot, INK2 if u == 0 else GRID, 1.3 if u == 0 else 1))
        body.append(t(sx(u), ybot + 16, f"{u:+d}" if u else "0", 10, MUTED, "middle"))
    col = {"data": C_DATA, "model": C_MODEL, "eval": C_EVAL}
    for i, (lab, kind, a, m, b, note) in enumerate(items):
        y = ytop + 12 + i * rh + rh / 2
        body.append(t(lx, y + 4, lab, 12, INK))
        body.append(line(sx(a), y, sx(b), y, col[kind], 7))
        body.append(f'<circle cx="{sx(m):.1f}" cy="{y:.1f}" r="6" fill="#FFFFFF" stroke="{col[kind]}" stroke-width="2.5"/>')
        body.append(t(px1 + 16, y + 4, note, 10.5, INK2))
    for k, (name, kind) in enumerate((("data", "data"), ("model", "model"), ("evaluation", "eval"))):
        x = lx + k * 120
        body.append(rect(x, H - 24, 12, 12, col[kind], "none", 2, 0))
        body.append(t(x + 18, H - 14, name, 10.5, INK2))
    return svg(W, H, "Ranking of all factors on one scale", body)


def main() -> None:
    write("factor_map.svg", factor_map())
    write("variance.svg", variance())
    write("data_factors.svg", data_factors())
    write("data_corr.svg", data_corr())
    write("components_forest.svg", components_forest())
    write("components_heatmap.svg", components_heatmap())
    write("eval_noise.svg", eval_noise())
    write("ranking.svg", ranking())
    write("best_family.svg", best_family())



# ------------------------------------------------------------------ 7. the best family, drawn
def arrow(x1, y1, x2, y2, color=INK2, sw=1.4) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" '
            f'marker-end="url(#ah)"/>')


def box(body, x, y, w, h, head, lines=(), color=INK2, fill="#FFFFFF", size=12) -> None:
    body.append(rect(x, y, w, h, fill, color, 6, 1.4))
    body.append(t(x + w / 2, y + 20, head, size, INK, "middle", "600"))
    for i, s in enumerate(lines):
        body.append(t(x + w / 2, y + 37 + 15 * i, s, 10.5, INK2, "middle"))


def best_family() -> str:
    W, H = 1120, 800
    defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M1 1L9 5L1 9z" fill="{INK2}"/></marker></defs>')
    body: list[str] = []
    title(body, "The best family so far: a RecurrenceNet trunk plus a window-level side path, joined at the head",
          "Shapes and parameter counts on UTD-MHAD (6 channels, 150 samples = 3 s at 50 Hz, 27 classes). "
          "RN + statistics = recstatnet (26.5k); RN + laws = lawnet_rec (27.6k).")
    trunk, side = C_MODEL, C_DATA
    # input
    box(body, 24, 96, 150, 250, "Window x", ["C channels × L", "samples", "", "UTD-MHAD:", "6 × 150", "(acc xyz, gyro xyz)"])
    # trunk row
    body.append(t(200, 92, "TEMPORAL TRUNK (per time step, then pooled)", 10.5, trunk, weight="700"))
    box(body, 200, 104, 140, 74, "Encoder", ["Linear C → 32", "per sample · 224"], trunk)
    box(body, 372, 98, 250, 86, "RecurrenceBlock × 3", ["soft recurrence-plot rows", "over 10 lags · 7.8k each", "(detail below)"], trunk)
    box(body, 654, 104, 150, 74, "Mean over time", ["150 × 32 → 32"], trunk)
    body.append(arrow(174, 141, 198, 141))
    body.append(arrow(340, 141, 370, 141))
    body.append(arrow(622, 141, 652, 141))
    # side row
    body.append(t(200, 218, "WINDOW-LEVEL SIDE PATH (whole window, per channel or 3-axis group, before any mixing): one of", 10.5,
                  side, weight="700"))
    box(body, 200, 230, 296, 116, "A. Channel statistics (recstatnet)",
        ["mean, std, min, max of each channel", "over the whole window → 4C = 24", "BatchNorm → Linear 24 → 32", "848 parameters"], side)
    box(body, 508, 230, 296, 116, "B. Laws (lawnet_rec)",
        ["per 3-axis group: ridge fit x(t) ≈ Aτ · x(t − τ),", "τ = 1, 2, 4, 8, 16 → trace, e2, det of each Aτ;",
         "covariance shape; canonical corr. of group", "pairs at lags −4, 0, +4 → MLP → 32 · 1,950"], side)
    body.append(arrow(174, 288, 198, 288))
    # head
    hx = 850
    body.append(rect(hx, 96, 246, 250, "#FFFFFF", INK, 8, 1.6))
    body.append(t(hx + 123, 118, "Head (joins the two paths)", 12, INK, "middle", "600"))
    for i, (a, b) in enumerate((("concat", "32 trunk + 32 side = 64"), ("LayerNorm", "64"),
                                ("Linear 64 → K", "1,755 parameters (K = 27)"), ("class scores", "softmax, cross-entropy"))):
        y = 136 + i * 50
        body.append(rect(hx + 18, y, 210, 40, "#FFFFFF", BORDER, 5, 1))
        body.append(t(hx + 123, y + 17, a, 11.5, INK, "middle", "600"))
        body.append(t(hx + 123, y + 32, b, 10, INK2, "middle"))
        if i:
            body.append(arrow(hx + 123, y - 10, hx + 123, y - 1))
    body.append(arrow(804, 141, hx - 2, 156))
    body.append(arrow(804, 288, hx - 2, 156))
    # block detail
    py = 372
    body.append(rect(24, py, W - 48, 272, "#FFFFFF", trunk, 8, 1.2))
    body.append(t(40, py + 24, "Inside one RecurrenceBlock (d = 32): each time step asks \"am I back where I was τ steps ago?\"",
                  12.5, INK, weight="600"))
    yy = py + 70
    box(body, 40, yy, 110, 60, "h(t)", ["32 values"], trunk)
    box(body, 176, yy, 110, 60, "LayerNorm", ["per time step"], INK2)
    box(body, 316, py + 46, 180, 52, "mix_in", ["Linear 32 → 32 = u(t)"], INK2, size=11.5)
    box(body, 316, py + 112, 180, 52, "embed", ["Linear 32 → 16 = 8 points z(t) in 2-D"], INK2, size=11.5)
    box(body, 528, py + 104, 270, 70, "soft recurrence, 10 lags", ["r(t, τ) = σ((ε − ‖z(t) − z(t − τ)‖) / 0.1)",
                                                                  "τ = 1, 2, 4, 6, 8, 12, 16, 24, 32, 48 → 80"], INK2, size=11.5)
    box(body, 528, py + 180, 270, 46, "line = r(t, τ) · r(t − 1, τ)", ["recurrence held two steps → 80"], INK2, size=11)
    box(body, 830, yy, 120, 60, "mix_out", ["[u, r, line] = 192", "Linear → 32, GELU"], INK2, size=11.5)
    box(body, 976, yy, 104, 60, "h(t) + ·", ["residual", "→ next block"], trunk, size=11.5)
    body.append(arrow(150, yy + 30, 174, yy + 30))
    body.append(arrow(286, yy + 30, 314, py + 72))
    body.append(arrow(286, yy + 30, 314, py + 138))
    body.append(arrow(496, py + 138, 526, py + 138))
    body.append(arrow(663, py + 174, 663, py + 178))
    body.append(arrow(496, py + 72, 828, yy + 22))
    body.append(arrow(798, py + 139, 828, yy + 34))
    body.append(arrow(798, py + 203, 828, yy + 46))
    body.append(arrow(950, yy + 30, 974, yy + 30))
    body.append(t(40, py + 256, "ε = learned radius per embedded point (8 values); points before t = τ get r = 0. "
                  "All 10 lags read the same 2-D embeddings, so the block compares the signal with itself, not with templates.",
                  10.5, MUTED))
    # results strip
    sy0 = 664
    cols = [
        ("Results (3 seeds, 10 datasets)", ["RN + statistics 74.70; RN + laws 74.42", "RecurrenceNet 72.01; S4D 71.29",
                                            "RN + laws − S4D +3.12, Holm p 0.049"]),
        ("Where each path pays", ["trunk: gait and gesture shape (+6 vs no memory)", "statistics: ActRecTut +21, SAD +3",
                                   "laws: SAD +5.8, DSADS +4.2, UniMiB +3.6"]),
        ("Cost and edge", ["26.5k / 27.6k parameters; ~5.7 MFLOPs per window", "side path = running sums per sample",
                           "laws: 3×3 lagged covariances, 3×3 solves per window"]),
    ]
    cw = (W - 48 - 2 * 16) / 3
    for i, (head, lines) in enumerate(cols):
        x = 24 + i * (cw + 16)
        body.append(rect(x, sy0, cw, 108, "#FFFFFF", BORDER, 6, 1))
        body.append(t(x + 14, sy0 + 22, head, 12, INK, weight="600"))
        for j, s in enumerate(lines):
            body.append(t(x + 14, sy0 + 44 + 18 * j, s, 10.8, INK2))
    return svg(W, H, "Structure of the best model family: RecurrenceNet trunk plus a window-level side path", body, defs)


if __name__ == "__main__":
    main()
