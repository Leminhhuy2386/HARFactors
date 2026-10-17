"""Figures and the table of the IEEE Sensors Letters manuscript, from saved results only (no training, no inference).

    fig01_variance_components.{pdf,png}  within-dataset variance shares by source and R with its 97.5 % interval, for
                                         the confirmatory test (H1, S2) and the pre-registered sensitivity analyses
    fig02_channels.{pdf,png}             Random Forest margin over the best deep model vs channel count, over the
                                         WHAR Arena paper's 30 datasets (re-analysis of published single runs)
    tab01_variance_components.{csv,tex}  the numbers behind Fig. 1

Every figure and table also writes its numbers to results/figure_data/ or results/table_data/, with provenance.
Colours: the dataviz reference hues in fixed slot order, at their deeper steps for print (blue 500, orange, aqua,
yellow, magenta, violet, red; validated with the dataviz validator, light surface, adjacent pairs: worst CVD delta E 8.4,
normal-vision 19.3, only yellow below 3:1 contrast, relieved by direct labels and Table I). The seed (residual) term is
neutral grey: it is unexplained variation, not a category.

    uv run python -m src.visualization.letter_figures
"""
from __future__ import annotations

import csv
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation import factor_analysis as FA  # noqa: E402
from src.utils import paths as PATHS  # noqa: E402
from src.utils import provenance as PV  # noqa: E402

CONF = PATHS.METRICS / "confirmatory"
FIG_DATA = PATHS.RESULTS / "figure_data"
TAB_DATA = PATHS.RESULTS / "table_data"
SLOTS = ["#256abf", "#d95926", "#199e70", "#c98500", "#d55181", "#4a3aa7", "#e34948"]  # slots 1-7, deeper steps
INK, INK2, GRID = "#0b0b0b", "#3d3c39", "#dedcd5"
RESIDUAL = "#8c8a84"  # neutral grey for the seed (residual) term
SOURCES = [("P", "test-subject group"), ("M", "model"), ("MD", "model × dataset"), ("MP", "model × split"),
           ("e", "seed (residual)")]
ROWS = [("H1", "S2 F1 (H1)"), ("sens1_S2_balanced_accuracy", "S2 bal. acc."), ("sens2_S1_macro_f1", "S1 F1"),
        ("sens2_S1_balanced_accuracy", "S1 bal. acc."), ("sens3_S1_without_actrectut", "S1 F1, no ActRecTut"),
        ("sens4_S3_macro_f1", "S3 F1 (17 models)")]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6,
                     "ytick.minor.width": 0.6, "lines.linewidth": 1.5, "grid.linewidth": 0.5,
                     "legend.frameon": False, "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def save(fig, name: str) -> list:
    PATHS.FIGURES.mkdir(exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = PATHS.FIGURES / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", metadata={"CreationDate": None} if ext == "pdf" else None)
        out.append(p)
    plt.close(fig)
    return out


def record(path, data, run_id, started, inputs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=1, default=float) + "\n")
    PV.write(path, run_id=run_id, started=started, inputs=inputs, produced_by="src/visualization/letter_figures.py")


def variance_rows() -> tuple[list[dict], list]:
    rows, inputs = [], []
    for key, label in ROWS:
        f = CONF / f"{key}.json"
        r = json.loads(f.read_text())
        inputs.append(f)
        comp = r["point"]["components"]
        total = sum(comp.values())
        b = r["bootstrap"]
        rows.append({"analysis": key, "label": label, "set": r["set"], "outcome": r["outcome"],
                     "datasets": len(r["datasets"]), "splits": r["n_splits"], "runs": r["n_runs"],
                     "components": comp, "shares": {k: comp.get(k, 0.0) / total for k, _ in SOURCES},
                     "R": r["point"]["R"], "R_interval": b["R_interval"], "pi_P": r["point"]["pi_P"],
                     "pi_P_interval": b["pi_P_interval"], "R_definition": r["point"]["R_definition"],
                     "decision": r.get("decision"), "excluded": r["excluded"]})
    return rows, inputs


def fig01(rows, run_id, started, inputs) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.45), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.16},
                               sharey=True)
    y = np.arange(len(rows))[::-1]
    colours = SLOTS[:4] + [RESIDUAL]
    for i, (k, name) in enumerate(SOURCES):
        left = np.array([sum(r["shares"][kk] for kk, _ in SOURCES[:i]) for r in rows])
        w = np.array([r["shares"][k] for r in rows])
        a.barh(y, w, left=left, height=0.64, color=colours[i], edgecolor="white", linewidth=1.0, label=name)
        for yy, l, ww in zip(y, left, w):
            if ww >= 0.08:  # direct labels: relief for the low-contrast slots
                a.text(l + ww / 2, yy, f"{100 * ww:.0f}", ha="center", va="center", fontsize=6.5,
                       fontweight="bold", color=INK if k == "MP" else "white")
    a.set_yticks(y, [r["label"] for r in rows])
    a.set_xlim(0, 1)
    a.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    a.set_xlabel("within-dataset variance (%)")
    a.legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.62, 1.0), frameon=False, handlelength=1.0,
             columnspacing=0.8, handletextpad=0.4)
    for yy, r in zip(y, rows):
        lo, hi = r["R_interval"]
        b.plot([lo, hi], [yy, yy], color=INK, linewidth=1.6, solid_capstyle="butt")
        b.plot([lo, hi], [yy, yy], "|", color=INK, markersize=5, markeredgewidth=1.2)
        b.plot([r["R"]], [yy], "o", color=SLOTS[0] if r["R_interval"][0] > 1 else SLOTS[1], markersize=6,
               markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    b.axvline(1, color=SLOTS[6], linewidth=1.0, linestyle=(0, (3, 2)))
    b.set_xscale("log")
    b.set_xlim(0.04, 30)
    b.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    b.set_xlabel("R, 97.5 % interval")
    b.tick_params(axis="y", length=0)
    b.grid(axis="x", color=GRID, linewidth=0.5)
    b.set_axisbelow(True)
    save(fig, "fig01_variance_components")
    record(FIG_DATA / "fig01_variance_components.json", {"rows": rows, "sources": SOURCES}, run_id, started, inputs)


def fig02(run_id, started) -> dict:
    """The Random Forest's margin over the best deep network against channel count, over the WHAR Arena paper's 30
    datasets. The memory contrasts that were panel (a) until 2026-10-17 left the letter: the architecture is the
    companion study's subject."""
    fac_f = PATHS.METRICS / "factors" / "data.json"
    fac = json.loads(fac_f.read_text())["data"]
    fig, b = plt.subplots(figsize=(3.5, 1.9))
    out = "RF − best deep (classical margin)"
    ds = sorted(fac["outcome"][out])
    x = [fac["pred"]["log2 channels"][d] for d in ds]
    yv = [fac["outcome"][out][d] for d in ds]
    ols = fac["ols"][out]
    j = 1 + ["log2 subjects", "log2 channels", "log2 classes", "log2 samples/window"].index("log2 channels")
    b.axhline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    b.scatter(x, yv, s=26, color=SLOTS[0], edgecolor="white", linewidth=0.8, zorder=3)
    names = {e["dataset"]: e["dataset_name"] for e in json.loads((PATHS.DOCS / "baselines_cited.json").read_text())["entries"]}
    for d, xx, yy in zip(ds, x, yv):
        if yy > 5:
            b.annotate(names.get(d, d), (xx, yy), xytext=(-4, 3), textcoords="offset points", fontsize=6, color=INK2,
                       ha="right")
    b.set_xlabel("log2 channel count")
    b.set_ylabel("RF − best deep, F1 points")
    pj = ols["p"][j]
    b.text(0.98, 0.04, f"OLS, other properties fixed:\n{ols['beta'][j]:+.1f} per sd, "
                       f"p {'< 0.001' if pj < 0.001 else f'= {pj:.3f}'}, R² {ols['r2']:.2f}",
           transform=b.transAxes, fontsize=6.5, va="bottom", ha="right", color=INK)
    b.set_ylim(min(yv) - 2, max(yv) + 5)
    b.grid(color=GRID, linewidth=0.5)
    b.set_axisbelow(True)
    save(fig, "fig02_channels")
    data = {"scatter": [{"dataset": d, "log2_channels": xx, "rf_minus_best_deep": yy} for d, xx, yy in zip(ds, x, yv)],
            "ols_log2_channels": {"beta_per_sd": ols["beta"][j], "p": ols["p"][j], "r2": ols["r2"]},
            "note": "re-analyses the WHAR Arena paper's published single runs (Figure 6): derived from quoted numbers"}
    record(FIG_DATA / "fig02_channels.json", data, run_id, started, [fac_f])
    return data


def tab01(rows, run_id, started, inputs) -> None:
    PATHS.TABLES.mkdir(exist_ok=True)
    head = ["analysis", "set", "outcome", "datasets", "splits", "runs", "s2_P", "s2_M", "s2_MD", "s2_MP", "s2_e", "R",
            "R_lo", "R_hi", "pi_P", "pi_P_lo", "pi_P_hi"]
    lines = []
    for r in rows:
        c = r["components"]
        lines.append([r["analysis"], r["set"], r["outcome"], r["datasets"], r["splits"], r["runs"],
                      *(round(c.get(k, float("nan")), 3) for k in ("P", "M", "MD", "MP", "e")),
                      round(r["R"], 3), round(r["R_interval"][0], 3), round(r["R_interval"][1], 3),
                      round(r["pi_P"], 3), round(r["pi_P_interval"][0], 3), round(r["pi_P_interval"][1], 3)])
    for path in (PATHS.TABLES / "tab01_variance_components.csv", TAB_DATA / "tab01_variance_components.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(lines)
    record(TAB_DATA / "tab01_variance_components.csv", None, run_id, started, inputs)
    fmt = lambda v: "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"  # noqa: E731
    tex = [r"\begin{tabular}{@{}lrrrrrrl@{}}", r"\toprule",
           r"Analysis & $n$ & $\sigma^2_P$ & $\sigma^2_M$ & $\sigma^2_{MD}$ & $\sigma^2_{MP}$ & $\sigma^2_e$ & $R$ [97.5\,\% int.] \\",
           r"\midrule"]
    for r in rows:
        c = r["components"]
        lo, hi = r["R_interval"]
        tex.append(f"{r['label']} & {r['runs']} & {fmt(c['P'])} & {fmt(c['M'])} & {fmt(c['MD'])} & "
                   f"{fmt(c.get('MP'))} & {fmt(c['e'])} & {r['R']:.2f} [{lo:.2f}, {hi:.2f}] \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (PATHS.TABLES / "tab01_variance_components.tex").write_text("\n".join(tex) + "\n")


def main() -> None:
    started = time.time()
    run_id = PV.new_run_id("letter_figures")
    rows, inputs = variance_rows()
    fig01(rows, run_id, started, inputs)
    fig02(run_id, started)
    tab01(rows, run_id, started, inputs)
    print("figures and table written; run", run_id)


if __name__ == "__main__":
    main()
