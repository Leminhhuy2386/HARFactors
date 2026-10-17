"""Figure 10 and Table IV of the manuscript: the subject-to-network ratio by network population, and the error budget
of a comparison between two networks. From saved results only (no training, no inference); exploratory analyses added
after the review of 2026-10-17.

    fig10_population_budget.{pdf,png}  (a) R' of S3 for the k best published networks (ranked on the Arena datasets
                                       outside S3) and for the 14 deep networks, against H1's R; (b) the variance
                                       components of S2 and of three S3 populations; (c) the smallest network
                                       difference a paired comparison detects, by test-subject groups and seeds
    tab04_error_budget.{csv,tex}       noise SD, detectable difference and reliability of comparison designs

Colours and marks follow src/visualization/letter_figures.py (same validated slots, same component colours as Fig. 1).

    uv run python -m src.visualization.contribution_figures
"""
from __future__ import annotations

import csv
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import paths as PATHS  # noqa: E402
from src.utils import provenance as PV  # noqa: E402
from src.visualization.letter_figures import FIG_DATA, GRID, INK, INK2, RESIDUAL, SLOTS, TAB_DATA, save  # noqa: E402

POP = PATHS.METRICS / "network_population" / "data.json"
EB = PATHS.METRICS / "error_budget" / "data.json"
H1 = PATHS.METRICS / "confirmatory" / "H1.json"
COMPONENTS = [("P", "test-subject group", SLOTS[0]), ("M", "network", SLOTS[1]), ("MD", "network × dataset", SLOTS[2]),
              ("MP", "network × split", SLOTS[3]), ("e", "seed (residual)", RESIDUAL)]
DESIGN_ROWS = [("unpaired_1x1", r"Unpaired, 1 $\times$ 1"), ("paired_1x1", r"Paired, 1 $\times$ 1"),
               ("paired_10x1", r"Paired, 10 $\times$ 1"), ("paired_10x3", r"Paired, 10 $\times$ 3 (protocol)"),
               ("paired_10xinf", r"Paired, 10 $\times$ $\infty$"), ("paired_20x3", r"Paired, 20 $\times$ 3")]


def record(path, data, run_id, started, inputs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=1, default=float) + "\n")
    PV.write(path, run_id=run_id, started=started, inputs=inputs, produced_by="src/visualization/contribution_figures.py")


def state_colour(lo: float, hi: float) -> str:
    return SLOTS[0] if lo > 1 else (SLOTS[1] if hi < 1 else RESIDUAL)


def fig10(run_id, started) -> None:
    pop, eb, h1 = (json.loads(f.read_text()) for f in (POP, EB, H1))
    P = pop["populations"]
    tops = sorted((v for k, v in P.items() if k.startswith("top")), key=lambda v: v["n_models"])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1.1, 1.05, 1.1],
                                                                            "wspace": 0.45})
    # (a) R' by population
    rows_a = []
    for v in tops + [P["deep"]]:
        deep = v is P["deep"]
        x = v["n_models"] + (0.38 if deep else 0.0)
        lo, hi = v["bootstrap"]["R_interval"]
        R = v["point"]["R"]
        a.plot([x, x], [lo, hi], color=INK, linewidth=1.2, solid_capstyle="butt")
        a.plot([x], [R], "D" if deep else "o", color=state_colour(lo, hi), markersize=5.2 if deep else 5.6,
               markeredgecolor="white", markeredgewidth=0.9, zorder=3)
        rows_a.append({"population": "deep" if deep else f"top{v['n_models']}", "n_models": v["n_models"], "R_prime": R,
                       "interval": [lo, hi]})
    a.annotate("14 deep\nonly", (14.38, P["deep"]["point"]["R"]), xytext=(9.3, 0.075), fontsize=6.3, color=INK2,
               arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.6})
    a.axhline(1, color=SLOTS[6], linewidth=1.0, linestyle=(0, (3, 2)))
    a.axhline(h1["point"]["R"], color=INK2, linewidth=0.8, linestyle=(0, (1, 1.5)))
    a.text(17.6, h1["point"]["R"] / 1.2, "S2, 4 shared-skeleton\nnetworks (H1)", fontsize=6.3, color=INK2, ha="right",
           va="top")
    a.set_yscale("log")
    a.set_ylim(0.04, 30)
    a.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    a.set_xlim(2.3, 17.9)
    a.set_xticks([3, 5, 8, 11, 14, 17])
    a.set_xlabel("networks compared (best k by\npublished macro-F1 elsewhere)")
    a.set_ylabel("R′ = subject / network variance")
    a.grid(axis="y", color=GRID, linewidth=0.5)
    a.set_axisbelow(True)
    a.set_title("(a)", loc="left", fontweight="bold")

    # (b) variance components, F1 points squared
    rows = [("S2 (H1)", h1["point"]["components"]),
            ("S3 best 5", P["top5"]["point"]["components"]),
            ("S3 deep", P["deep"]["point"]["components"]),
            ("S3 all 17", P["top17"]["point"]["components"])]
    y = np.arange(len(rows))[::-1]
    for i, (k, name, col) in enumerate(COMPONENTS):
        left = np.array([sum(r[1].get(kk, 0.0) for kk, _, _ in COMPONENTS[:i]) for r in rows])
        w = np.array([r[1].get(k, 0.0) for r in rows])
        b.barh(y, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0, label=name)
        for yy, l, ww in zip(y, left, w):
            if ww >= 18:
                b.text(l + ww / 2, yy, f"{ww:.0f}", ha="center", va="center", fontsize=6.3, fontweight="bold",
                       color=INK if k == "MP" else "white")
    b.set_yticks(y, [r[0] for r in rows])
    b.set_xlabel("variance (F1 points²)")
    b.tick_params(axis="y", length=0)
    b.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.35, -0.24), frameon=False, handlelength=1.0,
             columnspacing=0.8, handletextpad=0.4, fontsize=6.3)
    b.grid(axis="x", color=GRID, linewidth=0.5)
    b.set_axisbelow(True)
    b.set_title("(b)", loc="left", fontweight="bold")

    # (c) smallest detectable difference of a paired comparison, S2 components
    s2 = eb["analyses"]["S2"]
    ks = np.array(s2["paired_noise_sd_grid"]["k"])
    z = s2["z_sum"]
    curves = []
    for (n, lab), col in zip((("1", "1 seed"), ("3", "3 seeds"), ("inf", "∞ seeds")), (SLOTS[1], SLOTS[0], SLOTS[5])):
        mdd = z * np.array(s2["paired_noise_sd_grid"]["by_seeds"][n])
        inside = ks <= 10
        c.plot(ks[inside], mdd[inside], color=col, linewidth=1.6)
        c.plot(ks[ks >= 10], mdd[ks >= 10], color=col, linewidth=1.6, linestyle=(0, (3, 2)))
        c.plot([], [], color=col, linewidth=1.6, label=lab)
        curves.append({"seeds": n, "k": ks.tolist(), "detectable_difference": mdd.tolist()})
    c.hlines(s2["true_difference_sd"], 0.5, 20.4, color=INK2, linewidth=0.9, linestyle=(0, (1, 1.5)))
    c.text(20.8, s2["true_difference_sd"], "true\ndifference,\n1 SD", fontsize=6.0, color=INK2, va="center")
    c.legend(loc="upper right", frameon=False, handlelength=1.6, fontsize=6.3)
    c.axvline(10, color=GRID, linewidth=1.0, zorder=0)
    c.text(9.6, 17.6, "protocol\n(k = 10)", fontsize=6.3, color=INK2, va="top", ha="right")
    c.set_xlim(0.5, 26.5)
    c.set_ylim(0, 18)
    c.set_xticks([1, 5, 10, 15, 20])
    c.set_xlabel("test-subject groups (dashed: more subjects)")
    c.set_ylabel("detectable difference, F1 points")
    c.grid(color=GRID, linewidth=0.5)
    c.set_axisbelow(True)
    c.set_title("(c)", loc="left", fontweight="bold")
    save(fig, "fig10_population_budget")
    record(FIG_DATA / "fig10_population_budget.json",
           {"a": rows_a, "b": [{"row": r[0], "components": r[1]} for r in rows], "c": curves,
            "c_true_difference_sd": s2["true_difference_sd"], "c_alpha": eb["alpha"], "c_power": eb["power"]},
           run_id, started, [POP, EB, H1])


def tab04(run_id, started) -> None:
    eb = json.loads(EB.read_text())
    A = eb["analyses"]
    head = ["design", "k", "n", "paired"] + [f"{s}_{m}" for s in ("S2", "S1")
                                            for m in ("noise_sd", "detectable_difference", "reliability")]
    lines = []
    for key, _ in DESIGN_ROWS:
        d2, d1 = A["S2"]["designs"][key], A["S1"]["designs"][key]
        lines.append([key, d2["k"], d2["n"] if d2["n"] is not None else "inf", d2["paired"]]
                     + [round(d[m], 4) for d in (d2, d1) for m in ("noise_sd", "detectable_difference", "reliability")])
    PATHS.TABLES.mkdir(exist_ok=True)
    for path in (PATHS.TABLES / "tab04_error_budget.csv", TAB_DATA / "tab04_error_budget.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(lines)
    record(TAB_DATA / "tab04_error_budget.csv", None, run_id, started, [EB])
    tex = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{3}{c}{S2 (held-out)} & \multicolumn{3}{c}{S1 (development)} \\",
           r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
           r"Comparison & SD & Detect. & Rel. & SD & Detect. & Rel. \\", r"\midrule"]
    for key, label in DESIGN_ROWS:
        d2, d1 = A["S2"]["designs"][key], A["S1"]["designs"][key]
        tex.append(f"{label} & " + " & ".join(f"{d['noise_sd']:.1f} & {d['detectable_difference']:.1f} & "
                                              f"{d['reliability']:.2f}" for d in (d2, d1)) + r" \\")
    tex += [r"\midrule",
            f"True difference of two networks, SD & \\multicolumn{{3}}{{l}}{{{A['S2']['true_difference_sd']:.1f}}} & "
            f"\\multicolumn{{3}}{{l}}{{{A['S1']['true_difference_sd']:.1f}}} \\\\", r"\bottomrule", r"\end{tabular}"]
    (PATHS.TABLES / "tab04_error_budget.tex").write_text("\n".join(tex) + "\n")


def main() -> None:
    started = time.time()
    run_id = PV.new_run_id("contribution_figures")
    fig10(run_id, started)
    tab04(run_id, started)
    print("figure 10 and table IV written; run", run_id)


if __name__ == "__main__":
    main()
