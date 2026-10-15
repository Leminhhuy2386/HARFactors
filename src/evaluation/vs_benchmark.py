"""Head-to-head: our best family (RN + statistics, RN + laws) against every model of the WHAR Arena paper (Figure 6),
paired over the 10 datasets. No training.

For each paper model: mean difference (ours − paper) with a 95 % t-interval over datasets, datasets won/lost, Wilcoxon
p and Holm correction over the 17 paper models (within each of our models). Ours at seed 0 (the paper ran each model
once) and, as a check, the 3-seed mean. Also the rank of ours among the 17 paper models on each dataset.

    uv run python -m src.evaluation.vs_benchmark   ->   results/metrics/vs_benchmark/summary.md, figures/vs_benchmark.svg
"""

from __future__ import annotations

import json
import math
import statistics as st

from scipy import stats

from src.evaluation import compare_arena as CA
from src.visualization.build_factor_figures import BORDER, C_DATA, C_MODEL, GRID, INK, INK2, MUTED, line, mix, rect, svg, t, title, write
from src.utils import paths as PATHS  # noqa: E402

OUT = PATHS.METRICS / "vs_benchmark"
DS = CA.DATASETS
OURS = {"lawnet_rec": "RN + laws", "recstatnet": "RN + statistics"}
PARAMS = json.loads((PATHS.METRICS / "pareto" / "costs.json").read_text())["rows"] \
    if (PATHS.METRICS / "pareto" / "costs.json").exists() else {}


def gparams(m: str) -> float | None:
    rows = PARAMS.get(m)
    return math.exp(st.mean(math.log(rows[d]["params"]) for d in DS)) if rows else None


def compare(ours: dict[str, float]) -> dict[str, dict]:
    res = {}
    for a in CA.ARENA:
        diff = [ours[d] - CA.PAPER[d][a][0] for d in DS]
        m = st.mean(diff)
        h = stats.t.ppf(0.975, len(diff) - 1) * st.stdev(diff) / math.sqrt(len(diff))
        res[a] = {"mean": m, "lo": m - h, "hi": m + h, "won": sum(x > 0 for x in diff), "lost": sum(x < 0 for x in diff),
                  "p": float(stats.wilcoxon(diff).pvalue)}
    holm = CA.holm({a: r["p"] for a, r in res.items()})
    for a in res:
        res[a]["holm"] = holm[a]
    return res


def main() -> None:
    seed0 = {m: {d: CA.ours(m, d, 0) for d in DS} for m in OURS}
    seed3 = {m: {d: CA.ours_seed_mean(m, d) for d in DS} for m in OURS}
    res0 = {m: compare(seed0[m]) for m in OURS}
    res3 = {m: compare(seed3[m]) for m in OURS}
    paper_mean = {a: st.mean(CA.PAPER[d][a][0] for d in DS) for a in CA.ARENA}
    order = sorted(CA.ARENA, key=lambda a: -paper_mean[a])
    L = ["# Our best family against every WHAR Arena model (10 datasets, paired)", "",
         "Ours − paper, macro-F1 points. Seed 0 (as the paper); the 3-seed column is a check. Holm over the 17 paper models.", ""]
    for m, lab in OURS.items():
        L += [f"## {lab} (mean F1 seed 0 {st.mean(seed0[m].values()):.2f}, 3 seeds {st.mean(seed3[m].values()):.2f}; "
              f"params {gparams(m) / 1e3:.1f}k)", "",
              "| Paper model | Its mean F1 | Its params | Ours − it (seed 0) [95 % CI] | Won–lost | Wilcoxon p | Holm p | "
              "Ours − it (3 seeds) | Holm p (3 seeds) |", "|---|---|---|---|---|---|---|---|---|"]
        for a in order:
            r, r3 = res0[m][a], res3[m][a]
            p = gparams(a)
            L.append(f"| {a} | {paper_mean[a]:.2f} | {'–' if p is None else f'{p / 1e3:.0f}k'} | {r['mean']:+.2f} "
                     f"[{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['won']}–{r['lost']} | {r['p']:.3f} | {r['holm']:.3f} | "
                     f"{r3['mean']:+.2f} | {r3['holm']:.3f} |")
        L.append("")
    L += ["## Rank among the 17 paper models on each dataset (1 = above all of them; seed 0)", "",
          "| Model | " + " | ".join(CA.SHORT[d] for d in DS) + " | Mean rank | First places |", "|---|" + "---|" * (len(DS) + 2)]
    for m, lab in OURS.items():
        rk = [CA.arena_rank(seed0[m][d], d) for d in DS]
        L.append(f"| {lab} | " + " | ".join(f"#{r}" for r in rk) + f" | {st.mean(rk):.1f} | {sum(r == 1 for r in rk)} |")
    best = {d: max(CA.ARENA, key=lambda a: CA.PAPER[d][a][0]) for d in DS}
    L.append("| best paper model | " + " | ".join(f"{CA.PAPER[d][best[d]][0]:.1f} {best[d]}" for d in DS) + " | | |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "data.json").write_text(json.dumps({"seed0": res0, "seed3": res3, "paper_mean": paper_mean}, indent=1) + "\n")
    print("\n".join(L))
    write("vs_benchmark.svg", draw(res0, seed0, paper_mean, order))


def draw(res: dict, seed0: dict, paper_mean: dict, order: list[str]) -> str:
    W = 1120
    rh = 30
    H = 120 + len(order) * rh + 250
    body: list[str] = []
    title(body, "Our best family against every model of the WHAR Arena benchmark (10 datasets, paired, seed 0)",
          "Dot = mean over datasets of (ours − paper model); bar = 95 % t-interval; filled = Holm-corrected Wilcoxon p < 0.05 "
          "over the 17 comparisons. Right: datasets won–lost.")
    cols = {"lawnet_rec": C_MODEL, "recstatnet": mix(C_MODEL, "#FFFFFF", 0.45)}
    lx, px0, px1 = 24, 330, 820
    lo, hi = -10, 30
    sx = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)  # noqa: E731
    ytop = 96
    ybot = ytop + 18 + len(order) * rh
    for v in (-10, -5, 0, 5, 10, 15, 20, 25, 30):
        body.append(line(sx(v), ytop, sx(v), ybot, INK2 if v == 0 else GRID, 1.3 if v == 0 else 1))
        body.append(t(sx(v), ybot + 16, f"{v:+d}" if v else "0", 10, MUTED, "middle"))
    body.append(t((px0 + px1) / 2, ybot + 32, "ours − paper model, macro-F1 points (right = ours better)", 10.5, INK2, "middle"))
    body.append(t(lx, ytop + 6, "paper model (its mean F1, params)", 10.5, MUTED))
    body.append(t(px1 + 24, ytop + 6, "RN + laws", 10.5, cols["lawnet_rec"], weight="600"))
    body.append(t(px1 + 140, ytop + 6, "RN + statistics", 10.5, INK2, weight="600"))
    for i, a in enumerate(order):
        y = ytop + 18 + i * rh + rh / 2
        p = gparams(a)
        body.append(t(lx, y + 4, f"{a}  ({paper_mean[a]:.1f}{'' if p is None else f', {p / 1e3:.0f}k'})", 11.5, INK))
        for j, (m, col) in enumerate(cols.items()):
            r = res[m][a]
            yy = y - 5 + 10 * j
            body.append(line(sx(max(lo, r["lo"])), yy, sx(min(hi, r["hi"])), yy, col, 2.2))
            sig = r["holm"] < 0.05
            body.append(f'<circle cx="{sx(r["mean"]):.1f}" cy="{yy:.1f}" r="4.8" fill="{col if sig else "#FFFFFF"}" '
                        f'stroke="{col}" stroke-width="2"/>')
            body.append(t(px1 + 24 + 116 * j, y + 4, f"{r['mean']:+.1f}  {r['won']}–{r['lost']}", 11, INK2))
    # per-dataset ranks
    y0 = ybot + 56
    body.append(t(lx, y0, "Rank among the 17 paper models on each dataset (1 = above all of them), seed 0", 12, INK, weight="600"))
    cw = (W - 48 - 170) / len(DS)
    for j, d in enumerate(DS):
        body.append(t(194 + j * cw + cw / 2, y0 + 26, CA.SHORT[d], 10.5, INK2, "middle", "600"))
    for i, (m, lab) in enumerate((("lawnet_rec", "RN + laws"), ("recstatnet", "RN + statistics"))):
        y = y0 + 40 + i * 40
        body.append(t(lx, y + 21, lab, 11.5, INK))
        for j, d in enumerate(DS):
            r = CA.arena_rank(seed0[m][d], d)
            fill = mix("#FFFFFF", C_DATA, max(0.08, 1 - (r - 1) / 12)) if r <= 12 else "#F3E3EA"
            body.append(rect(194 + j * cw + 3, y + 4, cw - 6, 28, fill, "none", 4, 0))
            ink = "#FFFFFF" if r <= 3 else INK
            body.append(t(194 + j * cw + cw / 2, y + 23, f"#{r}  ({seed0[m][d]:.1f})", 10.5, ink, "middle", "600" if r == 1 else "normal"))
    body.append(t(lx, y0 + 140, "Lost ground: Opportunity (#11-13; the paper's MLP-HAR 74.7) and WISDM (#9-11; DANA 85.5). "
                  "First places: UTD-MHAD, UCI-HAR, UniMiB, ActRecTut, SAD, Daphnet.", 10.5, MUTED))
    return svg(W, H, "Head-to-head comparison of our best family with the WHAR Arena benchmark models", body)


if __name__ == "__main__":
    main()
