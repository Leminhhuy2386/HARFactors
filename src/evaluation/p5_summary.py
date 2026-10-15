"""P5 analysis (RESEARCH_PLAN.md §5b, pre-registered): the 5 models across the WHAR datasets.

Per dataset: for each split the mean over training seeds 0-2, then the mean over splits. Across
datasets: mean rank (1 = best), Friedman test, Wilcoxon signed-rank with Holm correction for every
pair, the Nemenyi critical difference, and the differences of the new families to S4D and Mamba
with a 95 % t-interval over datasets (non-inferior if the lower bound is above -1 point).
Only datasets where every model has every split for every seed enter the test; the others are
listed as incomplete.

    uv run python -m src.evaluation.p5_summary            -> results/metrics/p5/summary.md, results/metrics/p5/cd.svg
    uv run python -m src.evaluation.p5_summary recurrencenet   (add round-6/7 models) -> results/metrics/p5/summary_<ids>.md, cd_<ids>.svg
"""

from __future__ import annotations

import itertools
import json
import statistics as st
from pathlib import Path

from scipy import stats
from src.utils import paths as PATHS  # noqa: E402

HERE = Path(__file__).resolve().parents[2]  # project root
RES = PATHS.RUNS
WHAR = PATHS.EXTERNAL_RUNS  # the 17 paper models re-run on 4 datasets by the HAR reproduction
OUT = PATHS.METRICS / "p5"
MODELS = ["quantilenet", "hystnet_sel_clock", "s4d", "mamba", "hystnet_nomem"]
LABELS = {"quantilenet": "QuantileNet", "hystnet_sel_clock": "HysteresisNet sel+clock", "s4d": "S4D",
          "mamba": "Mamba", "hystnet_nomem": "No memory", "recurrencenet": "RecurrenceNet", "rainflownet": "RainflowNet",
          "ordinalnet": "OrdinalNet", "returnmapnet": "ReturnMapNet", "phaselocknet": "PhaseLockNet"}
NEW, OLD = ["quantilenet", "hystnet_sel_clock"], ["s4d", "mamba"]
# dataset -> number of subject splits under the protocol (min(10, subjects))
# mhealth was selected but the paper-era code cannot preprocess it (13 label ids vs 12 activities in its
# config; "Input validation failed"), so it is out by the eligibility rule itself.
N_SPLITS = {"utd_mhad": 8, "actrectut_gestures": 2, "opportunity": 4, "unimib_shar": 10, "uci_har": 10,
            "sad": 10, "hapt": 10, "dsads": 8, "wisdm": 10, "daphnet": 10}
DATASETS = list(N_SPLITS)
SEEDS = [0, 1, 2]
MARGIN = 1.0
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}  # Nemenyi q_alpha, alpha = 0.05


def runs(model: str, dataset: str) -> dict[int, dict[str, float]]:
    """seed -> split -> macro-F1 %."""
    base = RES / model / dataset
    out = {}
    for s in SEEDS:
        d = base / f"seed-{s}" if s else base
        out[s] = {p.stem: 100 * json.loads(p.read_text())["metrics"]["macro_f1"] for p in sorted(d.glob("split-*.json"))}
    return out


def dataset_scores(dataset: str) -> tuple[dict[str, float] | None, str]:
    r = {m: runs(m, dataset) for m in MODELS}
    splits = {f"split-{i:02d}" for i in range(N_SPLITS[dataset])}
    found = sum(len(set(r[m][s]) & splits) for m in MODELS for s in SEEDS)
    if found < len(splits) * len(MODELS) * len(SEEDS):
        return None, f"{found}/{len(splits) * len(MODELS) * len(SEEDS)} runs"
    return {m: st.mean(st.mean(r[m][s][sp] for s in SEEDS) for sp in splits) for m in MODELS}, f"{len(splits)} splits"


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def cd_svg(mean_rank: dict[str, float], cd: float, n: int) -> str:
    """Critical-difference diagram: models on a rank axis; bars join models closer than the CD."""
    k = len(mean_rank)
    w, left, right, top = 760, 210, 160, 74
    x = lambda r: left + (r - 1) / (k - 1) * (w - left - right)  # noqa: E731
    order = sorted(mean_rank, key=mean_rank.get)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {top + 40 + 22 * k}" font-family="sans-serif" '
             f'font-size="12"><style>text{{fill:currentColor}} line{{stroke:currentColor}}</style>',
             f'<text x="{w / 2}" y="14" text-anchor="middle">Mean rank over {n} datasets (1 = best); '
             f'Nemenyi CD = {cd:.2f}</text>',
             f'<line x1="{x(1)}" y1="{top}" x2="{x(k)}" y2="{top}"/>']
    for r in range(1, k + 1):
        parts.append(f'<line x1="{x(r)}" y1="{top - 5}" x2="{x(r)}" y2="{top}"/><text x="{x(r)}" y="{top - 8}" '
                     f'text-anchor="middle">{r}</text>')
    parts.append(f'<line x1="{x(1)}" y1="{top - 40}" x2="{x(1 + cd)}" y2="{top - 40}" stroke-width="3"/>'
                 f'<line x1="{x(1)}" y1="{top - 45}" x2="{x(1)}" y2="{top - 35}"/>'
                 f'<line x1="{x(1 + cd)}" y1="{top - 45}" x2="{x(1 + cd)}" y2="{top - 35}"/>'
                 f'<text x="{x(1 + cd) + 6}" y="{top - 36}">CD</text>')
    for i, m in enumerate(order):  # labels: better half on the left, worse half on the right
        y = top + 30 + 22 * (i if i < (k + 1) // 2 else k - 1 - i)
        xr = x(mean_rank[m])
        lx = left - 8 if i < (k + 1) // 2 else w - right + 8
        anchor = "end" if i < (k + 1) // 2 else "start"
        parts.append(f'<line x1="{xr}" y1="{top}" x2="{xr}" y2="{y}"/><line x1="{xr}" y1="{y}" x2="{lx}" y2="{y}"/>'
                     f'<text x="{lx + (-4 if anchor == "end" else 4)}" y="{y + 4}" text-anchor="{anchor}">'
                     f'{LABELS[m]} ({mean_rank[m]:.2f})</text>')
    groups, seen = [], set()
    for i in range(k):  # maximal cliques of consecutive models within the CD
        j = i
        while j + 1 < k and mean_rank[order[j + 1]] - mean_rank[order[i]] <= cd:
            j += 1
        if j > i and (i, j) not in seen and not any(a <= i and j <= b for a, b in seen):
            seen.add((i, j))
            groups.append((i, j))
    for g, (i, j) in enumerate(groups):
        y = top + 14 + 6 * g
        parts.append(f'<line x1="{x(mean_rank[order[i]]) - 3}" y1="{y}" x2="{x(mean_rank[order[j]]) + 3}" y2="{y}" '
                     f'stroke-width="4" stroke-linecap="round"/>')
    return "\n".join(parts + ["</svg>"])


def main() -> None:
    import sys

    global MODELS, NEW
    extra = sys.argv[1:]
    MODELS = MODELS + extra
    NEW = NEW + extra
    tag = ("_" + "_".join(extra)) if extra else ""
    OUT.mkdir(parents=True, exist_ok=True)
    scores, status = {}, {}
    for d in DATASETS:
        sc, msg = dataset_scores(d)
        status[d] = msg
        if sc:
            scores[d] = sc
    lines = ["# P5: the 5 models across WHAR datasets (paper-era code, WHAR Arena v1)", "",
             "Built by `p5_summary.py`. Macro-F1 (%): per split the mean over training seeds 0–2, then the mean over "
             "splits. Pre-registered design in `RESEARCH_PLAN.md` §5b (P5).", "",
             "| Dataset | Status | " + " | ".join(LABELS[m] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for d in DATASETS:
        cells = [f"{scores[d][m]:.2f}" if d in scores else "" for m in MODELS]
        if d in scores:
            best = max(scores[d].values())
            cells = [f"**{c}**" if scores[d][m] == best else c for c, m in zip(cells, MODELS)]
        lines.append(f"| {d} | {status[d]} | " + " | ".join(cells) + " |")
    n = len(scores)
    if n >= 2:
        ranks = {d: dict(zip(MODELS, stats.rankdata([-scores[d][m] for m in MODELS]))) for d in scores}
        mean_rank = {m: st.mean(ranks[d][m] for d in scores) for m in MODELS}
        mean_f1 = {m: st.mean(scores[d][m] for d in scores) for m in MODELS}
        fr = stats.friedmanchisquare(*[[scores[d][m] for d in scores] for m in MODELS]) if n >= 3 else None
        cd = Q05[len(MODELS)] * (len(MODELS) * (len(MODELS) + 1) / (6 * n)) ** 0.5
        lines += ["", f"## Ranking over the {n} complete datasets", "",
                  f"Friedman χ² = {fr.statistic:.2f}, p = {fr.pvalue:.3g}. " if fr else "",
                  f"Nemenyi critical difference (α = 0.05): {cd:.2f} ranks. Diagram: `results/p5/cd{tag}.svg`.", "",
                  "| Model | Mean rank | Mean macro-F1 | Wins (rank 1) |", "|---|---|---|---|"]
        for m in sorted(MODELS, key=mean_rank.get):
            lines.append(f"| {LABELS[m]} | {mean_rank[m]:.2f} | {mean_f1[m]:.2f} | "
                         f"{sum(ranks[d][m] == 1 for d in scores)} |")
        pairs = list(itertools.combinations(MODELS, 2))
        pv = {}
        for a, b in pairs:
            diff = [scores[d][a] - scores[d][b] for d in scores]
            pv[(a, b)] = stats.wilcoxon(diff).pvalue if any(diff) else 1.0
        adj = holm(pv)
        lines += ["", "## Pairwise, over datasets (Wilcoxon signed-rank, Holm-corrected)", "",
                  "| A − B | Mean Δ | 95 % CI | Wins/losses | p | p (Holm) |", "|---|---|---|---|---|---|"]
        for a, b in pairs:
            diff = [scores[d][a] - scores[d][b] for d in scores]
            half = stats.t.ppf(0.975, n - 1) * st.stdev(diff) / n ** 0.5
            lines.append(f"| {LABELS[a]} − {LABELS[b]} | {st.mean(diff):+.2f} | [{st.mean(diff) - half:+.2f}, "
                         f"{st.mean(diff) + half:+.2f}] | {sum(x > 0 for x in diff)}/{sum(x < 0 for x in diff)} | "
                         f"{pv[(a, b)]:.3g} | {adj[(a, b)]:.3g} |")
        lines += ["", "## Pre-registered claims", "", "| New − old | Mean Δ | 95 % CI | Non-inferior (margin 1) | "
                  "Better (CI above 0) |", "|---|---|---|---|---|"]
        for a in NEW:
            for b in OLD:
                diff = [scores[d][a] - scores[d][b] for d in scores]
                half = stats.t.ppf(0.975, n - 1) * st.stdev(diff) / n ** 0.5
                lo = st.mean(diff) - half
                lines.append(f"| {LABELS[a]} − {LABELS[b]} | {st.mean(diff):+.2f} | [{lo:+.2f}, {st.mean(diff) + half:+.2f}] | "
                             f"{'yes' if lo > -MARGIN else 'no'} | {'yes' if lo > 0 else 'no'} |")
        (OUT / f"cd{tag}.svg").write_text(cd_svg(mean_rank, cd, n), encoding="utf-8")
        whar = {}
        for d in scores:
            for p in sorted(WHAR.glob(f"*/{d}")):
                f1 = [json.loads(q.read_text())["metrics"]["macro_f1"] * 100 for q in sorted(p.glob("split-*.json"))]
                if f1 and len(f1) == len(list((RES / "s4d" / d).glob("split-*.json"))):
                    whar.setdefault(d, []).append((st.mean(f1), p.parent.name))
        if whar:
            lines += ["", "## Context: best of the 17 WHAR baselines (paper-era re-run, seed 0)", "",
                      "| Dataset | Best WHAR model | Its macro-F1 | Best of our 5 |", "|---|---|---|---|"]
            for d, v in whar.items():
                f1, name = max(v)
                lines.append(f"| {d} | {name} | {f1:.2f} | {max(scores[d].values()):.2f} |")
    (OUT / f"summary{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", OUT / f"summary{tag}.md")


if __name__ == "__main__":
    main()
