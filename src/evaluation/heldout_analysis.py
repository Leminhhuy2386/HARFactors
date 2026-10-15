"""Held-out confirmation for the paper (RESEARCH_PLAN.md §5p, pre-registered 2026-10-12). No training.

Primary, over the 8 held-out datasets, paired over datasets:
  1. lawnet_rec − S4D on 3-seed means: mean, 95 % t-interval, wins, Wilcoxon; "better" only if the lower bound > 0.
  2. lawnet_rec (seed 0) − each paper model; non-inferiority to TinyHAR, CNN-HAR and Triple-Cross-Attn with a 1-point
     margin (lower bound > −1).
Secondary: recstatnet under the same tests; lawnet_rec − RN; all 18 datasets together (labelled as including dev data).
A dataset counts only when every model has every split at every seed (no mixing of split sets).

    uv run python -m src.evaluation.heldout_analysis   ->   results/metrics/heldout/summary.md
"""

from __future__ import annotations

import json
import math
import statistics as st

from scipy import stats

from src.evaluation import compare_arena as CA
from src.utils import paths as PATHS  # noqa: E402

OUT = PATHS.METRICS / "heldout"
HELDOUT = ["pamap2", "up_fall", "uma_fall", "usc_had", "motion_sense", "wear", "uca_ehar", "w_har"]
SHORT = {"pamap2": "PAMAP2", "up_fall": "UP-Fall", "uma_fall": "UMA-Fall", "usc_had": "USC-HAD",
         "motion_sense": "MotionSense", "wear": "WEAR", "uca_ehar": "UCA-EHAR", "w_har": "W-HAR"}
UNSORTED = {"motion_sense", "usc_had"}  # parsers list files without sorting: compare with the paper by means only
OURS = ["lawnet_rec", "recstatnet", "recurrencenet", "s4d"]
LABEL = {"lawnet_rec": "RN + laws", "recstatnet": "RN + statistics", "recurrencenet": "RecurrenceNet", "s4d": "S4D"}
TOP3 = ["TinyHAR", "CNN-HAR", "Triple-Cross-Attn"]  # fixed in §5p before any held-out run
MARGIN = 1.0


def split_scores(m: str, d: str, seed: int) -> dict[str, float]:
    return CA.split_f1(CA.RES / m / d / f"seed-{seed}" if seed else CA.RES / m / d)


def complete(d: str) -> int | None:
    """Number of splits if every model has the same full set at every seed, else None."""
    sets = [set(split_scores(m, d, s)) for m in OURS for s in (0, 1, 2)]
    if not sets or not sets[0] or any(x != sets[0] for x in sets):
        return None
    n = len(sets[0])
    return n if sets[0] == {f"split-{i:02d}" for i in range(n)} else None


def mean_f1(m: str, d: str, seeds=(0, 1, 2)) -> float:
    return st.mean(st.mean(split_scores(m, d, s).values()) for s in seeds)


def paired(diff: list[float]) -> dict:
    m = st.mean(diff)
    h = stats.t.ppf(0.975, len(diff) - 1) * st.stdev(diff) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    p = float(stats.wilcoxon(diff).pvalue) if len(diff) >= 5 and any(diff) else float("nan")
    return {"mean": m, "lo": m - h, "hi": m + h, "won": sum(x > 0 for x in diff), "lost": sum(x < 0 for x in diff),
            "n": len(diff), "p": p}


def fmt(r: dict) -> str:
    return (f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], {r['won']}–{r['lost']} of {r['n']}, "
            f"Wilcoxon p {r['p']:.3f}" if not math.isnan(r["p"]) else
            f"{r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], {r['won']}–{r['lost']} of {r['n']}")


def main() -> None:
    ready = {d: complete(d) for d in HELDOUT}
    done = [d for d in HELDOUT if ready[d]]
    L = ["# Held-out confirmation (RESEARCH_PLAN.md §5p)", "",
         f"Complete datasets: {len(done)} of {len(HELDOUT)} "
         f"({', '.join(f'{SHORT[d]} ({ready[d]} splits)' for d in done) or 'none yet'}). "
         f"Pending: {', '.join(SHORT[d] for d in HELDOUT if not ready[d]) or 'none'}.", ""]
    if done:
        L += ["## Per dataset (macro-F1 %, mean over splits; ours: 3-seed mean, paper: one run)", "",
              "| Model | " + " | ".join(SHORT[d] + ("†" if d in UNSORTED else "") for d in done) + " | Mean |",
              "|---|" + "---|" * (len(done) + 1)]
        three = {m: {d: mean_f1(m, d) for d in done} for m in OURS}
        seed0 = {m: {d: mean_f1(m, d, (0,)) for d in done} for m in OURS}
        for m in OURS:
            L.append(f"| {LABEL[m]} | " + " | ".join(f"{three[m][d]:.1f}" for d in done) + f" | {st.mean(three[m].values()):.2f} |")
        best = {d: max(CA.ARENA, key=lambda a: CA.PAPER[d][a][0]) for d in done}
        for a in TOP3 + ["Random Forest"]:
            L.append(f"| {a} (paper) | " + " | ".join(f"{CA.PAPER[d][a][0]:.1f}" for d in done)
                     + f" | {st.mean(CA.PAPER[d][a][0] for d in done):.2f} |")
        L.append("| best paper model | " + " | ".join(f"{CA.PAPER[d][best[d]][0]:.1f} {best[d]}" for d in done) + " | |")
        L += ["", "† parser lists files unsorted: subject numbering may differ from the paper's machine; compared by means only.", ""]
        if len(done) >= 2:
            L += ["## Primary (pre-registered)", ""]
            r1 = paired([three["lawnet_rec"][d] - three["s4d"][d] for d in done])
            L.append(f"1. RN + laws − S4D (3 seeds): {fmt(r1)} → "
                     f"{'better than S4D' if r1['lo'] > 0 else 'not shown better than S4D'}.")
            L.append("2. RN + laws (seed 0) − paper models, non-inferiority margin −1 on the top three:")
            for a in sorted(CA.ARENA, key=lambda a: -st.mean(CA.PAPER[d][a][0] for d in done)):
                r = paired([seed0["lawnet_rec"][d] - CA.PAPER[d][a][0] for d in done])
                tag = (" — non-inferior" if r["lo"] > -MARGIN else " — non-inferiority not shown") if a in TOP3 else ""
                L.append(f"   - {a}: {fmt(r)}{tag}")
            L += ["", "## Secondary", ""]
            r = paired([three["recstatnet"][d] - three["s4d"][d] for d in done])
            L.append(f"- RN + statistics − S4D (3 seeds): {fmt(r)}")
            for a in TOP3:
                r = paired([seed0["recstatnet"][d] - CA.PAPER[d][a][0] for d in done])
                L.append(f"- RN + statistics (seed 0) − {a}: {fmt(r)}"
                         f"{' — non-inferior' if r['lo'] > -MARGIN else ' — non-inferiority not shown'}")
            L.append(f"- RN + laws − RecurrenceNet (3 seeds): {fmt(paired([three['lawnet_rec'][d] - three['recurrencenet'][d] for d in done]))}")
            L.append(f"- RecurrenceNet − S4D (3 seeds): {fmt(paired([three['recurrencenet'][d] - three['s4d'][d] for d in done]))}")
            dev = CA.DATASETS
            all18 = [three["lawnet_rec"][d] - three["s4d"][d] for d in done] + \
                    [CA.ours_seed_mean("lawnet_rec", d) - CA.ours_seed_mean("s4d", d) for d in dev]
            L.append(f"- All {len(all18)} datasets (includes the 10 development datasets): RN + laws − S4D {fmt(paired(all18))}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
