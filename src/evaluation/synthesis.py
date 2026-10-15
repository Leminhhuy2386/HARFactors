"""Synthesis of rounds 1-13: every finished model on the 10 dev datasets, the dataset properties, and which factors
go with which gains (RESULTS_SUMMARY.md §12). No training; reads finished result files only.

    uv run python -m src.evaluation.synthesis   ->   results/metrics/synthesis/summary.md, results/metrics/synthesis/data.json
"""

from __future__ import annotations

import json
import math
import statistics as st

from scipy import stats

from src.evaluation import compare_arena as CA
from src.utils import paths as PATHS  # noqa: E402

OUT = PATHS.METRICS / "synthesis"
DS = CA.DATASETS
# paper-era whar-datasets configs (cfg_<id>.py); ActRecTut channels from the paper's Table 3
PROPS = {  # sampling Hz, subjects, classes, channels, cfg window seconds (unused: the protocol uses 3 s)
    "utd_mhad": (50, 8, 27, 6, 2.0), "uci_har": (50, 30, 6, 9, 2.56), "unimib_shar": (50, 30, 17, 3, 3.02),
    "actrectut_gestures": (32, 2, 12, 15, 2.0), "opportunity": (30, 4, 5, 133, 1.0), "sad": (50, 10, 7, 60, 2.56),
    "hapt": (50, 30, 12, 6, 2.56), "dsads": (25, 8, 19, 45, 5.0), "wisdm": (20, 36, 6, 3, 5.0),
    "daphnet": (64, 10, 3, 9, 1.0)}
KIND = {"utd_mhad": "gesture", "actrectut_gestures": "gesture", "uci_har": "locomotion/ADL",
        "unimib_shar": "locomotion/ADL", "hapt": "locomotion/ADL", "sad": "locomotion/ADL", "dsads": "locomotion/ADL",
        "wisdm": "locomotion/ADL", "opportunity": "locomotion, many sensors", "daphnet": "freezing of gait"}
SEED3 = ["recstatnet", "lawnet_rec", "recurrencenet", "s4d", "hystnet_sel_clock", "lawnet", "statnet", "quantilenet",
         "mamba", "hystnet_nomem"]
SEED0 = ["stylealignnet", "lawnet_rand"]
LABEL = {"recstatnet": "RN + channel statistics", "lawnet_rec": "RN + LawNet", "recurrencenet": "RecurrenceNet (RN)",
         "s4d": "S4D", "hystnet_sel_clock": "HysteresisNet sel+clock", "lawnet": "LawNet (memoryless + laws)",
         "statnet": "Statistics (memoryless + stats)", "quantilenet": "QuantileNet", "mamba": "Mamba",
         "hystnet_nomem": "No memory (control)", "stylealignnet": "StyleAlignNet (seed 0)",
         "lawnet_rand": "LawNet, random triples (seed 0)"}
PROBE = ["root/et", "tree_abs/et", "tree_abs/lr"]


def probe(kind: str, d: str) -> float | None:
    p = PATHS.EXPERIMENTS / "tree_probe" / f"{d}.json"
    return json.loads(p.read_text())["mean"][kind] if p.exists() else None


def elastic(kind: str, d: str) -> float | None:
    files = sorted((PATHS.EXPERIMENTS / "elastic" / d).glob("split-*.json"))
    return st.mean(100 * json.loads(f.read_text())["macro_f1"][kind] for f in files) if len(files) >= CA.N_SPLITS[d] else None


def seed_values(m: str, d: str) -> list[float]:
    return [v for v in (CA.ours(m, d, s) for s in (0, 1, 2)) if v is not None]


def spearman(x: list[float], y: list[float]) -> str:
    r = stats.spearmanr(x, y)
    return f"{r.statistic:+.2f} (p {r.pvalue:.2f})"


def main() -> None:
    table = {m: {d: CA.ours_seed_mean(m, d) for d in DS} for m in SEED3}
    table |= {m: {d: CA.ours(m, d, 0) for d in DS} for m in SEED0}
    table |= {f"probe {k}": {d: probe(k, d) for d in DS} for k in PROBE}
    table["1-NN DTW std"] = {d: elastic("dtw_std", d) for d in DS}
    table["Paper best (17)"] = {d: max(CA.PAPER[d][a][0] for a in CA.ARENA) for d in DS}
    table["Paper RF"] = {d: CA.PAPER[d]["Random Forest"][0] for d in DS}
    table["Paper CNN-HAR"] = {d: CA.PAPER[d]["CNN-HAR"][0] for d in DS}
    table["Paper TinyHAR"] = {d: CA.PAPER[d]["TinyHAR"][0] for d in DS}
    L = []
    L += ["# Synthesis of rounds 1-13 (10 dev datasets)", "",
          "3-seed means where seeds 0-2 exist, else seed 0 (marked). Probe = segment statistics + classical classifier "
          "(no deep training; DSADS and Daphnet not run). Paper = WHAR Arena Figure 6 (one run).", "",
          "| Model | " + " | ".join(CA.SHORT[d] for d in DS) + " | Mean (10) | Mean w/o ActRecTut |",
          "|---|" + "---|" * (len(DS) + 2)]
    for m, row in table.items():
        cells = [f"{row[d]:.1f}" if row[d] is not None else "–" for d in DS]
        full = None not in row.values()
        mean10 = f"{st.mean(row.values()):.2f}" if full else "–"
        w = [row[d] for d in DS if d != "actrectut_gestures"]
        mean9 = f"{st.mean(w):.2f}" if None not in w else "–"
        L.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + f" | {mean10} | {mean9} |")

    # dataset properties
    L += ["", "## Dataset properties (paper-era configs)", "",
          "| Dataset | Kind | Hz | Subjects | Classes | Channels | Samples per 3-s window |", "|---|---|---|---|---|---|---|"]
    for d in DS:
        hz, subj, cls, ch, w = PROPS[d]
        L.append(f"| {CA.SHORT[d]} | {KIND[d]} | {hz} | {subj} | {cls} | {ch} | {3 * hz} |")

    # gains of each component, per dataset
    def diff(a: str, b: str) -> dict[str, float | None]:
        return {d: (table[a][d] - table[b][d]) if table[a][d] is not None and table[b][d] is not None else None for d in DS}

    gains = {
        "memory: RN − no memory": diff("recurrencenet", "hystnet_nomem"),
        "memory: S4D − no memory": diff("s4d", "hystnet_nomem"),
        "statistics on memoryless: statnet − no memory": diff("statnet", "hystnet_nomem"),
        "laws on memoryless: LawNet − no memory": diff("lawnet", "hystnet_nomem"),
        "statistics on RN: recstatnet − RN": diff("recstatnet", "recurrencenet"),
        "laws on RN: lawnet_rec − RN": diff("lawnet_rec", "recurrencenet"),
        "classical vs deep: probe root/et − RN": diff("probe root/et", "recurrencenet"),
        "classical vs deep: probe tree_abs/et − RN": diff("probe tree_abs/et", "recurrencenet"),
        "elastic vs deep: 1-NN DTW std − RN": diff("1-NN DTW std", "recurrencenet"),
        "RN − S4D": diff("recurrencenet", "s4d"),
        "S4D − Mamba": diff("s4d", "mamba"),
    }
    L += ["", "## Gain of each component per dataset (points of macro-F1)", "",
          "| Contrast | " + " | ".join(CA.SHORT[d] for d in DS) + " | Mean |", "|---|" + "---|" * (len(DS) + 1)]
    for k, g in gains.items():
        vals = [v for v in g.values() if v is not None]
        L.append(f"| {k} | " + " | ".join(f"{g[d]:+.1f}" if g[d] is not None else "–" for d in DS)
                 + f" | {st.mean(vals):+.2f} |")

    # factor correlations
    props = {"log2 subjects": {d: math.log2(PROPS[d][1]) for d in DS},
             "log2 channels": {d: math.log2(PROPS[d][3]) for d in DS},
             "samples/window (3 s × Hz)": {d: 3 * PROPS[d][0] for d in DS},
             "classes": {d: PROPS[d][2] for d in DS}}
    L += ["", "## Spearman correlation of each gain with dataset properties (n = 10, or 8 for the probe; "
          "descriptive only, no correction)", "",
          "| Contrast | " + " | ".join(props) + " |", "|---|" + "---|" * len(props)]
    for k, g in gains.items():
        ds = [d for d in DS if g[d] is not None]
        L.append(f"| {k} | " + " | ".join(spearman([p[d] for d in ds], [g[d] for d in ds]) for p in props.values()) + " |")

    # evaluation noise: seeds and people
    L += ["", "## Evaluation noise", "",
          "Seed spread = sample std over seeds 0-2 of the dataset mean. Person spread = sample std over subject splits "
          "(seed 0), the paper's ±.", "",
          "| Model | " + " | ".join(CA.SHORT[d] for d in DS) + " | Mean |", "|---|" + "---|" * (len(DS) + 1)]
    for m in ["recurrencenet", "s4d", "recstatnet"]:
        sd = {d: st.stdev(seed_values(m, d)) for d in DS}
        L.append(f"| seed spread, {LABEL[m]} | " + " | ".join(f"{sd[d]:.2f}" for d in DS) + f" | {st.mean(sd.values()):.2f} |")
    for m in ["recurrencenet", "recstatnet"]:
        ps = {d: st.stdev(list(CA.split_f1(CA.RES / m / d).values())) for d in DS}
        L.append(f"| person spread, {LABEL[m]} | " + " | ".join(f"{ps[d]:.1f}" for d in DS) + f" | {st.mean(ps.values()):.1f} |")
    gap = CA.subject_gap()
    if gap["recurrencenet"]["mean"] is not None:
        g = gap["recurrencenet"]["per_dataset"]
        L.append("| validation (seen people) − test (unseen), RN seed 0 | " + " | ".join(f"{g[d]:.1f}" for d in DS)
                 + f" | {st.mean(g.values()):.1f} |")

    # size vs accuracy in the paper table (deep models with a profiled size)
    deep = [a for a in CA.ARENA if a not in CA.CLASSICAL]
    sizes = {a: CA.PROFILE.get(("utd_mhad", CA.WHAR_IDS[a])) for a in deep}
    sized = [a for a in deep if sizes[a]]
    mean_f1 = {a: st.mean(CA.PAPER[d][a][0] for d in DS) for a in sized}
    L += ["", f"## Size vs accuracy, paper deep models on these 10 datasets (n = {len(sized)})", "",
          f"Spearman(log params, mean F1) = {spearman([math.log(sizes[a]) for a in sized], [mean_f1[a] for a in sized])}. "
          "Ours (20-38k params) for reference: RN 24.7k, recstatnet 26.5k, S4D 27.7k.", ""]
    for a in sorted(sized, key=lambda a: -mean_f1[a]):
        L.append(f"- {a}: {sizes[a] / 1e3:.1f}k params, mean F1 {mean_f1[a]:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "data.json").write_text(json.dumps({"table": table, "gains": gains}, indent=1) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
