"""What drives macro-F1: data factors and model components, with statistics (RESULTS_SUMMARY.md §13).
No training; reads finished result files and the WHAR Arena paper table.

A. Variance decomposition of our 3-seed runs: dataset, test person (split), model, model x dataset, model x person,
   seed.
B. Data factors over the paper's 30 datasets (17 models each): Spearman correlations and an OLS regression of the
   level of the deep models, the spread between models, and the Random Forest's margin, on dataset properties.
C. Model components: controlled contrasts between our models, paired on the same splits (seeds averaged within a
   split), per dataset with a t-interval over splits, then across datasets (mean, t-interval, wins, Wilcoxon), plus
   correlation of each effect with dataset properties.

    uv run python -m src.evaluation.factor_analysis   ->   results/metrics/factors/summary.md, results/metrics/factors/data.json
"""

from __future__ import annotations

import json
import math
import statistics as st

import numpy as np
from scipy import stats

from src.evaluation import compare_arena as CA
from src.utils import paths as PATHS  # noqa: E402

OUT = PATHS.METRICS / "factors"
DS = CA.DATASETS
# paper-era whar-datasets configs (cfg_<id>.py): sampling Hz, subjects, classes, cfg window seconds (not used:
# the WHAR Arena v1 protocol overrides it with 3-s windows for every dataset, whar_arena/training.py);
# channels from the paper's Table 3 (compare_arena.CHANNELS)
CFG = {
    "sad": (50, 10, 7, 2.56), "uci_har": (50, 30, 6, 2.56), "motion_sense": (50, 24, 6, 2.56), "wear": (50, 24, 18, 2),
    "dsads": (25, 8, 19, 5.0), "mhealth": (50, 10, 12, 2.56), "wisdm": (20, 36, 6, 5), "real_disp": (50, 17, 34, 3),
    "hapt": (50, 30, 12, 2.56), "hhar": (20, 9, 6, 5), "usc_had": (100, 14, 12, 1.28), "wisdm_19_watch": (20, 51, 18, 5),
    "opportunity": (30, 4, 5, 1), "harth": (50, 22, 12, 3), "gotov": (20, 30, 16, 2), "real_world": (50, 15, 8, 2.56),
    "har70": (50, 18, 7, 3), "pamap2": (100, 9, 13, 1.28), "up_fall": (20, 17, 11, 2), "utd_mhad": (50, 8, 27, 2),
    "uca_ehar": (25, 20, 12, 2), "unimib_shar": (50, 30, 17, 3.02), "real_life_har": (20, 17, 4, 2.56),
    "hang_time": (50, 24, 6, 2.56), "w_har": (250, 22, 9, 1.28), "uma_fall": (200, 19, 15, 3), "hugadb": (60, 18, 12, 2),
    "actrectut_gestures": (32, 2, 12, 2), "daphnet": (64, 10, 3, 1.0), "wisdm_19_phone": (20, 51, 18, 5)}
# total windows (train + val + test of one split) from the cloud warm-up logs; our 10 datasets only
WINDOWS = {"utd_mhad": 783, "uci_har": 8188, "unimib_shar": 7195, "actrectut_gestures": 798, "opportunity": 10550,
           "sad": 8260, "hapt": 9056, "dsads": 30096, "wisdm": 34097, "daphnet": 11160}
SEED3 = ["recurrencenet", "s4d", "hystnet_sel_clock", "quantilenet", "mamba", "hystnet_nomem", "recstatnet",
         "lawnet_rec", "lawnet", "statnet"]
NAME = {"recurrencenet": "RecurrenceNet", "s4d": "S4D", "hystnet_sel_clock": "HysteresisNet", "quantilenet": "QuantileNet",
        "mamba": "Mamba", "hystnet_nomem": "no memory", "recstatnet": "RN+stats", "lawnet_rec": "RN+laws",
        "lawnet": "laws (no memory)", "statnet": "stats (no memory)", "lawnet_rand": "laws, random groups",
        "recstat_mean": "RN+mean", "recstat_std": "RN+std", "recstat_range": "RN+min/max",
        "recnet_meanstd": "RN, mean+std pooling", "recnet_nonorm": "RN, no LayerNorm",
        "recnet_meanstd_nonorm": "RN, both"}
KILL = ["utd_mhad", "actrectut_gestures", "sad", "opportunity"]
CONTRASTS = [  # (group, a, b, seeds); effect = a − b
    ("Memory (any) vs none", "recurrencenet", "hystnet_nomem", 3),
    ("Memory (any) vs none", "s4d", "hystnet_nomem", 3),
    ("Memory (any) vs none", "hystnet_sel_clock", "hystnet_nomem", 3),
    ("Memory (any) vs none", "quantilenet", "hystnet_nomem", 3),
    ("Memory (any) vs none", "mamba", "hystnet_nomem", 3),
    ("Memory type (vs S4D)", "recurrencenet", "s4d", 3),
    ("Memory type (vs S4D)", "hystnet_sel_clock", "s4d", 3),
    ("Memory type (vs S4D)", "quantilenet", "s4d", 3),
    ("Memory type (vs S4D)", "mamba", "s4d", 3),
    ("Window statistics", "statnet", "hystnet_nomem", 3),
    ("Window statistics", "recstatnet", "recurrencenet", 3),
    ("Laws (lagged second-order)", "lawnet", "hystnet_nomem", 3),
    ("Laws (lagged second-order)", "lawnet_rec", "recurrencenet", 3),
    ("Laws (lagged second-order)", "lawnet_rec", "recstatnet", 3),
    ("Sensor grouping", "lawnet", "lawnet_rand", 1),
]
KILL_CONTRASTS = [  # kill set only, 3 seeds
    ("Which statistic", "recstat_mean", "recurrencenet"), ("Which statistic", "recstat_std", "recurrencenet"),
    ("Which statistic", "recstat_range", "recurrencenet"), ("Which statistic", "recstatnet", "recurrencenet"),
    ("Normalization / pooling", "recnet_meanstd", "recurrencenet"), ("Normalization / pooling", "recnet_nonorm", "recurrencenet"),
    ("Normalization / pooling", "recnet_meanstd_nonorm", "recurrencenet")]


def split_scores(m: str, d: str, seeds: int) -> dict[str, float] | None:
    """Per split, the mean over seeds 0..seeds-1; None unless every split and seed is there."""
    want = [f"split-{i:02d}" for i in range(CA.N_SPLITS[d])]
    per = []
    for s in range(seeds):
        sc = CA.split_f1(CA.RES / m / d / f"seed-{s}" if s else CA.RES / m / d)
        if not set(want) <= set(sc):
            return None
        per.append(sc)
    return {k: st.mean(p[k] for p in per) for k in want}


def t_ci(x: list[float]) -> tuple[float, float, float]:
    m = st.mean(x)
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * st.stdev(x) / math.sqrt(len(x))
    return m, m - h, m + h


def wilcoxon_p(x: list[float]) -> float:
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 5 and any(x) else float("nan")


def props10() -> dict[str, dict[str, float]]:
    return {"log2 subjects": {d: math.log2(CFG[d][1]) for d in DS},
            "log2 channels": {d: math.log2(CA.CHANNELS[d]) for d in DS},
            "log2 classes": {d: math.log2(CFG[d][2]) for d in DS},
            "samples/window (3 s × Hz)": {d: 3 * CFG[d][0] for d in DS},
            "log2 windows": {d: math.log2(WINDOWS[d]) for d in DS}}


# ---------------------------------------------------------------- A. variance decomposition
def variance_decomposition() -> tuple[list[str], dict]:
    y = {}  # (m, d, split, seed) -> F1
    for m in SEED3:
        for d in DS:
            for s in range(3):
                sc = CA.split_f1(CA.RES / m / d / f"seed-{s}" if s else CA.RES / m / d)
                for i in range(CA.N_SPLITS[d]):
                    y[(m, d, f"split-{i:02d}", s)] = sc[f"split-{i:02d}"]
    mu = st.mean(y.values())
    mean = lambda keys: st.mean(y[k] for k in keys)  # noqa: E731
    keys_d = {d: [k for k in y if k[1] == d] for d in DS}
    yd = {d: mean(keys_d[d]) for d in DS}
    yds = {(d, sp): mean([k for k in keys_d[d] if k[2] == sp]) for d in DS for sp in {k[2] for k in keys_d[d]}}
    ymd = {(m, d): mean([k for k in keys_d[d] if k[0] == m]) for m in SEED3 for d in DS}
    ymds = {(m, d, sp): mean([k for k in keys_d[d] if k[0] == m and k[2] == sp]) for m in SEED3 for d in DS
            for sp in {k[2] for k in keys_d[d]}}
    cm = {m: st.mean(ymd[(m, d)] - yd[d] for d in DS) for m in SEED3}
    ss = dict.fromkeys(["dataset", "test person (split)", "model", "model x dataset", "model x person", "seed"], 0.0)
    for (m, d, sp, s), v in y.items():
        ss["dataset"] += (yd[d] - mu) ** 2
        ss["test person (split)"] += (yds[(d, sp)] - yd[d]) ** 2
        ss["model"] += cm[m] ** 2
        ss["model x dataset"] += (ymd[(m, d)] - yd[d] - cm[m]) ** 2
        ss["model x person"] += (ymds[(m, d, sp)] - ymd[(m, d)] - yds[(d, sp)] + yd[d]) ** 2
        ss["seed"] += (v - ymds[(m, d, sp)]) ** 2
    total = sum((v - mu) ** 2 for v in y.values())
    within = {k: v for k, v in ss.items() if k != "dataset"}
    tw = sum(within.values())
    L = ["## A. Where the variance of macro-F1 comes from (our 10 models × 10 datasets × splits × 3 seeds)", "",
         f"{len(y)} runs. Sum of squares by source; shares of the total, and of the variance left once the dataset "
         "is fixed (what decides a ranking within a dataset).", "",
         "| Source | Share of total | Share within a dataset | Typical size (RMS, F1 points) |", "|---|---|---|---|"]
    for k, v in ss.items():
        within_share = f"{100 * v / tw:.1f} %" if k != "dataset" else "–"
        L.append(f"| {k} | {100 * v / total:.1f} % | {within_share} | {math.sqrt(v / len(y)):.2f} |")
    L.append(f"| sum of parts / total | {100 * sum(ss.values()) / total:.1f} % | | |")
    return L, {"ss": ss, "total": total, "n": len(y)}


# ---------------------------------------------------------------- B. data factors over 30 datasets
def data_factors() -> tuple[list[str], dict]:
    deep = [a for a in CA.ARENA if a not in CA.CLASSICAL]
    ds30 = [d for d in CA.PAPER if d != "Mean"]
    f = lambda d, a: CA.PAPER[d][a][0]  # noqa: E731
    outcome = {
        "median deep F1 (level)": {d: st.median(f(d, a) for a in deep) for d in ds30},
        "best − median deep (does the model matter)": {d: max(f(d, a) for a in deep) - st.median(f(d, a) for a in deep) for d in ds30},
        "RF − best deep (classical margin)": {d: f(d, "Random Forest") - max(f(d, a) for a in deep) for d in ds30},
        "mean split sd of deep models (person spread)": {d: st.mean(CA.PAPER[d][a][1] for a in deep) for d in ds30},
    }
    pred = {"log2 subjects": {d: math.log2(CFG[d][1]) for d in ds30},
            "log2 channels": {d: math.log2(CA.CHANNELS[d]) for d in ds30},
            "log2 classes": {d: math.log2(CFG[d][2]) for d in ds30},
            "log2 samples/window": {d: math.log2(3 * CFG[d][0]) for d in ds30}}  # 3-s windows: = sampling rate
    L = ["", "## B. Data factors over the paper's 30 datasets (17 models, one run each)", "",
         "Spearman ρ (p) between each dataset property and each outcome, n = 30:", "",
         "| Outcome | " + " | ".join(pred) + " |", "|---|" + "---|" * len(pred)]
    corr = {}
    for o, ov in outcome.items():
        cells = []
        for p, pv in pred.items():
            r = stats.spearmanr([pv[d] for d in ds30], [ov[d] for d in ds30])
            corr[(o, p)] = (float(r.statistic), float(r.pvalue))
            cells.append(f"{r.statistic:+.2f} ({r.pvalue:.3f})")
        L.append(f"| {o} | " + " | ".join(cells) + " |")
    # collinearity of the properties
    pk = list(pred)
    L += ["", "Correlations among the properties (Spearman ρ; large values mean their effects cannot be separated):", "",
          "| | " + " | ".join(pk) + " |", "|---|" + "---|" * len(pk)]
    for a in pk:
        L.append(f"| {a} | " + " | ".join(f"{stats.spearmanr([pred[a][d] for d in ds30], [pred[b][d] for d in ds30]).statistic:+.2f}"
                                         for b in pk) + " |")
    # OLS with standardized predictors
    use = ["log2 subjects", "log2 channels", "log2 classes", "log2 samples/window"]
    X = np.array([[pred[p][d] for p in use] for d in ds30])
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    X1 = np.column_stack([np.ones(len(ds30)), X])
    ols = {}
    for o in outcome:
        yv = np.array([outcome[o][d] for d in ds30])
        beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
        resid = yv - X1 @ beta
        dof = len(yv) - X1.shape[1]
        s2 = resid @ resid / dof
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
        tval = beta / se
        pval = 2 * stats.t.sf(np.abs(tval), dof)
        r2 = 1 - resid @ resid / ((yv - yv.mean()) @ (yv - yv.mean()))
        ols[o] = {"beta": beta.tolist(), "se": se.tolist(), "p": pval.tolist(), "r2": float(r2)}
    L += ["", "Multiple regression (OLS, n = 30, predictors standardized: coefficient = change in F1 points per 1 sd of "
          "the property, others held fixed; p in brackets):", "",
          "| Outcome | R² | " + " | ".join(use) + " |", "|---|---|" + "---|" * len(use)]
    for o, r in ols.items():
        L.append(f"| {o} | {r['r2']:.2f} | " + " | ".join(f"{b:+.2f} ({p:.3f})" for b, p in zip(r["beta"][1:], r["p"][1:])) + " |")
    return L, {"outcome": outcome, "pred": pred, "ols": ols, "corr": {f"{a}|{b}": v for (a, b), v in corr.items()}}


# ---------------------------------------------------------------- C. model components
def contrast(a: str, b: str, seeds: int, datasets: list[str]) -> dict:
    per = {}
    for d in datasets:
        sa, sb = split_scores(a, d, seeds), split_scores(b, d, seeds)
        if sa is None or sb is None:
            continue
        diffs = [sa[k] - sb[k] for k in sa]
        m, lo, hi = t_ci(diffs)
        per[d] = {"effect": m, "lo": lo, "hi": hi, "split_wins": sum(x > 0 for x in diffs), "splits": len(diffs)}
    eff = [per[d]["effect"] for d in per]
    m, lo, hi = t_ci(eff)
    return {"per_dataset": per, "mean": m, "lo": lo, "hi": hi, "wins": sum(x > 0 for x in eff), "n": len(eff),
            "p": wilcoxon_p(eff), "between_sd": st.stdev(eff) if len(eff) > 1 else float("nan")}


def components() -> tuple[list[str], dict]:
    res = {}
    short = {d: CA.SHORT[d][:6] for d in DS}
    L = ["", "## C. Model components: controlled contrasts, paired on the same splits", "",
         "Effect = a − b in F1 points; per dataset the mean over splits of the seed-averaged difference; across datasets "
         "the mean with a 95 % t-interval, datasets won, Wilcoxon p (no correction), and the between-dataset sd "
         "(heterogeneity). ★ = per-dataset 95 % interval over splits excludes 0.", "",
         "| Component | a − b | Seeds | " + " | ".join(short[d] for d in DS) + " | Mean [95 % CI] | Won | p | sd between datasets |",
         "|---|---|---|" + "---|" * len(DS) + "---|---|---|---|"]
    for group, a, b, seeds in CONTRASTS:
        c = contrast(a, b, seeds, DS)
        res[f"{a}-{b}"] = c | {"group": group}
        cells = []
        for d in DS:
            p = c["per_dataset"].get(d)
            if p is None:
                cells.append("–")
            else:
                sig = "★" if not math.isnan(p["lo"]) and (p["lo"] > 0 or p["hi"] < 0) else ""
                cells.append(f"{p['effect']:+.1f}{sig}")
        L.append(f"| {group} | {NAME[a]} − {NAME[b]} | {seeds} | " + " | ".join(cells)
                 + f" | {c['mean']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] | {c['wins']}/{c['n']} | {c['p']:.3f} | {c['between_sd']:.1f} |")
    L += ["", "Kill set only (UTD-MHAD, ActRecTut, SAD, Opportunity; 3 seeds):", "",
          "| Component | a − b | " + " | ".join(CA.SHORT[d] for d in KILL) + " | Mean | Without ActRecTut |", "|---|---|" + "---|" * (len(KILL) + 2)]
    for group, a, b in KILL_CONTRASTS:
        c = contrast(a, b, 3, KILL)
        res[f"kill:{a}-{b}"] = c | {"group": group}  # separate key: recstatnet − RN also exists over 10 datasets
        e = {d: c["per_dataset"][d]["effect"] for d in c["per_dataset"]}
        L.append(f"| {group} | {NAME[a]} − {NAME[b]} | " + " | ".join(f"{e[d]:+.1f}" if d in e else "–" for d in KILL)
                 + f" | {st.mean(e.values()):+.2f} | {st.mean(v for d, v in e.items() if d != 'actrectut_gestures'):+.2f} |")
    # moderators: which data properties go with each effect (n = 10)
    pr = props10()
    L += ["", "Which data properties go with each effect (Spearman ρ over the 10 datasets, p in brackets; descriptive):", "",
          "| a − b | " + " | ".join(pr) + " |", "|---|" + "---|" * len(pr)]
    for group, a, b, seeds in CONTRASTS:
        c = res[f"{a}-{b}"]
        ds = list(c["per_dataset"])
        if len(ds) < 6:
            continue
        L.append(f"| {NAME[a]} − {NAME[b]} | " + " | ".join(
            (lambda r: f"{r.statistic:+.2f} ({r.pvalue:.2f})")(stats.spearmanr([pv[d] for d in ds], [c['per_dataset'][d]['effect'] for d in ds]))
            for pv in pr.values()) + " |")
    # model size, paper deep models
    deep = [a for a in CA.ARENA if a not in CA.CLASSICAL]
    sizes = {a: CA.PROFILE.get(("utd_mhad", CA.WHAR_IDS[a])) for a in deep}
    sized = [a for a in deep if sizes[a]]
    ds30 = [d for d in CA.PAPER if d != "Mean"]
    mean30 = {a: st.mean(CA.PAPER[d][a][0] for d in ds30) for a in sized}
    mean10 = {a: st.mean(CA.PAPER[d][a][0] for d in DS) for a in sized}
    r30 = stats.spearmanr([math.log(sizes[a]) for a in sized], [mean30[a] for a in sized])
    r10 = stats.spearmanr([math.log(sizes[a]) for a in sized], [mean10[a] for a in sized])
    L += ["", f"Model size (paper deep models, n = {len(sized)}): Spearman(log params, mean F1) = {r30.statistic:+.2f} "
          f"(p {r30.pvalue:.2f}) over 30 datasets, {r10.statistic:+.2f} (p {r10.pvalue:.2f}) over our 10."]
    return L, res


def main() -> None:
    la, a = variance_decomposition()
    lb, b = data_factors()
    lc, c = components()
    lines = ["# What drives macro-F1: data factors and model components", ""] + la + lb + lc
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "data.json").write_text(json.dumps({"variance": a, "data": b, "components": c}, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
