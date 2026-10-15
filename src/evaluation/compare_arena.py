"""Our models against the WHAR Arena paper's benchmark table (Figure 6, 17 models x 30 datasets).

No training: reads finished results only. The paper's numbers come from
docs/sources/arena_extracted.json (copied from the HAR reproduction's tools/; Figure 6, matched 510/510 to whar-arena's
paper-benchmark.json). Our paper-era harness reproduces that table on the 4 dev datasets (checked
below against the 17 models re-run with the paper-era code, experiments/external_har_paper_era/whar_arena_v1), so a paper cell and one of our
cells share data, windows, splits and protocol.

Our models are compared at training seed 0, the protocol run, which is what the paper reports
(one run per split); the mean over seeds 0-2 is given alongside. "Rank among Arena" inserts one of
our models into the 17 paper models on that dataset (1 = above all of them).

    uv run python -m src.evaluation.compare_arena   -> results/metrics/arena/compare.md, results/metrics/arena/compare.json
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy import stats
from src.utils import paths as PATHS  # noqa: E402

HERE = Path(__file__).resolve().parents[2]  # project root
RES = PATHS.RUNS
WHAR = PATHS.EXTERNAL_RUNS  # the 17 paper models re-run on 4 datasets by the HAR reproduction
EXTRACTED = json.loads((PATHS.SOURCES / "arena_extracted.json").read_text())  # Figure 6
PAPER = EXTRACTED["f1"]
ARENA = [m for m in EXTRACTED["f1_models"] if m != "Mean"]
PROFILE = {(r["dataset"], r["model_id"]): r["params"] for r in json.loads((PATHS.METRICS / "profile.json").read_text())}
OUT = PATHS.METRICS / "arena"

WHAR_IDS = {
    "CNN-HAR": "cnn_har", "TinyHAR": "tinyhar", "Triple-Cross-Attn": "triple_cross_domain_attention",
    "DANA": "dana", "TinierHAR": "tinierhar", "Attend+Discriminate": "attend_discriminate",
    "DynamicWHAR": "dynamic_whar", "Random Forest": "random_forest",
    "DeepConvLSTM-Attn": "deepconv_lstm_attention", "MLP-HAR": "mlp_har",
    "DeepConvLSTM": "deepconv_lstm", "MLP-Mixer": "mlp_mixer", "Guan-LSTM": "lstms_ensemble",
    "DeepConvShallowLSTM": "deepconv_lstm_iswc", "SA-HAR": "sa_har", "SVM": "svm", "k-NN": "knn",
}
N_SPLITS = {"utd_mhad": 8, "uci_har": 10, "unimib_shar": 10, "actrectut_gestures": 2,
            "opportunity": 4, "sad": 10, "hapt": 10, "dsads": 8, "wisdm": 10, "daphnet": 10}
DEV = ["utd_mhad", "uci_har", "unimib_shar", "actrectut_gestures"]
HELD = [d for d in N_SPLITS if d not in DEV]
DATASETS = DEV + HELD
MAIN = ["recurrencenet", "s4d", "hystnet_sel_clock", "quantilenet", "mamba", "hystnet_nomem"]
LABELS = {
    "recurrencenet": "RecurrenceNet", "s4d": "S4D", "hystnet_sel_clock": "HysteresisNet sel+clock",
    "quantilenet": "QuantileNet", "mamba": "Mamba", "hystnet_nomem": "No memory (control)",
    "rainflownet": "RainflowNet", "ordinalnet": "OrdinalNet", "returnmapnet": "ReturnMapNet",
    "phaselocknet": "PhaseLockNet", "dualmem": "DualMemNet", "dualmem_path": "DualMemNet (path)",
    "pathquantilenet": "PathQuantileNet", "tropicalnet": "TropicalNet", "vortexnet": "VortexNet",
    "gru": "GRU", "hystnet": "HysteresisNet", "hystnet_clock": "HysteresisNet clock",
    "hystnet_fixed_r": "HysteresisNet fixed r", "hystnet_k6": "HysteresisNet k6", "hystnet_mm": "HysteresisNet mm",
    "hystnet_sel": "HysteresisNet sel", "hystnet_sel_k6": "HysteresisNet sel k6",
    "edge_ssm": "Edge-SSM v2", "edge_ssm_s": "Edge-SSM-S", "edge_ssm_ema": "Edge-SSM EMA",
    "edge_ssm_nophi": "Edge-SSM no phi", "edge_ssm_osc": "Edge-SSM osc",
    "edge_ssm_tau_fixed": "Edge-SSM tau fixed", "edge_ssm_tau_input": "Edge-SSM tau input",
}
SHORT = {"utd_mhad": "UTD-MHAD", "uci_har": "UCI-HAR", "unimib_shar": "UniMiB", "actrectut_gestures": "ActRecTut",
         "opportunity": "Opportunity", "sad": "SAD", "hapt": "HAPT", "dsads": "DSADS", "wisdm": "WISDM",
         "daphnet": "Daphnet"}


def split_f1(base: Path) -> dict[str, float]:
    return {p.stem: 100 * json.loads(p.read_text())["metrics"]["macro_f1"] for p in sorted(base.glob("split-*.json"))}


def ours(model: str, dataset: str, seed: int) -> float | None:
    """Mean macro-F1 % over the splits of one training seed; None unless every split is there."""
    base = RES / model / dataset
    sc = split_f1(base / f"seed-{seed}" if seed else base)
    want = {f"split-{i:02d}" for i in range(N_SPLITS[dataset])}
    return st.mean(sc[s] for s in want) if want <= set(sc) else None


def ours_seed_mean(model: str, dataset: str) -> float | None:
    v = [ours(model, dataset, s) for s in (0, 1, 2)]
    return st.mean(v) if None not in v else None


def params(model: str) -> int | None:
    """Parameter count on the UTD-MHAD configuration (6 channels, 27 classes)."""
    p = RES / model / "utd_mhad" / "split-00.json"
    if p.exists():
        return json.loads(p.read_text())["model"]["params"]
    return PROFILE.get(("utd_mhad", model))


def arena_rank(score: float, dataset: str) -> int:
    """1 + number of the 17 paper models strictly above `score` on `dataset`."""
    return 1 + sum(PAPER[dataset][m][0] > score for m in ARENA)


def ranks(scores: dict[str, float]) -> dict[str, float]:
    """Average ranks, 1 = best."""
    r = stats.rankdata([-v for v in scores.values()], method="average")
    return {k: float(v) for k, v in zip(scores, r)}


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def fidelity() -> list[dict]:
    """Our re-run of the 17 Arena models (paper-era code) against the paper, per dev dataset."""
    rows = []
    for d in DEV:
        pairs = []
        for m in ARENA:
            sc = split_f1(WHAR / WHAR_IDS[m] / d)
            if len(sc) == N_SPLITS[d]:
                pairs.append((PAPER[d][m][0], st.mean(sc.values())))
        delta = [o - p for p, o in pairs]
        rows.append({"dataset": d, "models": len(pairs), "rho": float(stats.spearmanr(*zip(*pairs)).statistic),
                     "mean_delta": st.mean(delta), "mean_abs_delta": st.mean(abs(x) for x in delta)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fid = fidelity()
    seed0 = {m: {d: ours(m, d, 0) for d in DATASETS} for m in MAIN}
    seedm = {m: {d: ours_seed_mean(m, d) for d in DATASETS} for m in MAIN}
    assert all(v is not None for m in MAIN for v in seed0[m].values()), "main models must be complete"

    # combined field: 17 paper models + our 6 at seed 0
    field = {m: {d: PAPER[d][m][0] for d in DATASETS} for m in ARENA} | seed0
    per_ds_rank = {d: ranks({m: field[m][d] for m in field}) for d in DATASETS}
    overall = []
    for m in field:
        is_ours = m in MAIN
        overall.append({
            "model": m, "label": LABELS.get(m, m), "ours": is_ours, "params": params(m if is_ours else WHAR_IDS[m]),
            "mean10": st.mean(field[m][d] for d in DATASETS),
            "mean_dev": st.mean(field[m][d] for d in DEV), "mean_held": st.mean(field[m][d] for d in HELD),
            "mean_no_art": st.mean(field[m][d] for d in DATASETS if d != "actrectut_gestures"),
            "mean_rank": st.mean(per_ds_rank[d][m] for d in DATASETS),
            "top3": sum(per_ds_rank[d][m] <= 3 for d in DATASETS),
            "seed_mean10": st.mean(seedm[m][d] for d in DATASETS) if is_ours else None,
            "arena_mean30": PAPER["Mean"][m][0] if not is_ours else None,
        })
    overall.sort(key=lambda r: r["mean_rank"])

    # per dataset: where each of our models would sit among the 17 paper models
    per_ds = []
    for d in DATASETS:
        best = max(ARENA, key=lambda m: PAPER[d][m][0])
        per_ds.append({
            "dataset": d, "group": "dev" if d in DEV else "held-out", "splits": N_SPLITS[d],
            "arena_best": best, "arena_best_f1": PAPER[d][best][0], "cnn_har": PAPER[d]["CNN-HAR"][0],
            "arena_median": st.median(PAPER[d][m][0] for m in ARENA),
            "ours": {m: {"f1": seed0[m][d], "rank": arena_rank(seed0[m][d], d), "seed_mean": seedm[m][d],
                         "rank_seed_mean": arena_rank(seedm[m][d], d)} for m in MAIN},
        })

    # paired over the 10 datasets: each of ours against the 5 best paper models (by mean over these 10),
    # TinierHAR (the only paper model at or below our size) and the best paper model per dataset
    top_arena = sorted(ARENA, key=lambda m: -st.mean(field[m][d] for d in DATASETS))[:5]
    top_arena += [] if "TinierHAR" in top_arena else ["TinierHAR"]
    paired = {}
    for o in MAIN:
        pvals = {}
        for a in top_arena + ["best"]:
            ref = [max(PAPER[d][m][0] for m in ARENA) if a == "best" else field[a][d] for d in DATASETS]
            diff = [field[o][d] - r for d, r in zip(DATASETS, ref)]
            paired[(o, a)] = {"mean": st.mean(diff), "median": st.median(diff),
                              "wins": sum(x > 0 for x in diff), "losses": sum(x < 0 for x in diff)}
            pvals[(o, a)] = float(stats.wilcoxon(diff).pvalue)
        for k, p in holm(pvals).items():  # Holm within each of our models (one row of the table)
            paired[k]["p_holm"] = p

    # every other model of ours with a complete seed-0 run on a dev dataset
    others = []
    for mdir in sorted(p.name for p in RES.iterdir()):
        if mdir in MAIN:
            continue
        cells = {d: ours(mdir, d, 0) for d in DEV}
        if any(v is not None for v in cells.values()):
            others.append({"model": mdir, "label": LABELS.get(mdir, mdir), "params": params(mdir),
                           "cells": {d: (None if v is None else {"f1": v, "rank": arena_rank(v, d)}) for d, v in cells.items()}})

    js = {"fidelity": fid, "overall": overall, "per_dataset": per_ds, "top_arena": top_arena,
          "paired": [{"ours": o, "arena": a, **v} for (o, a), v in paired.items()], "others": others,
          "channels": channel_analysis(), "subject_gap": subject_gap()}
    (OUT / "compare.json").write_text(json.dumps(js, indent=1))
    (OUT / "compare.md").write_text(render(js), encoding="utf-8")
    print((OUT / "compare.md").read_text(encoding="utf-8"))


# channels per dataset, Table 3 of the WHAR Arena paper (the code's configuration can differ slightly)
CHANNELS = {"wisdm": 3, "uci_har": 9, "actrectut_gestures": 15, "pamap2": 52, "opportunity": 133, "hhar": 6,
            "utd_mhad": 6, "hapt": 6, "mhealth": 23, "dsads": 45, "usc_had": 6, "sad": 60, "unimib_shar": 3,
            "daphnet": 9, "real_world": 81, "up_fall": 42, "motion_sense": 18, "uma_fall": 39, "real_disp": 117,
            "real_life_har": 15, "wisdm_19_phone": 6, "wisdm_19_watch": 6, "hugadb": 38, "harth": 6, "w_har": 7,
            "wear": 12, "har70": 6, "hang_time": 3, "uca_ehar": 7, "gotov": 28}
CLASSICAL = ("Random Forest", "SVM", "k-NN")


def channel_analysis() -> dict:
    """Paper table only, 30 datasets: Random Forest (tsfresh minimal features) minus the best deep model."""
    deep = [m for m in ARENA if m not in CLASSICAL]
    rows = []
    for d, c in sorted(CHANNELS.items(), key=lambda kv: kv[1]):
        rows.append({"dataset": d, "channels": c, "rf_minus_deep": PAPER[d]["Random Forest"][0] - max(PAPER[d][m][0] for m in deep),
                     "winner": max(ARENA, key=lambda m: PAPER[d][m][0])})
    res = stats.spearmanr([r["channels"] for r in rows], [r["rf_minus_deep"] for r in rows])
    return {"rows": rows, "rho": float(res.statistic), "p": float(res.pvalue)}


def subject_gap() -> dict:
    """Validation macro-F1 (held-back windows of the training subjects) minus test macro-F1 (unseen subjects), seed 0."""
    out = {}
    for m in MAIN:
        per = {}
        for d in DATASETS:
            js = [json.loads(p.read_text()) for p in sorted((RES / m / d).glob("split-*.json"))]
            js = [j for j in js if "val_macro_f1" in j["metrics"]]  # the earliest runs did not record it
            if len(js) == N_SPLITS[d]:  # never mix split sets
                per[d] = 100 * st.mean(j["metrics"]["val_macro_f1"] - j["metrics"]["macro_f1"] for j in js)
        out[m] = {"per_dataset": per, "mean": st.mean(per.values()) if len(per) == len(DATASETS) else None}
    return out


def fmt_params(p: int | None) -> str:
    if p is None:
        return "–"
    return f"{p / 1e6:.2f}M" if p >= 1e6 else f"{p / 1e3:.1f}k"


def render(js: dict) -> str:
    L = ["# Our models against the WHAR Arena paper table (Figure 6)", "",
         "Paper numbers: Figure 6 of the WHAR Arena paper (macro-F1 %, mean over subject splits, one run). "
         "Ours: same paper-era code, data, splits and protocol, training seed 0 (the protocol run); "
         "the mean over seeds 0-2 is in a separate column. No new training for this comparison.", "",
         "Datasets: the 10 of P5. Dev = used to screen rounds 6-7 (and UTD-MHAD for every earlier round); "
         "held-out = never used to choose anything. The paper's own mean (and the Leaderboard on the WHAR "
         "Arena website) is over 30 datasets; the means below are over these 10 only, which are easier on "
         "average (CNN-HAR 67.7 over 30, 73.2 over these 10, 65.0 over the other 20).", "",
         "## 0. Is a paper cell comparable to one of ours?", "",
         "Our harness re-ran all 17 Arena models on the 4 dev datasets (D:\\Projects\\HAR\\paper-era):", "",
         "| Dataset | Models | Spearman rho vs paper | Mean delta (ours - paper) | Mean abs delta |",
         "|---|---|---|---|---|"]
    for r in js["fidelity"]:
        L.append(f"| {SHORT[r['dataset']]} | {r['models']} | {r['rho']:.3f} | {r['mean_delta']:+.2f} | {r['mean_abs_delta']:.2f} |")
    L += ["", "Run-to-run noise between the paper's GPU and ours is about 1 point per cell (larger on "
          "ActRecTut, 2 splits). Classical models (RF, k-NN) match to the digit, so the splits are the same.", "",
          "## 1. All 23 models over the 10 datasets", "",
          "Ours at seed 0. Mean rank: per dataset among all 23, then averaged (1 = best). Top 3 = datasets "
          "where the model is in the first three. Params on the UTD-MHAD configuration.", "",
          "| # | Model | Params | Mean F1 (10) | Dev (4) | Held-out (6) | Without ActRecTut (9) | Mean rank /23 | Top 3 | Ours, mean seeds 0-2 | Paper mean (30 ds) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(js["overall"], 1):
        name = f"**{r['label']}**" if r["ours"] else r["label"]
        sm = f"{r['seed_mean10']:.2f}" if r["seed_mean10"] is not None else ""
        am = f"{r['arena_mean30']:.2f}" if r["arena_mean30"] is not None else ""
        L.append(f"| {i} | {name} | {fmt_params(r['params'])} | {r['mean10']:.2f} | {r['mean_dev']:.2f} | "
                 f"{r['mean_held']:.2f} | {r['mean_no_art']:.2f} | {r['mean_rank']:.2f} | {r['top3']} | {sm} | {am} |")
    L += ["", "## 2. Per dataset: our F1 and its rank among the 17 paper models", "",
          "Cell: seed-0 F1 (rank if inserted into the paper's 17; 1 = above all of them).", "",
          "| Dataset | Splits | Paper best | CNN-HAR | Paper median | " + " | ".join(LABELS[m] for m in MAIN) + " |",
          "|---|---|---|---|---|" + "---|" * len(MAIN)]
    for r in js["per_dataset"]:
        cells = " | ".join(f"{r['ours'][m]['f1']:.1f} (#{r['ours'][m]['rank']})" for m in MAIN)
        tag = "" if r["group"] == "dev" else " *"
        L.append(f"| {SHORT[r['dataset']]}{tag} | {r['splits']} | {r['arena_best_f1']:.1f} {r['arena_best']} | "
                 f"{r['cnn_har']:.1f} | {r['arena_median']:.1f} | {cells} |")
    L += ["", "\\* held-out dataset.", "",
          "Same table, our models at the mean over seeds 0-2:", "",
          "| Dataset | " + " | ".join(LABELS[m] for m in MAIN) + " |", "|---|" + "---|" * len(MAIN)]
    for r in js["per_dataset"]:
        L.append(f"| {SHORT[r['dataset']]} | " + " | ".join(
            f"{r['ours'][m]['seed_mean']:.1f} (#{r['ours'][m]['rank_seed_mean']})" for m in MAIN) + " |")
    top = js["top_arena"]
    L += ["", "## 3. Paired over the 10 datasets against the strongest paper models", "",
          "Cell: mean difference in F1 points (ours - paper model), wins-losses over the 10 datasets, "
          "Wilcoxon signed-rank p with Holm correction within each row. TinierHAR is added as the only paper "
          "model at or below our size. \"Best per dataset\" is the strongest paper model on each dataset "
          "(an oracle pick, biased upward).", "",
          "| Ours | " + " | ".join(top) + " | Best per dataset |", "|---|" + "---|" * (len(top) + 1)]
    pm = {(p["ours"], p["arena"]): p for p in js["paired"]}
    for o in MAIN:
        cells = []
        for a in top + ["best"]:
            p = pm[(o, a)]
            s = f"{p['mean']:+.2f} ({p['wins']}-{p['losses']})"
            if "p_holm" in p:
                s += f" p={p['p_holm']:.2f}"
            cells.append(s)
        L.append(f"| {LABELS[o]} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Our other models on the dev datasets (seed 0)", "",
          "Models without the full 10-dataset run: earlier rounds (UTD-MHAD only, some also ActRecTut/UniMiB) "
          "and rounds 6-7 that stopped at stage 2. Cell: F1 (rank among the paper's 17).", "",
          "| Model | Params | " + " | ".join(SHORT[d] for d in DEV) + " |", "|---|---|" + "---|" * len(DEV)]
    for r in sorted(js["others"], key=lambda r: -(r["cells"]["utd_mhad"] or {"f1": 0})["f1"]):
        cells = " | ".join("" if r["cells"][d] is None else f"{r['cells'][d]['f1']:.1f} (#{r['cells'][d]['rank']})" for d in DEV)
        L.append(f"| {r['label']} | {fmt_params(r['params'])} | {cells} |")
    best = {d: max(PAPER[d][m][0] for m in ARENA) for d in DEV}
    L += ["", "Paper best on the dev datasets: " + ", ".join(f"{SHORT[d]} {best[d]:.1f}" for d in DEV) + "."]

    ch = js["channels"]
    rows = ch["rows"]
    many = [r for r in rows if r["channels"] >= 20]
    few = [r for r in rows if r["channels"] < 20]
    L += ["", "## 5. Where the classical Random Forest wins (paper table, all 30 datasets)", "",
          "Random Forest on tsfresh minimal features minus the best of the 14 deep paper models, against the "
          "number of channels (paper Table 3). "
          f"Spearman rho = {ch['rho']:.2f} (p = {ch['p']:.1e}). Random Forest is the best model on "
          f"{sum(r['rf_minus_deep'] > 0 for r in many)} of {len(many)} datasets with at least 20 channels and on "
          f"{sum(r['rf_minus_deep'] > 0 for r in few)} of {len(few)} with fewer.", "",
          "| Dataset | Channels | RF - best deep | Best model |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['channels']} | {r['rf_minus_deep']:+.1f} | {r['winner']} |")

    gap = js["subject_gap"]
    L += ["", "## 6. Seen against unseen subjects (ours, seed 0)", "",
          "Validation macro-F1 (20 % of the training subjects' windows, which overlap training windows by 50 %, so "
          "it overstates pure subject shift) minus test macro-F1 (unseen subjects).", "",
          "| Model | Mean | " + " | ".join(SHORT[d] for d in DATASETS) + " |", "|---|---|" + "---|" * len(DATASETS)]
    for m in MAIN:
        g = gap[m]
        mean = "–" if g["mean"] is None else f"{g['mean']:.1f}"
        L.append(f"| {LABELS[m]} | {mean} | " + " | ".join(
            f"{g['per_dataset'][d]:.1f}" if d in g["per_dataset"] else "–" for d in DATASETS) + " |")
    L += ["", "– = not recorded (the earliest seed-0 runs on UTD-MHAD, ActRecTut and part of UniMiB predate the "
          "validation score in the result file)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
