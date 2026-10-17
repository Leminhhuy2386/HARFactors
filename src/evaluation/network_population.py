"""Exploratory: how the subject-to-network ratio of S3 depends on which published networks are compared.

The pre-registered sensitivity analysis 4 fits S3's 17 WHAR Arena models together (R' = 0.21). This analysis, which
was not pre-registered (added after the review of 2026-10-17), refits the same model
(src/evaluation/variance_components.py, unchanged) on sub-populations of those networks:

- "deep": the 14 deep networks (Random Forest, SVM and k-NN removed), a split fixed by model type;
- "top k", k = 3 .. 17: the k networks with the highest mean published macro-F1 on the 26 Arena datasets that are
  not in S3 (docs/sources/arena_extracted.json). The ranking reads no score of the S3 datasets, so the selection is
  independent of the runs analysed.

Each point estimate is fitted by statsmodels and by the independent REML (src/evaluation/reml_numpy.py); the solution
with the higher restricted likelihood counts and both are recorded (the rule of gate G5 since rerun 6). Intervals are
the pre-registered cluster bootstrap (1,000 resamples of splits within datasets, 97.5 %, seed 20261015), fitted by
statsmodels.

    uv run --no-sync python -m src.evaluation.network_population [--workers 16]
        -> results/metrics/network_population/data.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time

import numpy as np
import pandas as pd

from src.data.sets import S3, run_files
from src.evaluation import reml_numpy as RN
from src.evaluation import variance_components as V
from src.utils import paths as PATHS
from src.utils import provenance as PV

OUT = PATHS.METRICS / "network_population"
ARENA_FILE = PATHS.ROOT / "docs" / "sources" / "arena_extracted.json"
WHAR_IDS = {
    "CNN-HAR": "cnn_har", "TinyHAR": "tinyhar", "Triple-Cross-Attn": "triple_cross_domain_attention",
    "DANA": "dana", "TinierHAR": "tinierhar", "Attend+Discriminate": "attend_discriminate",
    "DynamicWHAR": "dynamic_whar", "Random Forest": "random_forest",
    "DeepConvLSTM-Attn": "deepconv_lstm_attention", "MLP-HAR": "mlp_har",
    "DeepConvLSTM": "deepconv_lstm", "MLP-Mixer": "mlp_mixer", "Guan-LSTM": "lstms_ensemble",
    "DeepConvShallowLSTM": "deepconv_lstm_iswc", "SA-HAR": "sa_har", "SVM": "svm", "k-NN": "knn",
}
CLASSICAL = {"random_forest", "svm", "knn"}
N_BOOT = 1000
K_MIN = 3


def ranking() -> tuple[list[str], dict[str, float], list[str]]:
    """Networks ordered by mean published macro-F1 over the Arena datasets outside S3."""
    arena = json.loads(ARENA_FILE.read_text())["f1"]
    others = sorted(d for d in arena if d != "Mean" and d not in S3["datasets"])
    score = {WHAR_IDS[m]: st.mean(arena[d][m][0] for d in others) for m in WHAR_IDS}
    order = sorted(score, key=lambda m: (-score[m], m))
    return order, score, others


def neg_reml_at(df: pd.DataFrame, comp: dict) -> float:
    ker = RN.kernels(df, has_seeds=False)
    names = list(ker)
    logv = np.log(np.maximum([comp[n] for n in names] + [comp["e"]], np.exp(RN.FLOOR)))
    X = pd.get_dummies(df["dataset"]).to_numpy(float)
    return float(RN.neg_reml(logv, df["y"].to_numpy(float), X, [ker[n] for n in names]))


def best_fit(df: pd.DataFrame) -> dict:
    sm, rn = V.fit(df, has_seeds=False), RN.fit(df, has_seeds=False)
    sm_neg = neg_reml_at(df, sm["components"]) if sm["converged"] else float("inf")
    rn_neg = neg_reml_at(df, rn["components"])
    chosen = "statsmodels" if sm_neg <= rn_neg else "numpy_reml"
    comp = (sm if chosen == "statsmodels" else rn)["components"]
    return {"chosen": chosen, "components": comp, **V.estimands(comp, has_seeds=False),
            "statsmodels": {"converged": sm["converged"], "neg_reml": sm_neg, "components": sm.get("components")},
            "numpy_reml": {"converged": rn["converged"], "neg_reml": rn_neg, "components": rn["components"]}}


def describe(df: pd.DataFrame) -> dict:
    per = {}
    for d, g in df.groupby("dataset", sort=True):
        per[d] = {"sd_split_means": float(g.groupby("split")["y"].mean().std()),
                  "sd_network_means": float(g.groupby("model")["y"].mean().std())}
    return per


def analyse(df_all: pd.DataFrame, models: list[str], workers: int) -> dict:
    df = df_all[df_all["model"].isin(models)].reset_index(drop=True)
    point = best_fit(df)
    boot = V.bootstrap(df, N_BOOT, has_seeds=False, workers=workers)
    lo, hi = boot["R_interval"]
    return {"models": models, "n_models": len(models), "n_runs": int(len(df)), "point": point,
            "bootstrap": {k: boot[k] for k in ("resamples", "converged", "dropped", "level", "R_interval",
                                               "pi_P_interval", "components_intervals", "seed")},
            "interval_above_1": bool(lo > 1), "interval_below_1": bool(hi < 1), "per_dataset": describe(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run_id = PV.new_run_id("network_population")
    order, score, others = ranking()
    config = {"set": "S3", "outcome": "macro_f1", "bootstrap": N_BOOT, "boot_seed": V.BOOT_SEED, "level": V.LEVEL,
              "ranking": "mean published macro-F1 on the Arena datasets outside S3", "k_min": K_MIN,
              "populations": ["deep (no Random Forest, SVM, k-NN)"] + [f"top {k}" for k in range(K_MIN, 18)]}
    (PATHS.EXPERIMENTS / run_id).mkdir()
    (PATHS.EXPERIMENTS / run_id / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    started = time.time()
    PV.ledger_start(run_id, config)
    try:
        df_all = V.load(S3, "macro_f1", list(S3["datasets"]))
        pops = {"deep": [m for m in order if m not in CLASSICAL]}
        pops |= {f"top{k}": order[:k] for k in range(K_MIN, len(order) + 1)}
        res = {}
        for name, models in pops.items():
            res[name] = analyse(df_all, models, a.workers)
            p = res[name]["point"]
            print(f"{name:7s} n={res[name]['n_runs']:3d} R'={p['R']:.2f} {res[name]['bootstrap']['R_interval']} "
                  f"chosen={p['chosen']} {({k: round(v, 1) for k, v in p['components'].items()})}", flush=True)
        tops = [res[f"top{k}"] for k in range(K_MIN, len(order) + 1)]
        summary = {
            "ranking": {"order": order, "mean_published_macro_f1_outside_S3": score, "arena_datasets_used": others},
            "populations": res,
            "largest_k_with_R_above_1": max((t["n_models"] for t in tops if t["point"]["R"] > 1), default=None),
            "top_k_with_interval_above_1": int(sum(t["interval_above_1"] for t in tops)),
            "top_k_with_interval_below_1": int(sum(t["interval_below_1"] for t in tops)),
            "top_k_populations": len(tops),
            "fits_by_numpy_reml": int(sum(r["point"]["chosen"] == "numpy_reml" for r in res.values())),
            "deep_share_MD_of_network_related": float(
                res["deep"]["point"]["components"]["MD"] / sum(res["deep"]["point"]["components"][c]
                                                                 for c in ("M", "MD", "e"))),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / "data.json"
        f.write_text(json.dumps(summary, indent=1) + "\n")
        PV.write(f, run_id=run_id, started=started, config=config, produced_by="src/evaluation/network_population.py",
                 inputs=run_files(S3) + [ARENA_FILE, PATHS.ROOT / "src/evaluation/network_population.py",
                                         PATHS.ROOT / "src/evaluation/variance_components.py",
                                         PATHS.ROOT / "src/evaluation/reml_numpy.py"])
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    PV.ledger_end(run_id, f"S3 by network population: {len(res)} fits with {N_BOOT} resamples each", started)


if __name__ == "__main__":
    main()
