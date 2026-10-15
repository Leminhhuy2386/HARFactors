"""The pre-registered variance-components analysis (docs/preregistration.md, § H1 and § Rules fixed now).

Model, fitted by REML with statsmodels MixedLM (one group, crossed variance components):

    y = dataset (fixed) + P[dataset:split] + M[model] + MD[model:dataset] + MP[model:dataset:split] + e

with e the variation between training seeds. Estimand R = s2_P / (s2_M + s2_MD + s2_MP); secondary
pi_P = s2_P / (s2_P + s2_M + s2_MD + s2_MP + s2_e). Interval: percentile 97.5 % interval from a cluster bootstrap that
resamples, within every dataset, as many splits as the dataset has (with replacement), each draw keeping all its models
and seeds under its own label. One-run-per-split sets (S3) have no seed replicate, so MP is dropped and e absorbs it.

Reading run files: `inclusion` reads split sizes and the protocol hash only; `load` reads the outcome metric.
"""
from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.data.sets import cells, run_path

PROTOCOL_HASH = "e8739ef2c5b36c99"
BOOT_SEED = 20261015
LEVEL = 0.975
MAX_DROPPED = 0.05


# --------------------------------------------------------------------------------------------- data
def inclusion(spec: dict) -> dict[str, dict]:
    """Pre-registered inclusion rule, per dataset: for every split, train/val/test sizes equal across all models and
    seeds, and every run has the paper protocol hash. Reads no metric."""
    sizes = defaultdict(lambda: defaultdict(set))
    bad_hash = defaultdict(int)
    missing = defaultdict(int)
    for m, d, s, i in cells(spec):
        f = run_path(spec, m, d, s, i)
        if not f.exists():
            missing[d] += 1
            continue
        r = json.loads(f.read_text())
        src = r["source"]
        sizes[d][i].add((src["train_size"], src["val_size"], src["test_size"]))
        bad_hash[d] += r.get("protocol_hash") != PROTOCOL_HASH
    out = {}
    for d in spec["datasets"]:
        disagree = sorted(i for i, v in sizes[d].items() if len(v) > 1)
        out[d] = {"included": not disagree and not bad_hash[d], "splits_with_size_disagreement": disagree,
                  "runs_with_other_protocol_hash": bad_hash[d], "missing_runs": missing[d]}
    return out


def load(spec: dict, outcome: str, datasets: list[str]) -> pd.DataFrame:
    """One row per stored run of `spec` on `datasets`: model, dataset, split, seed, y (outcome x 100)."""
    rows = []
    for m, d, s, i in cells(spec):
        if d not in datasets:
            continue
        f = run_path(spec, m, d, s, i)
        if f.exists():
            rows.append({"model": m, "dataset": d, "split": i, "seed": s,
                         "y": 100 * json.loads(f.read_text())["metrics"][outcome]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- model
def components(has_seeds: bool) -> list[str]:
    return ["P", "M", "MD", "MP"] if has_seeds else ["P", "M", "MD"]


def fit(df: pd.DataFrame, has_seeds: bool = True) -> dict:
    """REML variance components. Refits once with Powell's method if the default optimizers do not converge."""
    import statsmodels.formula.api as smf

    d = df.copy()
    d["g"] = 1
    d["ds_split"] = d["dataset"] + ":" + d["split"].astype(str)
    d["model_ds"] = d["model"] + ":" + d["dataset"]
    d["model_ds_split"] = d["model"] + ":" + d["ds_split"]
    vc = {"P": "0 + C(ds_split)", "M": "0 + C(model)", "MD": "0 + C(model_ds)"}
    if has_seeds:
        vc["MP"] = "0 + C(model_ds_split)"
    model = smf.mixedlm("y ~ 0 + C(dataset)", d, groups="g", vc_formula=vc, re_formula="0")
    result, method = None, "default"
    for method in ("default", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = model.fit(reml=True) if method == "default" else model.fit(reml=True, method="powell")
            except (np.linalg.LinAlgError, ValueError):
                result = None
        if result is not None and result.converged:
            break
    if result is None or not result.converged:
        return {"converged": False, "method": method}
    comp = dict(zip(model.exog_vc.names, (float(v) for v in result.vcomp)))
    comp["e"] = float(result.scale)
    return {"converged": True, "method": method, "components": comp, **estimands(comp, has_seeds)}


def estimands(comp: dict, has_seeds: bool) -> dict:
    model_related = comp["M"] + comp["MD"] + (comp["MP"] if has_seeds else comp["e"])
    within = sum(comp[k] for k in components(has_seeds)) + comp["e"]
    R = math.inf if model_related == 0 else comp["P"] / model_related
    return {"R": R, "pi_P": comp["P"] / within if within else float("nan"),
            "R_definition": "s2_P / (s2_M + s2_MD + s2_MP)" if has_seeds else "s2_P / (s2_M + s2_MD + s2_e)"}


# --------------------------------------------------------------------------------------------- bootstrap
def resamples(df: pd.DataFrame, n: int, seed: int = BOOT_SEED) -> list[list[tuple[str, int]]]:
    """Draw, per dataset, as many splits as it has, with replacement; one list of (dataset, split) per resample."""
    rng = np.random.default_rng(seed)
    per = {d: sorted(g["split"].unique()) for d, g in df.groupby("dataset", sort=True)}
    return [[(d, int(s)) for d in sorted(per) for s in rng.choice(per[d], size=len(per[d]), replace=True)]
            for _ in range(n)]


def _fit_resample(args) -> dict:
    df, draw, has_seeds = args
    parts = []
    by = {k: g for k, g in df.groupby(["dataset", "split"])}
    for j, (d, s) in enumerate(draw):
        g = by[(d, s)].copy()
        g["split"] = f"{s}#{j}"  # each draw is its own split level
        parts.append(g)
    return fit(pd.concat(parts, ignore_index=True), has_seeds)


def _one_blas_thread() -> None:  # worker processes: one BLAS thread each, or they oversubscribe the node's cores
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def bootstrap(df: pd.DataFrame, n: int, has_seeds: bool = True, workers: int = 1) -> dict:
    draws = resamples(df, n)
    args = [(df, dr, has_seeds) for dr in draws]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_one_blas_thread) as ex:
            fits = list(ex.map(_fit_resample, args, chunksize=max(1, n // (4 * workers))))
    else:
        fits = [_fit_resample(a) for a in args]
    ok = [f for f in fits if f["converged"]]
    q = ((1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2)
    interval = lambda key: [float(np.quantile([f[key] for f in ok], p)) for p in q] if ok else [None, None]  # noqa: E731
    return {"resamples": n, "converged": len(ok), "dropped": n - len(ok), "dropped_share": (n - len(ok)) / n,
            "level": LEVEL, "R_interval": interval("R"), "pi_P_interval": interval("pi_P"),
            "components_intervals": {k: [float(np.quantile([f["components"][k] for f in ok], p)) for p in q]
                                     for k in (components(has_seeds) + ["e"])} if ok else {},
            "refit_with_powell": sum(f["method"] == "powell" for f in ok), "seed": BOOT_SEED}


def decide(point: dict, boot: dict) -> str:
    """docs/preregistration.md: success / failure / inconclusive for H1."""
    if boot["dropped_share"] > MAX_DROPPED:
        return "inconclusive (more than 5 % of resamples dropped)"
    if point["R"] <= 1:
        return "failure"
    lo = boot["R_interval"][0]
    return "success" if lo is not None and lo > 1 else "inconclusive"
