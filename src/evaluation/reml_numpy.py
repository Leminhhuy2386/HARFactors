"""Second implementation of the variance-components fit, for verification (agents/05_verify.md).

Independent of statsmodels: the restricted log-likelihood of y ~ N(X b, V), V = sum_k s2_k K_k + s2_e I, with K_k the
0/1 matrix "rows i and j share a level of component k", maximised over the log-variances with scipy (L-BFGS-B):

    l_R = -1/2 [ log|V| + log|X' V^-1 X| + y' P y ],   P = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1

X is the dataset indicator matrix (dataset fixed). Components and the residual are those of
src/evaluation/variance_components.py; nothing is shared with it but the column names of the data frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg, optimize

FLOOR = -18.0  # log-variance floor (about 1.5e-8): lets a component go to its boundary


def _same_level(labels: pd.Series) -> np.ndarray:
    codes = pd.factorize(labels)[0]
    return (codes[:, None] == codes[None, :]).astype(float)


def kernels(df: pd.DataFrame, has_seeds: bool = True) -> dict[str, np.ndarray]:
    ds_split = df["dataset"] + ":" + df["split"].astype(str)
    k = {"P": _same_level(ds_split), "M": _same_level(df["model"]),
         "MD": _same_level(df["model"] + ":" + df["dataset"])}
    if has_seeds:
        k["MP"] = _same_level(df["model"] + ":" + ds_split)
    return k


def neg_reml(logv: np.ndarray, y: np.ndarray, X: np.ndarray, K: list[np.ndarray]) -> float:
    v = np.exp(logv)
    V = v[-1] * np.eye(len(y))
    for vk, Kk in zip(v[:-1], K):
        V += vk * Kk
    try:
        c = linalg.cho_factor(V, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return 1e300
    logdetV = 2 * np.log(np.diag(c[0])).sum()
    Vi_X = linalg.cho_solve(c, X, check_finite=False)
    Vi_y = linalg.cho_solve(c, y, check_finite=False)
    XtViX = X.T @ Vi_X
    c2 = linalg.cho_factor(XtViX, lower=True, check_finite=False)
    logdetXVX = 2 * np.log(np.diag(c2[0])).sum()
    beta = linalg.cho_solve(c2, X.T @ Vi_y, check_finite=False)
    r = y - X @ beta
    quad = float(r @ linalg.cho_solve(c, r, check_finite=False))
    return 0.5 * (logdetV + logdetXVX + quad)


def fit(df: pd.DataFrame, has_seeds: bool = True) -> dict:
    y = df["y"].to_numpy(float)
    X = pd.get_dummies(df["dataset"]).to_numpy(float)
    ker = kernels(df, has_seeds)
    names = list(ker)
    K = [ker[n] for n in names]
    start = np.log(np.full(len(names) + 1, max(np.var(y) / (len(names) + 1), 1e-3)))
    best = None
    for x0 in (start, start + 1.0, start - 1.0):  # a few starts; keep the best optimum
        r = optimize.minimize(neg_reml, x0, args=(y, X, K), method="L-BFGS-B",
                              bounds=[(FLOOR, 12.0)] * len(x0), options={"maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    v = np.exp(best.x)
    comp = {n: float(x) for n, x in zip(names, v[:-1])}
    comp = {k: (0.0 if np.log(x) <= FLOOR + 1e-6 else x) for k, x in comp.items()}
    comp["e"] = float(v[-1])
    model_related = comp["M"] + comp["MD"] + (comp["MP"] if has_seeds else comp["e"])
    within = sum(comp.values())
    return {"converged": bool(best.success), "neg_reml": float(best.fun), "components": comp,
            "R": float("inf") if model_related == 0 else comp["P"] / model_related, "pi_P": comp["P"] / within}


def permute_models(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Negative control: shuffle the model labels within every (dataset, split, seed) cell."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _, idx in out.groupby(["dataset", "split", "seed"]).groups.items():
        idx = list(idx)
        out.loc[idx, "model"] = rng.permutation(out.loc[idx, "model"].to_numpy())
    return out
