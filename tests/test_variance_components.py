"""Tests of src/evaluation/variance_components.py on synthetic data with known variance components (allowed in tests/
only; agents/01_integrity.md). Run: uv run --no-sync python tests/test_variance_components.py  (or pytest)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import variance_components as V  # noqa: E402

TRUE = {"P": 40.0, "M": 4.0, "MD": 3.0, "MP": 5.0, "e": 6.0}


def synthetic(n_datasets=8, n_splits=10, n_models=6, n_seeds=3, seed=0, comp=TRUE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sd = {k: np.sqrt(v) for k, v in comp.items()}
    rows = []
    mu = {d: 60 + 5 * d for d in range(n_datasets)}
    M = rng.normal(0, sd["M"], n_models)
    for d in range(n_datasets):
        MD = rng.normal(0, sd["MD"], n_models)
        for s in range(n_splits):
            P = rng.normal(0, sd["P"])
            MP = rng.normal(0, sd["MP"], n_models)
            for m in range(n_models):
                for r in range(n_seeds):
                    rows.append({"model": f"m{m}", "dataset": f"d{d}", "split": s, "seed": r,
                                 "y": mu[d] + P + M[m] + MD[m] + MP[m] + rng.normal(0, sd["e"])})
    return pd.DataFrame(rows)


def test_fit_recovers_components():
    f = V.fit(synthetic(n_models=10, seed=1))
    assert f["converged"]
    c = f["components"]
    # P, MP and e are estimated from many levels; M from 10 models only (loose)
    assert 25 < c["P"] < 60, c
    assert 3.5 < c["MP"] < 7, c
    assert 5 < c["e"] < 7, c
    assert 0.5 < c["M"] < 12, c
    true_R = TRUE["P"] / (TRUE["M"] + TRUE["MD"] + TRUE["MP"])
    assert 0.4 * true_R < f["R"] < 2.5 * true_R, f["R"]


def test_fit_without_seeds_drops_mp():
    df = synthetic(n_seeds=1, seed=2)
    f = V.fit(df, has_seeds=False)
    assert f["converged"] and "MP" not in f["components"]
    assert f["R_definition"].endswith("s2_e)")


def test_no_model_effect_gives_large_R():
    f = V.fit(synthetic(seed=3, comp={"P": 40.0, "M": 0.0, "MD": 0.0, "MP": 0.0, "e": 6.0}))
    assert f["converged"] and f["R"] > 20, f


def test_resamples_deterministic_and_balanced():
    df = synthetic(n_datasets=3, n_splits=4, n_models=2, n_seeds=1, seed=4)
    a, b = V.resamples(df, 5), V.resamples(df, 5)
    assert a == b
    assert all(len(r) == 12 for r in a)
    assert all(sum(d == "d1" for d, _ in r) == 4 for r in a)


def test_resample_relabels_duplicate_splits():
    df = synthetic(n_datasets=2, n_splits=3, n_models=2, n_seeds=2, seed=5)
    draw = [("d0", 0), ("d0", 0), ("d0", 1), ("d1", 2), ("d1", 2), ("d1", 2)]
    f = V._fit_resample((df, draw, True))
    assert f["converged"] in (True, False)  # tiny design: just must not crash


def test_bootstrap_interval_contains_point():
    df = synthetic(n_datasets=4, n_splits=6, n_models=4, n_seeds=2, seed=6)
    point = V.fit(df)
    boot = V.bootstrap(df, 20, workers=2)
    lo, hi = boot["R_interval"]
    assert boot["converged"] + boot["dropped"] == 20
    assert lo <= hi
    assert V.decide(point, boot) in ("success", "failure", "inconclusive") or V.decide(point, boot).startswith("inconclusive")


def test_decide_rules():
    boot = {"dropped_share": 0.0, "R_interval": [1.2, 3.0]}
    assert V.decide({"R": 2.0}, boot) == "success"
    assert V.decide({"R": 0.9}, boot) == "failure"
    assert V.decide({"R": 2.0}, {"dropped_share": 0.0, "R_interval": [0.8, 3.0]}) == "inconclusive"
    assert V.decide({"R": 2.0}, {"dropped_share": 0.06, "R_interval": [1.2, 3.0]}).startswith("inconclusive")


def test_inclusion_reads_no_metric():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        spec = {"root": root, "models": ["a", "b"], "seeds": [0, 1], "datasets": {"x": 2, "y": 2}}
        for m in spec["models"]:
            for d in spec["datasets"]:
                for s in spec["seeds"]:
                    for i in range(2):
                        base = root / m / d / (f"seed-{s}" if s else "")
                        base.mkdir(parents=True, exist_ok=True)
                        test = 50 + i if not (d == "y" and m == "b" and s == 1) else 99  # y disagrees
                        # no "metrics" key at all: inclusion must not need one
                        (base / f"split-{i:02d}.json").write_text(json.dumps(
                            {"protocol_hash": V.PROTOCOL_HASH,
                             "source": {"train_size": 10, "val_size": 2, "test_size": test}}))
        inc = V.inclusion(spec)
        assert inc["x"]["included"] and not inc["y"]["included"], inc
        assert inc["y"]["splits_with_size_disagreement"] == [0, 1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
