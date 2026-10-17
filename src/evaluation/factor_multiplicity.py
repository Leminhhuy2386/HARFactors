"""Exploratory: multiplicity and an order-statistic check for the regressions over the WHAR Arena's 30 datasets.

Not pre-registered (added after the review of 2026-10-17). src/evaluation/factor_analysis.py fits a frozen OLS
specification: 4 standardised dataset properties for each of 4 outcomes, i.e. 16 coefficient tests, reported without
correction. This script reads its output (results/metrics/factors/data.json) and

- applies Holm's step-down correction over the 16 coefficient p-values (intercepts excluded);
- checks the outcome "best - median deep network" for an order-statistic artefact: the maximum of 14 noisy means
  rises with the noise, so the gap is compared (Spearman, over the 30 datasets) with the outcome "mean split SD of
  the deep networks", the spread over test-subject groups.

    uv run --no-sync python -m src.evaluation.factor_multiplicity   ->   results/metrics/factor_multiplicity/data.json
"""
from __future__ import annotations

import json

from scipy import stats

from src.utils import paths as PATHS

SRC = PATHS.METRICS / "factors" / "data.json"
OUT = PATHS.METRICS / "factor_multiplicity"
PRED = ["log2 subjects", "log2 channels", "log2 classes", "log2 samples/window"]
GAP = "best − median deep (does the model matter)"
SPREAD = "mean split sd of deep models (person spread)"
ALPHA = 0.05


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    order = sorted(p, key=p.get)
    m, run, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def main() -> None:
    data = json.loads(SRC.read_text())["data"]
    raw = {f"{out} | {pr}": float(r["p"][j + 1]) for out, r in data["ols"].items() for j, pr in enumerate(PRED)}
    adj = holm(raw)
    tests = {k: {"p": raw[k], "p_holm": adj[k], "significant_holm": adj[k] < ALPHA} for k in sorted(raw)}
    gap, spread = data["outcome"][GAP], data["outcome"][SPREAD]
    ds = sorted(gap)
    r = stats.spearmanr([gap[d] for d in ds], [spread[d] for d in ds])
    out = {"tests": tests, "n_tests": len(tests), "alpha": ALPHA,
           "n_significant_holm": int(sum(t["significant_holm"] for t in tests.values())),
           "n_significant_uncorrected": int(sum(t["p"] < ALPHA for t in tests.values())),
           "gap_subjects_p_holm": adj[f"{GAP} | log2 subjects"],
           "rf_channels_p_holm": adj[f"RF − best deep (classical margin) | log2 channels"],
           "spread_channels_p_holm": adj[f"{SPREAD} | log2 channels"],
           "gap_vs_split_spread": {"datasets": len(ds), "rho": float(r.statistic), "p": float(r.pvalue)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "tests"}, indent=1, ensure_ascii=False))
    for k, t in sorted(tests.items(), key=lambda kv: kv[1]["p"])[:5]:
        print(f"{t['p']:.5f} -> {t['p_holm']:.4f}  {k}")


if __name__ == "__main__":
    main()
