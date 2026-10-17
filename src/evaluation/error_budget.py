"""Exploratory: what the variance components imply for comparing two networks (an error budget).

Not pre-registered (added after the review of 2026-10-17). Under the model of src/evaluation/variance_components.py,

    y = mu_d + P_ds + M_m + MD_md + MP_mds + e,

the difference between networks a and b on one dataset, averaged over k test-subject groups and n training seeds, is
the true difference (M_a - M_b) + (MD_ad - MD_bd), with variance 2 (s2_M + s2_MD) over pairs of networks, plus noise:

- paired (both networks on the same splits):      2 (s2_MP / k + s2_e / (k n));
- unpaired (each network on its own k splits with other test subjects, e.g. two papers that each report one
  hold-out split):                                2 ((s2_P + s2_MP) / k + s2_e / (k n)).

From the point estimates of an analysis this script derives the noise SD of a comparison for a grid of designs, the
smallest true difference a two-sided test at level ALPHA detects with power POWER ((z_{1-ALPHA/2} + z_POWER) x noise
SD), the reliability of a comparison (true variance over true plus noise variance), and the standard error of one
network's dataset mean as an estimate for new people ((s2_P + s2_MP) / k + s2_e / (k n)). Point estimates only:
the components' own uncertainty is not propagated. k above 10 means more test-subject groups of the same size, i.e.
more recorded subjects (an extrapolation beyond the protocol's k = 10).

    uv run --no-sync python -m src.evaluation.error_budget   ->   results/metrics/error_budget/data.json
"""
from __future__ import annotations

import json
import math

from scipy import stats

from src.utils import paths as PATHS

CONF = PATHS.METRICS / "confirmatory"
OUT = PATHS.METRICS / "error_budget"
ANALYSES = {"S2": "H1", "S1": "sens2_S1_macro_f1"}  # the confirmatory fit and its development counterpart
ALPHA, POWER = 0.05, 0.8
K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
N_GRID = [1, 3, 5, None]  # None: infinitely many seeds
DESIGNS = {  # name: (k, n, paired)
    "unpaired_1x1": (1, 1, False), "paired_1x1": (1, 1, True), "paired_10x1": (10, 1, True), "paired_10x3": (10, 3, True), "paired_10xinf": (10, None, True),
    "paired_20x3": (20, 3, True),
}


def noise_var(c: dict, k: int, n: int | None, paired: bool) -> float:
    seed = 0.0 if n is None else c["e"] / (k * n)
    return 2 * ((0.0 if paired else c["P"]) / k + c["MP"] / k + seed)


def budget(c: dict) -> dict:
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    true_var = 2 * (c["M"] + c["MD"])
    designs = {}
    for name, (k, n, paired) in DESIGNS.items():
        nv = noise_var(c, k, n, paired)
        designs[name] = {"k": k, "n": n, "paired": paired, "noise_sd": math.sqrt(nv),
                         "detectable_difference": z * math.sqrt(nv), "reliability": true_var / (true_var + nv)}
    grid = {("inf" if n is None else str(n)): [math.sqrt(noise_var(c, k, n, True)) for k in K_GRID] for n in N_GRID}
    se_mean = math.sqrt((c["P"] + c["MP"]) / 10 + c["e"] / 30)
    return {"components": c, "true_difference_sd": math.sqrt(true_var), "z_sum": z, "designs": designs,
            "paired_noise_sd_grid": {"k": K_GRID, "by_seeds": grid},
            "se_dataset_mean_10x3": se_mean,
            "seed_gain_3_to_inf": 1 - designs["paired_10xinf"]["noise_sd"] / designs["paired_10x3"]["noise_sd"],
            "group_gain_10_to_20": 1 - designs["paired_20x3"]["noise_sd"] / designs["paired_10x3"]["noise_sd"]}


def main() -> None:
    out = {"alpha": ALPHA, "power": POWER, "power_pct": 100 * POWER, "alpha_pct": 100 * ALPHA, "analyses": {}}
    for set_name, key in ANALYSES.items():
        r = json.loads((CONF / f"{key}.json").read_text())
        out["analyses"][set_name] = {"source": f"results/metrics/confirmatory/{key}.json", "n_runs": r["n_runs"],
                                     **budget(r["point"]["components"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(out, indent=1) + "\n")
    for s, b in out["analyses"].items():
        print(s, f"true sd {b['true_difference_sd']:.2f}",
              {k: (round(v["noise_sd"], 2), round(v["detectable_difference"], 2), round(v["reliability"], 3))
               for k, v in b["designs"].items()}, f"se mean {b['se_dataset_mean_10x3']:.2f}")


if __name__ == "__main__":
    main()
