"""Exploratory, descriptive: within each dataset, how far apart are the test-subject groups, and how far apart are the
networks? Reads the stored runs of S1 and S2 (datasets that pass the pre-registered inclusion rule).

Per dataset: the mean macro-F1 of every split (over networks and seeds) and of every network (over splits and seeds),
their standard deviations, the (network, split) cell means over seeds, and the mean seed sd within cells. Also the
validation macro-F1 that runs stored (not every run did) (validation windows come from the training subjects): its gap to the test macro-F1
(unseen subjects), and whether validation ranks the splits and the networks as the test does. S2 is read after the confirmatory
analysis (G4 seal, docs/preregistration.md); this read is logged.

    uv run python -m src.evaluation.split_spread   ->   results/metrics/split_spread/data.json
"""
from __future__ import annotations

import json
import statistics as st
import time

import pandas as pd
from scipy import stats

from src.data.sets import S1, S2, cells, run_path
from src.evaluation import variance_components as V
from src.utils import paths as PATHS

OUT = PATHS.METRICS / "split_spread"
NAME = {"utd_mhad": "UTD-MHAD", "uci_har": "UCI-HAR", "unimib_shar": "UniMiB-SHAR", "actrectut_gestures": "ActRecTut",
        "opportunity": "Opportunity", "sad": "SAD", "hapt": "HAPT", "dsads": "DSADS", "wisdm": "WISDM",
        "daphnet": "Daphnet", "pamap2": "PAMAP2", "up_fall": "UP-Fall", "uma_fall": "UMAFall", "usc_had": "USC-HAD",
        "motion_sense": "MotionSense", "wear": "WEAR", "uca_ehar": "UCA-EHAR", "w_har": "w-HAR"}


def load_val(spec: dict, dataset: str) -> pd.DataFrame:
    """The validation macro-F1 of every stored run of `dataset` that recorded one (x 100)."""
    rows = []
    for m, d, s, i in cells(spec):
        f = run_path(spec, m, d, s, i)
        if d == dataset and f.exists():
            met = json.loads(f.read_text())["metrics"]
            if "val_macro_f1" in met:
                rows.append({"model": m, "dataset": d, "split": i, "seed": s, "v": 100 * met["val_macro_f1"]})
    return pd.DataFrame(rows, columns=["model", "dataset", "split", "seed", "v"])


def main() -> None:
    with open(PATHS.RESULTS / "test_access.log", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | src/evaluation/split_spread.py | - | exploratory description "
                 f"of the stored S1 and S2 runs, after the confirmatory analysis\n")
    out = {}
    for set_name, spec in (("S1", S1), ("S2", S2)):
        inc = V.inclusion(spec)
        for d in spec["datasets"]:
            if not inc[d]["included"]:
                continue
            df = V.load(spec, "macro_f1", [d])
            split_means = df.groupby("split")["y"].mean()
            model_means = df.groupby("model")["y"].mean()
            cell = df.groupby(["model", "split"])["y"]
            seed_sd = cell.std().mean()
            cell_means = {m: {int(i): float(v) for (mm, i), v in cell.mean().items() if mm == m} for m in model_means.index}
            val = load_val(spec, d)
            both = df.merge(val, on=["model", "dataset", "split", "seed"])
            vs, ts_ = both.groupby("split")["v"].mean(), both.groupby("split")["y"].mean()
            vm, tm = both.groupby("model")["v"].mean(), both.groupby("model")["y"].mean()
            # only where every run of the dataset recorded a validation score (partial coverage would bias the means)
            validation = None if len(both) < len(df) else {"val_mean": float(both["v"].mean()), "test_mean": float(both["y"].mean()),
                          "gap": float(both["v"].mean() - both["y"].mean()),
                          "runs_with_val_above_test": int((both["v"] > both["y"]).sum()), "runs": int(len(both)),

                          "rho_split_means": float(stats.spearmanr(vs, ts_).statistic) if len(vs) > 3 else None,
                          "rho_network_means": float(stats.spearmanr(vm, tm).statistic) if len(vm) > 3 else None,
                          "val_network_means": {k: float(x) for k, x in vm.items()},
                          "best_network_by_val": str(vm.idxmax()), "best_network_by_test": str(tm.idxmax()),
                          "test_loss_choosing_by_val": float(tm.max() - tm[vm.idxmax()])}
            out[d] = {"set": set_name, "name": NAME[d], "splits": int(len(split_means)), "models": int(len(model_means)),
                      "validation": validation,
                      "split_means": {int(k): float(v) for k, v in split_means.items()},
                      "model_means": {k: float(v) for k, v in model_means.items()},
                      "cell_means": cell_means,
                      "sd_split_means": float(split_means.std()), "sd_model_means": float(model_means.std()),
                      "mean_seed_sd": float(seed_sd), "range_split_means": float(split_means.max() - split_means.min()),
                      "range_model_means": float(model_means.max() - model_means.min())}
    ratio = [v["sd_split_means"] / v["sd_model_means"] for v in out.values() if v["sd_model_means"] > 0]
    vals = [v["validation"] for v in out.values() if v["validation"] is not None]
    rn = [x["rho_network_means"] for x in vals if x["rho_network_means"] is not None]
    summary = {"datasets": out, "n_datasets": len(out),
               "validation": {"median_gap": float(st.median(x["gap"] for x in vals)),
                              "min_gap": float(min(x["gap"] for x in vals)), "max_gap": float(max(x["gap"] for x in vals)),
                              "datasets_val_above_test": int(sum(x["gap"] > 0 for x in vals)),
                              "runs_val_above_test": int(sum(x["runs_with_val_above_test"] for x in vals)),
                              "runs": int(sum(x["runs"] for x in vals)),
                              "datasets": len(vals),
                              "datasets_without_complete_val": sorted(d for d, v in out.items()
                                                                      if v["validation"] is None),
                              "median_rho_network_means": float(st.median(rn)), "datasets_with_rho_network": len(rn),
                              "datasets_best_by_val_is_best_by_test": int(sum(x["best_network_by_val"] ==
                                                                              x["best_network_by_test"] for x in vals)),
                              "median_test_loss_choosing_by_val": float(st.median(x["test_loss_choosing_by_val"]
                                                                                  for x in vals))},
               "datasets_split_sd_above_model_sd": int(sum(v["sd_split_means"] > v["sd_model_means"] for v in out.values())),
               "median_ratio_split_sd_to_model_sd": float(st.median(ratio))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "datasets"}, indent=1))
    for d, v in out.items():
        x = v["validation"]
        if x is None:
            print(f"{d:20s} validation macro-F1 missing in some runs: not analysed")
            continue
        print(f"{d:20s} val {x['val_mean']:5.1f} test {x['test_mean']:5.1f} gap {x['gap']:+5.1f}  rho(net) "
              f"{x['rho_network_means']}  best val {x['best_network_by_val']} / test {x['best_network_by_test']}")
    for d, v in out.items():
        print(f"{d:20s} {v['set']} splits {v['splits']:2d} sd(split) {v['sd_split_means']:6.2f}  sd(model) "
              f"{v['sd_model_means']:5.2f}  seed sd {v['mean_seed_sd']:5.2f}")


if __name__ == "__main__":
    main()
