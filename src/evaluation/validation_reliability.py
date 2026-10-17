"""Exploratory: is the protocol's validation score a worse guide to the networks' test ranking than the test itself?

Not pre-registered (added after the review of 2026-10-17). The harness draws each run's validation windows at random,
window by window, from the training subjects' windows (whar_datasets.splitting.splitter._get_train_val_indices; 3-s
windows with 50 % overlap), so validation measures windows of people, and often of stretches of signal, seen in
training. src/evaluation/split_spread.py reports the rank correlation between the networks' validation and test
means. This script adds the reference that correlation needs: how well the test ranking agrees with itself.

Per dataset (the datasets of S1 and S2 that pass the inclusion rule and whose runs all recorded a validation score):

- rho_val_test: Spearman correlation of the networks' validation means and test means (all splits and seeds);
- rho_test_seeds: test means from seed 0 against test means from seeds 1-2 (same splits, other training seeds);
- rho_test_halves: test means on even-numbered against odd-numbered splits (other test subjects; >= 4 splits);
- whether the network chosen by each criterion is the best on the other half.

S2 is read after the confirmatory analysis (G4 seal); the read is logged in results/test_access.log.

    uv run --no-sync python -m src.evaluation.validation_reliability
        -> results/metrics/validation_reliability/data.json
"""
from __future__ import annotations

import json
import statistics as st
import time

from scipy import stats

from src.data.sets import S1, S2
from src.evaluation import variance_components as V
from src.evaluation.split_spread import load_val
from src.utils import paths as PATHS

OUT = PATHS.METRICS / "validation_reliability"


def rho(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def dataset_row(df, val) -> dict:
    both = df.merge(val, on=["model", "dataset", "split", "seed"])
    vm = both.groupby("model")["v"].mean()
    tm = both.groupby("model")["y"].mean().reindex(vm.index)
    s0 = both[both.seed == 0].groupby("model")["y"].mean().reindex(vm.index)
    s12 = both[both.seed > 0].groupby("model")["y"].mean().reindex(vm.index)
    row = {"networks": int(len(vm)), "splits": int(both["split"].nunique()),
           "rho_val_test": rho(vm, tm), "rho_test_seeds": rho(s0, s12),
           "best_by_val_is_best_on_test": bool(vm.idxmax() == tm.idxmax()),
           "best_on_seed0_is_best_on_seeds12": bool(s0.idxmax() == s12.idxmax())}
    if row["splits"] >= 4:
        ev = both[both.split % 2 == 0].groupby("model")["y"].mean().reindex(vm.index)
        od = both[both.split % 2 == 1].groupby("model")["y"].mean().reindex(vm.index)
        row |= {"rho_test_halves": rho(ev, od), "best_on_even_is_best_on_odd": bool(ev.idxmax() == od.idxmax()),
                "odd_loss_choosing_by_val": float(od.max() - od[vm.idxmax()]),
                "odd_loss_choosing_by_even": float(od.max() - od[ev.idxmax()])}
    return row


def main() -> None:
    with open(PATHS.RESULTS / "test_access.log", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | src/evaluation/validation_reliability.py | - | exploratory "
                 f"reliability of the networks' test ranking (S1 and S2), after the confirmatory analysis\n")
    rows = {}
    for set_name, spec in (("S1", S1), ("S2", S2)):
        inc = V.inclusion(spec)
        for d in spec["datasets"]:
            if not inc[d]["included"]:
                continue
            df, val = V.load(spec, "macro_f1", [d]), load_val(spec, d)
            if len(val) < len(df):  # as split_spread: only datasets whose runs all recorded a validation score
                continue
            rows[d] = {"set": set_name, **dataset_row(df, val)}
    halves = [r for r in rows.values() if "rho_test_halves" in r]
    med = lambda key, rs: float(st.median(r[key] for r in rs))  # noqa: E731
    summary = {
        "datasets": rows, "n_datasets": len(rows), "n_datasets_with_halves": len(halves),
        "median_rho_val_test": med("rho_val_test", rows.values()),
        "median_rho_test_seeds": med("rho_test_seeds", rows.values()),
        "median_rho_test_halves": med("rho_test_halves", halves),
        "median_rho_val_test_where_halves": med("rho_val_test", halves),
        "datasets_val_below_halves": int(sum(r["rho_val_test"] < r["rho_test_halves"] for r in halves)),
        "datasets_val_below_seeds": int(sum(r["rho_val_test"] < r["rho_test_seeds"] for r in rows.values())),
        "best_by_val_is_best_on_test": int(sum(r["best_by_val_is_best_on_test"] for r in rows.values())),
        "best_on_seed0_is_best_on_seeds12": int(sum(r["best_on_seed0_is_best_on_seeds12"] for r in rows.values())),
        "best_on_even_is_best_on_odd": int(sum(r["best_on_even_is_best_on_odd"] for r in halves)),
        "median_odd_loss_choosing_by_val": med("odd_loss_choosing_by_val", halves),
        "median_odd_loss_choosing_by_even": med("odd_loss_choosing_by_even", halves),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "datasets"}, indent=1))
    for d, r in rows.items():
        print(f"{d:20s} {r['set']} n={r['networks']:2d} val/test {r['rho_val_test']:+.2f} seeds {r['rho_test_seeds']:+.2f} "
              f"halves {r.get('rho_test_halves', float('nan')):+.2f}")


if __name__ == "__main__":
    main()
