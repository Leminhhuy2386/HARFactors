"""Exploratory: the component contrasts of factor_analysis (§ C) without daphnet.

Gate G2 found that daphnet's stored runs disagree on split sizes: they ran on caches whose unsorted paper-era parser
numbered subjects differently, so a daphnet split id does not name the same test subjects across models and seeds.
The contrasts pair models by split, so on daphnet they are not paired by person. This repeats § C on the other 9
datasets. It is exploratory, like § C (docs/preregistration.md, § Exploratory).

    uv run python -m src.evaluation.contrasts_robustness   ->   results/metrics/contrasts_robustness/{data.json,summary.md}
"""
from __future__ import annotations

import json

from src.evaluation import factor_analysis as FA
from src.utils import paths as PATHS

OUT = PATHS.METRICS / "contrasts_robustness"
DROP = "daphnet"


def main() -> None:
    ds = [d for d in FA.DS if d != DROP]
    res, lines = {}, ["# Component contrasts without daphnet (exploratory)", "",
                      "Same contrasts and statistics as results/metrics/factors/summary.md § C, over the 9 datasets "
                      "whose stored runs agree on split sizes (daphnet excluded: its split ids do not name the same "
                      "test subjects across runs; results/gates/G2_data.json).", "",
                      "| Component | a − b | Seeds | 10 datasets: mean [95 % CI], won, p | 9 datasets: mean [95 % CI], won, p |",
                      "|---|---|---|---|---|"]
    for group, a, b, seeds in FA.CONTRASTS:
        full = FA.contrast(a, b, seeds, FA.DS)
        nine = FA.contrast(a, b, seeds, ds)
        res[f"{a}-{b}"] = {"group": group, "seeds": seeds, "all_10": full, "without_daphnet": nine}
        fmt = lambda c: f"{c['mean']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}], {c['wins']}/{c['n']}, {c['p']:.3f}"  # noqa: E731
        lines.append(f"| {group} | {FA.NAME[a]} − {FA.NAME[b]} | {seeds} | {fmt(full)} | {fmt(nine)} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "data.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
