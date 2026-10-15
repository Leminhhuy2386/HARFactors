"""P4 summary: multi-seed results of run_whar.py on one dataset (RESEARCH_PLAN.md §5b).

Seed 0 is experiments/whar_arena_v1/<model>/<dataset>/split-XX.json, seed s >= 1 is .../seed-<s>/split-XX.json.
The unit of analysis is the subject split: seeds are averaged within a split first, then models are
compared paired over splits (Wilcoxon signed-rank, wins/losses) as the plan fixes. The
non-inferiority check against S4D uses a margin of 1 point: the lower end of the 95 % t-interval
of the paired mean difference over splits must lie above -1.

    uv run python -m src.evaluation.seeds_summary [--dataset utd_mhad] [--models ...]   -> results/metrics/seeds/summary_<dataset>.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy import stats
from src.utils import paths as PATHS  # noqa: E402

HERE = Path(__file__).resolve().parents[2]  # project root
RES = PATHS.RUNS
OUT = PATHS.METRICS / "seeds"
DEFAULT_MODELS = ["quantilenet", "hystnet_sel_clock", "s4d", "mamba", "hystnet_nomem"]
REFS = ["s4d", "mamba", "hystnet_nomem"]
MARGIN = 1.0


def load(model: str, dataset: str) -> dict[str, dict[int, dict]]:
    """split -> seed -> result json."""
    base = RES / model / dataset
    runs: dict[str, dict[int, dict]] = {}
    for p in sorted(base.glob("split-*.json")) + sorted(base.glob("seed-*/split-*.json")):
        r = json.loads(p.read_text())
        seed = int(p.parent.name.split("-")[1]) if p.parent.name.startswith("seed-") else 0
        runs.setdefault(p.stem, {})[seed] = r
    return runs


def f1(r: dict) -> float:
    return 100 * r["metrics"]["macro_f1"]


def split_means(runs: dict[str, dict[int, dict]], seeds: set[int] | None = None) -> dict[str, float]:
    out = {}
    for split, by_seed in runs.items():
        vals = [f1(r) for s, r in by_seed.items() if seeds is None or s in seeds]
        if vals:
            out[split] = st.mean(vals)
    return out


def paired(a: dict[str, float], b: dict[str, float]) -> dict:
    d = [a[s] - b[s] for s in sorted(a) if s in b]
    n = len(d)
    mean = st.mean(d)
    sd = st.stdev(d) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5 if n > 1 else float("nan")
    p = stats.wilcoxon(d).pvalue if n > 1 and any(d) else float("nan")
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half, "wins": sum(x > 0 for x in d),
            "losses": sum(x < 0 for x in d), "p": p}


def summarize(dataset: str, models: list[str]) -> str:
    runs = {m: load(m, dataset) for m in models}
    runs = {m: r for m, r in runs.items() if r}
    seeds_all = sorted({s for r in runs.values() for by_seed in r.values() for s in by_seed})
    # only splits and seeds that every model has, so every comparison sees the same runs
    common_splits = sorted(set.intersection(*(set(r) for r in runs.values())))
    common_seeds = {s for s in seeds_all if all(s in runs[m][sp] for m in runs for sp in common_splits)}
    lines = [f"# P4: multi-seed results on {dataset}", "",
             "Built by `seeds_summary.py`. Macro-F1 (%), WHAR Arena v1 protocol. A training seed changes only the "
             "model initialization and the batch order; the splits, validation windows and normalization are the "
             "protocol's (seed 0). Seeds are averaged within each subject split; comparisons are paired over splits.",
             "", f"Splits used: {len(common_splits)}; seeds every model has on all of them: {sorted(common_seeds)}.", ""]
    rows = []
    for m, r in runs.items():
        sm = split_means({s: r[s] for s in common_splits}, common_seeds)
        per_seed = [st.mean(f1(r[sp][s]) for sp in common_splits) for s in sorted(common_seeds)]
        within = [st.stdev([f1(r[sp][s]) for s in common_seeds]) for sp in common_splits] if len(common_seeds) > 1 else [0]
        params = next(iter(next(iter(r.values())).values()))["model"]["params"]
        devices = sorted({x.get("device", "?") for by_seed in r.values() for x in by_seed.values()})
        rows.append((st.mean(sm.values()), m, params, sm, per_seed, st.mean(within), devices))
    rows.sort(reverse=True)
    lines += ["| Model | Params | Mean over splits (seed-averaged) | Std over splits | Mean seed std within a split | "
              "Mean per seed | Seed range |", "|---|---|---|---|---|---|---|"]
    for mean, m, params, sm, per_seed, within, devices in rows:
        lines.append(f"| {m} | {params / 1000:.1f}k | **{mean:.2f}** | {st.stdev(sm.values()):.1f} | {within:.2f} | "
                     + ", ".join(f"{v:.1f}" for v in per_seed)
                     + f" | {min(per_seed):.1f}–{max(per_seed):.1f} |")
    lines += ["", "Hardware per seed: seed 0 and 4 on the local GTX 1080 Ti, seeds 1–2 on Kaggle T4, seed 3 on a "
              "Colab GPU (see each result's `device`). Non-deterministic GPU kernels make results differ slightly "
              "across hardware; every seed ran all models on the same hardware.", ""]
    split_mean = {m: split_means({s: r[s] for s in common_splits}, common_seeds) for m, r in runs.items()}
    for ref in REFS:
        if ref not in runs:
            continue
        lines += [f"## Model − {ref}, paired over splits", "",
                  "| Model | Mean Δ | 95 % CI | Wins/losses | Wilcoxon p |" + (" Non-inferior (margin 1) |" if ref == "s4d" else ""),
                  "|---|---|---|---|---|" + ("---|" if ref == "s4d" else "")]
        for _, m, *_ in rows:
            if m == ref:
                continue
            c = paired(split_mean[m], split_mean[ref])
            ni = (" yes |" if c["lo"] > -MARGIN else " no |") if ref == "s4d" else ""
            lines.append(f"| {m} | {c['mean']:+.2f} | [{c['lo']:+.2f}, {c['hi']:+.2f}] | {c['wins']}/{c['losses']} | "
                         f"{c['p']:.3f} |" + ni)
        lines.append("")
    lines += ["## Per split (seed-averaged)", "", "| Split | " + " | ".join(m for _, m, *_ in rows) + " |",
              "|---|" + "---|" * len(rows)]
    for sp in common_splits:
        lines.append(f"| {sp} | " + " | ".join(f"{split_mean[m][sp]:.1f}" for _, m, *_ in rows) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="utd_mhad")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()
    text = summarize(args.dataset, args.models)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"summary_{args.dataset}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
