"""Exploratory: how far apart the four S2 networks are, and where they sit among the benchmark's published networks.

Not pre-registered (added on 2026-10-17 to make "comparable networks" concrete). On the datasets of H1 (the S2 datasets
that pass the inclusion rule), per network: the mean macro-F1 over splits and seeds per dataset, and its mean over the
datasets; per dataset: the range and SD of the four network means. The same quantities for the 17 WHAR Arena networks
come from the published table (docs/sources/arena_extracted.json: mean over splits of one run per split, same protocol
and code state). Comparing three-seed means with published single runs locates the population; it ranks no network.
Datasets whose parser lists files unsorted (usc_had) are compared by means only, which is all this script uses.

S2 is read after the confirmatory analysis (G4 seal); the read is logged in results/test_access.log.

    uv run --no-sync python -m src.evaluation.s2_positioning   ->   results/metrics/s2_positioning/data.json
"""
from __future__ import annotations

import json
import statistics as st
import time

from src.data.sets import S2
from src.evaluation import variance_components as V
from src.evaluation.split_spread import NAME
from src.utils import paths as PATHS

OUT = PATHS.METRICS / "s2_positioning"
ARENA_FILE = PATHS.ROOT / "docs" / "sources" / "arena_extracted.json"
DISPLAY = {"recurrencenet": "RecurrenceNet", "s4d": "S4D", "recstatnet": "RN + stats", "lawnet_rec": "RN + laws"}


def main() -> None:
    with open(PATHS.RESULTS / "test_access.log", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | src/evaluation/s2_positioning.py | - | exploratory "
                 f"description of the S2 network means against the published benchmark, after the confirmatory analysis\n")
    inc = V.inclusion(S2)
    datasets = [d for d in S2["datasets"] if inc[d]["included"]]
    df = V.load(S2, "macro_f1", datasets)
    cell = df.groupby(["dataset", "model"])["y"].mean()  # per dataset and network: mean over splits and seeds
    arena = json.loads(ARENA_FILE.read_text())["f1"]
    arena_models = [m for m in json.loads(ARENA_FILE.read_text())["f1_models"] if m != "Mean"]
    per_dataset = {}
    for d in datasets:
        ours = {DISPLAY[m]: float(cell[(d, m)]) for m in S2["models"]}
        pub = {m: float(arena[d][m][0]) for m in arena_models}
        per_dataset[d] = {"name": NAME[d], "networks": ours, "range_networks": max(ours.values()) - min(ours.values()),
                          "sd_networks": st.stdev(ours.values()), "published": pub,
                          "range_published": max(pub.values()) - min(pub.values()),
                          "best_published": max(pub.values())}
    ours_mean = {n: st.mean(per_dataset[d]["networks"][n] for d in datasets) for n in DISPLAY.values()}
    pub_mean = {m: st.mean(per_dataset[d]["published"][m] for d in datasets) for m in arena_models}
    pub_sorted = sorted(pub_mean.values(), reverse=True)
    rank = {n: 1 + sum(v > x for v in pub_sorted) for n, x in ours_mean.items()}  # position among the 17 if inserted
    widest = max(datasets, key=lambda d: per_dataset[d]["range_networks"])
    summary = {
        "datasets": datasets, "n_datasets": len(datasets), "per_dataset": per_dataset,
        "network_means": ours_mean, "published_means": pub_mean,
        "networks_mean_min": min(ours_mean.values()), "networks_mean_max": max(ours_mean.values()),
        "published_mean_min": min(pub_mean.values()), "published_mean_max": max(pub_mean.values()),
        "published_mean_5th": pub_sorted[4], "published_mean_12th": pub_sorted[11],
        "rank_among_published": rank, "best_rank": min(rank.values()), "worst_rank": max(rank.values()),
        "n_published": len(arena_models),
        "median_range_networks": float(st.median(per_dataset[d]["range_networks"] for d in datasets)),
        "max_range_networks": per_dataset[widest]["range_networks"], "widest_dataset": NAME[widest],
        "median_range_published": float(st.median(per_dataset[d]["range_published"] for d in datasets)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_dataset", "published_means")}, indent=1))


if __name__ == "__main__":
    main()
