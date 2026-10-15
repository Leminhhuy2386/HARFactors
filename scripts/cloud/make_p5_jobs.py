"""Write the P5 job files (RESEARCH_PLAN.md §5b, P5): 5 models x 10 datasets x training seeds 0-2.

Datasets were fixed before any P5 result, by a rule recorded in the plan: from the paper-era
BENCHMARK_DATASET_IDS, automatic download, no red/orange data defect in HAR/ATLAS.md §2, and at
most 2.2 M window-samples (windows x samples per 3-s window). UTD-MHAD already has seeds 0-4.

One (dataset, seed) runs on one machine, so every model of it shares the hardware:
- local GTX 1080 Ti: seed 0 of uci_har and unimib_shar, all seeds of actrectut_gestures (cached locally);
- Colab session 2: seed 1 of uci_har and unimib_shar; Colab session 3: seed 2 of them;
- Kaggle kernel A: seed 0 of the other seven; Kaggle kernel B: seed 1; Colab session 1: seed 2.
The local share was cut after the start: four Mamba processes saturate the 1080 Ti's memory bandwidth
(~700 s per UCI split instead of ~180 s alone), while an L4 runs four at nearly full speed.
Jobs are (dataset, seed, model). Cloud queues run longest first by an estimated cost (windows x samples
per window x measured ms per training step); the local queue interleaves models so that no more than
one Mamba runs at a time on the 1080 Ti.
All runs use EDGE_SSM_FAST=1 (same math, gradients equal up to float rounding).

    uv run python scripts/cloud/make_p5_jobs.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["quantilenet", "hystnet_sel_clock", "s4d", "mamba", "hystnet_nomem"]
MS_PER_STEP = {"quantilenet": 30, "hystnet_sel_clock": 32, "s4d": 9, "mamba": 40, "hystnet_nomem": 5}  # L = 150, GTX 1080 Ti
# windows (HAR/ATLAS.md §2), samples per 3-s window (native rate), number of splits (min(10, subjects))
DATASETS = {
    "actrectut_gestures": (812, 96, 2), "mhealth": (4348, 150, 10), "opportunity": (10642, 90, 4),
    "unimib_shar": (7207, 150, 10), "uci_har": (8188, 150, 10), "sad": (8330, 150, 10), "hapt": (9070, 150, 10),
    "dsads": (18240, 75, 8), "wisdm": (33724, 60, 10), "daphnet": (11165, 192, 10),
}
LOCAL = ["uci_har", "unimib_shar", "actrectut_gestures"]  # cached in D:\Projects\HAR\datasets_paper_era
CLOUD = [d for d in DATASETS if d not in LOCAL]


def cost(dataset: str, model: str) -> float:
    """Rough single-process hours: training windows over all splits / 64 x 50 epochs x step time."""
    windows, length, splits = DATASETS[dataset]
    steps = windows * 0.72 * splits / 64 * 50
    return steps * MS_PER_STEP[model] * length / 150 / 1000 / 3600


def jobs(datasets: list[str], seeds: list[int]) -> list[dict]:
    out = [{"models": [m], "dataset": d, "seed": s, "est_hours": round(cost(d, m), 2)}
           for d in datasets for s in seeds for m in MODELS]
    return sorted(out, key=lambda j: -j["est_hours"])


def interleave(job_list: list[dict]) -> list[dict]:
    """Round-robin over models (each model's jobs longest first), so the GPU-heavy Mamba jobs are spread out."""
    by_model = {m: [j for j in job_list if j["models"] == [m]] for m in MODELS}
    out = []
    while any(by_model.values()):
        for m in sorted(MODELS, key=lambda m: -sum(j["est_hours"] for j in by_model[m])):
            if by_model[m]:
                out.append(by_model[m].pop(0))
    return out


def main() -> None:
    env = {"EDGE_SSM_FAST": "1"}
    files = {
        "local_p5": {"experiment": "P5 local: seed 0 of uci_har and unimib_shar, all seeds of actrectut_gestures",
                     "max_parallel": 4, "env": env,
                     "jobs": interleave(jobs(["uci_har", "unimib_shar"], [0]) + jobs(["actrectut_gestures"], [0, 1, 2]))},
        "colab_p5_uu_s1": {"experiment": "P5 Colab session 2: seed 1 of uci_har and unimib_shar", "max_parallel": 4,
                           "env": env, "jobs": jobs(["uci_har", "unimib_shar"], [1])},
        "colab_p5_uu_s2": {"experiment": "P5 Colab session 3: seed 2 of uci_har and unimib_shar", "max_parallel": 4,
                           "env": env, "jobs": jobs(["uci_har", "unimib_shar"], [2])},
        "kaggle_p5_s0": {"experiment": "P5 Kaggle A: seed 0 of the datasets not cached locally", "max_parallel": 4,
                         "env": env, "jobs": jobs(CLOUD, [0])},
        "kaggle_p5_s1": {"experiment": "P5 Kaggle B: seed 1", "max_parallel": 4, "env": env, "jobs": jobs(CLOUD, [1])},
        "colab_p5_s2": {"experiment": "P5 Colab: seed 2", "max_parallel": 4, "env": env, "jobs": jobs(CLOUD, [2])},
    }
    for name, cfg in files.items():
        (HERE.parents[1] / "configs" / "cloud_jobs" / f"{name}.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        total = sum(j["est_hours"] for j in cfg["jobs"])
        print(f"{name}: {len(cfg['jobs'])} jobs, ~{total:.1f} process-hours, ~{total / cfg['max_parallel']:.1f} h wall")


if __name__ == "__main__":
    main()
