"""Write the round-6 stage-2 job files (RESEARCH_PLAN.md §5c): the 3 candidates on the 4 dev datasets, seeds 0-2.

Every (dataset, seed) runs on the GPU type the old models used for it in P4/P5, so paired comparisons
share hardware:
- local GTX 1080 Ti: utd_mhad seed 0, uci_har seed 0, unimib_shar seed 0, actrectut_gestures seeds 0-2;
- Colab T4: utd_mhad seeds 1-2 (Kaggle T4 in P4), uci_har and unimib_shar seed 1;
- Colab L4: uci_har and unimib_shar seed 2.
Cloud queues run longest first; the local queue interleaves models.

    uv run python scripts/cloud/make_r6_jobs.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["rainflownet", "ordinalnet", "recurrencenet"]
MS_PER_STEP = {"rainflownet": 62, "ordinalnet": 47, "recurrencenet": 36}  # GTX 1080 Ti, batch 64, L = 150
SIZE = {"utd_mhad": (796, 150, 8), "uci_har": (8188, 150, 10), "unimib_shar": (7207, 150, 10),
        "actrectut_gestures": (812, 96, 2)}


def cost(dataset: str, model: str) -> float:
    windows, length, splits = SIZE[dataset]
    return windows * 0.72 * splits / 64 * 50 * MS_PER_STEP[model] * length / 150 / 1000 / 3600


def jobs(pairs: list[tuple[str, int]]) -> list[dict]:
    out = [{"models": [m], "dataset": d, "seed": s, "est_hours": round(cost(d, m), 2)} for d, s in pairs for m in MODELS]
    return sorted(out, key=lambda j: -j["est_hours"])


def main() -> None:
    env = {"EDGE_SSM_FAST": "1"}
    local = jobs([("uci_har", 0), ("unimib_shar", 0), ("utd_mhad", 0)] + [("actrectut_gestures", s) for s in (0, 1, 2)])
    files = {
        "local_r6": {"experiment": "Round 6 stage 2, local GTX 1080 Ti", "max_parallel": 4, "env": env,
                     "jobs": [j for k in range(3) for j in local[k::3]]},  # spread models over the queue
        "colab_r6_t4": {"experiment": "Round 6 stage 2, Colab T4: utd_mhad seeds 1-2, uci_har and unimib_shar seed 1",
                        "max_parallel": 4, "env": env,
                        "jobs": jobs([("uci_har", 1), ("unimib_shar", 1), ("utd_mhad", 1), ("utd_mhad", 2)])},
        "colab_r6_l4": {"experiment": "Round 6 stage 2, Colab L4: uci_har and unimib_shar seed 2", "max_parallel": 4,
                        "env": env, "jobs": jobs([("uci_har", 2), ("unimib_shar", 2)])},
    }
    for name, cfg in files.items():
        (HERE.parents[1] / "configs" / "cloud_jobs" / f"{name}.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        total = sum(j["est_hours"] for j in cfg["jobs"])
        print(f"{name}: {len(cfg['jobs'])} jobs, ~{total:.1f} process-hours")


if __name__ == "__main__":
    main()
