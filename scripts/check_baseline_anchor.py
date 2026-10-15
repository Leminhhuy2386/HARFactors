"""Gate G3 (agents/03_data.md, Step 3): protocol identity of the quoted numbers, and an anchor of the evaluation stack.

- Protocol identity: every Figure 6 cell is classified in docs/baselines_cited.json ("exact" when the dataset's
  paper-era parser sorts its files, "partial" otherwise).
- Anchor (Step 3b): Random Forest, the cheapest baseline with no tuning, as re-run with this code state by the HAR
  reproduction (S3). Its mean macro-F1 over the splits of each of the 4 datasets is compared with Figure 6, with
  the template's default tolerance of +/-10 % relative, declared here before measuring.
- Supplementary, reported and not gating: (a) the authors' own per-split ActRecTut result files (pinned whar-arena
  repo) against the re-runs, model by model; (b) all 17 re-run models against Figure 6 on the 4 datasets
  (Spearman rho, mean absolute difference).

Note for the record: the HAR reproduction and Edge_SSM had compared these re-runs with Figure 6 before this gate.
The anchor's tolerance does not depend on that comparison: it is the template default. S2 (the sealed set) is not read.

    uv run --no-sync python scripts/check_baseline_anchor.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import S3, run_path  # noqa: E402

TOLERANCE = "anchor: |re-run mean - Figure 6| <= 10 % of the Figure 6 value, for Random Forest on each of the 4 datasets"
OFFICIAL = ROOT / "third_party/whar-arena/data/results/whar_arena_v1"


def f1(path: Path) -> float:
    return 100 * json.loads(path.read_text())["metrics"]["macro_f1"]


def main() -> None:
    cited = json.loads((ROOT / "docs/baselines_cited.json").read_text())
    fig6 = {(e["dataset"], e["model_id"]): e["value"] for e in cited["entries"]}
    match = {k: sum(e["protocol_match"] == k for e in cited["entries"]) for k in ("exact", "partial", "unknown")}

    rerun = {(m, d): st.mean(f1(run_path(S3, m, d, 0, i)) for i in range(n))
             for m in S3["models"] for d, n in S3["datasets"].items()}
    anchor = []
    for d in S3["datasets"]:
        paper, ours = fig6[(d, "random_forest")], rerun[("random_forest", d)]
        rel = abs(ours - paper) / paper
        anchor.append({"dataset": d, "figure6": paper, "rerun_mean": round(ours, 3), "abs_diff": round(ours - paper, 3),
                       "rel_diff": round(rel, 4), "within_tolerance": rel <= 0.10})

    agreement = []
    for d in S3["datasets"]:
        a = [fig6[(d, m)] for m in S3["models"]]
        b = [rerun[(m, d)] for m in S3["models"]]
        agreement.append({"dataset": d, "models": len(a), "spearman": round(float(stats.spearmanr(a, b).statistic), 3),
                          "mean_abs_diff": round(st.mean(abs(x - y) for x, y in zip(a, b)), 3)})

    official = []
    for f in sorted(OFFICIAL.glob("*/actrectut_gestures/split-*.json")):
        r = json.loads(f.read_text())
        mine = run_path(S3, r["model_id"], "actrectut_gestures", 0, int(r["split_id"][-2:]))
        if mine.exists():
            official.append({"model": r["model_id"], "split": r["split_id"], "authors": round(100 * r["metrics"]["macro_f1"], 4),
                             "rerun": round(f1(mine), 4), "diff": round(f1(mine) - 100 * r["metrics"]["macro_f1"], 4)})
    exact = sum(abs(o["diff"]) < 1e-3 for o in official)

    passed = all(a["within_tolerance"] for a in anchor)
    gate = {
        "gate": "G3_baseline", "stage": "03_data", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "quoted+anchor",
        "check": "Random Forest re-run with this code state reproduces its Figure 6 value on the 4 re-run datasets",
        "reference_value": {a["dataset"]: a["figure6"] for a in anchor},
        "reference_source": "WHAR Arena paper, Figure 6 (docs/baselines_cited.json; mean cross-checked against the "
                            "authors' snapshot)",
        "observed_value": {"anchor_random_forest": anchor, "protocol_match_counts": match,
                           "supplementary_17_models_vs_figure6": agreement,
                           "supplementary_authors_actrectut_files": {"pairs": len(official), "equal_to_1e-3": exact,
                                                                     "max_abs_diff": max((abs(o["diff"]) for o in official), default=None),
                                                                     "rows": official}},
        "tolerance": TOLERANCE, "verdict": "PASS" if passed else "FAIL",
        "test_read": {"scope": "none of the sealed set S2",
                      "justification": "the anchor reads S3 and the quoted table only; S2 stays sealed until G4"},
        "evidence": ["docs/baselines_cited.json", "experiments/external_har_paper_era/whar_arena_v1/random_forest",
                     "third_party/whar-arena/data/results/whar_arena_v1", "scripts/check_baseline_anchor.py"],
        "notes": "Quoted cells with protocol_match 'partial' (usc_had, motion_sense, dsads, daphnet) are compared by "
                 "dataset means only. Our 10 population models have no external value; they share this evaluation "
                 "code (whar_arena.training._fit_torch_split and _classification_metrics) with the re-runs.",
    }
    out = ROOT / "results" / "gates" / "G3_baseline.json"
    if out.exists():
        raise SystemExit("G3_baseline.json exists: a re-run must get its own file and state why")
    out.write_text(json.dumps(gate, indent=1) + "\n")
    print("wrote", out.relative_to(ROOT), "verdict", gate["verdict"])
    print(json.dumps({k: v for k, v in gate["observed_value"].items() if k != "supplementary_authors_actrectut_files"},
                     indent=1))
    print("authors' ActRecTut files:", len(official), "pairs,", exact, "equal to 1e-3")


if __name__ == "__main__":
    main()
