"""The quoted-numbers register of agents/03_data.md, Step 3d: docs/baselines_cited.json.

One entry per cell of the WHAR Arena paper's Figure 6 (30 datasets x 17 models; macro-F1 mean and split sd, one run
per split). The values come from docs/sources/arena_extracted.json, which the HAR reproduction extracted from the
figure. Every mean is cross-checked here against the authors' own snapshot of the figure in the pinned whar-arena
repo (apps/web/src/generated/paper-benchmark.json). That snapshot has no sd, so the sds rest on the extraction alone;
spot checks against the paper are recorded in the register.

Also records where each data-factor covariate comes from: channels from the paper's Table 3; sampling rate, subjects
and classes from the paper-era whar-datasets configs.

These are quoted numbers, not measurements: they never enter results/metrics/ and carry no provenance record.

    uv run --no-sync python scripts/build_baselines_cited.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compare_arena as CA  # noqa: E402
from src.evaluation import factor_analysis as FA  # noqa: E402

PAPER_KEY = "burzer2026whararena"  # paper/references_verified.json holds its resolved title and id
SNAPSHOT = ROOT / "third_party/whar-arena/apps/web/src/generated/paper-benchmark.json"
EXTRACTED = ROOT / "docs/sources/arena_extracted.json"
# paper-era parsers that list files unsorted: subject numbering (so per-split membership) can differ by machine
UNSORTED = {"usc_had", "motion_sense", "dsads", "daphnet"}
OUT = ROOT / "docs" / "baselines_cited.json"
SPOT = ROOT / "docs" / "sources" / "figure6_spot_checks.json"  # cells read by eye from the paper (check 13)


def main() -> None:
    ex = json.loads(EXTRACTED.read_text())
    snap = json.loads(SNAPSHOT.read_text())
    name_to_id = {m["name"]: m["id"] for m in snap["catalog"]["models"]}
    ds_name = {d["id"]: d["name"] for d in snap["catalog"]["datasets"]}
    snap_val = {(r["dataset_id"], r["model_id"]): 100 * r["metrics"]["macro_f1"] for r in snap["results"]}
    entries, mismatches = [], []
    for d in ex["dataset_order_f1"]:
        if d == "Mean":
            continue
        for name in ex["f1_models"]:
            if name == "Mean":
                continue
            mean, sd = ex["f1"][d][name]
            mid = name_to_id[name]
            s = snap_val[(d, mid)]
            if abs(s - mean) > 0.006:  # the snapshot stores fractions with 3-4 decimals, the figure 1-2 decimals
                mismatches.append({"dataset": d, "model": name, "extracted": mean, "snapshot": round(s, 4)})
            entries.append({
                "id": f"whar_arena_fig6/{d}/{mid}", "method": name, "model_id": mid,
                "value": mean, "split_sd": sd,
                "metric": "macro-F1 (%), harness definition (absent classes count as 0); mean and sd over the "
                          "leave-subjects-out splits; one training run per split",
                "dataset": d, "dataset_name": ds_name.get(d, d),
                "split": "WHAR Arena v1 protocol (k = 10 subject groups), mean over splits",
                "source": {"citation_key": PAPER_KEY, "location": f"Figure 6, row {ds_name.get(d, d)}, column {name}"},
                "protocol_match": "partial" if d in UNSORTED else "exact",
                "protocol_differences": ["paper-era parser lists files unsorted: subject numbering, so per-split "
                                         "membership, can differ by machine; compare dataset means only"]
                                        if d in UNSORTED else [],
                "read_on": "2026-10-15",
                "read_from": "docs/sources/arena_extracted.json (extracted from Figure 6 by the HAR reproduction); "
                             "mean cross-checked against the authors' snapshot "
                             "third_party/whar-arena/apps/web/src/generated/paper-benchmark.json (whar-arena 19b3019)",
                "snapshot_value": round(s, 4),
            })
    covariates = {d: {"channels": CA.CHANNELS[d], "channels_source": f"{PAPER_KEY}, Table 3",
                      "sampling_hz": FA.CFG[d][0], "subjects": FA.CFG[d][1], "classes": FA.CFG[d][2],
                      "hz_subjects_classes_source": f"third_party/whar-datasets (144e006) src/whar_datasets/config/"
                                                    f"cfg_{d}.py"}
                  for d in ex["dataset_order_f1"] if d != "Mean"}
    out = {"note": "Quoted numbers, not measurements of this project. Never write them into results/metrics/.",
           "cells": len(entries), "snapshot_mismatches": mismatches, "entries": entries,
           "covariates": covariates,
           "spot_checks_against_paper": json.loads(SPOT.read_text()) if SPOT.exists() else None}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("cells", len(entries), "snapshot mismatches", len(mismatches), "partial", sum(e["protocol_match"] == "partial"
                                                                                       for e in entries))


if __name__ == "__main__":
    main()
