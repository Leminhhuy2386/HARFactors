"""Gate G2 (agents/03_data.md, Step 2): the data this project analyses are what their documentation says.

The "data" are the stored run records of the three sets of the pre-registration (S1, S2, S3), the quoted Figure 6
table, and the paper-era caches the harness regenerates splits from. Checks:

1. completeness and schema: every run of S1, S2 and S3 exists, is complete, has the paper protocol hash and the three
   metric keys. S2's metric values are not read (only their presence and type): S2 is sealed until gate G4;
2. external reference values:
   a. the WHAR Arena authors' own ActRecTut result files (pinned whar-arena repo) give the split sizes the stored runs
      must have;
   b. UTD-MHAD: 861 data sequences, 8 subjects, 27 actions (Chen et al., ICIP 2015) against the cache;
   c. subjects per dataset in the paper-era configs against the subjects the harness finds in each cache;
   d. the Figure 6 extraction against the authors' snapshot of the figure (0 mismatches allowed);
3. split records: the stored runs of each (dataset, split) agree on their split sizes, and the harness regenerates
   those sizes from the caches on this machine. Disagreements are listed, not hidden: the pre-registered inclusion
   rule (docs/preregistration.md) excludes such datasets from the variance analyses.

Writes results/gates/G2_data.json. Tolerances are fixed below, before measuring.

    uv run --no-sync python scripts/validate_data.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import SETS, cells, run_path  # noqa: E402
from src.evaluation import factor_analysis as FA  # noqa: E402

PROTOCOL_HASH = "e8739ef2c5b36c99"
TOLERANCE = ("exact: all S1/S2/S3 runs present, complete, protocol hash e8739ef2c5b36c99, metric keys present; "
             "ActRecTut split sizes equal the authors' result files; UTD-MHAD cache has 861 sequences, 8 subjects, "
             "27 classes; subjects per cached dataset equal the paper-era config; 0 Figure 6 cells differing from the "
             "authors' snapshot by more than 0.006 points")
UTD_PUBLISHED = {"sequences": 861, "subjects": 8, "actions": 27,
                 "source": "Chen, Jafari, Kehtarnavaz, UTD-MHAD dataset paper, IEEE ICIP 2015 (paper/"
                           "references_verified.json); 864 = 8 x 27 x 4 minus 3 corrupted sequences"}
OFFICIAL = ROOT / "third_party/whar-arena/data/results/whar_arena_v1"


def completeness() -> dict:
    out = {}
    for name, spec in SETS.items():
        missing, bad = [], []
        for c in cells(spec):
            f = run_path(spec, *c)
            if not f.exists():
                missing.append(str(f.relative_to(ROOT)))
                continue
            r = json.loads(f.read_text())
            m = r.get("metrics", {})
            ok = (r.get("status") == "complete" and r.get("protocol_hash") == PROTOCOL_HASH
                  and r.get("model_id") == c[0] and r.get("dataset_id") == c[1] and r.get("split_id") == f"split-{c[3]:02d}"
                  and all(isinstance(m.get(k), float) for k in ("macro_f1", "accuracy", "balanced_accuracy")))
            if ok and name != "S2":  # S2 values stay unread until G4
                ok = all(0.0 <= m[k] <= 1.0 for k in ("macro_f1", "accuracy", "balanced_accuracy"))
            if not ok:
                bad.append(str(f.relative_to(ROOT)))
        out[name] = {"expected": len(cells(spec)), "missing": missing, "invalid": bad,
                     "metric_values_read": name != "S2"}
    return out


def official_actrectut() -> dict:
    ours = json.loads((ROOT / "data/processed/splits.json").read_text())["datasets"]["actrectut_gestures"]["splits"]
    rows, ok = [], True
    for f in sorted(OFFICIAL.glob("*/actrectut_gestures/split-*.json")):
        r = json.loads(f.read_text())
        s = r["source"]
        theirs = {"identifier": s["split_identifier"], "train_size": s["train_size"], "val_size": s["val_size"],
                  "test_size": s["test_size"]}
        mine = ours[r["split_id"]]["stored_sizes"]
        match = mine == [theirs]
        ok &= match
        rows.append({"file": str(f.relative_to(ROOT)), "theirs": theirs, "match": match})
    return {"official_files": len(rows), "all_match": ok and bool(rows),
            "sizes": sorted({json.dumps(r["theirs"], sort_keys=True) for r in rows})}


def utd_mhad() -> dict:
    h = json.loads((ROOT / "data/processed/harness_splits.json").read_text())["datasets"]["utd_mhad"]
    classes = FA.CFG["utd_mhad"][2]
    obs = {"sequences": h["sessions"], "subjects": h["subjects"], "actions": classes}
    return {"published": UTD_PUBLISHED, "observed": obs,
            "match": all(obs[k] == UTD_PUBLISHED[k] for k in ("sequences", "subjects", "actions"))}


def subjects_vs_config() -> dict:
    h = json.loads((ROOT / "data/processed/harness_splits.json").read_text())["datasets"]
    rows = {d: {"config": FA.CFG[d][1], "cache": v["subjects"], "splits": len(v["splits"]),
                "expected_splits": min(10, FA.CFG[d][1])} for d, v in h.items()}
    return {"rows": rows, "all_match": all(r["config"] == r["cache"] and r["splits"] == r["expected_splits"]
                                           for r in rows.values())}


def figure6() -> dict:
    b = json.loads((ROOT / "docs/baselines_cited.json").read_text())
    return {"cells": b["cells"], "snapshot_mismatches": len(b["snapshot_mismatches"]),
            "match": b["cells"] == 510 and not b["snapshot_mismatches"]}


def split_records() -> dict:
    s = json.loads((ROOT / "data/processed/splits.json").read_text())["datasets"]
    disagree = {d: [k for k, e in v["splits"].items() if not e["stored_runs_agree"]] for d, v in s.items()}
    harness_mismatch = {d: [k for k, e in v["splits"].items() if e.get("harness_matches_stored") is False]
                        for d, v in s.items()}
    return {"stored_runs_disagree": {d: k for d, k in disagree.items() if k},
            "harness_differs_from_stored": {d: k for d, k in harness_mismatch.items() if k},
            "note": "A dataset whose stored runs disagree on split sizes was run on caches with different subject "
                    "numbering: its split ids do not name the same test subjects across runs. The pre-registered "
                    "inclusion rule excludes it from the variance analyses (S1: daphnet; S2: motion_sense)."}


def main() -> None:
    manifest = json.loads((ROOT / "data/processed/manifest.json").read_text())
    obs = {"completeness": completeness(), "actrectut_vs_authors": official_actrectut(), "utd_mhad": utd_mhad(),
           "subjects_vs_config": subjects_vs_config(), "figure6_vs_snapshot": figure6(),
           "split_records": split_records(), "data_manifest_sha256": manifest["data_manifest_sha256"],
           "manifest_files": manifest["files"]}
    comp_ok = all(not v["missing"] and not v["invalid"] for v in obs["completeness"].values())
    passed = (comp_ok and obs["actrectut_vs_authors"]["all_match"] and obs["utd_mhad"]["match"]
              and obs["subjects_vs_config"]["all_match"] and obs["figure6_vs_snapshot"]["match"])
    gate = {
        "gate": "G2_data", "stage": "03_data", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "check": "the stored runs, the quoted table and the caches match their documentation and published values",
        "reference_value": {"utd_mhad": UTD_PUBLISHED, "actrectut_split_sizes": "WHAR Arena authors' result files",
                            "figure6": "authors' snapshot paper-benchmark.json"},
        "reference_source": "UTD-MHAD dataset paper (ICIP 2015); third_party/whar-arena (19b3019) data/results and "
                            "apps/web/src/generated/paper-benchmark.json; paper-era whar-datasets configs",
        "observed_value": obs, "tolerance": TOLERANCE, "verdict": "PASS" if passed else "FAIL",
        "evidence": ["data/processed/manifest.json", "data/processed/splits.json",
                     "data/processed/harness_splits.json", "docs/baselines_cited.json", "scripts/validate_data.py",
                     "scripts/export_splits.py", "scripts/build_data_records.py", "scripts/build_baselines_cited.py"],
        "notes": "data_manifest_sha256 above is the value every provenance record must carry. Split-size "
                 "disagreements (daphnet in S1, motion_sense in S2) are documented consequences of unsorted "
                 "paper-era parsers and are handled by the pre-registered inclusion rule; they do not contradict the "
                 "datasets' documentation.",
    }
    out = ROOT / "results" / "gates" / "G2_data.json"
    if out.exists():
        raise SystemExit("G2_data.json exists: a re-run must get its own file and state why")
    out.write_text(json.dumps(gate, indent=1) + "\n")
    print("wrote", out.relative_to(ROOT), "verdict", gate["verdict"])
    print(json.dumps({k: v for k, v in obs.items() if k not in ("subjects_vs_config",)}, indent=1)[:3000])


if __name__ == "__main__":
    main()
