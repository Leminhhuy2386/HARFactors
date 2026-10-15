"""Gate G4 (agents/04_research.md, adapted): seal the confirmatory analysis before S2's metrics are read for it.

In its template form G4 is not applicable here: no held-out test split is sealed, because the analysis runs over stored
runs. What is sealed instead is the pre-registration and the analysis code, by their sha256, before
scripts/run_confirmatory.py reads S2 (run_confirmatory refuses S2 without this file, and checks the hashes).

Refuses to seal if results/test_access.log already has an entry, or if the sealed files are not committed.

    uv run --no-sync python scripts/seal_g4.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import S2  # noqa: E402
from src.evaluation import variance_components as V  # noqa: E402
from src.utils.provenance import git_commit, sha256  # noqa: E402

SEALED = ["docs/preregistration.md", "src/evaluation/variance_components.py", "scripts/run_confirmatory.py",
          "src/data/sets.py", "src/utils/provenance.py", "src/utils/paths.py", "tests/test_variance_components.py"]
OUT = ROOT / "results" / "gates" / "G4_test_seal.json"
ACCESS_LOG = ROOT / "results" / "test_access.log"
PRIOR_READS = (
    "Before this seal, S2 metric values were read by code only through scripts/run_analyses.py ->"
    " src/evaluation/heldout_analysis.py, the imported exploratory analysis of the RILA-Net held-out confirmation, while"
    " checking that the imported aggregates reproduce (gate G1 jobs 86656, 86658, 86661 and diagnostic job 86657). It"
    " recomputes results/metrics/heldout/summary.md (per-dataset means and paired model differences), which was"
    " already known; it computes no variance component and no split-level spread. S2's per-dataset means were known"
    " before the pre-registration (docs/preregistration.md, § Data). scripts/validate_data.py checked only that S2's"
    " metric keys are present floats; src/evaluation/variance_components.inclusion read only split sizes and the"
    " protocol hash.")


def main() -> None:
    if OUT.exists():
        raise SystemExit("G4_test_seal.json exists: the seal is written once")
    if ACCESS_LOG.exists() and ACCESS_LOG.read_text().strip():
        raise SystemExit("refused: results/test_access.log already has entries")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", *SEALED], capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refused: commit the sealed files first:\n{dirty}")
    inc = V.inclusion(S2)
    gate = {
        "gate": "G4_test_seal", "stage": "04_research", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "template_form": "not_applicable: no held-out test split exists to seal; the analysis is over stored runs. "
                         "Sealed instead: the pre-registration and the analysis code (agents/04_research.md).",
        "check": "the pre-registration and the confirmatory analysis code are hashed before S2's metrics are read for "
                 "the confirmatory analysis",
        "reference_value": None, "reference_source": "docs/preregistration.md, § Rules fixed now (Sealing)",
        "observed_value": {
            "sealed_files": {f: sha256(ROOT / f) for f in SEALED},
            "git_commit": git_commit(),
            "test_access_log_before_seal": [],
            "prior_reads_declared": PRIOR_READS,
            "anchor_baseline_test_read": "none of S2 (G3 read S3 and the quoted table only)",
            "s2_inclusion": inc,
            "analyses_allowed_on_S2": ["H1", "sens1_S2_balanced_accuracy"],
        },
        "tolerance": "every sealed file hashed and committed; no entry in results/test_access.log before this "
                     "timestamp; every earlier read of S2 metric values declared",
        "verdict": "PASS",
        "evidence": SEALED + ["scripts/seal_g4.py"],
        "notes": "This is the first read of S2 for the confirmatory analysis. After this seal no change to the sealed "
                 "files is permitted; run_confirmatory.py refuses to read S2 if any sealed hash differs.",
    }
    OUT.write_text(json.dumps(gate, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), "commit", gate["observed_value"]["git_commit"])
    print("S2 inclusion:", {d: v["included"] for d, v in inc.items()})


if __name__ == "__main__":
    main()
