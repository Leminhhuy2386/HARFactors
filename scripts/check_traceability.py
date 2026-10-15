"""Gate G6 (agents/06_summary.md): every number of docs/summary.md traces to an artifact, and the check can fail.

1. `scripts/audit.py --traceability` re-derives every value of docs/summary_values.json from its artifact.
2. Negative control (agents/05_verify.md, Level 5: "edit a number in a metric file -> --traceability"): on a copy of
   the project, one confirmatory component is changed by 10 %; the copy must fail --traceability.
3. Every artifact a value comes from has a provenance record (or is a gate file).

    uv run --no-sync python scripts/check_traceability.py   ->   results/gates/G6_traceability.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "results" / "gates" / "G6_traceability.json"
TOLERANCE = "every summary value re-derives exactly; the tampered copy fails; every source artifact has provenance or is a gate"


def audit(root: Path) -> tuple[bool, str]:
    p = subprocess.run([sys.executable, "scripts/audit.py", "--traceability", "--root", str(root)], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode == 0, p.stdout


def main() -> None:
    if GATE.exists():
        raise SystemExit("G6 exists: a re-run must get its own file and state why")
    values = json.loads((ROOT / "docs/summary_values.json").read_text())
    ok, out = audit(ROOT)
    with tempfile.TemporaryDirectory() as t:
        copy = Path(t) / "proj"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("checkpoints", ".venv", "third_party", "whar_arena_v1",
                                                                  "external_har_paper_era", "paper", "figures"))
        base_ok, _ = audit(copy)  # the untampered copy must pass, or the control proves nothing
        f = copy / "results/metrics/confirmatory/H1.json"
        r = json.loads(f.read_text())
        r["point"]["components"]["P"] *= 1.1
        f.write_text(json.dumps(r, indent=1))
        tampered_ok, tampered_out = audit(copy)
    sources = sorted({v["artifact"] for v in values})
    # computed results need a provenance record (agents/01_integrity.md); gates and data records are exempt
    no_prov = [s for s in sources if s.startswith("results/") and not s.startswith("results/gates/")
               and not (ROOT / (s + ".provenance.json")).exists()]
    passed = ok and base_ok and not tampered_ok and not no_prov
    gate = {"gate": "G6_traceability", "stage": "06_summary", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "check": "every number in docs/summary.md re-derives from its artifact (scripts/audit.py --traceability)",
            "reference_value": None, "reference_source": "docs/summary_values.json, docs/artifact_index.md",
            "observed_value": {"values": len(values), "audit_output": out.strip().splitlines(),
                               "negative_control": {"tampering": "H1 s2_P edited by +10 % in a copy",
                                                    "untampered_passes": base_ok, "caught": base_ok and not tampered_ok,
                                                    "output": tampered_out.strip().splitlines()[:4]},
                               "source_artifacts": sources, "sources_without_provenance": no_prov},
            "tolerance": TOLERANCE, "verdict": "PASS" if passed else "FAIL",
            "evidence": ["docs/summary.md", "docs/summary_values.json", "docs/artifact_index.md", "scripts/build_summary.py",
                         "scripts/audit.py", "scripts/check_traceability.py"],
            "notes": "The 30-dataset values derive from the WHAR Arena paper's published single runs (quoted numbers, "
                     "docs/baselines_cited.json) through results/metrics/factors/data.json."}
    GATE.write_text(json.dumps(gate, indent=1) + "\n")
    print("G6", gate["verdict"], "values", len(values), "control caught", not tampered_ok, "no provenance", no_prov)


if __name__ == "__main__":
    main()
