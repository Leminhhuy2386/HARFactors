"""Gate G7 (agents/07_paper.md): the manuscript compiles, its citations and cross-references resolve, every bib entry
is verified, and every number it prints is in docs/summary_values.json (scripts/audit.py --manuscript).

Also compiles paper/manuscript.tex in place (latexmk) so that paper/manuscript.pdf is the checked document, and
records the exit code, the page count and the log's warnings.

    uv run --no-sync python scripts/check_manuscript.py   ->   results/gates/G7_compile_citations[_rerunK].json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "results" / "gates"
TOLERANCE = ("latexmk exit 0; zero undefined references and citations; every \\cite in references.bib; every bib entry "
             "verified with a matching title; every number in summary_values.json or a declared design constant; "
             "the format limits paper/format_limits.json enforces (IEEE Sensors Letters: 4 pages, 5 figures, abstract "
             "150 words; page and figure limits waived for the working draft by the user's decision of 2026-10-16)")


def main() -> None:
    reason = sys.argv[1] if len(sys.argv) > 1 else None
    out = GATES / "G7_compile_citations.json"
    if out.exists():
        if not reason:
            raise SystemExit("G7 exists: a re-run needs a reason (first argument)")
        k = 1
        while (GATES / f"G7_compile_citations_rerun{k}.json").exists():
            k += 1
        out = GATES / f"G7_compile_citations_rerun{k}.json"
    p = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"],
                       cwd=ROOT / "paper", capture_output=True, text=True)
    log = (ROOT / "paper/manuscript.log").read_text(errors="replace")
    pages = re.search(r"Output written on manuscript\.pdf \((\d+) page", log)
    warnings = [l for l in log.splitlines() if "Warning" in l]
    a = subprocess.run([sys.executable, "scripts/audit.py", "--manuscript", "--max-errors", "100"], cwd=ROOT,
                       capture_output=True, text=True)
    problems = [l.strip() for l in a.stdout.splitlines()[1:] if l.strip()]
    tex = (ROOT / "paper/manuscript.tex").read_text()
    m = re.search(r"\\begin\{abstract\}(?:\[[^\]]*\])?(.*?)\\end\{abstract\}", tex, re.S)
    words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\,]", " ", m.group(1)).replace("--", " ").split()) if m else None
    figures = len(re.findall(r"\\begin\{figure\*?\}", tex))
    lim_f = ROOT / "paper/format_limits.json"
    limits = json.loads(lim_f.read_text())["enforced"] if lim_f.exists() else {"pages": 4, "figures": 5,
                                                                                "abstract_words": 150}
    if words is None or words > limits["abstract_words"]:
        problems.append(f"abstract has {words} words (limit {limits['abstract_words']})")
    if limits["figures"] is not None and figures > limits["figures"]:
        problems.append(f"{figures} figures (limit {limits['figures']})")
    gate = {"gate": "G7_compile_citations", "stage": "07_paper", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "check": "the manuscript compiles; citations, references and numbers check out",
            "reference_value": None, "reference_source": "paper/references_verified.json, docs/summary_values.json",
            "observed_value": {"latexmk_exit_code": p.returncode, "pages": int(pages.group(1)) if pages else None,
                               "latex_warnings": warnings[:30], "audit_manuscript_exit": a.returncode,
                               "audit_problems": problems, "abstract_words": words, "figures": figures,
                               "format_limits_enforced": limits,
                               "class": re.search(r"\\documentclass(\[[^\]]*\])?\{([^}]+)\}", tex).group(2)},
            "tolerance": TOLERANCE,
            "verdict": "PASS" if p.returncode == 0 and a.returncode == 0 and not problems else "FAIL",
            "evidence": ["paper/manuscript.tex", "paper/manuscript.pdf", "paper/manuscript.log", "paper/references.bib",
                         "paper/format_limits.json",
                         "paper/references_verified.json", "scripts/audit.py", "scripts/check_manuscript.py"],
            "notes": "IEEE_lsens.cls from the IEEE Template Selector (2023 author templates) since rerun 3; IEEEtran "
                     "[journal,9pt] before."}
    if reason:
        gate["rerun_reason"] = reason
    out.write_text(json.dumps(gate, indent=1) + "\n")
    print(out.name, gate["verdict"], "pages", gate["observed_value"]["pages"], "problems", len(problems))
    print("\n".join(problems[:40]))


if __name__ == "__main__":
    main()
