"""Render paper/manuscript.tex from paper/manuscript_src.tex (agents/07_paper.md).

Every \\val{<id>} marker is replaced by the string docs/summary_values.json prints for <id>; an unknown id stops the
render. The letter's figures and tables are copied from figures/ and tables/ into paper/ (never redrawn here).

    uv run --no-sync python scripts/render_manuscript.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGS = ["fig01_variance_components.pdf", "fig02_channels.pdf", "fig03_design.pdf", "fig04_sensors.pdf",
        "fig05_architecture.pdf", "fig07_split_spread.pdf", "fig08_embedding.pdf", "fig09_validation.pdf",
        "fig10_population_budget.pdf"]
TABLES = ["tab01_variance_components.tex", "tab02_datasets.tex", "tab03_models.tex", "tab04_error_budget.tex"]


def main() -> None:
    values = {v["id"]: v["printed"] for v in json.loads((ROOT / "docs/summary_values.json").read_text())}
    src = (PAPER / "manuscript_src.tex").read_text()
    missing = sorted({m for m in re.findall(r"\\val\{([^}]+)\}", src) if m not in values})
    if missing:
        sys.exit(f"unknown value ids: {missing}")
    out = re.sub(r"\\val\{([^}]+)\}", lambda m: values[m.group(1)], src)
    out = out.replace("% Source of paper/manuscript.tex.", "% Rendered from paper/manuscript_src.tex by scripts/render_manuscript.py; do not edit.\n% Source note:", 1)
    (PAPER / "manuscript.tex").write_text(out)
    (PAPER / "figures").mkdir(exist_ok=True)
    for f in FIGS:
        shutil.copy2(ROOT / "figures" / f, PAPER / "figures" / f)
    for f in TABLES:
        shutil.copy2(ROOT / "tables" / f, PAPER / f)
    print("rendered paper/manuscript.tex with", len(set(re.findall(r"\\val\{([^}]+)\}", src))), "values")


if __name__ == "__main__":
    main()
