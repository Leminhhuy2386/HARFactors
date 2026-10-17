"""Figures and reports from saved results only: no training, no inference, no change to results/.
Figures go to figures/ (letter: fig0x_*.pdf), tables to tables/, their numbers to results/figure_data/ and
results/table_data/ with provenance.

    uv run python scripts/make_figures.py
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = [('src.visualization.build_factor_figures', []), ('src.visualization.letter_figures', []),
         ('src.visualization.detail_figures', []),
         ('src.visualization.contribution_figures', [])]

if __name__ == "__main__":
    for mod, args in STEPS:
        print("==", mod, *args, flush=True)
        sys.argv = [mod, *args]
        runpy.run_module(mod, run_name="__main__")
