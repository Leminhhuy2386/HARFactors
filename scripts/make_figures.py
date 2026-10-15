"""Figures and reports from saved results only: no training, no inference, no change to results/.
Figures go to figures/, HTML reports to docs/atlas/.

    uv run python scripts/make_figures.py
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = [('src.visualization.build_factor_figures', [])]

if __name__ == "__main__":
    for mod, args in STEPS:
        print("==", mod, *args, flush=True)
        sys.argv = [mod, *args]
        runpy.run_module(mod, run_name="__main__")
