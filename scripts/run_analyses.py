"""Recompute every cross-run aggregate in results/metrics/ from the stored runs, in dependency order. No training,
no inference: each step reads experiments/ (and docs/sources/) and writes results/metrics/<analysis>/.

    uv run python scripts/run_analyses.py [--only synthesis factor_analysis]
"""
import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = ["src.evaluation.compare_arena", "src.evaluation.synthesis", "src.evaluation.factor_analysis", "src.evaluation.heldout_analysis", "src.evaluation.vs_benchmark", "src.evaluation.seeds_summary", "src.evaluation.p5_summary"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for mod in STEPS:
        if a.only and mod.rsplit(".", 1)[-1] not in a.only:
            continue
        print("==", mod, flush=True)
        sys.argv = [mod]
        runpy.run_module(mod, run_name="__main__")
