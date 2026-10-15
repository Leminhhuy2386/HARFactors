"""Every project location in one place (agents/00_project_structure.md)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"            # per-run outputs (never aggregates)
RUNS = EXPERIMENTS / "whar_arena_v1"          # one json per model / dataset / [seed-s/] split, as run_whar.py writes
CHECKPOINTS = EXPERIMENTS / "checkpoints"     # <model>/<dataset>/seed-<s>/split-XX.pt
EXTERNAL_RUNS = EXPERIMENTS / "external_har_paper_era" / "whar_arena_v1"  # the 17 paper models, re-run by HAR
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"                 # cross-run aggregates, one folder per analysis
GATES = RESULTS / "gates"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
DOCS = ROOT / "docs"
SOURCES = DOCS / "sources"                    # external source files (the paper's Figure 6 numbers)
