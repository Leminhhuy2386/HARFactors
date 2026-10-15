"""The three sets of stored runs named in the pre-registration (docs/preregistration.md, § Data).

Paths only: nothing in this module reads a metric. S2 is the confirmatory set; its metrics are read once, after gate
G4 seals the analysis code.
"""
from __future__ import annotations

from pathlib import Path

from src.utils import paths as PATHS

POPULATION = ["recurrencenet", "s4d", "hystnet_sel_clock", "quantilenet", "mamba", "hystnet_nomem", "recstatnet",
              "lawnet_rec", "lawnet", "statnet"]
S1 = {  # development population: 10 models x 10 datasets x 82 splits x 3 seeds = 2,460 runs
    "root": PATHS.RUNS, "models": POPULATION, "seeds": [0, 1, 2],
    "datasets": {"utd_mhad": 8, "uci_har": 10, "unimib_shar": 10, "actrectut_gestures": 2, "opportunity": 4, "sad": 10,
                 "hapt": 10, "dsads": 8, "wisdm": 10, "daphnet": 10},
}
S2 = {  # held-out datasets: 4 models x 8 datasets x 79 splits x 3 seeds = 948 runs
    "root": PATHS.RUNS, "models": ["recurrencenet", "s4d", "recstatnet", "lawnet_rec"], "seeds": [0, 1, 2],
    "datasets": {"pamap2": 9, "up_fall": 10, "uma_fall": 10, "usc_had": 10, "motion_sense": 10, "wear": 10,
                 "uca_ehar": 10, "w_har": 10},
}
S3 = {  # the 17 WHAR Arena models re-run by the HAR reproduction: 17 x 4 datasets x 30 splits x 1 run = 510 runs
    "root": PATHS.EXTERNAL_RUNS, "seeds": [0],
    "models": ["attend_discriminate", "cnn_har", "dana", "deepconv_lstm", "deepconv_lstm_attention",
               "deepconv_lstm_iswc", "dynamic_whar", "knn", "lstms_ensemble", "mlp_har", "mlp_mixer", "random_forest",
               "sa_har", "svm", "tinierhar", "tinyhar", "triple_cross_domain_attention"],
    "datasets": {"utd_mhad": 8, "uci_har": 10, "unimib_shar": 10, "actrectut_gestures": 2},
}
SETS = {"S1": S1, "S2": S2, "S3": S3}


def run_path(spec: dict, model: str, dataset: str, seed: int, split: int) -> Path:
    """<root>/<model>/<dataset>/[seed-<s>/]split-XX.json, the layout run_whar.py writes (seed 0 at the top)."""
    base = spec["root"] / model / dataset
    return (base / f"seed-{seed}" if seed else base) / f"split-{split:02d}.json"


def cells(spec: dict) -> list[tuple[str, str, int, int]]:
    """Every (model, dataset, seed, split) of a set, in a fixed order."""
    return [(m, d, s, i) for m in spec["models"] for d, n in spec["datasets"].items() for s in spec["seeds"]
            for i in range(n)]


def run_files(spec: dict) -> list[Path]:
    return [run_path(spec, *c) for c in cells(spec)]
