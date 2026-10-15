"""Train and evaluate models under the WHAR Arena v1 protocol with the paper-era harness
(src/training/run_whar.py: splits, training loop and metric are the harness's own). One result json per
model / dataset / [seed-s/] split goes to experiments/whar_arena_v1/, weights to experiments/checkpoints/.

    uv run python scripts/train.py --models lawnet_rec --datasets utd_mhad [--splits 0 1] [--seed 1] [--max-epochs 2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.run_whar import main  # noqa: E402

if __name__ == "__main__":
    main()
