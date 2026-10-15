#!/usr/bin/env bash
# Smoke test on a compute node (never on the login node):
#   scripts/hpc/submit.sh smoke normal 00:30:00 -- bash scripts/hpc/smoke.sh      (CPU)
#   scripts/hpc/submit.sh smoke gpu    00:30:00 -- bash scripts/hpc/smoke.sh      (GPU)
# Checks torch and the device, one test file, that the analyses regenerate results/metrics, and a 1-epoch training
# whose results go to a temporary folder (nothing is added to experiments/).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
uv run --no-sync python - <<'EOF'
import torch
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev, "| op", (torch.ones(2, 2, device="cuda" if torch.cuda.is_available() else "cpu") @ torch.ones(2, 2, device="cuda" if torch.cuda.is_available() else "cpu")).sum().item())
EOF
uv run --no-sync python tests/test_round8.py
uv run --no-sync python scripts/run_analyses.py
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/smoke_XXXXXX")
RESULTS_DIR=$SCRATCH uv run --no-sync python scripts/train.py --models lawnet_rec --datasets utd_mhad --splits 0 --max-epochs 1
python3 - "$SCRATCH" <<'EOF'
import json, sys, glob
for f in glob.glob(sys.argv[1] + "/whar_arena_v1/*/*/split-00.json"):
    r = json.load(open(f))
    print("smoke run:", r["model_id"], r["dataset_id"], "sizes", r["source"]["train_size"], r["source"]["val_size"],
          r["source"]["test_size"], "macro_f1", round(r["metrics"]["macro_f1"], 4), "device", r["device"])
EOF
rm -rf "$SCRATCH"
echo "SMOKE OK"
