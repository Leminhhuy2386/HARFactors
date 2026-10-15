#!/usr/bin/env bash
# One-time setup of this project on the HPC login node (downloads and installs only; no training here):
#   bash scripts/hpc/setup.sh
# 1. uv (from PyPI into ~/.local/bin) and a managed Python 3.12;
# 2. the locked environment (.venv) from uv.lock, except the packages staged as local wheels in $WHEEL_DIR;
# 3. those packages from the local wheels, each of whose sha256 must equal the one uv.lock records for its file;
# 4. pristine checkouts of the pinned WHAR code in third_party/ (files copied from Windows carry CRLF line endings,
#    which git on Linux reports as local changes, and run_whar.py refuses modified WHAR code);
# 5. a check that the paper-era guard passes and the data are where run_whar.py looks.
#
# Why local wheels: uv.lock takes torch 2.14.0+cu126 from download-r2.pytorch.org, which the site's Sophos firewall
# blocks ("Blocked site" page instead of the wheel; checked 2026-10-15), and the nvidia-*/triton wheels (about 2.4 GB)
# come from files.pythonhosted.org at about 0.5 MB/s from the cluster. They are downloaded on the PC, checked against
# uv.lock, and copied to $WHEEL_DIR. Because the venv then differs from what uv would fetch, env.sh sets
# UV_NO_SYNC=1: `uv run` uses the environment as installed and never re-syncs.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/hpc/env.sh
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
WHEEL_DIR=${WHEEL_DIR:-/work/huy.leminh/.cache/whl}

if ! command -v uv >/dev/null 2>&1; then
    /opt/hpc/anaconda3/bin/python -m pip install --user --quiet uv
fi
uv --version
uv python install 3.12
PY=$(uv python find 3.12)

# every local wheel must be the file uv.lock names, with the same sha256; prints the package names
LOCAL=$("$PY" -I - "$WHEEL_DIR" <<'EOF'
import hashlib, sys, tomllib
from pathlib import Path
from urllib.parse import unquote
lock = tomllib.load(open("uv.lock", "rb"))
want = {unquote(w["url"].rsplit("/", 1)[1]): (p["name"], w["hash"].split(":", 1)[1])
        for p in lock["package"] for w in p.get("wheels", [])}
for f in sorted(Path(sys.argv[1]).glob("*.whl")):
    name, sha = want.get(f.name, (None, None))
    if name is None:
        sys.exit(f"{f.name}: not in uv.lock")
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        sys.exit(f"{f.name}: sha256 differs from uv.lock")
    print(name)
EOF
)
echo "local wheels match uv.lock:" $LOCAL
EXCLUDE=()
for n in $LOCAL; do EXCLUDE+=(--no-install-package "$n"); done
uv sync --frozen "${EXCLUDE[@]}"
uv pip install --python .venv/bin/python --no-deps "$WHEEL_DIR"/*.whl

for r in third_party/whar-datasets third_party/whar-models third_party/whar-arena; do
    git -C "$r" reset --hard --quiet
    test -z "$(git -C "$r" status --porcelain --untracked-files=no)" || { echo "$r is not clean" >&2; exit 1; }
    echo "$r at $(git -C "$r" rev-parse --short=7 HEAD), clean"
done

uv run --no-sync python - <<'EOF'
import torch
from src.training import run_whar as R
from whar_arena.protocol import load_protocol
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("paper-era guard:", R.check_paper_era(load_protocol()))
print("datasets dir:", R.DATASETS_DIR, "exists:", R.DATASETS_DIR.is_dir())
print("caches:", sorted(p.name for p in R.DATASETS_DIR.iterdir() if p.is_dir()))
EOF
echo "setup done. Run work through Slurm: scripts/hpc/submit.sh (see docs/hpc.md)."
