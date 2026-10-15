# Source this on the HPC (bash) before any project command:   source scripts/hpc/env.sh
# Paths and tools for this project on the cluster (login: hpc-gw, user huy.leminh). See docs/hpc.md.

# Shared WHAR data, the same layout as D:\Projects\HAR on the PC:
#   $HAR_DIR/datasets_paper_era/<id>   paper-era caches (what run_whar.py reads)
#   $HAR_DIR/datasets/<id>/data        raw files (only scripts/prepare_data.py reads them)
export HAR_DIR=/work/huy.leminh/code/Dr.Huy/HAR

# uv: binary in ~/.local/bin; cache and managed Pythons on /work, the same file system as the project's .venv, so
# environments are hard-linked from the cache instead of copied.
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/work/huy.leminh/.cache/uv
export UV_PYTHON_INSTALL_DIR=/work/huy.leminh/.local/share/uv/python
export UV_FROZEN=1          # install exactly uv.lock, never re-resolve on the cluster
export UV_NO_SYNC=1         # torch is installed from a checked local wheel (setup.sh): never re-sync

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
