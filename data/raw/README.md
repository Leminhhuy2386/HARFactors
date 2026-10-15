# data/raw: the raw downloads (shared, read-only, not copied here)

The raw files of the 30 WHAR datasets live in the HAR reproduction:
`D:\Projects\HAR\datasets\<id>\data` (`src/training/run_whar.py`: `RAW_DIR`; override the HAR folder with the
`HAR_DIR` environment variable). They are shared by every project that uses the WHAR protocol on this machine.

- Read-only. Never write, move, rename or delete anything there.
- The processed parts of `D:\Projects\HAR\datasets` (`sessions/`, `windows/`, ...) were built by the *latest* whar
  code and must never be read; `run_whar.py` refuses that directory as a cache.
- Sources: the URLs in the paper-era whar-datasets configs (`third_party/whar-datasets/src/whar_datasets/config/`);
  manual downloads (e.g. UCA-EHAR) were placed there by the HAR reproduction.

Nothing is stored in this folder on purpose; it exists so the layout of `agents/00_project_structure.md` holds.
