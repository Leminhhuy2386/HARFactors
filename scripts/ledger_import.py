"""Write the opening block of experiments/LEDGER.md: every run that already existed when this project was split off
from Edge_SSM (2026-10-14), one line per run, marked IMPORT. These runs were produced before the agents/ pipeline:
they carry the WHAR harness's own record (fingerprint, protocol hash, whar code commits, data directory, metrics),
but no <name>.provenance.json, no stored predictions and no START/END lines. Later runs append START/END lines below.

Runs only when LEDGER.md does not exist yet (the ledger is append-only).

    uv run python scripts/ledger_import.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import paths as P  # noqa: E402

LEDGER = P.EXPERIMENTS / "LEDGER.md"


def main() -> None:
    if LEDGER.exists():
        raise SystemExit(f"{LEDGER} exists; the ledger is append-only")
    lines = []
    for f in sorted(P.RUNS.rglob("split-*.json")):
        r = json.loads(f.read_text())
        rel = f.relative_to(P.EXPERIMENTS).as_posix()[:-5]
        seed = r.get("train_seed", 0)
        ck = P.CHECKPOINTS / r["model_id"] / r["dataset_id"] / f"seed-{seed}" / f"{r['split_id']}.pt"
        f1 = r.get("metrics", {}).get("macro_f1")
        lines.append(f"{rel} | IMPORT | fingerprint {r.get('fingerprint', '?')[:12]} | protocol {r.get('protocol_hash')} | "
                     f"data {r.get('datasets_dir')} | synthetic=false | macro_f1={f1} | "
                     f"checkpoint={'yes' if ck.exists() else 'no'} | status={r.get('status')}")
    other = Counter()
    for d in sorted(p for p in P.EXPERIMENTS.iterdir() if p.is_dir() and p.name not in ("whar_arena_v1", "checkpoints")):
        other[d.name] = sum(1 for x in d.rglob("*") if x.is_file())
    head = [
        "# Run ledger (append-only)",
        "",
        "Format of new runs (agents/01_integrity.md): `<run_id> | START | config <sha> | data <sha> | synthetic=<bool>`,",
        "then `<run_id> | END | <metric> | <duration> | status=<...>`. Never edit or delete a line.",
        "",
        "## Imported from Edge_SSM on 2026-10-14 (produced before this pipeline)",
        "",
        f"{len(lines)} WHAR-harness runs in `experiments/whar_arena_v1/` (one line each below). Each result json holds the",
        "harness record: fingerprint, protocol hash e8739ef2c5b36c99, whar code commits, data directory, split sizes,",
        "metrics. They have no provenance.json and no stored predictions; where `checkpoint=yes`, predictions can be",
        "regenerated (agents/05_verify.md). `data` is the cache directory at run time (cloud VM paths for cloud runs).",
        "",
        "Other imported folders (files): " + ", ".join(f"`{k}/` ({v})" for k, v in other.items()) + ".",
        "",
        "```",
        *lines,
        "```",
        "",
        "## New runs",
        "",
    ]
    LEDGER.write_text("\n".join(head), encoding="utf-8")
    print(f"wrote {LEDGER}: {len(lines)} imported runs")


if __name__ == "__main__":
    main()
