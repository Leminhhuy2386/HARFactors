"""Write the configuration of every network and of the protocol as files (configs/), from the code that built the runs.

- configs/models/<model>.json: the ten population networks (S1, S2): registry class and keyword arguments
  (src/models/registry.CONFIGS, the same dict every checkpoint and run record carries), where its checkpoints and run
  records are, and its parameter count with the reference input of gate G1;
- configs/models/whar/<model>.json: the 17 WHAR Arena models of S3, built by the pinned whar-models code with its
  defaults (no keyword arguments); run records only;
- configs/protocol_whar_arena_v1.yaml: the protocol file of the pinned whar-arena commit (copied, unchanged), with its
  hash, which every run record carries.

    uv run --no-sync python scripts/export_configs.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import POPULATION, S1, S2, S3  # noqa: E402
from src.models.registry import CONFIGS  # noqa: E402
from src.training import run_whar  # noqa: E402, F401  puts the pinned whar-arena on sys.path
from whar_arena.paths import PROTOCOL_PATH  # noqa: E402
from whar_arena.protocol import load_protocol  # noqa: E402

OUT = ROOT / "configs"


def main() -> None:
    anat = json.loads((ROOT / "results/metrics/model_anatomy/data.json").read_text())["structure"]
    (OUT / "models" / "whar").mkdir(parents=True, exist_ok=True)
    for m in POPULATION:
        cls, kwargs = CONFIGS[m]
        sets = [n for n, s in (("S1", S1), ("S2", S2)) if m in s["models"]]
        rec = {"model_id": m, "class": f"{cls.__module__}.{cls.__name__}", "config": kwargs,
               "built_by": "src.models.registry.build(model_id, input_channels=C, window_length=L, num_classes=K, "
                           "channel_names=...)",
               "sets": sets, "datasets": sorted({d for n, s in (("S1", S1), ("S2", S2)) if n in sets
                                                 for d in s["datasets"]}),
               "seeds": S1["seeds"],
               "checkpoints": f"experiments/checkpoints/{m}/<dataset>/seed-<s>/split-XX.pt (keys: model_id, config, "
                              "fingerprint, seed, macro_f1, device, state_dict)",
               "run_records": f"experiments/whar_arena_v1/{m}/<dataset>/[seed-<s>/]split-XX.json",
               "reference_input_gate_G1": {"input_channels": 6, "window_length": 150, "num_classes": 27,
                                           "params": anat[m]["params"], "embedding_dim": anat[m]["embedding_dim"]}}
        (OUT / "models" / f"{m}.json").write_text(json.dumps(rec, indent=1) + "\n")
    for m in S3["models"]:
        rec = {"model_id": m, "built_by": "whar_models.build_model(WHARModelID(model_id), input_channels=C, "
                                          "window_length=L, num_classes=K) at whar-models 9b65004 (defaults)",
               "sets": ["S3"], "datasets": list(S3["datasets"]), "seeds": S3["seeds"],
               "run_records": f"experiments/external_har_paper_era/whar_arena_v1/{m}/<dataset>/split-XX.json",
               "checkpoints": None, "note": "re-run by the HAR reproduction; run records only"}
        (OUT / "models" / "whar" / f"{m}.json").write_text(json.dumps(rec, indent=1) + "\n")
    shutil.copy2(PROTOCOL_PATH, OUT / "protocol_whar_arena_v1.yaml")
    p = load_protocol()
    (OUT / "protocol_whar_arena_v1.json").write_text(json.dumps(
        {"id": p.id, "digest": p.digest, "source": "whar-arena 19b3019, evaluation/protocol.yaml",
         "sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()}, indent=1) + "\n")
    print(f"wrote {len(POPULATION)} + {len(S3['models'])} model configs and the protocol (digest {p.digest})")


if __name__ == "__main__":
    main()
