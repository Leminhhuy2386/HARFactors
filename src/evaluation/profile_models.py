"""Model cost profile: trainable parameters, FLOPs per window and single-thread CPU latency per window.

Replaces results/metrics/profile.json, which was imported from Edge_SSM (its profile_models.py) without a provenance
record. Same measurement as src/evaluation/pareto.py: torch FlopCounterMode (matmul / conv / bmm only), batch 1, one
thread, median of 15 timed calls after 3 warm-up calls. Input: a zero window of each dataset's real shape (the shape
recorded in the stored runs). The imported file's streaming fields (per-sample latency of streaming models) are not
re-measured: nothing in this project reads them.

Before overwriting, the parameter counts of the imported file are compared with the new ones; the comparison is
written next to the profile (profile_import_check.json). Only `params` is read downstream (compare_arena.PROFILE).

    uv run python -m src.evaluation.profile_models   ->   results/metrics/profile.json, results/metrics/profile_import_check.json
"""
from __future__ import annotations

import json
import statistics as st
import time
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.evaluation import compare_arena as CA
from src.models.registry import build, is_whar_model
from src.utils import paths as PATHS

DATASETS = ["uci_har", "utd_mhad", "unimib_shar", "actrectut_gestures"]
MODELS = ["attend_discriminate", "cnn_har", "dana", "deepconv_lstm", "deepconv_lstm_attention", "deepconv_lstm_iswc",
          "dynamic_whar", "lstms_ensemble", "mlp_har", "mlp_mixer", "sa_har", "tinierhar", "tinyhar",
          "triple_cross_domain_attention", "dualmem", "dualmem_path", "edge_ssm", "edge_ssm_ema", "edge_ssm_nophi",
          "edge_ssm_osc", "edge_ssm_s", "edge_ssm_tau_fixed", "edge_ssm_tau_input", "gru", "hystnet", "hystnet_clock",
          "hystnet_fixed_r", "hystnet_k6", "hystnet_mm", "hystnet_nomem", "hystnet_sel", "hystnet_sel_clock",
          "hystnet_sel_k6", "mamba", "pathquantilenet", "quantilenet", "s4d", "tropicalnet", "vortexnet"]
OUT = PATHS.METRICS / "profile.json"
CHECK = PATHS.METRICS / "profile_import_check.json"


def shape(dataset: str) -> dict:
    m = json.loads((CA.RES / "recurrencenet" / dataset / "split-00.json").read_text())["model"]
    return {"c": m["input_channels"], "l": m["window_length"], "k": m["num_classes"]}


def measure(net: torch.nn.Module, x: torch.Tensor, repeats: int = 15) -> tuple[int, int, float]:
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            net(x)
        for _ in range(3):
            net(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            net(x)
            times.append(time.perf_counter() - t0)
    return params, int(fc.get_total_flops()), round(1e3 * st.median(times), 3)


def main() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    old = {(r["dataset"], r["model_id"]): r["params"] for r in json.loads(OUT.read_text())} if OUT.exists() else {}
    rows = []
    for d in DATASETS:
        sh = shape(d)
        for m in MODELS:
            torch.manual_seed(0)
            net = build(m, input_channels=sh["c"], window_length=sh["l"], num_classes=sh["k"]).eval()
            p, f, ms = measure(net, torch.zeros(1, sh["c"], sh["l"]))
            rows.append({"dataset": d, "model_id": m, "params": p, "flops_counted": f, "latency_ms_cpu1": ms,
                         "whar_model": is_whar_model(m)})
    check = {"imported_rows": len(old), "new_rows": len(rows),
             "params_equal": sum(old.get((r["dataset"], r["model_id"])) == r["params"] for r in rows),
             "params_differ": [{"dataset": r["dataset"], "model": r["model_id"], "imported": old.get((r["dataset"], r["model_id"])),
                                "measured": r["params"]} for r in rows
                               if (r["dataset"], r["model_id"]) in old and old[(r["dataset"], r["model_id"])] != r["params"]],
             "missing_from_import": [f"{r['dataset']}/{r['model_id']}" for r in rows if (r["dataset"], r["model_id"]) not in old]}
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    CHECK.write_text(json.dumps(check, indent=1) + "\n")
    print(json.dumps({k: v for k, v in check.items()}, indent=1))


if __name__ == "__main__":
    main()
