"""Reproducibility inventory of every stored run of S1 and S2 (verification; no training, no inference).

Per run (model, dataset, seed, split):
- the run record exists, and its checkpoint exists;
- the checkpoint's config equals the registry's (src/models/registry.CONFIGS) and the run record's, and its fingerprint
  equals the record's;
- the latest re-execution from the checkpoint (scripts/evaluate.py runs under experiments/, newest first per cell):
  exact (macro-F1 and accuracy equal the record), within the G5 tolerance (equal test size, at most 2 windows and
  2 F1 points apart), or outside, and on which cache it ran.
S3 (the 17 WHAR Arena models re-run by the HAR reproduction) has run records only; it is counted, not re-executed.

    uv run --no-sync python -m src.evaluation.reproducibility
        -> results/metrics/reproducibility/summary.json, docs/reproducibility_report.md
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import torch

from src.data.sets import S1, S2, S3, cells, run_path
from src.models.registry import CONFIGS
from src.utils import paths as PATHS
from src.utils import provenance as PV

OUT = PATHS.METRICS / "reproducibility"
REPORT = PATHS.DOCS / "reproducibility_report.md"
# evaluate runs on a cache whose order fits only part of their cells (several VMs): their cells use the per-checkpoint
# re-execution instead (scripts/reconstruct_order.py reexecute)
SUPERSEDED = ("_evaluate_dsads_colab_order",)
# runs from VMs whose listing order varies within their run set: per-checkpoint re-execution
PER_CHECKPOINT = {"daphnet": lambda dev: True, "motion_sense": lambda dev: True, "dsads": lambda dev: dev == "NVIDIA L4"}
EXCLUDED = {"daphnet": "S1", "motion_sense": "S2"}  # pre-registered inclusion rule


def within(r: dict) -> bool:
    if r["status"] != "evaluated" or r["test_size"] != r["stored_test_size"]:
        return False
    return round(abs(r["accuracy"] - r["stored_accuracy"]) * r["test_size"]) <= 2 and \
        abs(r["macro_f1"] - r["stored_macro_f1"]) * 100 <= 2


def latest_evaluations() -> tuple[dict, list]:
    """(model, dataset, seed, split) -> (evaluation record, run id, cache dir), the newest evaluate run first."""
    runs = sorted((d for d in PATHS.EXPERIMENTS.iterdir() if d.is_dir() and (d / "evaluated.json").exists()
                   and not d.name.endswith(SUPERSEDED)), key=lambda d: d.name, reverse=True)
    out, files = {}, []
    for d in runs:
        cfg = json.loads((d / "config.json").read_text())
        files.append(d / "evaluated.json")
        for r in json.loads((d / "evaluated.json").read_text()):
            key = (r["model"], r["dataset"], r["seed"], r["split"])
            if key not in out and r["status"] == "evaluated":
                out[key] = (r, d.name, cfg.get("datasets_dir"))
    return out, files


def main() -> None:
    started = time.time()
    run_id = PV.new_run_id("reproducibility")
    evals, eval_files = latest_evaluations()
    reexec = {}
    for d in PER_CHECKPOINT:
        f = OUT / f"reexecute_{d}.json"
        if f.exists():
            eval_files.append(f)
            for r in json.loads(f.read_text())["rows"]:
                reexec[(r["model"], d, r["seed"], r["split"])] = r
    rows, per = [], defaultdict(lambda: defaultdict(int))
    for set_name, spec in (("S1", S1), ("S2", S2)):
        for m, d, s, i in cells(spec):
            rec_f = run_path(spec, m, d, s, i)
            ck_f = PATHS.CHECKPOINTS / m / d / f"seed-{s}" / f"split-{i:02d}.pt"
            row = {"set": set_name, "model": m, "dataset": d, "seed": s, "split": i, "record": rec_f.exists(),
                   "checkpoint": ck_f.exists()}
            rec = json.loads(rec_f.read_text())
            if ck_f.exists():
                ck = torch.load(ck_f, map_location="cpu", weights_only=False)
                row["config_matches_registry"] = ck.get("config") == CONFIGS[m][1]
                row["config_matches_record"] = ck.get("config") == rec["model"]["config"]
                row["fingerprint_matches_record"] = ck.get("fingerprint") == rec.get("fingerprint")
            e = evals.get((m, d, s, i))
            pc = reexec.get((m, d, s, i)) if d in PER_CHECKPOINT and PER_CHECKPOINT[d](rec.get("device")) else None
            row["excluded_from_analyses"] = EXCLUDED.get(d) == set_name
            if pc:
                row.update({"reexecuted": True, "method": "per-checkpoint", "evaluate_run": f"reexecute_{d}.json",
                            "cache": None, "exact": pc["macro_f1"] == pc["stored_macro_f1"]
                            and pc["accuracy"] == pc["stored_accuracy"], "within_tolerance": pc["within_tolerance"],
                            "test_size_equal": pc["test_size_equal"],
                            "delta_macro_f1_points": pc["delta_macro_f1_points"], "delta_windows": pc["delta_windows"]})
            elif e and not (d in PER_CHECKPOINT and PER_CHECKPOINT[d](rec.get("device"))):
                r, rid, cache = e
                row.update({"reexecuted": True, "method": "cache", "evaluate_run": rid, "cache": cache,
                            "exact": r["macro_f1"] == r["stored_macro_f1"] and r["accuracy"] == r["stored_accuracy"],
                            "within_tolerance": within(r),
                            "delta_macro_f1_points": 100 * (r["macro_f1"] - r["stored_macro_f1"]),
                            "delta_windows": round(abs(r["accuracy"] - r["stored_accuracy"]) * r["test_size"])})
            else:
                row["reexecuted"] = False
            rows.append(row)
            p = per[(set_name, d)]
            p["runs"] += 1
            p["checkpoint"] += row["checkpoint"]
            p["config_ok"] += bool(row.get("config_matches_registry") and row.get("config_matches_record")
                                   and row.get("fingerprint_matches_record"))
            p["reexecuted"] += row["reexecuted"]
            p["per_checkpoint"] += row.get("method") == "per-checkpoint"
            p["excluded_from_analyses"] = int(row["excluded_from_analyses"])
            p["exact"] += row.get("exact", False)
            p["within_tolerance"] += row.get("within_tolerance", False)
            p["max_abs_delta_macro_f1"] = max(p["max_abs_delta_macro_f1"], abs(row.get("delta_macro_f1_points", 0.0)))
            p["max_delta_windows"] = max(p["max_delta_windows"], row.get("delta_windows", 0))
    s3 = sum(run_path(S3, *c).exists() for c in cells(S3))
    per_ds = {f"{k[0]}/{k[1]}": dict(v) for k, v in per.items()}
    total = {k: sum(v[k] for v in per.values()) for k in ("runs", "checkpoint", "config_ok", "reexecuted", "exact",
                                                           "within_tolerance")}
    summary = {"total_S1_S2": total, "per_dataset": per_ds,
               "S3": {"runs": len(cells(S3)), "records": s3, "checkpoints": 0,
                      "note": "17 WHAR Arena models re-run by the HAR reproduction: run records only"},
               "tolerance": "equal test size, at most 2 test windows and 2 macro-F1 points from the record",
               "evaluate_runs": sorted({r["evaluate_run"] for r in rows if r["reexecuted"]}), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "summary.json"
    analysed = [r for r in rows if not r["excluded_from_analyses"]]
    summary["analysed_runs"] = {"runs": len(analysed), "checkpoint": sum(r["checkpoint"] for r in analysed),
                                "reexecuted": sum(r["reexecuted"] for r in analysed),
                                "exact": sum(r.get("exact", False) for r in analysed),
                                "within_tolerance": sum(r.get("within_tolerance", False) for r in analysed),
                                "max_abs_delta_macro_f1": max(abs(r.get("delta_macro_f1_points", 0.0)) for r in analysed),
                                "median_abs_delta_macro_f1_outside": float(sorted(
                                    abs(r["delta_macro_f1_points"]) for r in analysed
                                    if r["reexecuted"] and not r["within_tolerance"])[len([1 for r in analysed if
                                    r["reexecuted"] and not r["within_tolerance"]]) // 2]) if any(
                                    r["reexecuted"] and not r["within_tolerance"] for r in analysed) else 0.0}
    rt_f = OUT / "retrained.json"
    if rt_f.exists():  # the runs without a checkpoint, retrained for one (scripts/retrain_missing.py)
        rt = json.loads(rt_f.read_text())
        summary["retrained"] = {"runs": len(rt["rows"]), "run_id": rt["run_id"],
                                "exact": sum(r["macro_f1"] == r["stored_macro_f1"] for r in rt["rows"]),
                                "max_abs_delta_macro_f1": max(abs(r["delta_macro_f1_points"]) for r in rt["rows"]),
                                "config_and_fingerprint_equal": sum(r["config_equal"] and r["fingerprint_equal"]
                                                                    for r in rt["rows"])}
        eval_files.append(rt_f)
    f.write_text(json.dumps(summary, indent=1) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=eval_files, produced_by="src/evaluation/reproducibility.py")
    lines = ["# Reproducibility of the stored runs", "",
             f"Of {total['runs']} stored runs of S1 and S2, {total['checkpoint']} have a checkpoint, "
             f"{total['config_ok']} of those carry the registry's config and the record's config and fingerprint, "
             f"{total['reexecuted']} were re-executed from their checkpoint, {total['exact']} reproduce their record "
             f"exactly and {total['within_tolerance']} within the G5 tolerance ({summary['tolerance']}). S3: "
             f"{s3} run records, no checkpoints.", "",
             "| Set/dataset | runs | checkpoint | config ok | re-executed | exact | within tol. | max abs dF1 | max "
             "d windows |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(per_ds.items()):
        lines.append(f"| {k} | {v['runs']} | {v['checkpoint']} | {v['config_ok']} | {v['reexecuted']} | {v['exact']} | "
                     f"{v['within_tolerance']} | {v['max_abs_delta_macro_f1']:.2f} | {v['max_delta_windows']} |")
    if "retrained" in summary:
        r = summary["retrained"]
        lines += ["", f"The {r['runs']} runs without a checkpoint were retrained for one (run {r['run_id']}, "
                      f"results/metrics/reproducibility/retrained.json; outputs under that run's directory, the stored "
                      f"records untouched): {r['exact']} reproduce their record's macro-F1 exactly, the largest difference "
                      f"is {r['max_abs_delta_macro_f1']:.3f} points, and {r['config_and_fingerprint_equal']} carry the "
                      f"record's config and fingerprint."]
    lines += ["", "Excluded from the analyses by the pre-registered inclusion rule: S1/daphnet, S2/motion_sense. Methods:"
              " 'cache' = scripts/evaluate.py on a cache in the order the run was built with (inferred from the "
              "checkpoints where the parser lists files unsorted); 'per-checkpoint' = scripts/reconstruct_order.py "
              "reexecute (test subjects inferred per checkpoint, z-score on all non-test windows).", "",
              f"Re-execution runs: {', '.join(summary['evaluate_runs'])}.", ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
