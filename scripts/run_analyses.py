"""Recompute every cross-run aggregate in results/metrics/ from the stored runs, in dependency order. No training,
no inference: each step reads experiments/ (and docs/sources/) and writes results/metrics/<analysis>/.

Each invocation is a run: experiments/<run_id>/ (config, log), START and END lines in experiments/LEDGER.md, and a
provenance record next to every file a step writes. The inputs are pinned by data/processed/manifest.json (its
data_manifest_sha256 covers every run file and the quoted table) and the step's source file.

    uv run python scripts/run_analyses.py [--only synthesis factor_analysis] [--no-record]
"""
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (module, arguments), in dependency order: compare_arena reads profile.json at import, vs_benchmark reads
# pareto/costs.json at import
STEPS = [("src.evaluation.profile_models", []), ("src.evaluation.compare_arena", []), ("src.evaluation.synthesis", []),
         ("src.evaluation.factor_analysis", []), ("src.evaluation.heldout_analysis", []),
         ("src.evaluation.pareto", []), ("src.evaluation.pareto", ["--heldout"]), ("src.evaluation.vs_benchmark", []),
         ("src.evaluation.seeds_summary", []), ("src.evaluation.p5_summary", []),
         ("src.evaluation.p5_summary", ["recurrencenet"]), ("src.evaluation.contrasts_robustness", []),
         ("src.evaluation.split_spread", []), ("src.evaluation.validation_reliability", []),
         ("src.evaluation.factor_multiplicity", []), ("src.evaluation.error_budget", [])]
# results a step reads besides the stored runs (they enter its provenance record)
EXTRA_INPUTS = {"src.evaluation.vs_benchmark": ["results/metrics/pareto/costs.json"],
                "src.evaluation.factor_multiplicity": ["results/metrics/factors/data.json"],
                "src.evaluation.error_budget": ["results/metrics/confirmatory/H1.json",
                                                "results/metrics/confirmatory/sens2_S1_macro_f1.json"]}


def written_since(t0: float) -> list[Path]:
    m = ROOT / "results" / "metrics"
    return sorted(p for p in m.rglob("*") if p.is_file() and not p.name.endswith(".provenance.json")
                  and p.stat().st_mtime >= t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-record", action="store_true", help="no run directory, ledger line or provenance (checks only)")
    a = ap.parse_args()
    steps = [s for s in STEPS if not a.only or s[0].rsplit(".", 1)[-1] in a.only]
    record = not a.no_record
    if record:
        from src.utils import provenance as PV
        run_id = PV.new_run_id("run_analyses")
        config = {"steps": [" ".join([m, *args]) for m, args in steps]}
        run_dir = ROOT / "experiments" / run_id
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
        started = time.time()
        PV.ledger_start(run_id, config)
    log = []
    try:
        for mod, args in steps:
            print("==", mod, *args, flush=True)
            t0 = time.time() - 1
            sys.argv = [mod, *args]
            try:
                runpy.run_module(mod, run_name="__main__")
            except SystemExit as e:  # some steps end with sys.exit(main()); only a non-zero code is a failure
                if e.code not in (0, None):
                    raise
            for name in [k for k in sys.modules if k.startswith("src.evaluation")]:
                del sys.modules[name]  # the next step re-imports and so reads what this step wrote
            if record:
                src = ROOT / (mod.replace(".", "/") + ".py")
                inputs = [ROOT / "data/processed/manifest.json", src]
                if mod != "src.evaluation.profile_models":  # every later step reads params via compare_arena.PROFILE
                    inputs.append(ROOT / "results/metrics/profile.json")
                inputs += [ROOT / p for p in EXTRA_INPUTS.get(mod, [])]
                for f in written_since(t0):
                    PV.write(f, run_id=run_id, started=started, config=config, produced_by=f"scripts/run_analyses.py ({mod}"
                             + (" " + " ".join(args) if args else "") + ")", inputs=inputs)
                    log.append(f"{mod}: {f.relative_to(ROOT)}")
    except BaseException as e:
        if record:
            PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    if record:
        (run_dir / "log.txt").write_text("\n".join(log) + "\n")
        PV.ledger_end(run_id, f"{len(log)} files", started)
