"""Run one pre-registered variance-components analysis (docs/preregistration.md) and write its result with provenance.

The analyses are fixed in ANALYSES below; nothing else can be run. Those that read S2 (the confirmatory set) refuse
to run unless results/gates/G4_test_seal.json exists and the sealed files still have the sealed hashes, log the
read to results/test_access.log before reading, and refuse to run twice.

    uv run --no-sync python scripts/run_confirmatory.py H1 --workers 32
    uv run --no-sync python scripts/run_confirmatory.py sens2_S1_macro_f1 --dev 20 --workers 8   # development, S1/S3 only:
                                                                                                # nothing is written to results/
Outputs: results/metrics/confirmatory/<analysis>.{json,md} (+ .provenance.json), experiments/<run_id>/ (config, log),
and START/END lines in experiments/LEDGER.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sets import SETS, run_path  # noqa: E402
from src.evaluation import variance_components as V  # noqa: E402
from src.utils import provenance as PV  # noqa: E402
from src.utils.provenance import sha256  # noqa: E402

OUT = ROOT / "results" / "metrics" / "confirmatory"
SEAL = ROOT / "results" / "gates" / "G4_test_seal.json"
ACCESS_LOG = ROOT / "results" / "test_access.log"
BOOTSTRAP = 1000  # docs/preregistration.md
# name: (set, outcome, datasets excluded beyond the inclusion rule, role)
ANALYSES = {
    "H1": ("S2", "macro_f1", [], "confirmatory test of H1"),
    "sens1_S2_balanced_accuracy": ("S2", "balanced_accuracy", [], "sensitivity 1"),
    "sens2_S1_macro_f1": ("S1", "macro_f1", [], "sensitivity 2"),
    "sens2_S1_balanced_accuracy": ("S1", "balanced_accuracy", [], "sensitivity 2"),
    "sens3_S1_without_actrectut": ("S1", "macro_f1", ["actrectut_gestures"], "sensitivity 3"),
    "sens4_S3_macro_f1": ("S3", "macro_f1", [], "sensitivity 4"),
}


def check_seal() -> dict:
    if not SEAL.exists():
        raise SystemExit("refused: S2 is sealed and results/gates/G4_test_seal.json does not exist")
    seal = json.loads(SEAL.read_text())
    if seal.get("verdict") != "PASS":
        raise SystemExit("refused: G4 is not PASS")
    changed = [f for f, h in seal["observed_value"]["sealed_files"].items() if sha256(ROOT / f) != h]
    if changed:
        raise SystemExit(f"refused: sealed files changed since G4: {changed}")
    return seal


def log_access(run_id: str, purpose: str) -> None:
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | scripts/run_confirmatory.py | {run_id} | {purpose}\n")


def report_md(name: str, res: dict) -> str:
    p, b = res["point"], res["bootstrap"]
    lo, hi = b["R_interval"]
    plo, phi = b["pi_P_interval"]
    L = [f"# {name}: {res['role']}", "",
         f"Set {res['set']}, outcome {res['outcome']} (x 100), {res['n_runs']} runs, {len(res['datasets'])} datasets "
         f"({', '.join(res['datasets'])}), {res['n_splits']} splits. Excluded: "
         f"{', '.join(f'{d} ({why})' for d, why in res['excluded'].items()) or 'none'}.", "",
         f"REML ({p['method']} optimizer), variance components (F1 points squared):", "",
         "| Component | Estimate | 97.5 % interval |", "|---|---|---|"]
    for k, v in p["components"].items():
        ci = b["components_intervals"].get(k, [None, None])
        L.append(f"| {k} | {v:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |" if ci[0] is not None else f"| {k} | {v:.3f} | – |")
    L += ["", f"R = {p['R_definition']} = {p['R']:.3f}, 97.5 % interval [{lo:.3f}, {hi:.3f}]",
          f"pi_P = {p['pi_P']:.3f}, 97.5 % interval [{plo:.3f}, {phi:.3f}]",
          f"Bootstrap: {b['resamples']} resamples, {b['converged']} converged, {b['dropped']} dropped "
          f"({100 * b['dropped_share']:.1f} %), {b['refit_with_powell']} refitted with Powell; seed {b['seed']}.", ""]
    if res.get("decision"):
        L.append(f"**Pre-registered decision for H1: {res['decision']}.** Secondary criterion pi_P lower bound >= 0.40: "
                 f"{'met' if plo is not None and plo >= 0.40 else 'not met'}.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", choices=list(ANALYSES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dev", type=int, default=0, help="development run with this many resamples; S1/S3 only; "
                                                      "prints, writes nothing to results/")
    a = ap.parse_args()
    set_name, outcome, extra_excluded, role = ANALYSES[a.analysis]
    spec = SETS[set_name]
    if a.dev and set_name == "S2":
        raise SystemExit("refused: no development runs on S2")
    run_id = PV.new_run_id(f"vc_{a.analysis}" + ("_dev" if a.dev else ""))
    config = {"analysis": a.analysis, "set": set_name, "outcome": outcome, "extra_excluded": extra_excluded,
              "bootstrap": a.dev or BOOTSTRAP, "boot_seed": V.BOOT_SEED, "level": V.LEVEL, "role": role}
    out_json, out_md = OUT / f"{a.analysis}.json", OUT / f"{a.analysis}.md"
    if set_name == "S2":
        seal = check_seal()
        if out_json.exists():
            raise SystemExit(f"refused: {out_json.relative_to(ROOT)} exists; S2 is analysed once")
    started = time.time()
    run_dir = ROOT / "experiments" / run_id
    if not a.dev:
        run_dir.mkdir()
        (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
        PV.ledger_start(run_id, config)
    try:
        analyse(a, spec, set_name, outcome, extra_excluded, role, run_id, run_dir, config, started,
                seal if set_name == "S2" else None, out_json, out_md)
    except BaseException as e:
        if not a.dev:
            PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise


def analyse(a, spec, set_name, outcome, extra_excluded, role, run_id, run_dir, config, started, seal, out_json,
            out_md) -> None:
    inc = V.inclusion(spec)  # sizes and protocol hash only
    excluded = {d: "split sizes disagree across runs (pre-registered inclusion rule)" for d, v in inc.items()
                if not v["included"]}
    excluded |= {d: "pre-registered sensitivity subset" for d in extra_excluded}
    datasets = [d for d in spec["datasets"] if d not in excluded]
    if set_name == "S2":
        log_access(run_id, f"pre-registered {a.analysis} ({role}); sealed by G4 at {seal['timestamp']}")
    df = V.load(spec, outcome, datasets)
    has_seeds = len(spec["seeds"]) > 1
    point = V.fit(df, has_seeds)
    if not point["converged"]:
        raise SystemExit("point fit did not converge")
    boot = V.bootstrap(df, a.dev or BOOTSTRAP, has_seeds, workers=a.workers)
    res = {"analysis": a.analysis, "role": role, "set": set_name, "outcome": outcome, "run_id": run_id,
           "inclusion": inc, "excluded": excluded, "datasets": datasets, "n_runs": len(df),
           "n_splits": int(df.groupby("dataset")["split"].nunique().sum()), "point": point, "bootstrap": boot,
           "decision": V.decide(point, boot) if a.analysis == "H1" else None,
           "per_dataset_split_means_sd": {d: float(g.groupby("split")["y"].mean().std()) for d, g in df.groupby("dataset")}}
    if a.dev:
        print(json.dumps({k: res[k] for k in ("analysis", "n_runs", "datasets", "excluded", "point", "bootstrap")},
                         indent=1, default=str))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=float) + "\n")
    out_md.write_text(report_md(a.analysis, res))
    inputs = [run_path(spec, r.model, r.dataset, r.seed, r.split) for r in df.itertuples()]
    for f in (out_json, out_md):
        PV.write(f, run_id=run_id, started=started, inputs=inputs, config=config,
                 produced_by="scripts/run_confirmatory.py")
    (run_dir / "log.txt").write_text(report_md(a.analysis, res))
    PV.ledger_end(run_id, f"R={point['R']:.3f} [{boot['R_interval'][0]:.3f}, {boot['R_interval'][1]:.3f}]"
                          + (f" decision={res['decision']}" if res["decision"] else ""), started)
    print(report_md(a.analysis, res))


if __name__ == "__main__":
    main()
