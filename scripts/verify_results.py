"""Gate G5 (agents/05_verify.md, adapted to HARFactors): prove by re-execution that the reported results are real.

Run as one ledger run. Steps:
 L1  existence: provenance of every result file (audit --provenance); figure and table data on disk.
 L2  well-formedness of every re-executed checkpoint: it loads strictly into the registry model; its parameter count
     equals the run record's; all weights finite; weights differ from a fresh initialisation with the run's seed;
     predictions neither constant nor equal to the truth; predictions of different seeds differ.
 L3  re-execution: test predictions regenerated from checkpoints (scripts/evaluate.py) for
       - every run of the confirmatory set S2 whose dataset has a cache here (up_fall, wear, uca_ehar: 360 of the
         828 runs H1 used), and
       - a seeded sample of 6 runs per re-executable development dataset of S1 (8 datasets), plus dsads (informative:
         its subject numbering may differ from the stored runs);
     metrics recomputed from the prediction files against the stored run records.
     Aggregates: H1 and every sensitivity analysis refitted by an independent REML (src/evaluation/reml_numpy.py).
     Figures and table regenerated into a temporary directory; their data compared with results/figure_data and
     results/table_data.
 L3.5 leakage: no test subject in train/val (scripts/export_splits.py asserts it for all 13 cached datasets); after
     the harness's normalisation every re-executed training split has per-channel mean 0 and sd 1 (a train-only fit);
     results/test_access.log audited against the G4 seal. The label-shuffle control needs training, which the user
     ruled out (2026-10-15): not run, and reported as not run.
 L5  negative controls: permuting model labels within every (dataset, split, seed) cell must remove the model share;
     tampered copies must fail the auditor.

Tolerances are fixed below, before measuring.

    WHAR_DATASETS_DIR=<private cache copy> uv run --no-sync python scripts/verify_results.py --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.sets import S1, S2, SETS  # noqa: E402
from src.evaluation import reml_numpy as RN  # noqa: E402
from src.evaluation import variance_components as V  # noqa: E402
from src.utils import provenance as PV  # noqa: E402

SEED = 20261015
# S2 datasets with a paper-era cache on the cluster; pamap2, uma_fall and w_har were built on the PC and copied on
# 2026-10-16 (user-approved). usc_had and motion_sense (unsorted parsers) are not re-executable here.
S2_REEXEC = ["up_fall", "wear", "uca_ehar", "pamap2", "uma_fall", "w_har"]
S1_REEXEC = ["utd_mhad", "uci_har", "unimib_shar", "actrectut_gestures", "opportunity", "sad", "hapt", "wisdm"]
INFORMATIVE = ["dsads"]
PER_DATASET = 6
TOL = {
    "reexecution": "a run matches when its test size equals the stored one, |delta accuracy| x test size <= 2 windows "
                   "and |delta macro-F1| <= 2 points (float32 inference on this CPU vs the original GPU); zero gating "
                   "runs may fail (dsads is informative)",
    "second_implementation": "every component within max(2 % of its value, 0.05) and R within 2 %",
    "normalisation": "every re-executed training split: per-channel |mean| <= 1e-3 and |sd - 1| <= 1e-2",
    "model_permutation": "with model labels permuted within (dataset, split, seed) cells, (s2_M + s2_MD + s2_MP) / "
                         "within-dataset variance <= 0.03",
    "negative_controls": "every tampered copy fails the check that guards it",
}
OUT_REPORT = ROOT / "docs" / "verification_report.md"
RERUN_REASON = (
    "Rerun 1, approved by the user on 2026-10-16. The first G5 FAILed for four reasons. (1) 7 imported files under "
    "results/metrics had no provenance; they are regenerated with provenance by their own scripts (profile.json by the "
    "new src/evaluation/profile_models.py). (2) One run differed from its record by exactly 2 windows, the declared "
    "limit, and the float32 comparison counted it as above the limit; the window difference is now counted as an "
    "integer, as the tolerance states. (3) The recompute negative control changed one wrong prediction into another "
    "wrong one in a degenerate file (dsads, constant predictions), so no metric changed and --recompute passed; metric "
    "records now carry the prediction file's sha256, --recompute checks it, and the control alters a correct "
    "prediction in a non-degenerate file. (4) The provenance control was void because of (1); it now also requires "
    "the deleted file to be named. The tolerances are unchanged. Rerun 3 (approved by the user on 2026-10-16): adds "
    "the label-shuffle control of Level 3.5.3 (scripts/label_shuffle_control.py), which had not been run; nothing "
    "else changes. Rerun 4 (approved by the user on 2026-10-16): pamap2, uma_fall and w_har received paper-era caches "
    "(built on the PC with scripts/prepare_data.py, split sizes checked, copied to the cluster), so their S2 runs are "
    "re-executed too; the set of re-executed cells grows accordingly and nothing else changes. Rerun 2: rerun 1 FAILed only because --recompute "
    "required the new hash field also in the 414 metric records of the first re-execution run "
    "(20261015_184242_evaluate_G5), written before the field existed, so even the untampered copy failed. Records "
    "without the field are now checked by recomputation only, records with it by hash and recomputation.")
GATE = ROOT / "results" / "gates" / "G5_results_authenticity.json"


RUNNING = {"id": ""}


def audit(*flags: str, root: Path = ROOT) -> tuple[bool, str]:
    import os
    env = os.environ | {"HARF_RUNNING_RUN_ID": RUNNING["id"]}
    p = subprocess.run([sys.executable, "scripts/audit.py", *flags, "--root", str(root)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    return p.returncode == 0, p.stdout + p.stderr


def log_s2_reread(purpose: str) -> None:
    with open(ROOT / "results" / "test_access.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | scripts/verify_results.py | {RUNNING['id']} | verification "
                f"(G5): re-read the S2 run records for {purpose}\n")


def select_cells() -> tuple[list, list]:
    rng = random.Random(SEED)
    gating = [(m, d, s, i) for m in S2["models"] for d in S2_REEXEC for s in S2["seeds"] for i in range(S2["datasets"][d])]
    informative = []
    for d in S1_REEXEC + INFORMATIVE:
        pool = [(m, d, s, i) for m in S1["models"] for s in S1["seeds"] for i in range(S1["datasets"][d])]
        pick = rng.sample(pool, PER_DATASET)
        (informative if d in INFORMATIVE else gating).extend(pick)
    return gating, informative


def run_matches(r: dict) -> bool:
    if r["status"] != "evaluated" or r["test_size"] != r["stored_test_size"]:
        return False
    windows = round(abs(r["accuracy"] - r["stored_accuracy"]) * r["test_size"])  # an integer count of windows
    return windows <= 2 and abs(r["macro_f1"] - r["stored_macro_f1"]) * 100 <= 2


def level2(r: dict) -> list[str]:
    bad = []
    if r["params"] != r["stored_params"]:
        bad.append("parameter count")
    if not r["finite"]:
        bad.append("non-finite weights")
    if not r["differs_from_init"]:
        bad.append("weights equal a fresh initialisation")
    if r["predicted_classes"] < 2:
        bad.append("constant predictions")
    if r["equal_to_truth"]:
        bad.append("predictions identical to the labels")
    if not r["checkpoint_fingerprint_matches"]:
        bad.append("checkpoint fingerprint differs from the run record")
    return bad


def seeds_differ(eval_dir: Path) -> dict:
    """Prediction files of different seeds of the same (model, dataset, split) must differ."""
    groups = {}
    for f in (eval_dir / "predictions").rglob("split-*.csv"):
        model, dataset, seed_dir = f.parts[-4], f.parts[-3], f.parts[-2]
        groups.setdefault((model, dataset, f.name), []).append(f.read_bytes())
    multi = [v for v in groups.values() if len(v) > 1]
    identical = sum(len(set(v)) < len(v) for v in multi)
    return {"groups_with_several_seeds": len(multi), "groups_with_identical_seed_predictions": identical}


def second_implementation() -> list[dict]:
    rows = []
    for f in sorted(p for p in (ROOT / "results/metrics/confirmatory").glob("*.json") if not p.name.endswith(".provenance.json")):
        r = json.loads(f.read_text())
        has_seeds = len(SETS[r["set"]]["seeds"]) > 1
        df = V.load(SETS[r["set"]], r["outcome"], r["datasets"])
        alt = RN.fit(df, has_seeds)
        ok = True
        diffs = {}
        for k, v in r["point"]["components"].items():
            a = alt["components"][k]
            diffs[k] = {"statsmodels": v, "numpy_reml": a}
            ok &= abs(a - v) <= max(0.02 * abs(v), 0.05)
        ok &= abs(alt["R"] - r["point"]["R"]) <= 0.02 * r["point"]["R"]
        rows.append({"analysis": r["analysis"], "components": diffs, "R_statsmodels": r["point"]["R"],
                     "R_numpy_reml": alt["R"], "converged": alt["converged"], "match": bool(ok)})
    return rows


def model_permutation() -> list[dict]:
    rows = []
    for name, spec in (("S2", S2), ("S1", S1)):
        inc = V.inclusion(spec)
        df = V.load(spec, "macro_f1", [d for d, v in inc.items() if v["included"]])
        perm = RN.permute_models(df, SEED)
        f = V.fit(perm, True)
        c = f["components"]
        within = sum(c.values())
        share = (c["M"] + c["MD"] + c["MP"]) / within
        real = V.fit(df, True)["components"]
        rows.append({"set": name, "permuted_model_share": share, "permuted_components": c,
                     "real_model_share": (real["M"] + real["MD"] + real["MP"]) / sum(real.values()),
                     "caught": share <= 0.03})
    return rows


def regenerate_figures() -> dict:
    """Level 3.4: rebuild figure and table data in a temporary directory and compare with the stored files."""
    from src.utils import paths as PATHS
    from src.visualization import letter_figures as LF
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        saved = (PATHS.FIGURES, PATHS.TABLES, LF.FIG_DATA, LF.TAB_DATA)
        PATHS.FIGURES, PATHS.TABLES, LF.FIG_DATA, LF.TAB_DATA = t / "fig", t / "tab", t / "fd", t / "td"
        try:
            LF.main()
        finally:
            PATHS.FIGURES, PATHS.TABLES, LF.FIG_DATA, LF.TAB_DATA = saved
        out = {}
        for sub, stored_dir in (("fd", ROOT / "results/figure_data"), ("td", ROOT / "results/table_data")):
            for f in sorted((t / sub).iterdir()):
                if f.name.endswith(".provenance.json"):
                    continue
                stored = stored_dir / f.name
                out[str(stored.relative_to(ROOT))] = stored.exists() and stored.read_bytes() == f.read_bytes()
        return out


def negative_controls(eval_dir: Path) -> list[dict]:
    """Tamper with copies (never the real artifacts) and check that the guard fails."""
    rows = []
    with tempfile.TemporaryDirectory() as t:
        copy = Path(t) / "proj"
        ignore = shutil.ignore_patterns("checkpoints", ".venv", "third_party", "paper", "figures", "docs")
        shutil.copytree(ROOT, copy, ignore=ignore, symlinks=True)
        (copy / "third_party").symlink_to(ROOT / "third_party")
        (copy / "experiments" / "checkpoints").symlink_to(ROOT / "experiments" / "checkpoints")
        rel_eval = eval_dir.relative_to(ROOT)
        base = {flag: audit(flag, root=copy)[0] for flag in ("--recompute", "--provenance", "--ledger", "--seeds")}
        # 1. alter one prediction value -> --recompute. A non-degenerate S2 file; a correct prediction made wrong, so
        #    that the metrics change too (the file hash in the metric record catches any edit regardless)
        pf = sorted((copy / rel_eval / "predictions" / "s4d" / "wear").rglob("split-*.csv"))[0]
        k = json.loads(Path(str(pf).replace("/predictions/", "/metrics/")).with_suffix(".json").read_text())["num_classes"]
        rows_ = list(csv.reader(open(pf)))
        j = next(i for i in range(1, len(rows_)) if rows_[i][1] == rows_[i][2])
        rows_[j][2] = str((int(rows_[j][2]) + 1) % k)  # a correctly classified window now predicted as another class
        with open(pf, "w", newline="") as f:
            csv.writer(f).writerows(rows_)
        ok, _ = audit("--recompute", root=copy)
        rows.append({"tampering": "one prediction value altered", "check": "--recompute",
                     "untampered_passes": base["--recompute"], "caught": base["--recompute"] and not ok})
        shutil.copy2(ROOT / rel_eval / pf.relative_to(copy / rel_eval), pf)
        # 2. delete a provenance record -> --provenance
        pv = next((copy / "results/metrics/confirmatory").glob("*.json.provenance.json"))
        pv.unlink()
        ok, out = audit("--provenance", root=copy)
        named = str(pv.relative_to(copy)).removesuffix(".provenance.json") in out
        rows.append({"tampering": "a provenance record deleted", "check": "--provenance",
                     "untampered_passes": base["--provenance"], "deleted_file_named": named,
                     "caught": not ok and named})
        shutil.copy2(ROOT / pv.relative_to(copy), pv)
        # 3. remove a run directory listed in the ledger -> --ledger
        shutil.rmtree(copy / rel_eval)
        ok, _ = audit("--ledger", root=copy)
        rows.append({"tampering": "a ledger run directory removed", "check": "--ledger",
                     "untampered_passes": base["--ledger"], "caught": base["--ledger"] and not ok})
        shutil.copytree(ROOT / rel_eval, copy / rel_eval)
        # 4. edit a stored confirmatory component -> --seeds
        cf = copy / "results/metrics/confirmatory/H1.json"
        r = json.loads(cf.read_text())
        r["point"]["components"]["P"] *= 1.1
        cf.write_text(json.dumps(r, indent=1))
        ok, _ = audit("--seeds", root=copy)
        rows.append({"tampering": "a stored variance component edited", "check": "--seeds",
                     "untampered_passes": base["--seeds"], "caught": base["--seeds"] and not ok})
    # 5. a checkpoint replaced by a fresh initialisation -> --reproduce
    rows.append(reproduce_control(eval_dir))
    # 6. hand-edit table data -> Level 3.4; 7. corrupt a metric file, regenerate: figure data must change
    rows += figure_controls()
    # 8. move one test window into train -> split-disjointness check; 9. normalise with all windows -> norm check
    rows += leakage_controls()
    return rows


def reproduce_control(eval_dir: Path) -> dict:
    import torch
    from scripts import evaluate as E
    from src.models.registry import build
    f = sorted((eval_dir / "predictions" / "s4d").rglob("split-*.csv"))[0]
    model, dataset, seed_dir = f.parts[-4], f.parts[-3], f.parts[-2]
    seed, split = int(seed_dir.split("-")[1]), int(f.stem.split("-")[1])
    real_ck = E.checkpoint_path(model, dataset, seed, split)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        fake_root = t / "ck"
        fake = fake_root / model / dataset / seed_dir / real_ck.name
        fake.parent.mkdir(parents=True)
        ck = torch.load(real_ck, map_location="cpu", weights_only=False)
        rec = json.loads(Path(str(f).replace("/predictions/", "/metrics/")).with_suffix(".json").read_text())
        stored = json.loads(E.stored_run(model, dataset, seed, split).read_text())
        torch.manual_seed(seed)
        net = build(model, input_channels=stored["model"]["input_channels"],
                    window_length=stored["model"]["window_length"], num_classes=rec["num_classes"])
        ck["state_dict"] = net.state_dict()
        torch.save(ck, fake)
        saved, saved_log, saved_rel = E.PATHS.CHECKPOINTS, E.ACCESS_LOG, E.PV.rel
        E.PATHS.CHECKPOINTS, E.ACCESS_LOG = fake_root, t / "access.log"
        E.PV.rel = lambda q: str(q)  # the fake checkpoint lives outside the project
        try:
            E.evaluate([(model, dataset, seed, split)], t / "out", "negative_control", 0.0)
        finally:
            E.PATHS.CHECKPOINTS, E.ACCESS_LOG, E.PV.rel = saved, saved_log, saved_rel
        new = t / "out" / "predictions" / model / dataset / seed_dir / f.name
        caught = list(csv.reader(open(new))) != list(csv.reader(open(f)))
    return {"tampering": "a checkpoint replaced by a fresh initialisation", "check": "--reproduce (re-execution)",
            "caught": caught}


def figure_controls() -> list[dict]:
    from src.utils import paths as PATHS
    from src.visualization import letter_figures as LF
    rows = []
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        td = t / "td_stored"
        shutil.copytree(ROOT / "results/table_data", td)
        f = td / "tab01_variance_components.csv"
        lines = f.read_text().splitlines()
        cells = lines[1].split(",")
        cells[6] = str(float(cells[6]) + 1.0)
        lines[1] = ",".join(cells)
        f.write_text("\n".join(lines) + "\n")
        saved = (PATHS.FIGURES, PATHS.TABLES, LF.FIG_DATA, LF.TAB_DATA, LF.CONF, LF.PV.rel)
        PATHS.FIGURES, PATHS.TABLES, LF.FIG_DATA, LF.TAB_DATA = t / "fig", t / "tab", t / "fd", t / "td"
        LF.PV.rel = lambda q: str(q)  # temporary inputs live outside the project
        try:
            LF.main()
            regen = (t / "td" / f.name).read_bytes()
            untampered = regen == (ROOT / "results/table_data" / f.name).read_bytes()
            rows.append({"tampering": "a value in results/table_data hand-edited", "check": "Level 3.4 regeneration",
                         "untampered_passes": untampered, "caught": untampered and regen != f.read_bytes()})
            # corrupt a metric file, regenerate: the figure data must change
            conf = t / "conf"
            shutil.copytree(ROOT / "results/metrics/confirmatory", conf)
            h = json.loads((conf / "H1.json").read_text())
            h["point"]["components"]["P"] *= 2
            (conf / "H1.json").write_text(json.dumps(h))
            LF.CONF = conf
            before = (t / "fd" / "fig01_variance_components.json").read_bytes()
            LF.main()
            after = (t / "fd" / "fig01_variance_components.json").read_bytes()
            rows.append({"tampering": "a metric file corrupted, figures regenerated", "check": "Level 3.4: figure data "
                         "must change", "caught": before != after})
        finally:
            PATHS.FIGURES, PATHS.TABLES, LF.FIG_DATA, LF.TAB_DATA, LF.CONF, LF.PV.rel = saved
    return rows


def disjoint(window_subject: dict, train: list, val: list, test: list) -> bool:
    tv = {window_subject[i] for i in train + val}
    return not tv & {window_subject[i] for i in test} and not (set(train) | set(val)) & set(test)


def leakage_controls() -> list[dict]:
    from src.training import run_whar as R
    ds = R.prepare_dataset("utd_mhad")
    wdf, sdf = ds["window_df"], ds["session_df"]
    subj = dict(zip(sdf["session_id"], sdf["subject_id"]))
    ws = {i: subj[s] for i, s in zip(wdf.index, wdf["session_id"])}
    sp = ds["splits"][0]
    real = disjoint(ws, list(sp.train_indices), list(sp.val_indices), list(sp.test_indices))
    moved = disjoint(ws, list(sp.train_indices) + [sp.test_indices[0]], list(sp.val_indices), list(sp.test_indices))
    _, dl, _ = R.prepare_split(ds, sp)
    x_tr, x_te = dl["train"].dataset.tensors[1], dl["test"].dataset.tensors[1]
    ok_real = float(x_tr.mean(dim=(0, 1)).abs().max()) <= 1e-3
    allx = __import__("torch").cat([x_tr, x_te])  # a "preprocessor" fitted on train + test, applied to train
    wrong = (x_tr - allx.mean(dim=(0, 1))) / allx.std(dim=(0, 1))
    ok_wrong = float(wrong.mean(dim=(0, 1)).abs().max()) <= 1e-3
    return [{"tampering": "one test window moved into train", "check": "split disjointness (Level 3.5.1)",
             "real_passes": real, "caught": real and not moved},
            {"tampering": "normalisation fitted on train + test", "check": "train-only normalisation (Level 3.5.2)",
             "real_passes": ok_real, "caught": ok_real and not ok_wrong}]


def label_shuffle(run_id: str | None):
    """Level 3.5.3: the label-shuffle control's verdict, read from its own ledger run."""
    if not run_id:
        return "not run"
    s = json.loads((ROOT / "experiments" / run_id / "summary.json").read_text())
    return {k: v for k, v in s.items()} | {"run_dir": f"experiments/{run_id}"}


def access_log_audit() -> dict:
    seal = json.loads((ROOT / "results/gates/G4_test_seal.json").read_text())
    lines = [l for l in (ROOT / "results/test_access.log").read_text().splitlines() if l.strip()]
    before = [l for l in lines if l.split(" | ")[0] < seal["timestamp"]]
    s2 = [l for l in lines if "run_confirmatory.py" in l]
    return {"entries": len(lines), "before_seal": before, "confirmatory_reads": s2,
            "verification_reads": sum("evaluate.py" in l for l in lines), "seal_timestamp": seal["timestamp"],
            "ok": not before and all(any(a in l for a in ("H1", "sens1_S2")) for l in s2)}


def label_shuffle_lines(ls) -> str:
    if not isinstance(ls, dict):
        return "- Label-shuffle control: not run."
    return (f"- Label-shuffle control ({ls['run_dir']}): S4D trained on permuted training labels, {ls['runs']} runs on "
            f"UP-Fall and UCA-EHAR. Mean test macro-F1 {ls['mean_shuffled_test_macro_f1']:.2f} against a chance level of "
            f"{ls['mean_chance_test_macro_f1']:.2f} and {ls['mean_stored_test_macro_f1']:.2f} for the true-label runs; "
            f"validation {ls['mean_shuffled_val_macro_f1']:.2f} vs chance {ls['mean_chance_val_macro_f1']:.2f} vs "
            f"{ls['mean_stored_val_macro_f1']:.2f}. Collapsed to chance on every run: {ls['all_collapsed']}.")


def write_report(gate: dict, gate_path: Path) -> None:
    """docs/verification_report.md: for a reader who did not follow the run."""
    o = gate["observed_value"]
    l3 = o["L3_reexecution"]
    neg = o["L5_negative_controls"]
    perm = o["L5_model_permutation"]
    second = o["L3_second_implementation"]
    caught = sum(r["caught"] for r in neg)
    acc = o["L3_5_leakage"]["access_log"]
    if gate["verdict"] == "PASS":
        head = (f"All {l3['evaluated']} re-executed runs ({l3['s2_runs_reexecuted']} of them from the confirmatory set S2) "
                f"regenerate their stored metrics from their checkpoints within the declared tolerance, "
                f"{l3['exact_match']} of them exactly. An independent REML gives the same variance components for all "
                f"{len(second)} analyses, permuting model labels removes the model share, the training-split "
                f"normalisation is a train-only fit, and the test-access log matches the G4 seal. {caught} of "
                f"{len(neg)} negative controls were caught by the auditor.")
        ls = o["L3_5_leakage"]["label_shuffle_control"]
        if isinstance(ls, dict):
            head += (f" The label-shuffle control collapses to chance: test macro-F1 "
                     f"{ls['mean_shuffled_test_macro_f1']:.1f} against {ls['mean_chance_test_macro_f1']:.1f} by chance "
                     f"and {ls['mean_stored_test_macro_f1']:.1f} with the true labels.")
    else:
        head = "The verification FAILed; the failing checks are listed below. These results must not be used until resolved."
    L = ["# Verification report (gate G5)", "", head, "",
         f"Gate file: `{gate_path.relative_to(ROOT)}` (verdict {gate['verdict']}). Earlier G5 files are kept; "
         "`G5_results_authenticity.json` (FAIL) explains the first run.", "",
         "## What was re-executed", "",
         f"- {l3['gating_cells']} runs, chosen before measuring: every run of S2 on the datasets with a cache on the "
         f"cluster ({', '.join(S2_REEXEC)}: all 4 models x 3 seeds x every split) and a seeded sample of 6 runs on each "
         "of the 8 re-executable development datasets.",
         f"- Exact equality with the stored macro-F1 and accuracy: {l3['exact_match']}; within the tolerance: "
         f"{l3['within_tolerance']}; outside: {len(l3['mismatches'])}. Largest |delta macro-F1|: "
         f"{l3['max_abs_delta_macro_f1_points']:.3f} points (float32 inference on a CPU node vs the original GPU).",
         "- Not re-executable here, verified by their stored records only: S2's usc_had (unsorted parser, no matching "
         "cache); the six runs without a checkpoint (S4D on ActRecTut x2, Mamba on ActRecTut and UniMiB x4); dsads "
         "(below); motion_sense and daphnet (excluded from the analyses).",
         "- dsads (informative, not gating): " + ", ".join(
             f"{d['model']} seed {d['seed']} split {d['split']}: {100 * d['macro_f1']:.1f} vs stored "
             f"{100 * d['stored_macro_f1']:.1f}" for d in l3["informative_dsads"]) +
         ". On this cluster's dsads cache the checkpoints score at chance, so the cache's windows or channel order "
         "differ from the ones the stored runs used. dsads stays in S1 on the strength of its records.", "",
         "## Checkpoints and predictions (Level 2)", "",
         f"- {o['L2_wellformed']['checked']} checkpoints load strictly, match the recorded parameter count, are finite, "
         "differ from a fresh initialisation with the run's seed, and carry the run record's fingerprint. Problems: "
         f"{len(o['L2_wellformed']['problems'])}.",
         f"- Predictions of different seeds differ: {o['L2_wellformed']['seeds']['groups_with_identical_seed_predictions']} "
         f"identical groups of {o['L2_wellformed']['seeds']['groups_with_several_seeds']}.", "",
         "## Aggregates (Level 3)", "", "| Analysis | R (statsmodels) | R (numpy REML) | match |", "|---|---|---|---|"]
    L += [f"| {s['analysis']} | {s['R_statsmodels']:.4f} | {s['R_numpy_reml']:.4f} | {s['match']} |" for s in second]
    L += ["", "Figure and table data regenerated in a temporary directory: " +
          ", ".join(f"{k} {'identical' if v else 'DIFFERENT'}" for k, v in o["L3_4_figures"].items()) + ".", "",
          "## Leakage (Level 3.5)", "",
          "- Subject-disjoint splits: asserted for all 13 cached datasets when the splits were exported (G2).",
          f"- Train-only normalisation: per-channel mean 0 and sd 1 on every re-executed training split; problems: "
          f"{len(o['L3_5_leakage']['normalisation_problems'])}.",
          f"- Test-access log: {acc['entries']} entries, none before the seal ({acc['seal_timestamp']}); the confirmatory "
          "reads are the two pre-registered S2 analyses.",
          label_shuffle_lines(o["L3_5_leakage"]["label_shuffle_control"]), "",
          "## Negative controls (Level 5)", "", "| Tampering | Check | Caught |", "|---|---|---|"]
    L += [f"| {r['tampering']} | {r['check']} | {r['caught']} |" for r in neg]
    L += [f"| model labels permuted within cells ({p['set']}) | model share {p['permuted_model_share']:.4f} "
          f"(real {p['real_model_share']:.3f}) | {p['caught']} |" for p in perm]
    L += ["", "The traceability control (an edited number in a metric file) belongs to gate G6.", "",
          "## Auditor", "", "```", *[x for x in o["audit"]["output"] if x.startswith(("PASS", "FAIL", "SKIPPED"))], "```", ""]
    if gate.get("rerun_reason"):
        L += ["## Why this is a rerun", "", gate["rerun_reason"], ""]
    OUT_REPORT.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--evaluated", default=None, help="run_id of a scripts/evaluate.py run over select_cells()")
    ap.add_argument("--label-shuffle", default=None, help="run_id of a scripts/label_shuffle_control.py run")
    a = ap.parse_args()
    gate_path = GATE
    if GATE.exists():  # never edit a gate file: a re-run gets its own name and states why
        k = 1
        while (GATE.parent / f"G5_results_authenticity_rerun{k}.json").exists():
            k += 1
        gate_path = GATE.parent / f"G5_results_authenticity_rerun{k}.json"
    from scripts import evaluate as E
    run_id = PV.new_run_id("verify_G5")
    RUNNING["id"] = run_id
    started = time.time()
    eval_dir = ROOT / "experiments" / run_id
    eval_dir.mkdir()
    gating, informative = select_cells()
    config = {"tolerances": TOL, "seed": SEED, "gating_cells": len(gating), "informative_cells": len(informative)}
    (eval_dir / "config.json").write_text(json.dumps(config | {"cells": gating + informative}, indent=1) + "\n")
    PV.ledger_start(run_id, config)
    try:
        obs = {}
        ok_prov, prov_out = audit("--provenance")
        obs["L1_provenance"] = {"pass": ok_prov, "output": prov_out.strip().splitlines()[:12]}
        if a.evaluated:  # re-execution done by its own ledger run over exactly these cells
            prior = ROOT / "experiments" / a.evaluated
            if [tuple(c) for c in json.loads((prior / "config.json").read_text())["cells"]] != gating + informative:
                raise SystemExit("the evaluated run did not cover select_cells()")
            res = json.loads((prior / "evaluated.json").read_text())
            eval_dir = prior
        else:
            res = E.evaluate(gating + informative, eval_dir, run_id, started)
            (eval_dir / "evaluated.json").write_text(json.dumps(res, indent=1) + "\n")
        key = lambda r: (r["model"], r["dataset"], r["seed"], r["split"])  # noqa: E731
        gset = set(gating)
        g = [r for r in res if key(r) in gset]
        inf = [r for r in res if key(r) not in gset]
        evaluated = [r for r in g if r["status"] == "evaluated"]
        mism = [r for r in evaluated if not run_matches(r)]
        exact = [r for r in evaluated if r["macro_f1"] == r["stored_macro_f1"] and r["accuracy"] == r["stored_accuracy"]]
        l2 = {f"{r['model']}/{r['dataset']}/seed-{r['seed']}/split-{r['split']:02d}": level2(r) for r in evaluated}
        l2_bad = {k: v for k, v in l2.items() if v}
        norm_bad = [key(r) for r in evaluated if r["train_norm_mean_abs"] > 1e-3 or r["train_norm_std_dev"] > 1e-2]
        obs["L2_wellformed"] = {"checked": len(l2), "problems": l2_bad, "seeds": seeds_differ(eval_dir)}
        obs["L3_reexecution"] = {
            "gating_cells": len(g), "evaluated": len(evaluated),
            "no_checkpoint": [key(r) for r in g if r["status"] == "no checkpoint"],
            "exact_match": len(exact), "within_tolerance": len(evaluated) - len(mism),
            "mismatches": [{k: r[k] for k in ("model", "dataset", "seed", "split", "macro_f1", "stored_macro_f1",
                                              "accuracy", "stored_accuracy", "test_size", "stored_test_size")}
                           for r in mism],
            "max_abs_delta_macro_f1_points": max((abs(r["macro_f1"] - r["stored_macro_f1"]) * 100 for r in evaluated),
                                                 default=None),
            "s2_runs_reexecuted": sum(r["dataset"] in S2_REEXEC for r in evaluated),
            "informative_dsads": [{k: r.get(k) for k in ("model", "seed", "split", "status", "macro_f1",
                                                         "stored_macro_f1", "test_size", "stored_test_size")}
                                  for r in inf]}
        log_s2_reread("the second implementation of every confirmatory analysis")
        obs["L3_second_implementation"] = second_implementation()
        obs["L3_4_figures"] = regenerate_figures()
        obs["L3_5_leakage"] = {"normalisation_problems": norm_bad, "access_log": access_log_audit(),
                               "split_disjointness": "asserted for all 13 cached datasets by scripts/export_splits.py "
                                                     "(data/processed/harness_splits.json)",
                               "label_shuffle_control": label_shuffle(a.label_shuffle)}
        log_s2_reread("the model-label permutation control")
        obs["L5_model_permutation"] = model_permutation()
        log_s2_reread("the auditor's --seeds check and its negative control")
        obs["L5_negative_controls"] = negative_controls(eval_dir)
        ok_rest, rest_out = audit("--ledger", "--data", "--synthetic", "--recompute", "--reproduce", "--seeds")
        obs["audit"] = {"pass": ok_rest, "output": rest_out.strip().splitlines()}
        ls = obs["L3_5_leakage"]["label_shuffle_control"]
        passed = (ok_prov and (not isinstance(ls, dict) or ls["caught"]) and not mism and not l2_bad and not norm_bad and obs["L2_wellformed"]["seeds"]["groups_with_identical_seed_predictions"] == 0
                  and all(r["match"] for r in obs["L3_second_implementation"])
                  and all(obs["L3_4_figures"].values()) and obs["L3_5_leakage"]["access_log"]["ok"]
                  and all(r["caught"] for r in obs["L5_model_permutation"])
                  and all(r["caught"] for r in obs["L5_negative_controls"]) and ok_rest)
        gate = {"gate": "G5_results_authenticity", "stage": "05_verify", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "check": "every reported result re-executes from its checkpoint and source files, and no leakage path "
                         "is open", "reference_value": None, "reference_source": "the stored run records and checkpoints",
                "observed_value": obs, "tolerance": TOL, "verdict": "PASS" if passed else "FAIL",
                "evidence": [str(eval_dir.relative_to(ROOT)), "docs/verification_report.md", "scripts/verify_results.py",
                             "scripts/evaluate.py", "src/evaluation/reml_numpy.py", "scripts/audit.py"],
                "notes": "Runs without a checkpoint and datasets without a cache here cannot be re-executed; they "
                         "are listed in the report."}
        if gate_path != GATE:
            gate["rerun_reason"] = RERUN_REASON
        gate_path.write_text(json.dumps(gate, indent=1, default=str) + "\n")
        write_report(gate, gate_path)
        PV.ledger_end(run_id, f"G5 {gate['verdict']}", started)
        print("G5", gate["verdict"])
        print(json.dumps({k: (v if k != "L5_negative_controls" else v) for k, v in obs.items()
                          if k not in ("L2_wellformed",)}, indent=1, default=str)[:20000])
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise


if __name__ == "__main__":
    main()
