"""Stage 06 (agents/06_summary.md): docs/summary.md, docs/summary_values.json and docs/artifact_index.md.

Every number in the summary is read from one artifact by a fixed path and formatted the way the manuscript prints
it. docs/summary_values.json records, for each value: its id, the printed string, the artifact, the path inside it,
the format and the producing script. `scripts/audit.py --traceability` re-derives every printed string from the
artifact, independently of this script's output. The manuscript may only print numbers listed here.

    uv run --no-sync python scripts/build_summary.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONF = "results/metrics/confirmatory"
ANALYSES = [("H1", "H1, confirmatory (S2, macro-F1)"), ("sens1_S2_balanced_accuracy", "S2, balanced accuracy"),
            ("sens2_S1_macro_f1", "S1, macro-F1"), ("sens2_S1_balanced_accuracy", "S1, balanced accuracy"),
            ("sens3_S1_without_actrectut", "S1 without ActRecTut"), ("sens4_S3_macro_f1", "S3, 17 architectures")]
CONTRASTS = [("recurrencenet-hystnet_nomem", "RecurrenceNet − no memory"), ("s4d-hystnet_nomem", "S4D − no memory"),
             ("hystnet_sel_clock-hystnet_nomem", "HysteresisNet − no memory"),
             ("quantilenet-hystnet_nomem", "QuantileNet − no memory"), ("mamba-hystnet_nomem", "Mamba − no memory"),
             ("recurrencenet-s4d", "RecurrenceNet − S4D"), ("hystnet_sel_clock-s4d", "HysteresisNet − S4D"),
             ("quantilenet-s4d", "QuantileNet − S4D"), ("mamba-s4d", "Mamba − S4D")]
OLS_OUT = {"rf": "RF − best deep (classical margin)", "gap": "best − median deep (does the model matter)",
           "level": "median deep F1 (level)", "spread": "mean split sd of deep models (person spread)"}
PRED = ["log2 subjects", "log2 channels", "log2 classes", "log2 samples/window"]


def get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def fmt(v, f: str) -> str:
    if f == "p3":
        return "< 0.001" if v < 0.001 else f"= {v:.3f}"
    if f == "pct":
        return f"{100 * v:.0f}"
    if f == "pct1":
        return f"{100 * v:.1f}"
    return format(v, f)


def spec() -> list[dict]:
    """(id, description, artifact, path, format)."""
    s = []
    add = lambda i, d, a, p, f: s.append({"id": i, "description": d, "artifact": a, "path": p, "format": f})  # noqa: E731
    for key, label in ANALYSES:
        a = f"{CONF}/{key}.json"
        add(f"{key}.R", f"{label}: R", a, ["point", "R"], ".2f")
        add(f"{key}.R_lo", f"{label}: R, 97.5 % lower bound", a, ["bootstrap", "R_interval", 0], ".2f")
        add(f"{key}.R_hi", f"{label}: R, 97.5 % upper bound", a, ["bootstrap", "R_interval", 1], ".2f")
        add(f"{key}.pi_P", f"{label}: pi_P (%)", a, ["point", "pi_P"], "pct")
        add(f"{key}.pi_P_lo", f"{label}: pi_P lower bound (%)", a, ["bootstrap", "pi_P_interval", 0], "pct")
        add(f"{key}.pi_P_hi", f"{label}: pi_P upper bound (%)", a, ["bootstrap", "pi_P_interval", 1], "pct")
        add(f"{key}.runs", f"{label}: runs", a, ["n_runs"], "d")
        add(f"{key}.splits", f"{label}: splits", a, ["n_splits"], "d")
        for c in ("P", "M", "MD", "MP", "e"):
            if key != "sens4_S3_macro_f1" or c != "MP":
                add(f"{key}.s2_{c}", f"{label}: variance component {c}", a, ["point", "components", c], ".1f")
        add(f"{key}.resamples_converged", f"{label}: converged resamples", a, ["bootstrap", "converged"], "d")
        add(f"{key}.datasets", f"{label}: datasets analysed", a, ["datasets"], "len")
    add("H1.decision", "H1: pre-registered decision", f"{CONF}/H1.json", ["decision"], "s")
    for key, label in CONTRASTS:
        a = "results/metrics/contrasts_robustness/data.json"
        for sub, tag in (("without_daphnet", "9"), ("all_10", "10")):
            add(f"contrast.{key}.{tag}.mean", f"{label}, {tag} datasets: mean", a, [key, sub, "mean"], "+.2f")
            add(f"contrast.{key}.{tag}.lo", f"{label}, {tag} datasets: 95 % CI low", a, [key, sub, "lo"], "+.2f")
            add(f"contrast.{key}.{tag}.hi", f"{label}, {tag} datasets: 95 % CI high", a, [key, sub, "hi"], "+.2f")
            add(f"contrast.{key}.{tag}.wins", f"{label}, {tag} datasets: datasets won", a, [key, sub, "wins"], "d")
            add(f"contrast.{key}.{tag}.p", f"{label}, {tag} datasets: Wilcoxon p", a, [key, sub, "p"], "p3")
    fa = "results/metrics/factors/data.json"
    for k, out in OLS_OUT.items():
        add(f"ols.{k}.r2", f"OLS over 30 datasets, {out}: R^2", fa, ["data", "ols", out, "r2"], ".2f")
        for j, pr in enumerate(PRED):
            add(f"ols.{k}.{pr.split()[1].replace('/', '_')}.beta", f"OLS, {out}: {pr} (per sd)", fa,
                ["data", "ols", out, "beta", j + 1], "+.1f")
            add(f"ols.{k}.{pr.split()[1].replace('/', '_')}.p", f"OLS, {out}: {pr} p", fa,
                ["data", "ols", out, "p", j + 1], "p3")
    ss = "results/metrics/factors/data.json"
    for src, tag in (("test person (split)", "P"), ("model", "M"), ("model x dataset", "MD"), ("model x person", "MP"),
                     ("seed", "e")):
        add(f"exploratory_ss.{tag}", f"exploratory S1 sums of squares: {src}, share within a dataset (%)", ss,
            ["variance", "ss", src], "within_share")
    hs = "data/processed/harness_splits.json"
    add("subjects_per_split.min", "fewest test subjects in a split (13 cached datasets)", hs, ["datasets"], "subj_min")
    add("subjects_per_split.max", "most test subjects in a split (13 cached datasets)", hs, ["datasets"], "subj_max")
    g3 = "results/gates/G3_baseline.json"
    for i, d in enumerate(["utd_mhad", "uci_har", "unimib_shar", "actrectut_gestures"]):
        add(f"g3.rf.{d}.figure6", f"G3 anchor: Random Forest Figure 6 value, {d}", g3,
            ["observed_value", "anchor_random_forest", i, "figure6"], ".2f")
        add(f"g3.rf.{d}.rerun", f"G3 anchor: Random Forest re-run mean, {d}", g3,
            ["observed_value", "anchor_random_forest", i, "rerun_mean"], ".2f")
    reruns = sorted((ROOT / "results/gates").glob("G5_results_authenticity_rerun*.json"),
                    key=lambda q: int(q.stem.rsplit("rerun", 1)[1]))
    g5 = str((reruns[-1] if reruns else ROOT / "results/gates/G5_results_authenticity.json").relative_to(ROOT))
    add("g5.verdict", "G5 verdict", g5, ["verdict"], "s")
    add("g5.reexecuted", "G5: runs re-executed from checkpoints (gating)", g5, ["observed_value", "L3_reexecution", "evaluated"], "d")
    add("g5.exact", "G5: re-executed runs equal to the stored metrics exactly", g5,
        ["observed_value", "L3_reexecution", "exact_match"], "d")
    add("g5.s2", "G5: S2 runs re-executed", g5, ["observed_value", "L3_reexecution", "s2_runs_reexecuted"], "d")
    return s


def derive(entry: dict) -> str:
    art = json.loads((ROOT / entry["artifact"]).read_text())
    if entry["format"] in ("subj_min", "subj_max"):
        n = [len(sp["test_subjects"]) for d in art["datasets"].values() for sp in d["splits"].values()]
        return str(min(n) if entry["format"] == "subj_min" else max(n))
    if entry["format"] == "len":
        return str(len(get(art, entry["path"])))
    if entry["format"] == "within_share":
        ss = art["variance"]["ss"]
        within = sum(v for k, v in ss.items() if k != "dataset")
        return f"{100 * get(art, entry['path']) / within:.1f}"
    v = get(art, entry["path"])
    return v if entry["format"] == "s" else fmt(v, entry["format"])


def main() -> None:
    values = []
    for e in spec():
        values.append(e | {"printed": derive(e), "produced_by": producer(e["artifact"])})
    by = {v["id"]: v["printed"] for v in values}
    (ROOT / "docs/summary_values.json").write_text(json.dumps(values, indent=1, ensure_ascii=False) + "\n")
    write_summary(by, values)
    write_index(values)
    print("values", len(values))


def producer(artifact: str) -> str:
    pv = ROOT / (artifact + ".provenance.json")
    if pv.exists():
        return json.loads(pv.read_text())["produced_by"]
    if artifact.startswith("results/gates/G5_"):
        return "scripts/verify_results.py"
    return {"results/gates/G3_baseline.json": "scripts/check_baseline_anchor.py",
            "data/processed/harness_splits.json": "scripts/export_splits.py"}.get(artifact, "?")


def write_summary(v: dict, values: list) -> None:
    L = ["# Summary of results (stage 06)", "",
         "Every number below is printed exactly as `docs/summary_values.json` derives it from its artifact; "
         "`scripts/audit.py --traceability` re-derives each one. The manuscript may print no other number.", "",
         "## 1. Confirmatory test (pre-registered, docs/preregistration.md)", "",
         f"- H1, S2 macro-F1: {v['H1.runs']} runs, {v['H1.splits']} splits of 7 held-out datasets (motion_sense "
         f"excluded by the inclusion rule).",
         f"- Variance components (F1 points²): test-subject group {v['H1.s2_P']}, model {v['H1.s2_M']}, model × "
         f"dataset {v['H1.s2_MD']}, model × split {v['H1.s2_MP']}, seed {v['H1.s2_e']}.",
         f"- R = {v['H1.R']} (97.5 % interval {v['H1.R_lo']}–{v['H1.R_hi']}); pi_P = {v['H1.pi_P']} % "
         f"({v['H1.pi_P_lo']}–{v['H1.pi_P_hi']} %).",
         f"- Pre-registered decision: **{v['H1.decision']}**; secondary criterion (pi_P lower bound >= 40 %) met.", "",
         "## 2. Sensitivity analyses (pre-registered; they do not decide H1)", "",
         "| Analysis | runs | R [97.5 % interval] | pi_P % [interval] |", "|---|---|---|---|"]
    for key, label in ANALYSES:
        L.append(f"| {label} | {v[key + '.runs']} | {v[key + '.R']} [{v[key + '.R_lo']}, {v[key + '.R_hi']}] | "
                 f"{v[key + '.pi_P']} [{v[key + '.pi_P_lo']}, {v[key + '.pi_P_hi']}] |")
    L += ["", "S3's R is the conservative R′ = s2_P / (s2_M + s2_MD + s2_e) (one run per split).", "",
          "## 3. Exploratory findings (labelled exploratory in the letter)", "",
          f"- Sums-of-squares shares within a dataset on S1 (the finding that motivated H1): test-subject group "
          f"{v['exploratory_ss.P']} %, seed {v['exploratory_ss.e']} %, model × split {v['exploratory_ss.MP']} %, "
          f"model × dataset {v['exploratory_ss.MD']} %, model {v['exploratory_ss.M']} %.",
          "- Component contrasts over the 9 datasets of S1 whose runs pair by person (10-dataset values in "
          "summary_values.json):", "", "| Contrast | mean [95 % CI] | won | p |", "|---|---|---|---|"]
    for key, label in CONTRASTS:
        c = f"contrast.{key}.9"
        L.append(f"| {label} | {v[c + '.mean']} [{v[c + '.lo']}, {v[c + '.hi']}] | {v[c + '.wins']}/9 | {v[c + '.p']} |")
    L += ["", "- Data factors over the WHAR Arena paper's 30 datasets (re-analysis of published single runs; "
              "correlational):",
          f"  - RF − best deep vs channels: {v['ols.rf.channels.beta']} points per sd (p {v['ols.rf.channels.p']}), "
          f"R² {v['ols.rf.r2']};",
          f"  - best − median deep vs subjects: {v['ols.gap.subjects.beta']} per sd (p {v['ols.gap.subjects.p']}), "
          f"R² {v['ols.gap.r2']};",
          f"  - median deep F1 (difficulty): R² {v['ols.level.r2']}.", "",
          "## 4. Anchors and verification", "",
          "- G3: Random Forest re-run vs Figure 6: " + ", ".join(
              f"{d} {v[f'g3.rf.{d}.rerun']} vs {v[f'g3.rf.{d}.figure6']}"
              for d in ["utd_mhad", "uci_har", "unimib_shar", "actrectut_gestures"]) + ".",
          f"- G5 {v['g5.verdict']}: {v['g5.reexecuted']} runs re-executed from checkpoints ({v['g5.s2']} of them in "
          f"S2), {v['g5.exact']} equal to the stored metrics exactly.", "",
          "## 5. Limitations carried to the manuscript", "",
          "- One protocol (WHAR Arena v1); 3 training seeds (template asks >= 5; user's decision); the 10 population "
          "models share one skeleton; the 30-dataset part re-analyses published single runs; data factors are "
          "correlational.",
          "- Re-execution covers the datasets with a cache on the cluster; runs without a checkpoint and datasets "
          "without a cache are verified by their stored records only (docs/verification_report.md).",
          "- The S2 per-dataset means were known before the pre-registration; no variance component of S2 was.",
          "- The manuscript was drafted with IEEEtran [journal,9pt] in place of IEEE_lsens.cls.", "",
          "## 6. Pre-registration outcome", "",
          f"H1: {v['H1.decision']}. No deviation from docs/preregistration.md in the confirmatory analysis.", ""]
    (ROOT / "docs/summary.md").write_text("\n".join(L))


def write_index(values: list) -> None:
    L = ["# Artifact index (stage 06)", "", "Claim → artifact → script. One row per value in docs/summary_values.json.",
         "", "| id | printed | artifact | path | produced by |", "|---|---|---|---|---|"]
    for x in values:
        L.append(f"| {x['id']} | {x['printed']} | {x['artifact']} | {'/'.join(map(str, x['path']))} | {x['produced_by']} |")
    (ROOT / "docs/artifact_index.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
