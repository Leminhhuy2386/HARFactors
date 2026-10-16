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
    add("runs.imported", "stored WHAR-harness runs imported (experiments/whar_arena_v1)", "data/processed/manifest.json",
        ["entries"], "count_runs")
    for name in ("S1", "S2", "S3"):
        add(f"runs.{name}", f"runs in set {name}", "data/processed/splits.json", ["sets", name, "runs"], "d")
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
    g5_json = json.loads((ROOT / g5).read_text())
    rb = ["observed_value", "L3_reexecution", "informative_rebuilt"]
    if isinstance(g5_json["observed_value"]["L3_reexecution"].get("informative_rebuilt"), dict):
        for k, f, lab in (("runs", "d", "S2 runs re-executed on rebuilt caches"), ("exact", "d", "of them exact"),
                          ("outside_tolerance", "d", "of them outside the gating tolerance"),
                          ("max_abs_dataset_mean_delta_macro_f1", ".2f", "largest |dataset-mean delta macro-F1|"),
                          ("max_abs_delta_macro_f1", ".2f", "largest |single-run delta macro-F1|")):
            add(f"g5.rebuilt.{k}", f"G5 informative, rebuilt caches: {lab}", g5, rb + [k], f)
    ls = ["observed_value", "L3_5_leakage", "label_shuffle_control"]
    if isinstance(get(g5_json, ls), dict):
        add("ls.runs", "label-shuffle control: runs", g5, ls + ["runs"], "d")
        for k, lab in (("mean_shuffled_test_macro_f1", "shuffled-label test macro-F1"),
                       ("mean_chance_test_macro_f1", "chance test macro-F1"),
                       ("mean_stored_test_macro_f1", "true-label test macro-F1 (stored runs)"),
                       ("mean_shuffled_val_macro_f1", "shuffled-label validation macro-F1"),
                       ("mean_chance_val_macro_f1", "chance validation macro-F1"),
                       ("mean_stored_val_macro_f1", "true-label validation macro-F1 (stored runs)")):
            add(f"ls.{k}", f"label-shuffle control: mean {lab}", g5, ls + [k], ".1f")
        add("ls.all_collapsed", "label-shuffle control: every run collapsed to chance", g5, ls + ["all_collapsed"], "s")
    s += detail_spec()
    return s


def detail_spec() -> list[dict]:
    """Descriptive and exploratory values of the expanded manuscript (datasets, networks, memories, split spread,
    embeddings) and every number of Tables II and III. Each artifact enters only once it exists."""
    s = []
    add = lambda i, d, a, p, f: s.append({"id": i, "description": d, "artifact": a, "path": p, "format": f})  # noqa: E731
    have = lambda a: (ROOT / a).exists()  # noqa: E731
    t2 = "results/table_data/tab02_datasets.json"
    if have(t2):
        for i, r in enumerate(json.loads((ROOT / t2).read_text())["rows"]):
            for k, f in (("channels", "d"), ("subjects", "d"), ("classes", "d"), ("hz", "g"), ("windows", ",d"),
                         ("splits", "d")):
                add(f"tab02.{r['dataset']}.{k}", f"Table II, {r['dataset']}: {k}", t2, ["rows", i, k], f)
        for k, f, lab in (("channels_min", "d", "fewest channels"), ("channels_max", "d", "most channels"),
                          ("subjects_min", "d", "fewest subjects"), ("subjects_max", "d", "most subjects"),
                          ("classes_min", "d", "fewest classes"), ("classes_max", "d", "most classes"),
                          ("hz_min", "g", "lowest sampling rate"), ("hz_max", "g", "highest sampling rate"),
                          ("windows_total", ",d", "windows over the 18 datasets"), ("datasets", "d", "datasets")):
            add(f"datasets.{k}", f"S1 and S2 datasets: {lab}", t2, ["summary", k], f)
    t3 = "results/table_data/tab03_models.json"
    if have(t3):
        for i, r in enumerate(json.loads((ROOT / t3).read_text())["rows"]):
            for k, f in (("d_model", "d"), ("layers", "d"), ("embedding_dim", "d"), ("params_ref", ",d"),
                         ("params_min", ",d"), ("params_max", ",d")):
                add(f"tab03.{r['model']}.{k}", f"Table III, {r['model']}: {k}", t3, ["rows", i, k], f)
        for k, f, lab in (("params_ref_min", ",d", "fewest parameters, reference input"),
                          ("params_ref_max", ",d", "most parameters, reference input"),
                          ("params_all_min", ",d", "fewest parameters over all datasets"),
                          ("params_all_max", ",d", "most parameters over all datasets")):
            add(f"models.{k}", f"ten networks: {lab}", t3, ["summary", k], f)
    an = "results/metrics/model_anatomy/data.json"
    if have(an):
        add("learned.checkpoints", "learned memory parameters: checkpoints read", an, ["learned", "checkpoints_read"], ",d")
        for k, lab in (("s4d", "S4D time constant / window"), ("s4d_a_real", "S4D decay rate"),
                       ("hystnet_sel_clock", "play base width"), ("quantilenet_tau", "quantile level"),
                       ("quantilenet_eta", "quantile step"), ("recurrencenet", "recurrence radius")):
            for q in ("p05", "p50", "p95"):
                add(f"learned.{k}.{q}", f"learned {lab}: {q}", an, ["learned", k, "quantiles", q], ".2g")
                add(f"learned.{k}.init_{q}", f"initial {lab}: {q}", an, ["learned", k, "init_quantiles", q], ".2g")
            add(f"learned.{k}.n", f"learned {lab}: values", an, ["learned", k, "n"], ",d")
        for k, lab in (("quantilenet_tau", "quantile level"), ("recurrencenet", "recurrence radius"),
                       ("s4d_a_real", "S4D decay rate (log10 ratio)")):
            for q in ("p50", "p95"):
                add(f"learned.{k}.change_{q}", f"change from initialisation, {lab}: {q}", an,
                    ["learned", k, "exact_change", "quantiles", q], ".2g")
            add(f"learned.{k}.change_below_005", f"change from initialisation below 0.05, {lab} (%)", an,
                ["learned", k, "exact_change", "share_below_0.05"], "pct")
        add("learned.s4d.above_window", "S4D modes whose time constant exceeds the window (%)", an,
            ["learned", "s4d_fraction_of_window_above_1"], "pct")
        add("learned.s4d.init_above_window", "S4D modes whose initial time constant exceeds the window (%)", an,
            ["learned", "s4d_init_fraction_of_window_above_1"], "pct")
    sp = "results/metrics/split_spread/data.json"
    if have(sp):
        add("spread.datasets", "split spread: datasets", sp, ["n_datasets"], "d")
        add("spread.split_above_model", "split spread: datasets whose split-mean sd exceeds the network-mean sd", sp,
            ["datasets_split_sd_above_model_sd"], "d")
        add("spread.median_ratio", "split spread: median ratio of split-mean sd to network-mean sd", sp,
            ["median_ratio_split_sd_to_model_sd"], ".1f")
        for d in ("uci_har", "up_fall"):
            for k in ("sd_split_means", "sd_model_means", "range_split_means", "range_model_means", "mean_seed_sd"):
                add(f"spread.{d}.{k}", f"split spread, {d}: {k}", sp, ["datasets", d, k], ".1f")
        if "validation" in json.loads((ROOT / sp).read_text()):
            for k, f in (("median_gap", ".1f"), ("min_gap", ".1f"), ("max_gap", ".1f"), ("datasets_val_above_test", "d"),
                         ("runs_val_above_test", ",d"), ("runs", ",d"), ("median_rho_network_means", "+.2f"),
                         ("datasets_with_rho_network", "d"), ("datasets_best_by_val_is_best_by_test", "d"),
                         ("datasets", "d"),
                         ("median_test_loss_choosing_by_val", ".1f")):
                add(f"val.{k}", f"validation vs test: {k.replace('_', ' ')}", sp, ["validation", k], f)
    em = "results/metrics/embedding/summary.json"
    if have(em):
        for k, f in (("datasets", "d"), ("datasets_with_more_than_3_splits", "d"),
                     ("pooled_within_dataset_rank_rho_embedding_ccs_f1", "+.2f"),
                     ("pooled_within_dataset_rank_p_embedding", "p3"),
                     ("pooled_within_dataset_rank_rho_sensor_ccs_f1", "+.2f"),
                     ("pooled_within_dataset_rank_p_sensor", "p3"), ("median_dataset_rho_embedding", "+.2f"),
                     ("median_dataset_rho_sensor", "+.2f"), ("datasets_negative_rho_embedding", "d"),
                     ("datasets_negative_rho_sensor", "d"), ("median_subject_probe_over_chance_embedding", ".1f"),
                     ("median_subject_probe_over_chance_sensor", ".1f"), ("median_activity_probe_embedding", "pct"),
                     ("probe_cases", ",d"), ("regenerated_cases", ",d"), ("regenerated_exact", ",d"),
                     ("max_abs_regenerated_minus_stored_macro_f1", ".2f"),
                     ("pooled_within_dataset_rank_rho_embedding_ccs_per_dim_f1", "+.2f"),
                     ("pooled_within_dataset_rank_p_embedding_per_dim", "p3"), ("network_dataset_cases", "d"),
                     ("median_network_dataset_rho_embedding", "+.2f"), ("network_dataset_negative_rho", "d")):
            add(f"emb.{k}", f"embedding analysis: {k.replace('_', ' ')}", em, [k], f)
    ep = "results/metrics/embedding/uci_har.json"
    if have(ep):
        add("emb.probe_windows", "embedding analysis: most training windows in a probe", ep, ["probe_windows"], ",d")
    ts = "results/metrics/embedding/tsne_case.json"
    if have(ts):
        add("emb.tsne.split", "t-SNE case: split", ts, ["case", "split"], "d")
        add("emb.tsne.train_windows", "t-SNE case: training windows shown", ts, ["train", "activity"], "len")
        add("emb.tsne.test_windows", "t-SNE case: test windows shown", ts, ["test", "activity"], "len")
    return s


def derive(entry: dict) -> str:
    art = json.loads((ROOT / entry["artifact"]).read_text())
    if entry["format"] == "count_runs":
        return str(sum(k.startswith("experiments/whar_arena_v1/") for k in art["entries"]))
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
    return (v if isinstance(v, str) else str(v)) if entry["format"] == "s" else fmt(v, entry["format"])


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
          f"S2), {v['g5.exact']} equal to the stored metrics exactly.",
          (f"- Informative: {v['g5.rebuilt.runs']} S2 runs re-executed on caches rebuilt on another machine "
           f"(pamap2, uma_fall, w_har): {v['g5.rebuilt.exact']} exact, {v['g5.rebuilt.outside_tolerance']} outside the "
           f"gating tolerance; dataset means within {v['g5.rebuilt.max_abs_dataset_mean_delta_macro_f1']} F1 points; "
           f"largest single-run difference {v['g5.rebuilt.max_abs_delta_macro_f1']}." if "g5.rebuilt.runs" in v
           else "- No S2 run re-executed on a rebuilt cache."),
          (f"- Label-shuffle control (S4D, {v['ls.runs']} runs on UP-Fall and UCA-EHAR, permuted training labels): test "
           f"macro-F1 {v['ls.mean_shuffled_test_macro_f1']} against chance {v['ls.mean_chance_test_macro_f1']} and "
           f"{v['ls.mean_stored_test_macro_f1']} with true labels; validation {v['ls.mean_shuffled_val_macro_f1']} vs "
           f"{v['ls.mean_chance_val_macro_f1']} vs {v['ls.mean_stored_val_macro_f1']}; collapsed on every run: "
           f"{v['ls.all_collapsed']}." if "ls.runs" in v else "- Label-shuffle control: not run."), "",
          "## 5. Excluded runs (agents/08_review.md, check 9)", "",
          f"- The ledger imports {v['runs.imported']} WHAR-harness runs. The letter analyses S1 ({v['runs.S1']} runs) and "
          f"S2 ({v['runs.S2']} runs), and S3 ({v['runs.S3']} re-runs of the benchmark's models, in "
          "experiments/external_har_paper_era).",
          "- The other imported runs are Edge_SSM's development history: models outside the population, and training "
          "seeds 3-4 on UTD-MHAD for five models. The letter does not use them; some enter the imported exploratory "
          "aggregates (results/metrics/factors: the kill-set and sensor-grouping contrasts; synthesis, p5, seeds).",
          f"- Excluded by the pre-registered inclusion rule: daphnet in S1 (S1 analyses use {v['sens2_S1_macro_f1.runs']} "
          f"of {v['runs.S1']} runs) and motion_sense in S2 (H1 uses {v['H1.runs']} of {v['runs.S2']} runs).",
          "- Analysis runs in the ledger: development runs (none wrote to results/), crashed verification runs "
          "(recorded with status=crashed), and the G5 runs whose gates are kept (G5 FAIL, rerun1 FAIL, rerun2 PASS).", "",
          "## 6. Limitations carried to the manuscript", "",
          "- One protocol (WHAR Arena v1); 3 training seeds (template asks >= 5; user's decision); the 10 population "
          "models share one skeleton; the 30-dataset part re-analyses published single runs; data factors are "
          "correlational.",
          "- Re-execution covers the datasets with a cache on the cluster; runs without a checkpoint and datasets "
          "without a cache are verified by their stored records only (docs/verification_report.md).",
          "- The S2 per-dataset means were known before the pre-registration; no variance component of S2 was.",
          "- The manuscript was drafted with IEEEtran [journal,9pt] in place of IEEE_lsens.cls.", "",
          "## 7. Pre-registration outcome", "",
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
