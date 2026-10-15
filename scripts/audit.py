"""The user's single entry point for checking this project without trusting its author (agents/01_integrity.md).

    uv run --no-sync python scripts/audit.py --all              # every check implemented so far
    uv run --no-sync python scripts/audit.py --provenance --ledger --data --synthetic

Each check prints PASS or FAIL with its reasons. Exit code 1 if any check fails. Checks that later stages add are listed
as SKIPPED until they exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_DIRS = ["results/metrics", "results/predictions", "results/figure_data", "results/table_data"]
IMPORTED_DIRS = {"checkpoints", "elastic", "error_analysis", "external_har_paper_era", "logs", "person_probe",
                 "style_probe", "tree_probe", "value_probe", "whar_arena_v1"}  # experiments/: IMPORT block of the ledger
RUN_ID = re.compile(r"^\d{8}_\d{6}_[A-Za-z0-9_.-]+$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_files() -> list[Path]:
    files = [p for d in RESULT_DIRS for p in (ROOT / d).rglob("*") if p.is_file()]
    for run in (ROOT / "experiments").iterdir():
        if run.is_dir() and RUN_ID.match(run.name):
            files += [p for sub in ("metrics", "predictions") for p in (run / sub).rglob("*") if p.is_file()]
    return sorted(p for p in files if not p.name.endswith(".provenance.json") and p.name != ".gitkeep")


def g2_manifest() -> str | None:
    g = ROOT / "results/gates/G2_data.json"
    return json.loads(g.read_text())["observed_value"]["data_manifest_sha256"] if g.exists() else None


# ------------------------------------------------------------------------------------------------- checks
def check_provenance() -> list[str]:
    """Every computed result has a provenance record whose output and input hashes match the files on disk."""
    errs = []
    for f in result_files():
        pv = f.with_name(f.name + ".provenance.json")
        if not pv.exists():
            errs.append(f"no provenance: {f.relative_to(ROOT)}")
            continue
        rec = json.loads(pv.read_text())
        if rec.get("output_sha256") != sha256(f):
            errs.append(f"output hash mismatch: {f.relative_to(ROOT)}")
        for inp, h in rec.get("input_files", {}).items():
            p = ROOT / inp
            if not p.exists():
                errs.append(f"input missing: {inp} (for {f.relative_to(ROOT)})")
            elif "sha256:" + sha256(p) != h:
                errs.append(f"input changed since: {inp} (for {f.relative_to(ROOT)})")
    return errs


def check_ledger() -> list[str]:
    """Ledger, run files and run directories agree exactly; every START has an END."""
    errs = []
    lines = (ROOT / "experiments/LEDGER.md").read_text(encoding="utf-8").splitlines()
    imported = {l.split(" | ")[0].strip() for l in lines if " | IMPORT | " in l}
    on_disk = {str(p.relative_to(ROOT / "experiments").with_suffix("")).replace("\\", "/")
               for p in (ROOT / "experiments/whar_arena_v1").rglob("split-*.json")}
    errs += [f"run file not in ledger: {r}" for r in sorted(on_disk - imported)]
    errs += [f"ledger entry without run file: {r}" for r in sorted(imported - on_disk)]
    first = lambda l: l.split(" | ")[0].strip()  # noqa: E731
    starts = [first(l) for l in lines if " | START | " in l and RUN_ID.match(first(l))]
    ends = {first(l) for l in lines if " | END " in l and RUN_ID.match(first(l))}
    running = os.environ.get("HARF_RUNNING_RUN_ID", "")  # the run that calls the audit has not ended yet
    errs += [f"START without END: {r}" for r in starts if r not in ends and r != running]
    errs += [f"run listed twice: {r}" for r in {r for r in starts if starts.count(r) > 1}]
    for d in sorted((ROOT / "experiments").iterdir()):
        if not d.is_dir() or d.name in IMPORTED_DIRS:
            continue
        if not RUN_ID.match(d.name):
            errs.append(f"directory is neither imported nor a run_id: experiments/{d.name}")
        elif d.name not in starts:
            errs.append(f"run directory not in ledger: experiments/{d.name}")
    for r in starts:
        if not (ROOT / "experiments" / r).is_dir():
            errs.append(f"ledger run without directory: experiments/{r}")
    return errs


def check_data() -> list[str]:
    """The manifest re-hashes to G2's value, and every provenance record carries that value."""
    errs = []
    want = g2_manifest()
    if want is None:
        return ["results/gates/G2_data.json missing"]
    man = json.loads((ROOT / "data/processed/manifest.json").read_text())
    for path, h in man["entries"].items():
        p = ROOT / path
        if not p.exists():
            errs.append(f"data file missing: {path}")
        elif sha256(p) != h:
            errs.append(f"data file changed: {path}")
    listing = "".join(f"{k}  {v}\n" for k, v in sorted(man["entries"].items()))
    if hashlib.sha256(listing.encode()).hexdigest() != want or man["data_manifest_sha256"] != want:
        errs.append("manifest does not hash to G2's data_manifest_sha256")
    for f in result_files():
        pv = f.with_name(f.name + ".provenance.json")
        if pv.exists() and json.loads(pv.read_text()).get("data_manifest_sha256") != want:
            errs.append(f"provenance with another data manifest: {f.relative_to(ROOT)}")
    return errs


def check_synthetic() -> list[str]:
    """No reported result descends from data/synthetic/."""
    errs = []
    for f in result_files():
        pv = f.with_name(f.name + ".provenance.json")
        if not pv.exists():
            continue
        rec = json.loads(pv.read_text())
        if rec.get("synthetic"):
            errs.append(f"synthetic result: {f.relative_to(ROOT)}")
        errs += [f"input from data/synthetic: {f.relative_to(ROOT)}" for i in rec.get("input_files", {})
                 if i.startswith("data/synthetic")]
    return errs


def prediction_files() -> list[Path]:
    return sorted(p for run in (ROOT / "experiments").iterdir() if run.is_dir() and RUN_ID.match(run.name)
                  for p in (run / "predictions").rglob("split-*.csv"))


def check_recompute() -> list[str]:
    """Every metric file next to a prediction file recomputes from it (harness metric function, exact)."""
    from scripts.evaluate import metrics_from_csv
    errs = []
    for f in prediction_files():
        mf = Path(str(f).replace("/predictions/", "/metrics/")).with_suffix(".json")
        if not mf.exists():
            errs.append(f"no metric file for {f.relative_to(ROOT)}")
            continue
        rec = json.loads(mf.read_text())
        if rec.get("predictions_sha256") != sha256(f):
            errs.append(f"prediction file differs from the one its metrics were computed from: {f.relative_to(ROOT)}")
        num_classes = int(rec.get("num_classes") or 0) or None
        m = metrics_from_csv(f, num_classes or _classes(rec["dataset"]))
        for k in ("macro_f1", "accuracy", "balanced_accuracy"):
            if m[k] != rec[k]:
                errs.append(f"{k} does not recompute: {mf.relative_to(ROOT)} ({rec[k]} vs {m[k]})")
    return errs


def _classes(dataset: str) -> int:
    from src.evaluation import factor_analysis as FA
    return FA.CFG[dataset][2]


def check_reproduce(k: int = 3, seed: int = 20261015) -> list[str]:
    """For k sampled prediction files, the checkpoint re-generates the stored predictions exactly."""
    import csv
    import random
    import tempfile
    from scripts import evaluate as E
    files = prediction_files()
    if not files:
        return ["no prediction files to reproduce"]
    errs = []
    for f in random.Random(seed).sample(files, min(k, len(files))):
        model, dataset, seed_dir = f.parts[-4], f.parts[-3], f.parts[-2]
        cell = (model, dataset, int(seed_dir.split("-")[1]), int(f.stem.split("-")[1]))
        with tempfile.TemporaryDirectory() as t:
            E.ACCESS_LOG = Path(t) / "access.log"  # the audit's own re-reads are not part of the record
            E.evaluate([cell], Path(t), "audit_reproduce", 0.0)
            new = Path(t) / "predictions" / model / dataset / seed_dir / f.name
            if not new.exists():
                errs.append(f"could not re-generate {f.relative_to(ROOT)}")
            elif list(csv.reader(open(new))) != list(csv.reader(open(f))):
                errs.append(f"checkpoint does not reproduce {f.relative_to(ROOT)}")
    return errs


def check_seeds() -> list[str]:
    """The confirmatory point estimates recompute from the per-run files (sealed code, exact to 1e-9)."""
    from src.data.sets import SETS
    from src.evaluation import variance_components as V
    errs = []
    for f in sorted(p for p in (ROOT / "results/metrics/confirmatory").glob("*.json") if not p.name.endswith(".provenance.json")):
        r = json.loads(f.read_text())
        df = V.load(SETS[r["set"]], r["outcome"], r["datasets"])
        p = V.fit(df, len(SETS[r["set"]]["seeds"]) > 1)
        for kk, v in r["point"]["components"].items():
            if abs(p["components"][kk] - v) > 1e-9 * max(1.0, abs(v)):
                errs.append(f"{f.name}: component {kk} recomputes to {p['components'][kk]} (stored {v})")
        if len(df) != r["n_runs"]:
            errs.append(f"{f.name}: {len(df)} runs now, {r['n_runs']} stored")
    return errs


def _get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _printed(entry: dict) -> str:
    """Re-derive a summary value from its artifact (independent of scripts/build_summary.py)."""
    art = json.loads((ROOT / entry["artifact"]).read_text())
    f = entry["format"]
    if f in ("subj_min", "subj_max"):
        sizes = sorted(len(s["test_subjects"]) for ds in art["datasets"].values() for s in ds["splits"].values())
        return str(sizes[0] if f == "subj_min" else sizes[-1])
    if f == "len":
        return str(len(_get(art, entry["path"])))
    if f == "within_share":
        ss = art["variance"]["ss"]
        return f"{100 * _get(art, entry['path']) / sum(v for k, v in ss.items() if k != 'dataset'):.1f}"
    v = _get(art, entry["path"])
    if f == "s":
        return v
    if f == "pct":
        return f"{100 * v:.0f}"
    if f == "p3":
        return "< 0.001" if v < 0.001 else f"= {v:.3f}"
    return format(v, f)


def check_traceability() -> list[str]:
    """Every value in docs/summary_values.json re-derives from its artifact; every printed value is in summary.md or
    the artifact index."""
    vf = ROOT / "docs/summary_values.json"
    if not vf.exists():
        return ["docs/summary_values.json missing"]
    values = json.loads(vf.read_text())
    summary = (ROOT / "docs/summary.md").read_text()
    index = (ROOT / "docs/artifact_index.md").read_text()
    errs = []
    for e in values:
        try:
            now = _printed(e)
        except (KeyError, IndexError, FileNotFoundError) as ex:
            errs.append(f"{e['id']}: cannot re-derive ({type(ex).__name__}: {ex})")
            continue
        if now != e["printed"]:
            errs.append(f"{e['id']}: printed {e['printed']} but the artifact gives {now}")
        if f"| {e['id']} | {e['printed']} |" not in index:
            errs.append(f"{e['id']}: not in docs/artifact_index.md")
    return errs


# numbers the manuscript may print that are design constants, not results (source in brackets)
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "17", "20", "30", "40", "50", "95", "97.5", "100",
             "1000", "2016", "2026", "1.0"}
# [protocol: 3-s windows, 50 % overlap, k = 10, 20 % validation, cfg.seed 0, 100 epochs; pre-registration: 97.5 %,
#  1,000 resamples, 40 % secondary threshold; design: 3 seeds, 4/7/8/9/10 datasets, 17 WHAR models, 30 datasets]


def check_manuscript() -> list[str]:
    import re
    import shutil
    import subprocess
    import tempfile
    errs = []
    paper = ROOT / "paper"
    tex = paper / "manuscript.tex"
    if not tex.exists():
        return ["paper/manuscript.tex missing"]
    with tempfile.TemporaryDirectory() as t:
        shutil.copytree(paper, Path(t) / "paper")
        p = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"],
                           cwd=Path(t) / "paper", capture_output=True, text=True)
        log = (Path(t) / "paper" / "manuscript.log").read_text(errors="replace") if (Path(t) / "paper" / "manuscript.log").exists() else ""
        if p.returncode:
            errs.append(f"latexmk exit code {p.returncode}")
        for pat in ("undefined references", "Citation `", "Reference `", "There were undefined"):
            if pat in log:
                errs.append(f"LaTeX log: '{pat}' present")
        pages = re.search(r"Output written on manuscript.pdf \((\d+) page", log)
        if pages and int(pages.group(1)) > 4:
            errs.append(f"{pages.group(1)} pages > 4 (IEEE Sensors Letters limit)")
    src = tex.read_text()
    body = src.split("\\begin{document}", 1)[-1]
    bib = (paper / "references.bib").read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", body) for k in g.split(",")}
    errs += [f"\\cite{{{k}}} not in references.bib" for k in sorted(cited - bib_keys)]
    ver = json.loads((paper / "references_verified.json").read_text())
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    for k in sorted(bib_keys):
        m = re.search(r"@\w+\{" + re.escape(k) + r",(.*?)\n\}", bib, re.S)
        title = re.search(r"title\s*=\s*\{(.*)\},?\s*$", m.group(1), re.M) if m else None
        if k not in ver or ver[k].get("status") != "verified":
            errs.append(f"bib entry {k} has no verified record")
        elif title and norm(title.group(1)) != norm(ver[k]["resolved_title"]):
            errs.append(f"bib title of {k} differs from the resolved title")
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    errs += [f"\\ref{{{r}}} undefined" for r in sorted(refs - labels)]
    errs += [f"{l} never referenced" for l in sorted(labels) if (l.startswith("fig:") or l.startswith("tab:"))
             and l not in refs]
    values = {e["printed"].lstrip("=< ").lstrip("+") for e in json.loads((ROOT / "docs/summary_values.json").read_text())}
    values |= {v.replace("-", "−") for v in values} | {v.lstrip("−-") for v in values}
    for inc in re.findall(r"\\input\{([^}]+)\}", body):  # tables are \input files: check their numbers too
        f = paper / (inc if inc.endswith(".tex") else inc + ".tex")
        body += "\n" + (f.read_text() if f.exists() else "")
    text = re.sub(r"(?<!\\)%.*", "", body)  # comments, not escaped percent signs
    text = re.sub(r"\\(cite|ref|label|includegraphics|input|bibliography|bibliographystyle|begin|end|cline|"
                  r"multicolumn|setlength|vspace|hspace|resizebox)(\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\{[0-9.]+(pt|in|cm|em|mm)\}|[0-9.]+(pt|in|cm|em|mm|\\columnwidth|\\textwidth|\\linewidth)", " ", text)
    for num in re.findall(r"(?<![\w.])[−-]?\d+(?:[.,]\d+)?", text):
        n_ = num.replace(",", "").lstrip("−-")
        if n_ not in values and n_ not in CONSTANTS:
            errs.append(f"number {num} in the manuscript is not in docs/summary_values.json")
    return errs


CHECKS = {"provenance": check_provenance, "ledger": check_ledger, "data": check_data, "synthetic": check_synthetic,
          "recompute": check_recompute, "reproduce": check_reproduce, "seeds": check_seeds,
          "traceability": check_traceability, "manuscript": check_manuscript}
LATER = ["authenticity"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    for name in [*CHECKS, *LATER]:
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--max-errors", type=int, default=20)
    ap.add_argument("--root", type=Path, default=None, help="audit another copy of the project (negative controls)")
    a = ap.parse_args()
    if a.root:
        global ROOT
        ROOT = a.root.resolve()
    chosen = [n for n in CHECKS if a.all or getattr(a, n)]
    failed = False
    for n in chosen:
        errs = CHECKS[n]()
        failed |= bool(errs)
        print(f"{'FAIL' if errs else 'PASS'}  --{n}" + (f"  ({len(errs)} problems)" if errs else ""))
        for e in errs[:a.max_errors]:
            print("      ", e)
        if len(errs) > a.max_errors:
            print(f"       ... and {len(errs) - a.max_errors} more")
    for n in LATER:
        if a.all or getattr(a, n):
            print(f"SKIPPED --{n}  (not implemented yet: added by a later stage)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
