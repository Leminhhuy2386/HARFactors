"""Gate G1 (agents/02_plan.md, Step 6): every planned component imports and runs on the hardware used here.

Checks, by executing them: Python and library versions; a tensor op on every available device; the paper-era guard;
one forward pass of each of the 10 population models at the UTD-MHAD shape; loading one checkpoint per model; the
harness data pipeline on UTD-MHAD (split 0), with the shared cache unchanged before and after; that
scripts/run_analyses.py regenerates results/metrics/ byte for byte (originals restored afterwards whatever happens);
and that the LaTeX toolchain compiles a stub of the manuscript class. No training, no inference on data.

Writes results/gates/G1_environment.json. Tolerance and substitutes are fixed in this file, before measuring.

    uv run --no-sync python scripts/check_environment.py          (on a compute node: scripts/hpc/submit.sh)
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOLERANCE = ("zero planned components missing without a recorded substitute; results/metrics/ regenerated identically "
             "once line endings are normalised, with every floating-point value in a JSON file equal to a relative "
             "1e-9; the shared data cache unchanged by the pipeline")
# Why the tolerance is not 'byte for byte' (rerun 1, 2026-10-15): the first G1 run (G1_environment.json, FAIL)
# compared bytes. Its regenerated files differed from the stored ones (written on the Windows PC) only by CRLF vs LF
# line endings (11 files) and, in results/metrics/factors/data.json, by the last 1-2 digits of OLS coefficients (BLAS/
# LAPACK of the PC vs the Linux node; e.g. 65.436 vs 65.43599999999999). Every rounded number in the .md summaries was
# identical. The comparison below states that equivalence explicitly; anything else still fails.
RERUN_REASON = ("Rerun 1. The first run FAILed on three components. paper_era_guard and models failed through bugs in "
                "this script: whar_arena was imported before run_whar puts it on sys.path, and the forward pass fed "
                "(B, L, C) instead of the harness's (B, C, L). analyses_reproduce failed on bytes: line endings "
                "(CRLF on the PC, LF here) and last-digit floating-point differences in factors/data.json. The tolerance "
                "was changed after that result to 'identical modulo line endings, JSON floats to relative 1e-9'. This is "
                "a declared deviation, reported to the user. Rerun 2: rerun 1 still failed on factors/data.json "
                "because the comparison treated NaN (e.g. Wilcoxon p for n < 5) as unequal to NaN; a diagnostic showed "
                "all 48 differing values within 1.5e-14 relative. NaN == NaN is now equal; the tolerance is unchanged "
                "from rerun 1.")
REL_TOL = 1e-9
# Substitutes declared before measuring (agents/01_system.md: no silent substitution).
SUBSTITUTES = {
    "IEEE_lsens.cls": "IEEEtran.cls [journal,9pt] for drafting. The official IEEE Sensors Letters class could not be "
                      "obtained from the cluster (Overleaf, IEEE and arXiv are behind the site's TLS-inspecting "
                      "firewall). Swap in IEEE_lsens.cls from the IEEE/Overleaf template before submission and re-check "
                      "the 4-page limit.",
}
POPULATION = ["recurrencenet", "s4d", "hystnet_sel_clock", "quantilenet", "mamba", "hystnet_nomem", "recstatnet",
              "lawnet_rec", "lawnet", "statnet"]
LIBS = ["numpy", "scipy", "pandas", "statsmodels", "sklearn", "matplotlib", "torch"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_manifest(d: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


def check(name: str, fn, out: dict, failed: list) -> None:
    try:
        out[name] = fn()
    except Exception as e:  # recorded, never hidden
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        failed.append(name)


def versions() -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform(), "host": platform.node()}
    for lib in LIBS:
        v[lib] = importlib.import_module(lib).__version__
    return v


def devices() -> dict:
    import torch
    r = {"cpu": float((torch.ones(2, 2) @ torch.ones(2, 2)).sum())}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        r["cuda"] = {"name": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
                     "op": float((torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).sum())}
    else:
        r["cuda"] = None
    return r


def guard() -> dict:
    from src.training import run_whar as R  # first: it puts the pinned whar-arena source on sys.path
    from whar_arena.protocol import load_protocol
    return {"commits": R.check_paper_era(load_protocol()), "datasets_dir": str(R.DATASETS_DIR)}


def models() -> dict:
    import torch

    from src.models.registry import build
    r = {}
    for m in POPULATION:
        torch.manual_seed(0)
        net = build(m, input_channels=6, window_length=150, num_classes=27).eval()
        with torch.no_grad():  # the harness feeds (B, C, L): whar_arena.training._to_model_inputs transposes (B, L, C)
            y = net(torch.zeros(2, 6, 150))
        r[m] = {"params": sum(p.numel() for p in net.parameters()), "out": list(y.shape)}
        assert list(y.shape) == [2, 27], (m, y.shape)
    return r


def checkpoints() -> dict:
    import torch

    from src.models.registry import build
    r = {}
    for m in POPULATION:
        f = ROOT / "experiments" / "checkpoints" / m / "utd_mhad" / "seed-0" / "split-00.pt"
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net = build(m, input_channels=6, window_length=150, num_classes=27)
        net.load_state_dict(ck["state_dict"], strict=True)
        r[m] = {"file": str(f.relative_to(ROOT)), "stored_macro_f1": ck.get("macro_f1"), "device": ck.get("device")}
    return r


def pipeline() -> dict:
    from src.training import run_whar as R
    cache = R.DATASETS_DIR / "utd_mhad"
    before = tree_manifest(cache)
    ds = R.prepare_dataset("utd_mhad")
    _, dl, shape = R.prepare_split(ds, ds["splits"][0])
    after = tree_manifest(cache)
    sizes = {k: len(v.dataset) for k, v in dl.items()}
    assert before == after, "the harness changed the shared cache"
    return {"shape_L_C_classes": list(shape), "split0_sizes": sizes, "cache_files": len(before), "cache_unchanged": True}


def same_json(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, float) and isinstance(b, float) and a != a and b != b:  # both NaN
            return True
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and (
            a == b or abs(a - b) <= REL_TOL * max(abs(a), abs(b)))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return a == b


def analyses() -> dict:
    metrics = ROOT / "results" / "metrics"
    backup = Path(tempfile.mkdtemp(prefix="g1_metrics_"))
    shutil.copytree(metrics, backup / "metrics")
    old = backup / "metrics"
    names = sorted(str(p.relative_to(metrics)) for p in metrics.rglob("*") if p.is_file())
    try:
        proc = subprocess.run([sys.executable, "scripts/run_analyses.py"], cwd=ROOT, capture_output=True, text=True)
        new = {n: (metrics / n).read_bytes() for n in sorted(str(p.relative_to(metrics)) for p in metrics.rglob("*")
                                                             if p.is_file())}
        identical, line_endings, float_digits, differ = [], [], [], []
        for n in names:
            a, b = (old / n).read_bytes(), new.get(n)
            if b is None:
                differ.append(n)
            elif a == b:
                identical.append(n)
            elif a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
                line_endings.append(n)
            elif n.endswith(".json") and same_json(json.loads(a), json.loads(b)):
                float_digits.append(n)
            else:
                differ.append(n)
        added = sorted(set(new) - set(names))
    finally:  # leave results/metrics exactly as found
        shutil.rmtree(metrics)
        shutil.copytree(old, metrics)
        shutil.rmtree(backup)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not differ and not added, f"differ {differ}, added {added}"
    return {"files": len(names), "byte_identical": identical, "line_endings_only": line_endings,
            "json_floats_within_rel_1e-9": float_digits, "differ": differ, "added": added,
            "exit_code": proc.returncode}


def latex() -> dict:
    stub = r"""\documentclass[journal,9pt]{IEEEtran}
\begin{document}
\title{Stub}\author{G1}\maketitle
\begin{abstract}Environment check.\end{abstract}
Text $x^2$.
\end{document}
"""
    d = Path(tempfile.mkdtemp(prefix="g1_tex_"))
    (d / "stub.tex").write_text(stub)
    proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "stub.tex"], cwd=d,
                          capture_output=True, text=True)
    ok = proc.returncode == 0 and (d / "stub.pdf").exists()
    ver = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    shutil.rmtree(d)
    assert ok, proc.stdout[-2000:]
    return {"pdflatex": shutil.which("pdflatex"), "version": ver, "class": "IEEEtran [journal,9pt]", "exit_code": 0}


def vcs() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
    if head.returncode:
        return {"vcs": "none"}
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"vcs": "git", "commit": head.stdout.strip(), "dirty_files": len(dirty.splitlines())}


def main() -> None:
    t0 = time.time()
    obs, failed = {}, []
    for name, fn in [("versions", versions), ("devices", devices), ("paper_era_guard", guard), ("models", models),
                     ("checkpoints", checkpoints), ("data_pipeline", pipeline), ("analyses_reproduce", analyses),
                     ("latex", latex), ("vcs", vcs)]:
        check(name, fn, obs, failed)
        print(name, "FAILED" if name in failed else "ok", flush=True)
    obs["failed_components"] = failed
    obs["accelerator_used"] = ("CUDA " + obs["devices"]["cuda"]["name"]) if isinstance(obs.get("devices"), dict) \
        and obs["devices"].get("cuda") else "CPU only (no GPU on this node; the analyses and checkpoint inference of "\
        "these small models run on CPU)"
    gate = {
        "gate": "G1_environment", "stage": "02_plan",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "check": "all planned components import and execute on the available hardware",
        "reference_value": None,
        "reference_source": "docs/research_plan.md and docs/preregistration.md (components the plan names)",
        "observed_value": obs,
        "tolerance": TOLERANCE,
        "substitutes": SUBSTITUTES,
        "verdict": "PASS" if not failed else "FAIL",
        "evidence": ["scripts/check_environment.py", "scripts/hpc/setup.sh", "uv.lock"],
        "notes": f"Ran on {platform.node()} via Slurm in {time.time() - t0:.0f} s; SLURM_JOB_ID="
                 f"{os.environ.get('SLURM_JOB_ID')}.",
    }
    out = ROOT / "results" / "gates" / "G1_environment.json"
    if out.exists():  # never edit a gate file; a re-run gets its own name and states why
        k = 1
        while (out.parent / f"G1_environment_rerun{k}.json").exists():
            k += 1
        out = out.parent / f"G1_environment_rerun{k}.json"
        gate["rerun_reason"] = RERUN_REASON
    out.write_text(json.dumps(gate, indent=1, default=str) + "\n")
    print("wrote", out.relative_to(ROOT), "verdict", gate["verdict"])


if __name__ == "__main__":
    main()
