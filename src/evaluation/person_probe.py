"""Where is the headroom? Person-level probes on fixed features (RESEARCH_PLAN.md §5n). No network is trained.

The factor analysis (§5l) puts 61 % of the within-dataset variance on the test person and 14.5 points between seen and
unseen people. This probe asks how much of that gap two kinds of access to the new person would recover, with the
same fixed per-channel features as `value_probe.py` (level3 + statistics) and the same linear probe:
- base: train on the training subjects (train + val windows), test on each test subject's windows;
- shots_k: k labelled windows per class of the test subject are added to training (k = 1, 3; 3 random draws), and
  the subject's remaining windows are scored; base is re-scored on exactly those windows for a paired comparison;
- unlabelled: every subject's features are standardized with that subject's own mean and std (training subjects with
  theirs, the test subject with the mean and std of its unlabelled test windows), i.e. what a device could do after
  observing its wearer without labels;
- unlabelled+shots_1: both.
Macro-F1 uses the harness definition over all classes, per test subject, then averaged over the subjects of a split.

    uv run python -m src.evaluation.person_probe
    uv run python -m src.evaluation.person_probe --summarize
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training import run_whar as R
from src.evaluation.value_probe import LOCAL_OK, features

OUT = R.RESULTS_DIR / "person_probe"
SHOTS = [1, 3]
DRAWS = 3
FEATS = "level3+stats"


def ordered(loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    """(x (N, C, L), y) of a split part in split-index order, transformed as the harness feeds models."""
    y, x = loader.dataset.tensors
    return R.training._to_model_inputs(x, cfg).double().numpy(), R.training._target_labels(y).numpy()


def fit_predict(ftr: np.ndarray, ytr: np.ndarray, fte: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(ftr)
    clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
    return clf.fit(scaler.transform(ftr), ytr).predict(scaler.transform(fte))


def per_subject_standardize(f: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.empty_like(f)
    for s in np.unique(subjects):
        m = subjects == s
        out[m] = (f[m] - f[m].mean(axis=0)) / (f[m].std(axis=0) + 1e-6)
    return out


def run(datasets: list[str]) -> None:
    for d in datasets:
        ds = R.prepare_dataset(d)
        cfg = ds["cfg"]
        nc = int(cfg.num_of_activities)

        def f1(y, p):
            return R.training._classification_metrics(y.tolist(), p.tolist(), nc)["macro_f1"]

        for i, split in enumerate(ds["splits"]):
            out = OUT / d / f"split-{i:02d}.json"
            if out.exists():
                continue
            t0 = time.time()
            loader, dl, _ = R.prepare_split(ds, split)
            # windows as the harness normalizes them, in split-index order (the train DataLoader shuffles, so read its
            # TensorDataset directly; that keeps the subject ids aligned)
            xtr, ytr = ordered(dl["train"], cfg)
            xva, yva = ordered(dl["val"], cfg)
            xte, yte = ordered(dl["test"], cfg)
            s_fit = np.concatenate([[loader.get_item(int(j))[1] for j in split.train_indices],
                                    [loader.get_item(int(j))[1] for j in split.val_indices]])
            s_te = np.asarray([loader.get_item(int(j))[1] for j in split.test_indices])
            assert len(s_te) == len(yte)
            xfit, yfit = np.concatenate([xtr, xva]), np.concatenate([ytr, yva])
            dx_std = np.diff(xfit, axis=-1).std(axis=(0, 2)) + 1e-8
            ffit, fte = features(xfit, dx_std)[FEATS], features(xte, dx_std)[FEATS]
            ffit_u, fte_u = per_subject_standardize(ffit, s_fit), per_subject_standardize(fte, s_te)
            base_pred = fit_predict(ffit, yfit, fte)
            unl_pred = fit_predict(ffit_u, yfit, fte_u)
            res = {"dataset": d, "split": i, "subjects": {}}
            rng = np.random.default_rng([i, 7])
            for subj in np.unique(s_te):
                m = np.where(s_te == subj)[0]
                r = {"n": int(len(m)), "base": f1(yte[m], base_pred[m]), "unlabelled": f1(yte[m], unl_pred[m])}
                for k in SHOTS:
                    for mode in ("shots", "unlabelled+shots") if k == 1 else ("shots",):
                        paired_base, adapted = [], []
                        for _ in range(DRAWS):
                            shot = np.concatenate([rng.permutation(m[yte[m] == c])[:k] for c in np.unique(yte[m])])
                            rest = np.setdiff1d(m, shot)
                            if mode == "shots":
                                pred = fit_predict(np.concatenate([ffit, fte[shot]]), np.concatenate([yfit, yte[shot]]),
                                                   fte[rest])
                                paired_base.append(f1(yte[rest], base_pred[rest]))
                            else:
                                pred = fit_predict(np.concatenate([ffit_u, fte_u[shot]]),
                                                   np.concatenate([yfit, yte[shot]]), fte_u[rest])
                                paired_base.append(f1(yte[rest], base_pred[rest]))
                            adapted.append(f1(yte[rest], pred))
                        r[f"{mode}_{k}"] = st.mean(adapted)
                        r[f"base_on_rest_{mode}_{k}"] = st.mean(paired_base)
                res["subjects"][str(subj)] = r
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=2) + "\n")
            rs = list(res["subjects"].values())
            print(f"{d} split {i}: base {100 * st.mean(x['base'] for x in rs):.1f} | unlabelled "
                  f"{100 * st.mean(x['unlabelled'] for x in rs):.1f} | 1 shot {100 * st.mean(x['shots_1'] for x in rs):.1f}"
                  f" (base on rest {100 * st.mean(x['base_on_rest_shots_1'] for x in rs):.1f}) | 3 shots "
                  f"{100 * st.mean(x['shots_3'] for x in rs):.1f} | unl+1 {100 * st.mean(x['unlabelled+shots_1'] for x in rs):.1f}"
                  f" ({time.time() - t0:.0f}s)", flush=True)


def summarize() -> None:
    from src.evaluation import p5_summary as P
    keys = [("base", "base"), ("unlabelled", "base"), ("shots_1", "base_on_rest_shots_1"),
            ("shots_3", "base_on_rest_shots_3"), ("unlabelled+shots_1", "base_on_rest_unlabelled+shots_1")]
    lines = ["# Person-level headroom on fixed features (linear probe, no network)", "",
             f"Features: {FEATS} (value_probe.py). Macro-F1 %, per test subject then mean over subjects and splits. "
             "Each adapted score is paired with the base model on exactly the windows it is scored on (shots are "
             "removed from scoring); the gain column is that paired difference.", "",
             "| Dataset | base | unlabelled (gain) | 1 shot/class (gain) | 3 shots/class (gain) | unlabelled + 1 shot (gain) |",
             "|---|---|---|---|---|---|"]
    gains = {k: [] for k, _ in keys[1:]}
    bases = []
    for d in LOCAL_OK:
        rs = [json.loads(f.read_text()) for f in sorted((OUT / d).glob("split-*.json"))]
        if len(rs) < P.N_SPLITS[d]:
            continue
        subj = [s for r in rs for s in r["subjects"].values()]
        cells = [f"{100 * st.mean(s['base'] for s in subj):.1f}"]
        bases.append(100 * st.mean(s["base"] for s in subj))
        for k, b in keys[1:]:
            a = 100 * st.mean(s[k] for s in subj)
            g = 100 * st.mean(s[k] - s[b] for s in subj)
            gains[k].append(g)
            cells.append(f"{a:.1f} ({g:+.1f})")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    if bases:
        lines.append(f"| mean ({len(bases)}) | {st.mean(bases):.1f} | " + " | ".join(
            f"({st.mean(v):+.1f})" for v in gains.values()) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=LOCAL_OK)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.datasets)


if __name__ == "__main__":
    main()
