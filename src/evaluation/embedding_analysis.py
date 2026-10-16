"""Exploratory: what the networks' embeddings say about the test-subject effect (inference only, no training).

For every stored seed-0 checkpoint of the population models on the datasets whose cluster cache the stored runs can be
re-executed on (S1: 8 datasets, 10 models; S2: up_fall, wear, uca_ehar, 4 models), the harness rebuilds each split
and the network embeds its windows. The embedding is the vector the classification head receives (pooled, after the
final LayerNorm), taken with a forward pre-hook on `model.head`; every population model ends in that head.

Per (model, dataset, split):
- test macro-F1 of the regenerated predictions (equals the stored run within the G5 tolerance);
- class-conditional shift of the test-subject group from the training subjects:
  CCS = mean over classes present in both of ||mu_test,k - mu_train,k|| / sqrt(tr Sigma_train,k / d),
  in the embedding space and in a sensor-feature space (per-channel mean, sd, min, max of the normalised window);
- subject information in the training embeddings: balanced accuracy of a linear probe (logistic regression) that
  predicts the training subject, with folds grouped by session so that overlapping windows never straddle folds;
  the same for the activity. Chance for subjects is 1 / number of training subjects.
At most PROBE_WINDOWS training windows enter a probe (a fixed, seeded subsample).

This analysis was not pre-registered (docs/preregistration.md, § Exploratory).

    WHAR_DATASETS_DIR=<private cache copy> uv run --no-sync python -m src.evaluation.embedding_analysis --datasets uci_har sad
    -> results/metrics/embedding/<dataset>.json (one per dataset, with provenance), aggregated by --aggregate
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.sets import S1, S2
from src.models.registry import build
from src.training import run_whar as R
from src.utils import paths as PATHS
from src.utils import provenance as PV
from whar_arena import training as T

OUT = PATHS.METRICS / "embedding"
SEED = 20261016
PROBE_WINDOWS = 4000
PROBE_FOLDS = 5
S1_DATASETS = ["utd_mhad", "uci_har", "unimib_shar", "actrectut_gestures", "opportunity", "sad", "hapt", "wisdm"]
S2_DATASETS = ["up_fall", "wear", "uca_ehar"]
TSNE_CASE = ("uci_har", "s4d", 0)  # one illustrative case: 6 activities, 3 test subjects, 27 training subjects
TSNE_TRAIN = 3000


def models_for(dataset: str) -> list[str]:
    return S1["models"] if dataset in S1_DATASETS else S2["models"]


@torch.no_grad()
def embed(net: torch.nn.Module, loader, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings (input of the head), labels and predictions of every window, in loader order."""
    grabbed = []
    hook = net.head.register_forward_pre_hook(lambda m, inp: grabbed.append(inp[0].detach().cpu()))
    ys, ps = [], []
    for labels, inputs in loader:
        ys.append(T._target_labels(labels))
        ps.append(net(T._to_model_inputs(inputs, cfg)).argmax(dim=1))
    hook.remove()
    return torch.cat(grabbed).numpy(), torch.cat(ys).numpy(), torch.cat(ps).numpy()


def sensor_features(loader) -> np.ndarray:
    """Per-channel mean, sd, min and max of each normalised window (B, L, C) -> (B, 4C)."""
    out = []
    for _, x in loader:
        out.append(torch.cat([x.mean(1), x.std(1), x.amin(1), x.amax(1)], dim=1))
    return torch.cat(out).numpy()


def ccs(train: np.ndarray, y_train: np.ndarray, test: np.ndarray, y_test: np.ndarray) -> float:
    vals = []
    for k in np.intersect1d(np.unique(y_train), np.unique(y_test)):
        a, b = train[y_train == k], test[y_test == k]
        if len(a) < 3 or len(b) < 1:
            continue
        spread = math.sqrt(max(np.trace(np.cov(a, rowvar=False)) / a.shape[1], 1e-12))
        vals.append(float(np.linalg.norm(b.mean(0) - a.mean(0)) / spread))
    return float(np.mean(vals)) if vals else float("nan")


def probe(x: np.ndarray, target: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> float:
    """Balanced accuracy of a linear probe, folds grouped by session."""
    if len(x) > PROBE_WINDOWS:
        keep = np.sort(rng.choice(len(x), PROBE_WINDOWS, replace=False))
        x, target, groups = x[keep], target[keep], groups[keep]
    if len(np.unique(target)) < 2:
        return float("nan")
    folds = min(PROBE_FOLDS, len(np.unique(groups)))
    pred = np.empty_like(target)
    for tr, te in GroupKFold(n_splits=folds).split(x, target, groups):
        sc = StandardScaler().fit(x[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=300).fit(sc.transform(x[tr]), target[tr])
        pred[te] = clf.predict(sc.transform(x[te]))
    return float(balanced_accuracy_score(target, pred))


def analyse_dataset(dataset: str, run_id: str, started: float) -> dict:
    ds = R.prepare_dataset(dataset)
    cfg, wdf, sdf = ds["cfg"], ds["window_df"], ds["session_df"]
    subject_of_session = dict(zip(sdf["session_id"], sdf["subject_id"]))
    rows, inputs = [], []
    for i, split in enumerate(ds["splits"]):
        _, dl, shape = R.prepare_split(ds, split)
        L, C, K = shape
        names = R.channel_names(cfg, C)
        train_sessions = wdf.loc[list(split.train_indices), "session_id"].to_numpy()
        train_subjects = np.array([subject_of_session[s] for s in train_sessions])
        rng = np.random.default_rng(SEED + i)
        # dl["train"] shuffles: read the training windows in index order, aligned with train_sessions
        ordered = torch.utils.data.DataLoader(dl["train"].dataset, batch_size=512, shuffle=False)
        f_tr, f_te = sensor_features(ordered), sensor_features(dl["test"])
        y_tr_raw = torch.cat([T._target_labels(y) for y, _ in ordered]).numpy()
        y_te = torch.cat([T._target_labels(y) for y, _ in dl["test"]]).numpy()
        sensor = {"ccs": ccs(f_tr, y_tr_raw, f_te, y_te),
                  "subject_probe": probe(f_tr, train_subjects, train_sessions, rng),
                  "activity_probe": probe(f_tr, y_tr_raw, train_sessions, rng)}
        for m in models_for(dataset):
            ck = PATHS.CHECKPOINTS / m / dataset / "seed-0" / f"split-{i:02d}.pt"
            if not ck.exists():
                rows.append({"model": m, "split": i, "status": "no checkpoint"})
                continue
            state = torch.load(ck, map_location="cpu", weights_only=False)
            torch.manual_seed(0)
            net = build(m, input_channels=C, window_length=L, num_classes=K, channel_names=names)
            net.load_state_dict(state["state_dict"], strict=True)
            net.eval()
            e_tr, y_tr, _ = embed(net, ordered, cfg)
            e_te, y_te2, p_te = embed(net, dl["test"], cfg)
            if (dataset, m, i) == TSNE_CASE:
                tsne_case(e_tr, y_tr, train_subjects, e_te, y_te2, wdf, split, subject_of_session, run_id, started, ck)
            f1 = T._classification_metrics(y_te2.tolist(), p_te.tolist(), K)["macro_f1"]
            stored = json.loads((PATHS.RUNS / m / dataset / f"split-{i:02d}.json").read_text())["metrics"]["macro_f1"]
            rows.append({"model": m, "split": i, "status": "ok", "test_subjects": len({subject_of_session[s] for s in
                         wdf.loc[list(split.test_indices), "session_id"]}),
                         "train_subjects": int(len(np.unique(train_subjects))), "embedding_dim": int(e_tr.shape[1]),
                         "macro_f1": 100 * f1, "stored_macro_f1": 100 * stored,
                         "embedding_ccs": ccs(e_tr, y_tr, e_te, y_te2),
                         "embedding_subject_probe": probe(e_tr, train_subjects, train_sessions, rng),
                         "embedding_activity_probe": probe(e_tr, y_tr, train_sessions, rng),
                         "sensor_ccs": sensor["ccs"], "sensor_subject_probe": sensor["subject_probe"],
                         "sensor_activity_probe": sensor["activity_probe"],
                         "subject_chance": 1 / len(np.unique(train_subjects))})
            inputs.append(ck)
        print(f"{dataset} split {i}: done", flush=True)
    out = {"dataset": dataset, "seed": SEED, "probe_windows": PROBE_WINDOWS, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{dataset}.json"
    f.write_text(json.dumps(out, indent=1) + "\n")
    stored_runs = [PATHS.RUNS / r["model"] / dataset / f"split-{r['split']:02d}.json" for r in rows if r["status"] == "ok"]
    PV.write(f, run_id=run_id, started=started, inputs=inputs + stored_runs, config={"dataset": dataset, "seed": SEED},
             produced_by="src/evaluation/embedding_analysis.py")
    return out


def tsne_case(e_tr, y_tr, s_tr, e_te, y_te, wdf, split, subject_of_session, run_id, started, ck) -> None:
    """2-D t-SNE of the training (subsample) and test embeddings of TSNE_CASE, saved for the figure."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(len(e_tr), min(TSNE_TRAIN, len(e_tr)), replace=False))
    s_te = np.array([subject_of_session[s] for s in wdf.loc[list(split.test_indices), "session_id"]])
    x = np.vstack([e_tr[keep], e_te])
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(x)
    n = len(keep)
    out = {"case": {"dataset": TSNE_CASE[0], "model": TSNE_CASE[1], "split": TSNE_CASE[2]}, "seed": SEED,
           "train": {"xy": xy[:n].tolist(), "activity": y_tr[keep].tolist(), "subject": s_tr[keep].tolist()},
           "test": {"xy": xy[n:].tolist(), "activity": y_te.tolist(), "subject": s_te.tolist()}}
    f = OUT / "tsne_case.json"
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=[ck], config={"case": list(TSNE_CASE), "seed": SEED},
             produced_by="src/evaluation/embedding_analysis.py (t-SNE case)")


def within_rank(values: dict) -> dict:
    """Rank-transform per dataset (so datasets of different difficulty can be pooled)."""
    return {k: stats.rankdata(v) / len(v) for k, v in values.items()}


def aggregate(run_id: str, started: float) -> None:
    """Across datasets: does the test group's shift track the split's macro-F1? How much subject identity is decodable?"""
    files = sorted(p for p in OUT.glob("*.json") if p.stem in S1_DATASETS + S2_DATASETS)
    per_ds, probes = {}, []
    for f in files:
        d = json.loads(f.read_text())
        ok = [r for r in d["rows"] if r["status"] == "ok"]
        splits = sorted({r["split"] for r in ok})
        by = lambda key: [st.mean(r[key] for r in ok if r["split"] == s) for s in splits]  # noqa: E731
        f1, ce, cs = by("macro_f1"), by("embedding_ccs"), by("sensor_ccs")
        rho = lambda a: float(stats.spearmanr(a, f1).statistic) if len(splits) > 3 else float("nan")  # noqa: E731
        per_ds[d["dataset"]] = {"splits": len(splits), "mean_macro_f1_per_split": f1, "embedding_ccs_per_split": ce,
                                "sensor_ccs_per_split": cs, "rho_embedding_ccs_f1": rho(ce), "rho_sensor_ccs_f1": rho(cs)}
        for r in ok:
            probes.append({"dataset": d["dataset"], "model": r["model"], "subject_probe": r["embedding_subject_probe"],
                           "sensor_subject_probe": r["sensor_subject_probe"], "chance": r["subject_chance"],
                           "activity_probe": r["embedding_activity_probe"]})
    usable = {d: v for d, v in per_ds.items() if v["splits"] > 3}
    pooled_e = stats.spearmanr(np.concatenate([stats.rankdata(v["embedding_ccs_per_split"]) for v in usable.values()]),
                               np.concatenate([stats.rankdata(v["mean_macro_f1_per_split"]) for v in usable.values()]))
    pooled_s = stats.spearmanr(np.concatenate([stats.rankdata(v["sensor_ccs_per_split"]) for v in usable.values()]),
                               np.concatenate([stats.rankdata(v["mean_macro_f1_per_split"]) for v in usable.values()]))
    rho_e = [v["rho_embedding_ccs_f1"] for v in usable.values()]
    rho_s = [v["rho_sensor_ccs_f1"] for v in usable.values()]
    ratio = [p["subject_probe"] / p["chance"] for p in probes if not math.isnan(p["subject_probe"])]
    sratio = [p["sensor_subject_probe"] / p["chance"] for p in probes if not math.isnan(p["sensor_subject_probe"])]
    summary = {
        "datasets": len(per_ds), "datasets_with_more_than_3_splits": len(usable),
        "pooled_within_dataset_rank_rho_embedding_ccs_f1": float(pooled_e.statistic),
        "pooled_within_dataset_rank_p_embedding": float(pooled_e.pvalue),
        "pooled_within_dataset_rank_rho_sensor_ccs_f1": float(pooled_s.statistic),
        "pooled_within_dataset_rank_p_sensor": float(pooled_s.pvalue),
        "median_dataset_rho_embedding": float(np.median(rho_e)), "median_dataset_rho_sensor": float(np.median(rho_s)),
        "datasets_negative_rho_embedding": int(sum(r < 0 for r in rho_e)),
        "datasets_negative_rho_sensor": int(sum(r < 0 for r in rho_s)),
        "median_subject_probe_over_chance_embedding": float(np.median(ratio)),
        "median_subject_probe_over_chance_sensor": float(np.median(sratio)),
        "median_activity_probe_embedding": float(np.median([p["activity_probe"] for p in probes])),
        "probe_cases": len(probes), "per_dataset": per_ds, "probes": probes,
    }
    f = OUT / "summary.json"
    f.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=files, produced_by="src/evaluation/embedding_analysis.py (aggregate)")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_dataset", "probes")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    run_id = PV.new_run_id("embedding" + ("_aggregate" if a.aggregate else "_" + "_".join(a.datasets)))
    run_dir = PATHS.EXPERIMENTS / run_id
    run_dir.mkdir()
    config = {"datasets": a.datasets, "aggregate": a.aggregate, "seed": SEED, "probe_windows": PROBE_WINDOWS,
              "datasets_dir": str(R.DATASETS_DIR)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    started = time.time()
    PV.ledger_start(run_id, config)
    try:
        for d in a.datasets:
            with open(PATHS.RESULTS / "test_access.log", "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | src/evaluation/embedding_analysis.py | {run_id} | "
                         f"exploratory embedding analysis: test windows of {d} embedded by stored checkpoints\n")
            analyse_dataset(d, run_id, started)
        if a.aggregate:
            aggregate(run_id, started)
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    PV.ledger_end(run_id, "embedding analysis " + (" ".join(a.datasets) or "aggregate"), started)


if __name__ == "__main__":
    sys.exit(main())
