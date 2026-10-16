"""Verification aid: recover the file-listing order a dataset's stored runs were built with (no training).

Some whar-datasets parsers number activities and subjects in the order `os.listdir` returns (dsads, daphnet, usc_had,
motion_sense). The stored runs of these datasets were built on cloud VMs (Colab, Kaggle), whose listing order is not
the one of the Windows PC (sorted) or of this cluster (Lustre), so a cache built here numbers the classes and the
subjects differently and the checkpoints cannot be re-executed on it.

The checkpoints themselves record that order:
- the label mapping: a network is near-perfect on its own training subjects, so matching its output units to the
  cache's labels (Hungarian assignment on the confusion matrix over all windows) gives, for each output unit k, the
  activity it stands for; the k-th activity is the k-th one the VM listed;
- the subject order: the harness's group k-fold puts subject index i in test group i (fewer subjects than 10 groups),
  so the subject on which split i's network is least accurate is the VM's subject i.
Every seed-0 checkpoint of every network votes; the result must be unanimous. The windows are normalised with the
statistics of one split of the existing cache, which is enough for argmax decisions but not for exact metrics: the
order is then checked by rebuilding the cache in that order and re-executing the checkpoints (scripts/evaluate.py).

    WHAR_DATASETS_DIR=<private cache copy> uv run --no-sync python scripts/reconstruct_order.py infer dsads
        -> results/metrics/reproducibility/order_dsads.json
    WHAR_DATASETS_DIR=<new private dir> uv run --no-sync python scripts/reconstruct_order.py build dsads
        -> a cache in the inferred order, and results/metrics/reproducibility/build_dsads_<file order>.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from src.data.sets import S1, S2  # noqa: E402
from src.models.registry import build  # noqa: E402
from src.training import run_whar as R  # noqa: E402
from src.utils import paths as PATHS  # noqa: E402
from src.utils import provenance as PV  # noqa: E402
from whar_arena import training as T  # noqa: E402

OUT = PATHS.METRICS / "reproducibility"


def spec_of(dataset: str) -> dict:
    return S1 if dataset in S1["datasets"] else S2


@torch.no_grad()
def all_windows(dataset: str):
    """Every window of the existing cache: inputs, labels, subject ids (cache numbering), normalised with split 0."""
    ds = R.prepare_dataset(dataset)
    wdf, sdf = ds["window_df"], ds["session_df"]
    subj = dict(zip(sdf["session_id"], sdf["subject_id"]))
    split = ds["splits"][0]
    _, dl, shape = R.prepare_split(ds, split)
    xs, ys, ss = [], [], []
    for part, idx in (("train", split.train_indices), ("val", split.val_indices), ("test", split.test_indices)):
        loader = torch.utils.data.DataLoader(dl[part].dataset, batch_size=512, shuffle=False)
        for labels, inputs in loader:
            ys.append(T._target_labels(labels))
            xs.append(T._to_model_inputs(inputs, ds["cfg"]))
        ss += [subj[s] for s in wdf.loc[list(idx), "session_id"]]
    return ds, torch.cat(xs), torch.cat(ys).numpy(), np.array(ss), shape


def selected(spec: dict, dataset: str, m: str, s: int, i: int, seeds, device) -> bool:
    """A run belongs to the group whose order is inferred: its seed and the GPU its record names (one cloud VM)."""
    if seeds is not None and s not in seeds:
        return False
    if device is None:
        return True
    f = PATHS.RUNS / m / dataset / (f"seed-{s}" if s else "") / f"split-{i:02d}.json"
    return json.loads(f.read_text()).get("device") == device


def infer(dataset: str, run_id: str, started: float, seeds=None, device=None, tag: str = "") -> dict:
    ds, x, y, s, (L, C, K) = all_windows(dataset)
    names = R.channel_names(ds["cfg"], C)
    act = pd.read_csv(Path(R.DATASETS_DIR) / dataset / "metadata" / "activity_df.csv")
    act_name = dict(zip(act["activity_id"], act["activity_name"]))
    spec = spec_of(dataset)
    votes, rows, inputs = [], [], []
    for m in spec["models"]:
        for s_, i in [(s_, i) for s_ in (seeds if seeds is not None else [0]) for i in range(spec["datasets"][dataset])]:
            ck = PATHS.CHECKPOINTS / m / dataset / f"seed-{s_}" / f"split-{i:02d}.pt"
            if not ck.exists() or not selected(spec, dataset, m, s_, i, seeds, device):
                continue
            torch.manual_seed(0)
            net = build(m, input_channels=C, window_length=L, num_classes=K, channel_names=names)
            net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"], strict=True)
            net.eval()
            with torch.no_grad():
                pred = torch.cat([net(x[j:j + 1024]).argmax(1) for j in range(0, len(x), 1024)]).numpy()
            conf = np.zeros((K, K))
            np.add.at(conf, (pred, y), 1)  # rows: output unit, columns: cache label
            unit, label = linear_sum_assignment(-conf)
            mapping = dict(zip(unit.tolist(), label.tolist()))
            mapped = np.array([mapping[p] for p in pred])
            acc = {int(k): float((mapped[s == k] == y[s == k]).mean()) for k in np.unique(s)}
            test_subject = min(acc, key=acc.get)
            votes.append((tuple(mapping[k] for k in range(K)), i, test_subject))
            rows.append({"model": m, "seed": s_, "split": i, "unit_to_cache_label": [mapping[k] for k in range(K)],
                         "accuracy_by_cache_subject": acc, "least_accurate_subject": int(test_subject),
                         "mapped_accuracy": float((mapped == y).mean())})
            inputs.append(ck)
        print(f"{dataset} {m}: done", flush=True)
    label_votes = Counter(v[0] for v in votes)
    order, n = label_votes.most_common(1)[0]
    subject_of_split = {}
    for i in range(spec["datasets"][dataset]):
        c = Counter(v[2] for v in votes if v[1] == i)
        subject_of_split[i] = {"subject": int(c.most_common(1)[0][0]), "votes": dict(c)}
    out = {"dataset": dataset, "group": {"seeds": seeds if seeds is not None else [0], "device": device, "tag": tag},
           "cache": str(Path(R.DATASETS_DIR) / dataset), "checkpoints": len(votes),
           "label_order_unanimous": n == len(votes), "label_order_votes": n,
           "vm_label_order_as_cache_labels": list(order),
           "vm_label_order_names": [act_name[j] for j in order],
           "vm_subject_order_as_cache_subjects": [subject_of_split[i]["subject"] for i in sorted(subject_of_split)],
           "subject_votes": subject_of_split,
           "subject_order_unanimous": all(len(v["votes"]) == 1 for v in subject_of_split.values()),
           "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"order_{dataset}{'_' + tag if tag else ''}.json"
    f.write_text(json.dumps(out, indent=1) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=inputs,
             config={"dataset": dataset, "mode": "infer", "seeds": seeds, "device": device, "tag": tag},
             produced_by="scripts/reconstruct_order.py")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "subject_votes")}, indent=1))
    return out


def listing_rules(dataset: str, order: dict, file_order: str):
    """Directory -> name order the VM listed, for the directories the dataset's parser lists (dsads so far)."""
    if dataset != "dsads":
        raise NotImplementedError(dataset)
    from whar_datasets.config.cfg_dsads import ACTIVITY_MAP
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731  cache names are canonicalised
    by_name = {norm(v): k for k, v in ACTIVITY_MAP.items()}
    acts = [f"a{by_name[norm(nm)]:02d}" for nm in order["vm_label_order_names"]]
    subs = [f"p{k + 1}" for k in order["vm_subject_order_as_cache_subjects"]]  # cache built sorted: p1..p8 -> 0..7
    base = Path(R.DATASETS_DIR).resolve() / dataset / "data" / "data"

    def rule(path: Path, names: list) -> list:
        if path == base:
            return acts + sorted(set(names) - set(acts))
        if path.parent == base:
            return [s for s in subs if s in names] + sorted(set(names) - set(subs))
        if path.parent.parent == base:
            return sorted(names) if file_order == "sorted" else names
        return names
    return rule


def build_cache(dataset: str, file_order: str, run_id: str, started: float, tag: str = "") -> dict:
    """Rebuild the cache of `dataset` in R.DATASETS_DIR (a private directory) in the inferred listing order, from hard
    links to HAR's raw files (scripts/prepare_data.py), and compare the split sizes with the stored runs."""
    import os
    import scripts.prepare_data as PD
    order_f = OUT / f"order_{dataset}{'_' + tag if tag else ''}.json"
    order = json.loads(order_f.read_text())
    rule = listing_rules(dataset, order, file_order)
    original = os.listdir

    def listdir(path="."):
        names = original(path)
        return rule(Path(path).resolve(), names)

    before = PD.manifest(PD.HAR_RAW / dataset)
    PD.link_raw(dataset)
    os.listdir = listdir
    try:
        PD.build(dataset)
    finally:
        os.listdir = original
    if PD.manifest(PD.HAR_RAW / dataset) != before:
        raise RuntimeError("HAR raw files changed")
    ds = R.prepare_dataset(dataset)
    spec = spec_of(dataset)
    ref = order["rows"][0]
    sizes = []
    for i, split in enumerate(ds["splits"]):
        got = [len(split.train_indices), len(split.val_indices), len(split.test_indices)]
        s0 = ref["seed"]
        src = json.loads((PATHS.RUNS / ref["model"] / dataset / (f"seed-{s0}" if s0 else "") / f"split-{i:02d}.json")
                         .read_text())["source"]
        want = [src["train_size"], src["val_size"], src["test_size"]]
        sizes.append({"split": i, "rebuilt": got, "stored": want, "equal": got == want})
    out = {"dataset": dataset, "datasets_dir": str(R.DATASETS_DIR), "file_order": file_order,
           "activity_dirs": rule(Path(R.DATASETS_DIR).resolve() / dataset / "data" / "data", []),
           "split_sizes": sizes, "all_sizes_equal": all(s["equal"] for s in sizes)}
    f = OUT / f"build_{dataset}{'_' + tag if tag else ''}_{file_order}.json"
    f.write_text(json.dumps(out, indent=1) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=[order_f],
             config={"dataset": dataset, "mode": "build", "file_order": file_order, "tag": tag},
             produced_by="scripts/reconstruct_order.py")
    print(json.dumps({k: v for k, v in out.items() if k != "split_sizes"}, indent=1), flush=True)
    return out


def reexecute(dataset: str, run_id: str, started: float) -> dict:
    """Per checkpoint, for runs whose VM order cannot be rebuilt as one cache (several VMs per run set): infer its
    label mapping and its test subjects from the checkpoint as in `infer`, check that those subjects' windows number
    exactly the record's test size, normalise with the harness's train-only z-score fitted on every other window (the
    run's training and validation windows), classify the test windows and compare the metrics with the record."""
    from whar_datasets.splitting.split import Split
    ds, x, y, s, (L, C, K) = all_windows(dataset)
    names = R.channel_names(ds["cfg"], C)
    wdf, sdf = ds["window_df"], ds["session_df"]
    subj_of_window = wdf["session_id"].map(dict(zip(sdf["session_id"], sdf["subject_id"]))).to_numpy()
    subjects = sorted(np.unique(subj_of_window).tolist())
    count = {q: int((subj_of_window == q).sum()) for q in subjects}
    k = min(int(ds["cfg"].num_folds), len(subjects))
    spec = spec_of(dataset)
    cache, rows, inputs = {}, [], []
    for m in spec["models"]:
        for s_ in spec["seeds"]:
            for i in range(spec["datasets"][dataset]):
                ck = PATHS.CHECKPOINTS / m / dataset / f"seed-{s_}" / f"split-{i:02d}.pt"
                rec_f = PATHS.RUNS / m / dataset / (f"seed-{s_}" if s_ else "") / f"split-{i:02d}.json"
                if not ck.exists():
                    continue
                rec = json.loads(rec_f.read_text())
                torch.manual_seed(0)
                net = build(m, input_channels=C, window_length=L, num_classes=K, channel_names=names)
                net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"], strict=True)
                net.eval()
                with torch.no_grad():
                    pred = torch.cat([net(x[j:j + 1024]).argmax(1) for j in range(0, len(x), 1024)]).numpy()
                conf = np.zeros((K, K))
                np.add.at(conf, (pred, y), 1)
                unit, label = linear_sum_assignment(-conf)
                mapping = dict(zip(unit.tolist(), label.tolist()))
                inv = {v: kk for kk, v in mapping.items()}
                mapped = np.array([mapping[p_] for p_ in pred])
                acc = {int(q): float((mapped[s == q] == y[s == q]).mean()) for q in np.unique(s)}
                n_test = len(range(i, len(subjects), k))
                # the record's test size pins the test group down when one subset of n_test subjects has exactly that
                # many windows (a structural fact of the split, not a metric); otherwise the least accurate subjects
                want = rec["source"]["test_size"]
                fits = [c for c in itertools.combinations(subjects, n_test) if sum(count[q] for q in c) == want]
                if len(fits) == 1:
                    test_subjects, how = sorted(fits[0]), "test size"
                elif fits:
                    test_subjects = sorted(min(fits, key=lambda c: np.mean([acc[q] for q in c])))
                    how = "test size, then accuracy"
                else:
                    test_subjects, how = sorted(sorted(acc, key=acc.get)[:n_test]), "accuracy"
                key = tuple(test_subjects)
                if key not in cache:
                    test_idx = wdf.index[np.isin(subj_of_window, test_subjects)].tolist()
                    rest = wdf.index[~np.isin(subj_of_window, test_subjects)].tolist()
                    split = Split(identifier=f"reexec_{dataset}_" + "_".join(map(str, key)), train_indices=rest,
                                  val_indices=rest[:1], test_indices=test_idx)
                    _, dl, _ = R.prepare_split(ds, split)
                    xt, yt = [], []
                    for labels, inputs_ in dl["test"]:
                        yt.append(T._target_labels(labels))
                        xt.append(T._to_model_inputs(inputs_, ds["cfg"]))
                    cache[key] = (torch.cat(xt), torch.cat(yt).numpy())
                xt, yt = cache[key]
                with torch.no_grad():
                    pt = torch.cat([net(xt[j:j + 1024]).argmax(1) for j in range(0, len(xt), 1024)]).numpy()
                y_vm = np.array([inv[v] for v in yt])
                met = T._classification_metrics(y_vm.tolist(), pt.tolist(), K)
                st_ = rec["metrics"]
                win = round(abs(met["accuracy"] - st_["accuracy"]) * len(yt))
                rows.append({"model": m, "seed": s_, "split": i, "device": rec.get("device"),
                             "test_subjects_cache_ids": list(key), "test_group_identified_by": how,
                             "test_size": len(yt),
                             "stored_test_size": rec["source"]["test_size"],
                             "test_size_equal": len(yt) == rec["source"]["test_size"],
                             "macro_f1": met["macro_f1"], "stored_macro_f1": st_["macro_f1"],
                             "accuracy": met["accuracy"], "stored_accuracy": st_["accuracy"],
                             "delta_macro_f1_points": 100 * (met["macro_f1"] - st_["macro_f1"]), "delta_windows": win,
                             "within_tolerance": len(yt) == rec["source"]["test_size"] and win <= 2
                             and abs(met["macro_f1"] - st_["macro_f1"]) * 100 <= 2})
                inputs.append(ck)
            print(f"{dataset} {m} seed {s_}: done", flush=True)
    d = [abs(r["delta_macro_f1_points"]) for r in rows]
    out = {"dataset": dataset, "method": "per-checkpoint label mapping and test group (by the record's test size where "
                                         "it is unique, else by accuracy); train-only z-score on all non-test windows",
           "runs": len(rows), "identified_by": dict(Counter(r["test_group_identified_by"] for r in rows)),
           "test_size_equal": sum(r["test_size_equal"] for r in rows),
           "within_tolerance": sum(r["within_tolerance"] for r in rows),
           "median_abs_delta_macro_f1": float(np.median(d)) if d else None, "max_abs_delta_macro_f1": max(d, default=None),
           "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows])) * 100 if rows else None,
           "mean_stored_macro_f1": float(np.mean([r["stored_macro_f1"] for r in rows])) * 100 if rows else None,
           "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"reexecute_{dataset}.json"
    f.write_text(json.dumps(out, indent=1) + "\n")
    PV.write(f, run_id=run_id, started=started, inputs=inputs, config={"dataset": dataset, "mode": "reexecute"},
             produced_by="scripts/reconstruct_order.py")
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "rows"}, indent=1), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["infer", "build", "reexecute"])
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--file-order", default="sorted", choices=["sorted", "as_listed"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="infer: the seeds of the group (default 0)")
    ap.add_argument("--device", default=None, help="infer: only runs whose record names this GPU")
    ap.add_argument("--tag", default="", help="name of the group (output files order_<dataset>_<tag>.json)")
    a = ap.parse_args()
    run_id = PV.new_run_id("reconstruct_order_" + "_".join(a.datasets) + (f"_{a.tag}" if a.tag else ""))
    run_dir = PATHS.EXPERIMENTS / run_id
    run_dir.mkdir()
    config = {"mode": a.mode, "datasets": a.datasets, "datasets_dir": str(R.DATASETS_DIR), "file_order": a.file_order,
              "seeds": a.seeds, "device": a.device, "tag": a.tag}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    started = time.time()
    PV.ledger_start(run_id, config)
    try:
        for d in a.datasets:
            with open(PATHS.RESULTS / "test_access.log", "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | scripts/reconstruct_order.py | {run_id} | "
                         f"verification: every window of {d} classified by the stored checkpoints (listing order)\n")
            if a.mode == "infer":
                infer(d, run_id, started, a.seeds, a.device, a.tag)
            elif a.mode == "reexecute":
                reexecute(d, run_id, started)
            else:
                build_cache(d, a.file_order, run_id, started, a.tag)
    except BaseException as e:
        PV.ledger_end(run_id, "-", started, status=f"crashed({type(e).__name__})")
        raise
    PV.ledger_end(run_id, f"{a.mode}: {' '.join(a.datasets)}", started)


if __name__ == "__main__":
    main()
