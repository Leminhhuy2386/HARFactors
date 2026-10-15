"""Train Edge-SSM and the extra sequence baselines under the WHAR Arena v1 protocol, with the
code state of the paper (whar-datasets 144e006, whar-models 9b65004, whar-arena 19b3019), so
the numbers sit next to the 17 models of the paper (experiments/external_har_paper_era/whar_arena_v1: the same 17 re-run with
this code on 4 datasets). The WHAR code is the pinned copy in third_party/.

Taken unchanged from the harness (whar_arena.training): dataset config (3 s windows, 50 %
overlap, 10 subject groups, 20 % validation, global z-score fitted on train), the splits, the
training loop `_fit_torch_split` (AdamW 1e-3, weight decay 1e-4, cosine LR, class-weighted
cross-entropy, early stopping on validation macro-F1 with patience 10, best-state restore,
<= 100 epochs) and the metric (macro-F1 averaged over all activities of the dataset).

One difference, for speed only: each split is read into tensors once and served by a
TensorDataset instead of TorchAdapter's per-window __getitem__. The loader is built with the
same batch size, the same shuffle and a generator seeded the same way, so the batches are
identical. The global RNG is reseeded right before the model is built, as TorchAdapter does.

Training seeds (--seed, for multi-seed runs): the protocol's seed (cfg.seed = 0) keeps fixing the
data side (subject splits, validation windows, normalization), so every seed is trained and tested
on the same windows. A training seed s != 0 only changes the model initialization and the order
of the training batches. Seed 0 is the protocol run itself: same calls, same fingerprint, same
result path as before. Results of seed s >= 1 go to <model>/<dataset>/seed-<s>/split-XX.json, a
subdirectory that the split-*.json globs of the other scripts do not see.

Every trained model is also saved (state_dict) under experiments/checkpoints/, so robustness tests
can score it again without retraining.

Usage:
  uv run python scripts/train.py --models edge_ssm gru --datasets uci_har utd_mhad
  uv run python scripts/train.py --models edge_ssm --datasets uci_har --splits 0 --max-epochs 2
  uv run python scripts/train.py --models quantilenet s4d --datasets utd_mhad --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # project root
ARENA_SRC = Path(os.environ.get("WHAR_ARENA_SRC", HERE / "third_party" / "whar-arena" / "src"))
sys.path.insert(0, str(ARENA_SRC))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from whar_arena import training  # noqa: E402
from whar_arena.protocol import load_protocol  # noqa: E402
from whar_datasets import (  # noqa: E402
    LKSOSplitter,
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
)

from src.models.hysteresis import fast_paths  # noqa: E402
from src.models.model import EdgeSSMClassifier  # noqa: E402
from src.models.registry import CONFIGS, build, describe, is_whar_model  # noqa: E402

# Data shared with the HAR reproduction (D:\Projects\HAR, read-only except for adding a new paper-era cache):
# datasets/<id>/data holds the raw downloads of the 30 datasets, datasets_paper_era/<id> the caches built by the
# paper-era whar-datasets code. Only the paper-era caches are read; datasets/ holds the latest code's cache.
HAR_DIR = Path(os.environ.get("HAR_DIR", r"D:\Projects\HAR"))
DATASETS_DIR = Path(os.environ.get("WHAR_DATASETS_DIR", HAR_DIR / "datasets_paper_era"))
RAW_DIR = HAR_DIR / "datasets"  # raw files in <id>/data
EXTRA_DATASETS_DIR = DATASETS_DIR  # kept for older scripts: one cache directory now
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", HERE / "experiments"))  # per-run outputs
DATASETS = ["uci_har", "utd_mhad", "unimib_shar", "actrectut_gestures"]

# Only the code state of the WHAR Arena paper is allowed: its 17 baseline results and the paper's
# tables were produced with it. The latest GitHub versions changed the splits (strict train/val
# purge), the splitter and parsers, so their numbers are not comparable.
PAPER_ERA_COMMITS = {"whar-datasets": "144e006", "whar-models": "9b65004", "whar-arena": "19b3019"}
PAPER_PROTOCOL_HASH = "e8739ef2c5b36c99"  # whar_arena_v1 as in the paper (same hash as the baseline results)
LATEST_CODE_CACHE = RAW_DIR  # its sessions/windows were built by the latest code: never read them


def check_paper_era(protocol) -> dict[str, str]:
    """Refuse to run unless the WHAR code, protocol and data directory are the paper's version."""
    import subprocess

    import whar_arena
    import whar_datasets
    import whar_models

    found = {}
    for name, module in (("whar-datasets", whar_datasets), ("whar-models", whar_models), ("whar-arena", whar_arena)):
        root = Path(module.__file__).resolve().parents[2]
        git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        head = git("rev-parse", "--short=7", "HEAD")
        if head != PAPER_ERA_COMMITS[name]:
            raise RuntimeError(f"{name} at {root} is at {head or '?'}, not the paper's {PAPER_ERA_COMMITS[name]}")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"{name} at {root} has local modifications; the paper's code must be unmodified")
        found[name] = head
    if protocol.digest != PAPER_PROTOCOL_HASH:
        raise RuntimeError(f"protocol hash {protocol.digest} != the paper's {PAPER_PROTOCOL_HASH}")
    if DATASETS_DIR.resolve() == LATEST_CODE_CACHE.resolve():
        raise RuntimeError(f"{DATASETS_DIR} holds the latest-code cache; use {HAR_DIR / 'datasets_paper_era'}")
    return found


def datasets_dir_for(dataset_id: str) -> Path:
    """DATASETS_DIR, or the locally built paper-era cache for a dataset DATASETS_DIR does not hold."""
    if not (DATASETS_DIR / dataset_id).is_dir() and (EXTRA_DATASETS_DIR / dataset_id / "windows").is_dir():
        return EXTRA_DATASETS_DIR
    return DATASETS_DIR


def seed_everything(seed: int) -> None:
    # The harness's own seeding, with warn_only=True like the paper-era runner.
    training._seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def tensor_dataloaders(adapter: TorchAdapter, split, batch_size: int, train_seed: int) -> dict[str, DataLoader]:
    def materialize(indices):
        items = [adapter[i] for i in indices]
        return TensorDataset(torch.stack([y for y, _ in items]), torch.stack([x for _, x in items]))

    g = adapter.generator  # seeded with cfg.seed by TorchAdapter, shared as in get_dataloaders
    if train_seed != adapter.cfg.seed:  # another training seed: only the batch order changes
        g.manual_seed(train_seed)
    return {
        "train": DataLoader(materialize(split.train_indices), batch_size, True, generator=g),
        "val": DataLoader(materialize(split.val_indices), batch_size, False, generator=g),
        "test": DataLoader(materialize(split.test_indices), batch_size, False, generator=g),
    }


def fingerprint(protocol, model_id, dataset_id, split_id, epochs, patience, seed=0) -> str:
    cls_name, kwargs = describe(model_id)
    payload = {
        "protocol": protocol.digest, "model": model_id, "class": cls_name, "config": kwargs,
        "dataset": dataset_id, "split": split_id, "epochs": epochs, "patience": patience,
    }
    if seed:  # seed 0 keeps the fingerprints of the results made before --seed existed
        payload["train_seed"] = seed
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def tau_summary(model, cfg) -> dict | None:
    if not isinstance(model, EdgeSSMClassifier):
        return None
    tau = model.tau_seconds(cfg.sampling_freq)  # (channels, hidden)
    channels = list(cfg.selected_channels or cfg.available_channels)
    q = torch.tensor([0.1, 0.5, 0.9])
    per_channel = {
        (channels[i] if i < len(channels) else f"ch{i}"): [round(v, 4) for v in torch.quantile(tau[i], q).tolist()]
        for i in range(tau.shape[0])
    }
    return {
        "unit": "seconds", "quantiles": [0.1, 0.5, 0.9],
        "all": [round(v, 4) for v in torch.quantile(tau.flatten(), q).tolist()],
        "per_input_channel": per_channel,
    }


def prepare_dataset(dataset_id: str, *, max_epochs=None, patience=None, seed: int = 0) -> dict:
    """Protocol, dataset config and splits, exactly as the harness builds them."""
    protocol = load_protocol()
    whar_code = check_paper_era(protocol)
    datasets_dir = datasets_dir_for(dataset_id)
    cfg = training._configure_dataset_cfg(
        protocol=protocol, dataset_id=WHARDatasetID(dataset_id), datasets_dir=datasets_dir,
        max_epochs=max_epochs,
    )
    training_cfg = training._deep_training_cfg(protocol)
    seed_everything(cfg.seed)
    pre = PreProcessingPipeline(cfg)
    _, session_df, window_df = pre.run()
    return {
        "dataset_id": dataset_id, "protocol": protocol, "cfg": cfg, "training_cfg": training_cfg, "pre": pre,
        "whar_code": whar_code, "datasets_dir": str(datasets_dir),
        "session_df": session_df, "window_df": window_df,
        "splits": LKSOSplitter(cfg).get_splits(session_df, window_df),
        "epochs": int(max_epochs or cfg.num_epochs),
        "patience": int(patience if patience is not None else training_cfg["early_stopping"]["patience"]),
        "seed": int(seed),  # training seed; the data side always uses cfg.seed
    }


def prepare_split(ds: dict, split) -> tuple[Loader, dict[str, DataLoader], tuple[int, int, int]]:
    """Train-only post-processing for one split, then tensor dataloaders and (L, C, classes)."""
    cfg = ds["cfg"]
    post = PostProcessingPipeline(cfg, ds["pre"], ds["window_df"], split.train_indices)
    samples = post.run()
    loader = Loader(ds["session_df"], ds["window_df"], post.samples_dir, samples)
    adapter = TorchAdapter(cfg, loader, split)
    dataloaders = tensor_dataloaders(adapter, split, cfg.batch_size, ds.get("seed", 0))
    return loader, dataloaders, training._infer_dataset_shape(dataloaders, cfg)


def channel_names(cfg, input_channels: int) -> list[str] | None:
    """Names of the model's input channels, in tensor order (select_channels keeps cfg's order)."""
    names = list(cfg.selected_channels or cfg.available_channels)
    return names if len(names) == input_channels else None


def fit(model_id: str, ds: dict, split, split_id: str, loader, dataloaders, shape):
    """Build the model with the harness's seeding and train it with the harness's loop."""
    cfg = ds["cfg"]
    window_length, input_channels, num_classes = shape
    seed = ds.get("seed", 0)  # training seed (0 = the protocol's cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build(model_id, input_channels=input_channels, window_length=window_length,
                  num_classes=num_classes, channel_names=channel_names(cfg, input_channels))
    metrics = training._fit_torch_split(
        cfg=cfg, model=model, dataloaders=dataloaders, loader=loader, split=split,
        split_id=split_id, max_epochs=ds["epochs"], patience=ds["patience"],
        weight_decay=float(ds["training_cfg"]["weight_decay"]),
    )
    return model, metrics


def run(model_id: str, dataset_id: str, *, max_epochs, patience, split_filter, seed=0,
        need_checkpoint=False) -> None:
    ds = prepare_dataset(dataset_id, max_epochs=max_epochs, patience=patience, seed=seed)
    protocol, cfg = ds["protocol"], ds["cfg"]
    epochs, patience = ds["epochs"], ds["patience"]

    for split_index, split in enumerate(ds["splits"]):
        if split_filter is not None and split_index not in split_filter:
            continue
        split_id = f"split-{split_index:02d}"
        out_path = result_path(ds, model_id, split_id)
        fp = fingerprint(protocol, model_id, dataset_id, split_id, epochs, patience, seed)
        done = out_path.exists() and json.loads(out_path.read_text()).get("fingerprint") == fp
        if done and not (need_checkpoint and not checkpoint_path(ds, model_id, split_id).exists()):
            print(f"{model_id}/{dataset_id}/{split_id} seed {seed}: done already, skipping", flush=True)
            continue

        loader, dataloaders, shape = prepare_split(ds, split)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, metrics = fit(model_id, ds, split, split_id, loader, dataloaders, shape)
        if done:  # retrained only for the checkpoint: keep the result file, report reproduction
            old = json.loads(out_path.read_text())["metrics"]["macro_f1"]
            save_checkpoint(ds, model_id, split_id, model, metrics, fp)
            print(f"=== {model_id}/{dataset_id}/{split_id} seed {seed}: checkpoint saved; macro_f1 "
                  f"{metrics.macro_f1:.4f} vs stored {old:.4f} -> "
                  f"{'reproduced' if abs(metrics.macro_f1 - old) < 1e-9 else 'DIFFERENT'}", flush=True)
            continue
        write_result(model_id, ds, split_id, model, metrics, dataloaders, shape, time.time() - t0)


def result_path(ds: dict, model_id: str, split_id: str) -> Path:
    base = RESULTS_DIR / ds["protocol"].id / model_id / ds["dataset_id"]
    seed = ds.get("seed", 0)
    return (base / f"seed-{seed}" if seed else base) / f"{split_id}.json"


def checkpoint_path(ds: dict, model_id: str, split_id: str) -> Path:
    return (RESULTS_DIR / "checkpoints" / model_id / ds["dataset_id"] / f"seed-{ds.get('seed', 0)}"
            / f"{split_id}.pt")


def save_checkpoint(ds: dict, model_id: str, split_id: str, model, metrics, fp: str) -> None:
    """Trained weights (best epoch restored) plus what is needed to rebuild and check the model."""
    path = checkpoint_path(ds, model_id, split_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": model_id, "config": describe(model_id)[1], "fingerprint": fp,
                "seed": ds.get("seed", 0), "macro_f1": metrics.macro_f1,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def write_result(model_id, ds, split_id, model, metrics, dataloaders, shape, seconds) -> None:
    """Result json of one trained split (also used by warp_test.py, which trains the same way)."""
    protocol, cfg = ds["protocol"], ds["cfg"]
    dataset_id = ds["dataset_id"]
    window_length, input_channels, num_classes = shape
    fp = fingerprint(protocol, model_id, dataset_id, split_id, ds["epochs"], ds["patience"], ds.get("seed", 0))
    out_path = result_path(ds, model_id, split_id)
    # validation score of the restored best model: the only number used to choose variants
    val = training._evaluate(model, dataloaders["val"], device=next(model.parameters()).device,
                             cfg=cfg, criterion=None)
    result = {
        "protocol_id": protocol.id,
        "protocol_hash": protocol.digest,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "train_seed": ds.get("seed", 0),
        "fingerprint": fp,
        "status": "complete",
        "metrics": {
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "val_macro_f1": val["macro_f1"],
        },
        "source": {
            "kind": "local_training",
            "framework": "torch",
            "runner": "HARFactors/src/training/run_whar.py",
            "split_identifier": metrics.split_identifier,
            "train_size": metrics.train_size,
            "val_size": metrics.val_size,
            "test_size": metrics.test_size,
            "best_epoch": metrics.best_epoch,
            "epochs_ran": metrics.epochs_ran,
        },
        "model": {
            "class": describe(model_id)[0],
            "config": describe(model_id)[1],
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_channels": input_channels,
            "window_length": window_length,
            "num_classes": num_classes,
            "sampling_freq": cfg.sampling_freq,
        },
        "whar_code": ds["whar_code"],  # commits of the WHAR libraries actually used (checked = paper's)
        "datasets_dir": ds["datasets_dir"],
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "train_seconds": round(seconds, 1),
        "fast_paths": fast_paths(),  # EDGE_SSM_FAST: same math, gradients equal up to float rounding
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
        "tau": tau_summary(model, cfg),
        # learned play-operator widths per block (HysteresisNet), quantiles 0.1 / 0.5 / 0.9
        "hysteresis_widths": model.width_summary() if hasattr(model, "width_summary") else None,
        # round 8: 3-axis sensors found from the channel names (indices into the input channels)
        "sensor_triples": [list(t) for t in model.triples] if hasattr(model, "triples") else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    save_checkpoint(ds, model_id, split_id, model, metrics, fp)
    print(f"=== {model_id}/{dataset_id}/{split_id} seed {ds.get('seed', 0)}: macro_f1={metrics.macro_f1:.4f} "
          f"best_epoch={metrics.best_epoch} epochs={metrics.epochs_ran} {seconds:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="ids from edge_ssm/registry.py, or WHAR Arena model ids (to check the runner)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--splits", nargs="+", type=int, help="split indices to run (default: all)")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed: model init and batch order (0 = the protocol run)")
    ap.add_argument("--need-checkpoint", action="store_true",
                    help="retrain finished splits that have no saved checkpoint (result file kept)")
    args = ap.parse_args()
    unknown = [m for m in args.models if m not in CONFIGS and not is_whar_model(m)]
    if unknown:
        ap.error(f"unknown model ids: {unknown}")
    for dataset_id in args.datasets:
        for model_id in args.models:
            print(f"##### {model_id} / {dataset_id}", flush=True)
            t0 = time.time()
            try:
                run(model_id, dataset_id, max_epochs=args.max_epochs, patience=args.patience,
                    split_filter=set(args.splits) if args.splits is not None else None, seed=args.seed,
                    need_checkpoint=args.need_checkpoint)
            except Exception as exc:  # keep going with the other pairs, as the harness runners do
                traceback.print_exc()
                print(f"##### FAILED {model_id}/{dataset_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"##### done {model_id}/{dataset_id} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
