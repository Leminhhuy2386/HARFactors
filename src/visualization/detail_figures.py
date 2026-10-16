"""Further figures and tables of the manuscript: study design, sensors, network structure, memories, split spread and
embeddings. From saved results only (no training, no inference); descriptive or exploratory unless stated.

    fig03_design            (a) pipeline from datasets to the variance decomposition; (b) which runs exist (networks x
                            datasets, by run set); (c) the person-disjoint test groups of UCI-HAR
    fig04_sensors           sensor configuration of the 18 datasets: device, modalities, channels, subjects, classes,
                            sampling rate (S1 and S2)
    fig05_architecture      the shared skeleton of the ten networks, the residual block and its memory variants
    fig06_memory            (a-c) the memory operators on a synthetic input (illustration); (d-f) the memory parameters
                            the networks learned (all stored checkpoints of eight S1 datasets)
    fig07_split_spread      (a) per dataset, the spread of the split means against that of the network means;
                            (b, c) network x split cell means for UCI-HAR (S1) and UP-Fall (S2)
    fig08_embedding         (a, b) t-SNE of one case by activity and by subject; (c) rank correlation between the test
                            group's class-conditional shift and the split's macro-F1; (d) subject decodability
    fig09_validation        per dataset: validation macro-F1 (training subjects) against test macro-F1 (unseen
                            subjects), and whether validation ranks the networks as the test does
    tab02_datasets.tex      dataset table (S1, S2)
    tab03_models.tex        network table

Every figure and table writes its numbers to results/figure_data/ or results/table_data/, with provenance.

    uv run --no-sync python -m src.visualization.detail_figures [--only fig03 fig04 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from src.data.sets import S1, S2, S3  # noqa: E402
from src.evaluation import factor_analysis as FA  # noqa: E402
from src.utils import paths as PATHS  # noqa: E402
from src.utils import provenance as PV  # noqa: E402
from src.visualization.letter_figures import FIG_DATA, GRID, INK, INK2, SLOTS, TAB_DATA, save  # noqa: E402

SENSORS = PATHS.DOCS / "sources" / "arena_table3_sensors.json"
HSPLITS = PATHS.ROOT / "data" / "processed" / "harness_splits.json"
ANAT = PATHS.METRICS / "model_anatomy" / "data.json"
SPREAD = PATHS.METRICS / "split_spread" / "data.json"
EMB = PATHS.METRICS / "embedding"
ACTS = PATHS.DOCS / "sources" / "uci_har_activities.json"
CONF = PATHS.METRICS / "confirmatory"
EXTRA = "#8a63d2"  # sixth categorical colour (t-SNE activities only)
LIGHT = "#ecebe8"
EXCLUDED = {"daphnet": "S1", "motion_sense": "S2"}  # pre-registered inclusion rule (split sizes disagree across runs)
NET = {"recurrencenet": "RecurrenceNet", "s4d": "S4D", "hystnet_sel_clock": "HysteresisNet", "quantilenet": "QuantileNet",
       "mamba": "Mamba", "hystnet_nomem": "No memory", "recstatnet": "RN + stats", "lawnet_rec": "RN + laws",
       "lawnet": "Laws (no mem.)", "statnet": "Stats (no mem.)"}
MEMORY = {"recurrencenet": "recurrence", "s4d": "S4D (diagonal SSM)", "hystnet_sel_clock": "play operators",
          "quantilenet": "quantile trackers", "mamba": "Mamba (S6)", "hystnet_nomem": "none (play width 0)",
          "recstatnet": "recurrence", "lawnet_rec": "recurrence", "lawnet": "none (plain block)",
          "statnet": "none (plain block)"}
SIDE = {"recstatnet": "channel stats", "statnet": "channel stats", "lawnet_rec": "law invariants",
        "lawnet": "law invariants"}
DEVICES = [("Phone", "phone"), ("Watch", "watch"), ("Glasses", "glasses"), ("Helmet", "helmet"), ("Node", "IMU node")]
MODALITIES = ["Acc", "Gyr", "Mag", "Baro", "ALS", "Stretch", "ECG", "EEG"]


def record(path, data, run_id, started, inputs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=1, default=float) + "\n")
    PV.write(path, run_id=run_id, started=started, inputs=inputs, produced_by="src/visualization/detail_figures.py")


def panel(ax, letter: str) -> None:
    ax.set_title(f"({letter})", loc="left", fontweight="bold")


def box(ax, x, y, w, h, title, body="", fc="white", ec=INK2, fs=6.3, title_fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015", fc=fc, ec=ec, lw=0.7,
                                transform=ax.transAxes, clip_on=False))
    ax.annotate(title, xy=(x, y + h), xycoords=ax.transAxes, xytext=(3.5, -3.5), textcoords="offset points",
                fontweight="bold", va="top", ha="left", fontsize=title_fs, color=INK, linespacing=1.15)
    if body:
        drop = 3.5 + 1.3 * title_fs * (title.count("\n") + 1) + 1.5
        ax.annotate(body, xy=(x, y + h), xycoords=ax.transAxes, xytext=(3.5, -drop), textcoords="offset points",
                    va="top", ha="left", fontsize=fs, color=INK, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8, "shrinkA": 0, "shrinkB": 0,
                            "mutation_scale": 7})


def dataset_rows() -> tuple[list[dict], list]:
    """One row per dataset of S1 and S2, measured from the run records and the harness splits where possible."""
    sens = json.loads(SENSORS.read_text())["datasets"]
    anat = json.loads(ANAT.read_text())
    hs = json.loads(HSPLITS.read_text())["datasets"]
    rows = []
    for set_name, spec in (("S1", S1), ("S2", S2)):
        for d in spec["datasets"]:
            sz = anat["sizes"][d]["recurrencenet"]
            rows.append({"dataset": d, "name": sens[d]["name"], "set": set_name, "device": sens[d]["device"],
                         "modalities": sens[d]["modalities"], "channels": sz["input_channels"],
                         "table3_channels": sens[d]["channels"],
                         "subjects": hs[d]["subjects"] if d in hs else FA.CFG[d][1],
                         "subjects_source": "harness splits" if d in hs else "whar-datasets config",
                         "classes": sz["num_classes"], "hz": sz["sampling_freq"], "window": sz["window_length"],
                         "windows": sz["windows"], "splits": sz["splits"], "networks": len(spec["models"]),
                         "excluded": EXCLUDED.get(d) == set_name})
    rows.sort(key=lambda r: (r["set"], r["channels"], r["name"]))
    return rows, [SENSORS, ANAT, HSPLITS, PATHS.ROOT / "src" / "evaluation" / "factor_analysis.py"]


# ---------------------------------------------------------------------------------------------------------- fig03

def fig03(run_id, started) -> None:
    hs = json.loads(HSPLITS.read_text())["datasets"]
    conf = {k: json.loads((CONF / f"{k}.json").read_text()) for k in ("H1", "sens2_S1_macro_f1", "sens4_S3_macro_f1")}
    fig = plt.figure(figsize=(7.16, 4.7))
    gs = fig.add_gridspec(1, 2, left=0.14, right=0.98, bottom=0.17, top=0.6, width_ratios=[1.25, 1], wspace=0.28)
    a = fig.add_axes([0.0, 0.69, 1.0, 0.27])
    a.axis("off")
    a.text(0.0, 1.06, "(a)", fontweight="bold", fontsize=8, transform=a.transAxes, va="bottom")
    s1, s2, s3 = (sum(n for n in s["datasets"].values()) for s in (S1, S2, S3))
    n1 = len(S1["models"]) * s1 * len(S1["seeds"])
    n2 = len(S2["models"]) * s2 * len(S2["seeds"])
    n3 = len(S3["models"]) * s3
    stages = [
        ("Datasets and networks", f"S1, development: {len(S1['datasets'])} data-\nsets × {len(S1['models'])} networks × "
         f"{len(S1['seeds'])} seeds\n= {n1:,} runs\nS2, held-out: {len(S2['datasets'])} datasets\n× {len(S2['models'])} "
         f"networks × {len(S2['seeds'])} seeds\n= {n2:,} runs\nS3: {len(S3['datasets'])} datasets × {len(S3['models'])} "
         f"WHAR\nArena models × 1 run = {n3}"),
        ("Person-disjoint splits", "subjects in k ≤ 10 groups\n(harness, LKSO); each\ngroup is the test set\nonce; "
         "training and\nvalidation windows from\nthe other subjects"),
        ("Training (stored runs)", "3-s windows; z-score\nfitted on the training\nwindows; early stopping\non "
         "validation; the same\nsplits for every network\nand seed"),
        ("Outcome per run", "macro-F1 over the test\ngroup's windows (absent\nclasses count 0);\nbalanced accuracy as "
         "a\nsensitivity outcome"),
        ("Variance decomposition", "crossed random effects\nwithin each dataset:\n" r"$y=\mu_d+P+M+MD+MP+e$" "\n"
         r"$R=\sigma^2_P/(\sigma^2_M+\sigma^2_{MD}+\sigma^2_{MP})$" "\nH1 (confirmatory):\nR > 1 on S2"),
    ]
    w, gap, y0, h = 0.184, 0.0195, 0.0, 1.0
    for i, (title, body) in enumerate(stages):
        x = 0.003 + i * (w + gap)
        box(a, x, y0, w, h, title, body, fc="#f6f5f2" if i != 4 else "#eef4fc", fs=5.9, title_fs=6.8)
        if i < len(stages) - 1:
            arrow(a, x + w + 0.002, y0 + h / 2, x + w + gap - 0.002, y0 + h / 2)

    b = fig.add_subplot(gs[0, 0])
    panel(b, "b")
    datasets = list(S1["datasets"]) + list(S2["datasets"])
    names = json.loads(SENSORS.read_text())["datasets"]
    models = S1["models"] + ["whar17"]
    grid = np.zeros((len(models), len(datasets)))
    text = {}
    for j, d in enumerate(datasets):
        for i, m in enumerate(models):
            for k, spec in enumerate((S1, S2, S3), start=1):
                inside = (d in spec["datasets"]) and (m in spec.get("models", []) or (m == "whar17" and spec is S3))
                if inside:
                    grid[i, j] = k if grid[i, j] == 0 else grid[i, j] + 10 * k  # S1 and S3 share 4 datasets
                    n = spec["datasets"][d] * len(spec["seeds"])
                    text[(i, j)] = str(n)
    colours = {1: SLOTS[0], 2: SLOTS[1], 3: SLOTS[2]}
    for i in range(len(models)):
        for j in range(len(datasets)):
            v = int(grid[i, j])
            if v == 0:
                b.add_patch(Rectangle((j, i), 1, 1, fc="white", ec=GRID, lw=0.4))
                continue
            c = colours[v % 10] if v < 10 else colours[3]
            excl = EXCLUDED.get(datasets[j]) == ("S1" if v == 1 else "S2" if v == 2 else "")
            b.add_patch(Rectangle((j, i), 1, 1, fc=c, ec="white", lw=0.6, alpha=0.35 if excl else 0.9,
                                  hatch="////" if excl else None))
            b.text(j + 0.5, i + 0.5, text[(i, j)], ha="center", va="center", fontsize=4.6,
                   color="white" if not excl else INK)
    b.set_xlim(0, len(datasets))
    b.set_ylim(len(models), 0)
    b.set_xticks(np.arange(len(datasets)) + 0.5, [names[d]["name"] for d in datasets], rotation=60, ha="right",
                 fontsize=5.8)
    b.set_yticks(np.arange(len(models)) + 0.5, [NET.get(m, "17 WHAR models") for m in models], fontsize=6)
    b.tick_params(length=0)
    for s in b.spines.values():
        s.set_visible(False)
    for k, (lab, c) in enumerate((("S1 (development)", SLOTS[0]), ("S2 (held-out)", SLOTS[1]),
                                  ("S3 (WHAR Arena models)", SLOTS[2]))):
        b.text(0.08 + 0.29 * k, 1.03, "■", color=c, transform=b.transAxes, fontsize=7, va="bottom")
        b.text(0.115 + 0.29 * k, 1.03, lab, color=INK, transform=b.transAxes, fontsize=6.0, va="bottom")
    b.text(1.0, -0.40, "cell: splits × seeds; hatched: excluded by the pre-registered inclusion rule",
           transform=b.transAxes, fontsize=5.6, color=INK2, ha="right")

    c = fig.add_subplot(gs[0, 1])
    panel(c, "c")
    u = hs["uci_har"]
    n_subj = u["subjects"]
    mat = np.zeros((len(u["splits"]), n_subj))
    for i, (sid, s) in enumerate(sorted(u["splits"].items())):
        for t in s["test_subjects"]:
            mat[i, t] = 1
    cmap = LinearSegmentedColormap.from_list("t", [LIGHT, SLOTS[0]])
    c.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    c.set_xticks(np.arange(0, n_subj, 5), [str(k) for k in range(0, n_subj, 5)])
    c.set_yticks(np.arange(len(u["splits"])), [f"{i}" for i in range(len(u["splits"]))])
    c.set_xlabel(f"UCI-HAR subject (of {n_subj})")
    c.set_ylabel("split")
    c.tick_params(length=0)
    for s in c.spines.values():
        s.set_visible(False)
    for x, sq, lab, col in ((0.30, "■", "test group", SLOTS[0]), (0.58, "■", "training and validation", GRID)):
        c.text(x, 1.03, sq, transform=c.transAxes, ha="left", fontsize=7, color=col, va="bottom")
        c.text(x + 0.045, 1.03, lab, transform=c.transAxes, ha="left", fontsize=6.3, color=INK, va="bottom")
    save(fig, "fig03_design")
    record(FIG_DATA / "fig03_design.json",
           {"stages": stages, "runs": {"S1": n1, "S2": n2, "S3": n3},
            "analysed": {k: v["n_runs"] for k, v in conf.items()},
            "grid_models": models, "grid_datasets": datasets, "grid_set_code": grid.tolist(),
            "uci_har_test_subjects": {k: v["test_subjects"] for k, v in u["splits"].items()}},
           run_id, started, [HSPLITS, SENSORS, *(CONF / f"{k}.json" for k in conf)])


# ---------------------------------------------------------------------------------------------------------- fig04

def fig04(run_id, started) -> list[dict]:
    rows, inputs = dataset_rows()
    order = [r for r in rows if r["set"] == "S1"] + [r for r in rows if r["set"] == "S2"]
    fig = plt.figure(figsize=(7.16, 3.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[2.0, 1.15, 1, 1, 1, 1], wspace=0.12)
    y = np.arange(len(order))
    colour = lambda r: SLOTS[0] if r["set"] == "S1" else SLOTS[1]  # noqa: E731
    a = fig.add_subplot(gs[0])
    cols = [lab for _, lab in DEVICES] + MODALITIES
    for j, col in enumerate(cols):
        a.axvline(j, color=GRID, lw=0.4, zorder=0)
    for i, r in enumerate(order):
        a.axhline(i, color=GRID, lw=0.3, zorder=0)
        present = [k for k, (key, _) in enumerate(DEVICES) if key in r["device"]]
        present += [len(DEVICES) + k for k, m in enumerate(MODALITIES) if m in r["modalities"]]
        a.scatter(present, [i] * len(present), s=16, color=colour(r), alpha=0.4 if r["excluded"] else 1.0,
                  edgecolor="white", linewidth=0.5, zorder=3)
    a.axvline(len(DEVICES) - 0.5, color=INK2, lw=0.6)
    a.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=6)
    a.set_yticks(y, [r["name"] + (" (excl.)" if r["excluded"] else "") for r in order], fontsize=6.3)
    a.set_ylim(len(order) - 0.5, -0.5)
    a.set_xlim(-0.6, len(cols) - 0.4)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("(a) device, modalities", loc="left", fontweight="bold", fontsize=7)
    n1 = sum(r["set"] == "S1" for r in order)
    a.axhline(n1 - 0.5, color=INK, lw=0.8)
    a.text(-0.78, (n1 - 1) / 2, "S1", rotation=90, va="center", ha="center", fontsize=7, fontweight="bold",
           color=SLOTS[0], transform=a.get_yaxis_transform())
    a.text(-0.78, n1 + (len(order) - n1 - 1) / 2, "S2", rotation=90, va="center", ha="center", fontsize=7,
           fontweight="bold", color=SLOTS[1], transform=a.get_yaxis_transform())
    bars = [("channels", "(b) channels", True), ("subjects", "(c) subjects", False), ("classes", "(d) classes", False),
            ("hz", "(e) rate, Hz", True), ("windows", "(f) windows", True)]
    for k, (key, title, log) in enumerate(bars):
        ax = fig.add_subplot(gs[k + 1], sharey=a)
        vals = [r[key] for r in order]
        ax.barh(y, vals, height=0.66, color=[colour(r) for r in order],
                alpha=None, edgecolor="white", linewidth=0.4)
        for yy, r in zip(y, order):
            if r["excluded"]:
                ax.barh([yy], [r[key]], height=0.66, color="white", alpha=0.55, hatch="////", edgecolor=colour(r),
                        linewidth=0.0)
            ax.text(r[key] * (1.08 if log else 1.0) + (0 if log else max(vals) * 0.02), yy,
                    f"{r[key]:,}" if key == "windows" else f"{r[key]:g}", va="center", ha="left", fontsize=5.3,
                    color=INK2)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(min(vals) / 1.5, max(vals) * (12 if key == "windows" else 6))
        else:
            ax.set_xlim(0, max(vals) * 1.32)
        ax.tick_params(axis="y", length=0, labelleft=False)
        ax.tick_params(axis="x", labelsize=5.8)
        ax.axhline(n1 - 0.5, color=INK, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7)
    save(fig, "fig04_sensors")
    record(FIG_DATA / "fig04_sensors.json", {"rows": order}, run_id, started, inputs)
    return order


def tab02(order, run_id, started) -> None:
    PATHS.TABLES.mkdir(exist_ok=True)
    head = ["dataset", "set", "device", "modalities", "channels", "subjects", "classes", "hz", "window", "windows",
            "splits", "networks", "excluded"]
    path = TAB_DATA / "tab02_datasets.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in order:
            w.writerow([r["dataset"], r["set"], "+".join(r["device"]), "+".join(r["modalities"]), r["channels"],
                        r["subjects"], r["classes"], r["hz"], r["window"], r["windows"], r["splits"], r["networks"],
                        r["excluded"]])
    record(path, None, run_id, started, [ANAT, SENSORS, HSPLITS])
    summary = {"datasets": len(order), "windows_total": int(sum(r["windows"] for r in order))}
    for k in ("channels", "subjects", "classes", "hz"):
        summary[f"{k}_min"], summary[f"{k}_max"] = min(r[k] for r in order), max(r[k] for r in order)
    record(TAB_DATA / "tab02_datasets.json", {"rows": order, "summary": summary}, run_id, started,
           [ANAT, SENSORS, HSPLITS])
    dev = {"Node": "node", "Phone": "phone", "Watch": "watch", "Glasses": "glasses", "Helmet": "helmet"}
    tex = [r"\begin{tabular}{@{}llllrrrrrr@{}}", r"\toprule",
           r"Dataset & Set & Device & Modalities & $C$ & Subj. & Cl. & Hz & Windows & Splits \\", r"\midrule"]
    for i, r in enumerate(order):
        if i and r["set"] != order[i - 1]["set"]:
            tex.append(r"\midrule")
        name = r["name"] + (r"$^\dagger$" if r["excluded"] else "")
        tex.append(f"{name} & {r['set']} & {', '.join(dev[d] for d in r['device'])} & "
                   f"{', '.join(r['modalities'])} & {r['channels']} & {r['subjects']} & {r['classes']} & {r['hz']:g} & "
                   f"{r['windows']:,} & {r['splits']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (PATHS.TABLES / "tab02_datasets.tex").write_text("\n".join(tex) + "\n")


# ---------------------------------------------------------------------------------------------------------- fig05

def fig05(run_id, started) -> dict:
    anat = json.loads(ANAT.read_text())
    st = anat["structure"]
    fig = plt.figure(figsize=(7.16, 3.3))
    a = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    a.axis("off")
    # (a) skeleton, top row
    a.text(0.005, 0.975, "(a) shared skeleton", fontweight="bold", fontsize=8, va="top")
    top = [("Window", "C channels ×\nL samples\n(3 s)"), ("Encoder", "linear C→D\nper sample\n(S4D: 1×1 conv)"),
           ("N residual\nblocks", "memory varies\n(panel b)"), ("Mean over\ntime", "D values"),
           ("Concat", "D (+32 side\npath)"), ("LayerNorm", ""), ("Linear head", "K classes")]
    w, h, y0, x0, gap = 0.112, 0.22, 0.68, 0.012, 0.0285
    xs = []
    for i, (t, body) in enumerate(top):
        x = x0 + i * (w + gap)
        xs.append(x)
        fc = "#eef4fc" if i == 2 else "#f6f5f2"
        box(a, x, y0, w, h, t, body, fc=fc, fs=6.0, title_fs=6.8)
        if i:
            arrow(a, xs[i - 1] + w + 0.002, y0 + h / 2, x - 0.002, y0 + h / 2)
    # side path below the concat box
    box(a, 0.36, 0.465, 0.31, 0.15, "Side path (4 networks)",
        "per-channel mean, sd, min, max → BN → 32, or\nlaw invariants of each 3-axis sensor → MLP → 32", fs=5.6,
        title_fs=6.3, fc="#fdf1ea")
    arrow(a, xs[0] + w / 2, y0, xs[0] + w / 2, 0.54)
    arrow(a, xs[0] + w / 2, 0.54, 0.358, 0.54)
    arrow(a, xs[4] + w / 2, 0.617, xs[4] + w / 2, y0 - 0.002)
    # embedding tap after the LayerNorm
    ex = xs[6] - gap / 2
    a.plot([ex, ex], [y0 + h + 0.01, y0 + h + 0.055], color=SLOTS[1], lw=1.2, transform=a.transAxes)
    a.text(ex, y0 + h + 0.06, "embedding (head input)", color=SLOTS[1], fontsize=6.3, ha="center", va="bottom",
           fontweight="bold")
    # (b) residual block
    a.text(0.005, 0.375, "(b) residual block", fontweight="bold", fontsize=8, va="top")
    blk = [("Layer-\nNorm", ""), ("Linear", "mix in: u"), ("Memory", "m(u), per\nchannel"), ("Concat", "[u, m(u)]"),
           ("Linear", "mix out"), ("GELU", ""), ("+ x", "residual")]
    w2, h2, y2 = 0.067, 0.12, 0.205
    bx = []
    for i, (t_, body) in enumerate(blk):
        x = 0.012 + i * (w2 + 0.012)
        bx.append(x)
        box(a, x, y2, w2, h2, t_, body, fc="#eef4fc" if i == 2 else "white", fs=5.6, title_fs=6.2)
        if i:
            arrow(a, bx[i - 1] + w2 + 0.001, y2 + h2 / 2, x - 0.001, y2 + h2 / 2)
    lines = [r"recurrence: $r_t(\tau)=\sigma((\epsilon-\Vert z_t-z_{t-\tau}\Vert)/T)$ and $r_t(\tau)\,r_{t-1}(\tau)$, ten lags",
             r"play operator: $y_t=\min(\max(y_{t-1},u_t-r),u_t+r)$, four widths $r$ ($r=0$: no memory)",
             r"quantile tracker: $q_t=q_{t-1}+\eta\,(\tau-\mathbf{1}[u_t<q_{t-1}])$, four levels $\tau$",
             "S4D block: LayerNorm, diagonal state-space convolution + skip, GELU, 1×1 conv, GLU, + x",
             "Mamba block: LayerNorm, in-projection, depthwise conv, SiLU, selective scan, gate, out-projection, + x"]
    for k, line in enumerate(lines):
        a.text(0.012, 0.165 - 0.033 * k, line, fontsize=5.6, color=INK, va="top")
    # (c) memory variants and sizes
    a.text(0.60, 0.375, "(c) the ten networks (reference input of gate G1)", fontweight="bold", fontsize=8, va="top")
    shown = {"none (play width 0)": "none (width 0)", "none (plain block)": "none (plain)"}
    yy = 0.325
    a.text(0.60, yy, "network", fontsize=5.6, va="top", color=INK2)
    a.text(0.705, yy, "memory + side path", fontsize=5.6, va="top", color=INK2)
    a.text(0.995, yy, "D, N, parameters", fontsize=5.6, va="top", ha="right", color=INK2)
    yy -= 0.031
    for m in S1["models"]:
        s = st[m]
        side = SIDE.get(m, "")
        a.text(0.60, yy, NET[m], fontsize=5.8, fontweight="bold", va="top", color=INK)
        a.text(0.705, yy, shown.get(MEMORY[m], MEMORY[m]) + (f" + {side}" if side else ""), fontsize=5.6, va="top",
               color=INK)
        a.text(0.995, yy, f"{s['d_model']}, {s['layers']}, {s['params']:,}", fontsize=5.6, va="top", ha="right",
               color=INK)
        yy -= 0.029
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    save(fig, "fig05_architecture")
    data = {"structure": st, "memory": MEMORY, "side": SIDE, "operators": lines}
    record(FIG_DATA / "fig05_architecture.json", data, run_id, started, [ANAT])
    return data


def tab03(run_id, started) -> None:
    anat = json.loads(ANAT.read_text())
    st, sizes = anat["structure"], anat["sizes"]
    rows = []
    for m in S1["models"]:
        p = [v[m]["params"] for v in sizes.values() if m in v]
        s = st[m]
        rows.append({"model": m, "name": NET[m], "memory": MEMORY[m], "side": SIDE.get(m, "none"), "d_model": s["d_model"],
                     "layers": s["layers"], "embedding_dim": s["embedding_dim"], "params_ref": s["params"],
                     "params_min": min(p), "params_max": max(p), "datasets": len(p), "s2": s["s2"],
                     "params_by_part": s["params_by_part"]})
    path = TAB_DATA / "tab03_models.json"
    summary = {"params_ref_min": min(r["params_ref"] for r in rows), "params_ref_max": max(r["params_ref"] for r in rows),
               "params_all_min": min(r["params_min"] for r in rows), "params_all_max": max(r["params_max"] for r in rows)}
    record(path, {"rows": rows, "summary": summary, "reference_input": anat["reference_input"]}, run_id, started, [ANAT])
    tex = [r"\begin{tabular}{@{}llllrrrr@{}}", r"\toprule",
           r"Network & Memory & Side path & Sets & $D{\times}N$ & Emb. & Params (ref.) & Range over datasets \\",
           r"\midrule"]
    for r in rows:
        tex.append(f"{r['name']} & {r['memory']} & {r['side']} & {'S1, S2' if r['s2'] else 'S1'} & "
                   f"{r['d_model']}$\\times${r['layers']} & {r['embedding_dim']} & {r['params_ref']:,} & "
                   f"{r['params_min']:,}--{r['params_max']:,} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (PATHS.TABLES / "tab03_models.tex").write_text("\n".join(tex) + "\n")


# ---------------------------------------------------------------------------------------------------------- fig06

def fig06(run_id, started) -> dict:
    anat = json.loads(ANAT.read_text())
    toy, ln = anat["toy"], anat["learned"]
    t = np.array(toy["t"])
    u = np.array(toy["u"])
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9), gridspec_kw={"hspace": 0.95, "wspace": 0.32})
    a, b, c = axes[0]
    for ax in (a, b):
        ax.plot(t, u, color=INK, lw=1.0, label="input u")
        ax.set_xlabel("time, s (synthetic input)")
        ax.grid(color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    for k, (r, y) in enumerate(zip(toy["play_widths"], toy["play"])):
        a.plot(t, y, color=SLOTS[k], lw=1.0, label=f"r = {r:g}")
    a.legend(ncol=3, fontsize=5.5, frameon=False, loc="upper left", handlelength=1.2, columnspacing=0.8)
    a.set_ylim(min(u) - 0.3, max(u) + 1.4)
    a.set_title("(a) play operators", loc="left", fontweight="bold")
    for k, ((tau, eta), q) in enumerate(zip(toy["trackers"], toy["quantile"])):
        b.plot(t, q, color=SLOTS[k], lw=1.0, label=fr"$\tau$={tau:g}, $\eta$={eta:g}")
    b.legend(ncol=2, fontsize=5.5, frameon=False, loc="upper left", handlelength=1.2, columnspacing=0.8)
    b.set_ylim(min(u) - 0.3, max(u) + 1.4)
    b.set_title("(b) quantile trackers", loc="left", fontweight="bold")
    rec = np.array(toy["recurrence"])
    cmap = LinearSegmentedColormap.from_list("r", ["white", SLOTS[0]])
    c.imshow(rec, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
             extent=[t[0], t[-1], len(toy["recurrence_lags"]) - 0.5, -0.5])
    c.set_yticks(range(len(toy["recurrence_lags"])), [str(v) for v in toy["recurrence_lags"]], fontsize=5.5)
    c.set_ylabel("lag τ, samples")
    c.set_xlabel("time, s (synthetic input)")
    c.set_title("(c) soft recurrence rows", loc="left", fontweight="bold")
    d, e, f = axes[1]

    def hist(ax, key, title, xlabel, colour=SLOTS[0]):
        h = ln[key]
        edges = np.array(h["hist_edges"])
        x = edges if h["scale"] == "linear" else 10 ** edges
        ax.bar(x[:-1], h["hist"], width=np.diff(x), align="edge", color=colour, edgecolor="white", linewidth=0.3,
               label="learned")
        ax.stairs(h["init_hist"], x, color=INK, lw=0.8, label="at initialisation")
        if h["scale"] != "linear":
            ax.set_xscale("log")
        ax.axvline(h["quantiles"]["p50"], color=INK, lw=0.8, ls=(0, (3, 2)))
        note = f"median {h['quantiles']['p50']:.3g} (initial {h['init_quantiles']['p50']:.3g})"
        if "exact_change" in h:
            note += f"\nmedian change {h['exact_change']['quantiles']['p50']:.2g}"
        ax.text(0.02, 1.0, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color=INK2)
        ax.set_title(title, loc="left", fontweight="bold", pad=14 if "\n" not in note else 21)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)

    hist(d, "s4d", "(d) S4D mode time constants", "time constant / window length")
    d.axvline(1, color=SLOTS[1], lw=0.8)
    d.text(1.08, 0.04, "window", color=SLOTS[1], fontsize=5.6, transform=d.get_xaxis_transform())
    hist(e, "hystnet_sel_clock", "(e) play operator widths", "base width r (normalised units)", colour=SLOTS[2])
    hist(f, "quantilenet_tau", "(f) quantile levels", r"level $\tau$", colour=SLOTS[3])
    f.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.2)
    save(fig, "fig06_memory")
    data = {"toy": toy, "learned": {k: v for k, v in ln.items()}, "note": toy["note"]}
    record(FIG_DATA / "fig06_memory.json", data, run_id, started, [ANAT])
    return data


# ---------------------------------------------------------------------------------------------------------- fig07

def fig07(run_id, started) -> dict:
    sp = json.loads(SPREAD.read_text())
    ds = sp["datasets"]
    order = sorted(ds, key=lambda d: ds[d]["sd_split_means"])
    fig = plt.figure(figsize=(7.16, 3.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.6, 1], wspace=0.42, hspace=0.55)
    a = fig.add_subplot(gs[:, 0])
    for i, d in enumerate(order):
        v = ds[d]
        sm = list(v["split_means"].values())
        mm = list(v["model_means"].values())
        a.plot([min(sm), max(sm)], [i, i], color=SLOTS[0], lw=0.8, alpha=0.5)
        a.scatter(sm, [i + 0.12] * len(sm), s=9, color=SLOTS[0], edgecolor="white", linewidth=0.3, zorder=3)
        a.scatter(mm, [i - 0.12] * len(mm), s=11, marker="D", color=SLOTS[1], edgecolor="white", linewidth=0.3,
                  zorder=3)
        a.text(102, i, f"{v['sd_split_means']:.1f} / {v['sd_model_means']:.1f}", va="center", fontsize=5.6,
               color=INK2)
    a.set_yticks(range(len(order)), [f"{ds[d]['name']} ({ds[d]['set']})" for d in order], fontsize=6)
    a.set_xlim(min(min(ds[d]["split_means"].values()) for d in order) - 3, 121)
    a.set_xticks([20, 40, 60, 80, 100])
    a.spines["bottom"].set_bounds(a.get_xlim()[0], 100)
    a.set_xlabel("macro-F1, %")
    a.grid(axis="x", color=GRID, lw=0.4)
    a.set_axisbelow(True)
    a.scatter([], [], s=9, color=SLOTS[0], label="split means (over networks, seeds)")
    a.scatter([], [], s=11, marker="D", color=SLOTS[1], label="network means (over splits, seeds)")
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.11), ncol=2, fontsize=5.6, frameon=False,
             handletextpad=0.3, columnspacing=0.8)
    a.set_ylim(-0.7, len(order) - 0.3)
    a.text(102, len(order) - 0.25, "sd split /\nnetwork", ha="left", va="bottom", fontsize=5.6, color=INK2)
    a.set_title("(a)", loc="left", fontweight="bold")
    cmap = LinearSegmentedColormap.from_list("div", [SLOTS[1], "#fbfaf8", SLOTS[0]])
    heat = {}
    for k, (d, letter) in enumerate((("uci_har", "b"), ("up_fall", "c"))):
        ax = fig.add_subplot(gs[k, 1])
        v = ds[d]
        models = list(v["cell_means"])
        splits = sorted(int(s) for s in next(iter(v["cell_means"].values())))
        m = np.array([[v["cell_means"][mm][str(s)] if str(s) in v["cell_means"][mm] else v["cell_means"][mm][s]
                       for s in splits] for mm in models])
        dev = m - m.mean()
        lim = np.abs(dev).max()
        im = ax.imshow(dev, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(models)), [NET[mm] for mm in models], fontsize=5.4)
        ax.set_xticks(range(len(splits)), [str(s) for s in splits], fontsize=5.6)
        ax.set_xlabel("split (test-subject group)")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label("F1 − dataset mean", fontsize=5.6)
        cb.outline.set_visible(False)
        ax.set_title(f"({letter}) {v['name']} ({v['set']}): network × split", loc="left", fontweight="bold")
        heat[d] = {"models": models, "splits": splits, "cell_means": m.tolist()}
    save(fig, "fig07_split_spread")
    data = {"order": order, "per_dataset": {d: {k: ds[d][k] for k in ("set", "name", "sd_split_means", "sd_model_means",
                                                                      "mean_seed_sd", "range_split_means",
                                                                      "range_model_means")} for d in order},
            "heatmaps": heat, "summary": {k: v for k, v in sp.items() if k != "datasets"}}
    record(FIG_DATA / "fig07_split_spread.json", data, run_id, started, [SPREAD])
    return data


# ---------------------------------------------------------------------------------------------------------- fig08

def fig08(run_id, started) -> dict:
    summ_f, tsne_f = EMB / "summary.json", EMB / "tsne_case.json"
    summ, ts = json.loads(summ_f.read_text()), json.loads(tsne_f.read_text())
    names = {d: v["name"] for d, v in json.loads(SENSORS.read_text())["datasets"].items()}
    fig = plt.figure(figsize=(7.16, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.95, 1.05], wspace=0.72, hspace=0.42)
    tr, te = ts["train"], ts["test"]
    acts = json.loads(ACTS.read_text())["activities"] if ts["case"]["dataset"] == "uci_har" else {}
    xy_tr, xy_te = np.array(tr["xy"]), np.array(te["xy"])
    cats = sorted(set(tr["activity"]) | set(te["activity"]))
    pal = (SLOTS + [EXTRA, INK2, "#6fb7e8", "#f2a07b"])[:len(cats)]
    a = fig.add_subplot(gs[0, 0])
    for k, cl in enumerate(cats):
        mt = np.array(tr["activity"]) == cl
        me = np.array(te["activity"]) == cl
        a.scatter(xy_tr[mt, 0], xy_tr[mt, 1], s=1.5, color=pal[k], alpha=0.25, linewidth=0, rasterized=True)
        a.scatter(xy_te[me, 0], xy_te[me, 1], s=5, color=pal[k], edgecolor=INK, linewidth=0.25, rasterized=True,
                  label=acts.get(str(cl), f"class {cl}"))
    for k, cl in enumerate(cats):  # direct labels at each class's centre
        pts = np.vstack([xy_tr[np.array(tr["activity"]) == cl], xy_te[np.array(te["activity"]) == cl]])
        cx, cy = np.median(pts, axis=0)
        a.text(cx, cy, acts.get(str(cl), f"class {cl}"), fontsize=5.4, ha="center", va="center", color=INK,
               fontweight="bold", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    a.set_xticks([])
    a.set_yticks([])
    case = ts["case"]
    a.set_title(f"(a) {names[case['dataset']]}, {NET[case['model']]}, split {case['split']}, by class", loc="left",
                fontweight="bold", fontsize=7)
    b = fig.add_subplot(gs[1, 0])
    b.scatter(xy_tr[:, 0], xy_tr[:, 1], s=1.5, color=GRID, alpha=0.6, linewidth=0, rasterized=True,
              label="training subjects")
    subs = sorted(set(te["subject"]))
    for k, s in enumerate(subs):
        m = np.array(te["subject"]) == s
        b.scatter(xy_te[m, 0], xy_te[m, 1], s=5, color=SLOTS[k % 5], edgecolor=INK, linewidth=0.25, rasterized=True,
                  label=f"test subject {s}")
    b.set_xticks([])
    b.set_yticks([])
    b.legend(fontsize=5.3, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.01), ncol=2, markerscale=1.5,
             handletextpad=0.1, columnspacing=0.6)
    b.set_title("(b) the same, by subject", loc="left", fontweight="bold", fontsize=7)

    per = summ["per_dataset"]
    usable = [d for d in per if per[d]["splits"] > 3]
    order = sorted(usable, key=lambda d: per[d]["rho_embedding_ccs_f1"])
    c = fig.add_subplot(gs[:, 1])
    for i, d in enumerate(order):
        re_, rs = per[d]["rho_embedding_ccs_f1"], per[d]["rho_sensor_ccs_f1"]
        c.plot([rs, re_], [i, i], color=GRID, lw=1.2, zorder=1)
        c.scatter([rs], [i], s=14, color=SLOTS[2], edgecolor="white", linewidth=0.4, zorder=3)
        c.scatter([re_], [i], s=14, color=SLOTS[0], edgecolor="white", linewidth=0.4, zorder=3)
    c.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
    c.set_yticks(range(len(order)), [names[d] for d in order], fontsize=6)
    c.set_xlim(-1.05, 1.05)
    c.set_xlabel("Spearman ρ over splits:\ntest-group shift vs split F1")
    c.scatter([], [], s=14, color=SLOTS[0], label="embedding")
    c.scatter([], [], s=14, color=SLOTS[2], label="sensor features")
    c.legend(fontsize=5.6, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.17), handletextpad=0.2)
    c.grid(axis="x", color=GRID, lw=0.4)
    c.set_axisbelow(True)
    c.set_title("(c) shift vs accuracy", loc="left", fontweight="bold", fontsize=7)

    dprobe = {}
    for p in summ["probes"]:
        dprobe.setdefault(p["dataset"], []).append(p)
    pord = sorted(dprobe, key=lambda d: np.median([p["subject_probe"] / p["chance"] for p in dprobe[d]]))
    e = fig.add_subplot(gs[:, 2])
    for i, d in enumerate(pord):
        emb = [p["subject_probe"] / p["chance"] for p in dprobe[d] if not math.isnan(p["subject_probe"])]
        sen = [p["sensor_subject_probe"] / p["chance"] for p in dprobe[d] if not math.isnan(p["sensor_subject_probe"])]
        e.scatter(emb, i + 0.15 + np.zeros(len(emb)), s=4, color=SLOTS[0], alpha=0.35, linewidth=0)
        e.scatter([np.median(emb)], [i + 0.15], s=18, color=SLOTS[0], edgecolor="white", linewidth=0.4, zorder=3)
        e.scatter([np.median(sen)], [i - 0.15], s=18, color=SLOTS[2], marker="s", edgecolor="white", linewidth=0.4,
                  zorder=3)
    e.axvline(1, color=INK, lw=0.8, ls=(0, (3, 2)))
    e.set_xscale("log")
    e.set_yticks(range(len(pord)), [names[d] for d in pord], fontsize=6)
    e.set_xlabel("subject probe:\nbalanced accuracy / chance")
    e.scatter([], [], s=14, color=SLOTS[0], label="embedding (median, cases)")
    e.scatter([], [], s=14, marker="s", color=SLOTS[2], label="sensor features (median)")
    e.legend(fontsize=5.6, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.17), handletextpad=0.2)
    e.grid(axis="x", color=GRID, lw=0.4)
    e.set_axisbelow(True)
    e.set_title("(d) subject identity", loc="left", fontweight="bold", fontsize=7)
    save(fig, "fig08_embedding")
    data = {"tsne_case": ts["case"], "per_dataset_rho": {d: {k: per[d][k] for k in ("rho_embedding_ccs_f1",
                                                                                   "rho_sensor_ccs_f1", "splits")}
                                                         for d in per},
            "probe_ratio_median": {d: {"embedding": float(np.median([p["subject_probe"] / p["chance"] for p in dprobe[d]])),
                                       "sensor": float(np.median([p["sensor_subject_probe"] / p["chance"] for p in dprobe[d]]))}
                                   for d in dprobe},
            "summary": {k: v for k, v in summ.items() if k not in ("per_dataset", "probes")}}
    record(FIG_DATA / "fig08_embedding.json", data, run_id, started, [summ_f, tsne_f, SENSORS, ACTS])
    return data


# ---------------------------------------------------------------------------------------------------------- fig09

def fig09(run_id, started) -> dict:
    sp = json.loads(SPREAD.read_text())
    ds = sp["datasets"]
    order = sorted((d for d in ds if ds[d]["validation"] is not None), key=lambda d: ds[d]["validation"]["gap"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.55})
    for i, d in enumerate(order):
        x = ds[d]["validation"]
        a.plot([x["test_mean"], x["val_mean"]], [i, i], color=GRID, lw=1.4, zorder=1)
        a.scatter([x["val_mean"]], [i], s=16, facecolor="white", edgecolor=SLOTS[0], linewidth=1.0, zorder=3)
        a.scatter([x["test_mean"]], [i], s=16, color=SLOTS[0], edgecolor="white", linewidth=0.4, zorder=3)
        a.text(101, i, f"{x['gap']:+.1f}", va="center", fontsize=5.6, color=INK2)
    a.set_yticks(range(len(order)), [f"{ds[d]['name']} ({ds[d]['set']})" for d in order], fontsize=6)
    a.set_xlim(min(ds[d]["validation"]["test_mean"] for d in order) - 4, 100)
    a.set_ylim(-0.7, len(order) - 0.3)
    a.text(101, len(order) - 0.25, "gap", ha="left", va="bottom", fontsize=5.6, color=INK2)
    a.set_xlabel("macro-F1, % (mean over runs)")
    a.scatter([], [], s=16, facecolor="white", edgecolor=SLOTS[0], label="validation (training subjects)")
    a.scatter([], [], s=16, color=SLOTS[0], label="test (unseen subjects)")
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=2, fontsize=5.6, frameon=False, handletextpad=0.3)
    a.grid(axis="x", color=GRID, lw=0.4)
    a.set_axisbelow(True)
    a.set_title("(a) validation on training subjects vs test", loc="left", fontweight="bold")
    rows = [d for d in order if ds[d]["validation"]["rho_network_means"] is not None]
    for i, d in enumerate(rows):
        x = ds[d]["validation"]
        same = x["best_network_by_val"] == x["best_network_by_test"]
        b.barh([i], [x["rho_network_means"]], height=0.62, color=SLOTS[0] if ds[d]["set"] == "S1" else SLOTS[1],
               edgecolor="white", linewidth=0.4)
        b.text(1.08, i, "same" if same else f"−{x['test_loss_choosing_by_val']:.1f}", va="center", fontsize=5.6,
               color=INK2)
    b.axvline(0, color=INK, lw=0.8)
    b.set_xlim(-1.05, 1.05)
    b.set_ylim(-0.7, len(rows) - 0.3)
    b.set_yticks(range(len(rows)), [f"{ds[d]['name']} ({ds[d]['set']})" for d in rows], fontsize=6)
    b.set_xlabel("Spearman ρ: network means, validation vs test")
    b.text(1.08, len(rows) - 0.25, "best by val.", ha="left", va="bottom", fontsize=5.6, color=INK2)
    b.grid(axis="x", color=GRID, lw=0.4)
    b.set_axisbelow(True)
    b.set_title("(b) does validation rank the networks?", loc="left", fontweight="bold")
    save(fig, "fig09_validation")
    data = {"order": order, "per_dataset": {d: ds[d]["validation"] for d in order},
            "summary": sp["validation"]}
    record(FIG_DATA / "fig09_validation.json", data, run_id, started, [SPREAD])
    return data


FIGS = {"fig03": fig03, "fig04": fig04, "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08, "fig09": fig09}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    started = time.time()
    run_id = PV.new_run_id("detail_figures")
    for key, fn in FIGS.items():
        if a.only and key not in a.only:
            continue
        out = fn(run_id, started)
        if key == "fig04":
            tab02(out, run_id, started)
        if key == "fig05":
            tab03(run_id, started)
        print(key, "written", flush=True)
    print("run", run_id)


if __name__ == "__main__":
    main()
