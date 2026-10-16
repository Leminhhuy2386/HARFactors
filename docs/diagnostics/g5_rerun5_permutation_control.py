"""Diagnose the S2 model-permutation control of G5 rerun5 (read-only: no project file is written)."""
import hashlib, platform, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path.cwd()))
from src.data.sets import S2
from src.evaluation import reml_numpy as RN
from src.evaluation import variance_components as V
print(platform.node(), subprocess.run("lscpu | grep 'Model name'", shell=True, capture_output=True, text=True).stdout.strip())
inc = V.inclusion(S2)
df = V.load(S2, "macro_f1", [d for d, v in inc.items() if v["included"]])
perm = RN.permute_models(df, 20261015)
print("rows", len(df), "relabelled rows", int((perm["model"] != df["model"]).sum()),
      "perm sha", hashlib.sha256(perm.to_csv(index=False).encode()).hexdigest()[:16])
y = perm["y"].to_numpy(float); X = pd.get_dummies(perm["dataset"]).to_numpy(float)
ker = RN.kernels(perm, True); names = list(ker); K = [ker[n] for n in names]
def nr(comp):
    return RN.neg_reml(np.log(np.maximum([comp[n] for n in names] + [comp["e"]], np.exp(RN.FLOOR))), y, X, K)
f = V.fit(perm, True)
print("statsmodels", f["method"], {k: round(v, 4) for k, v in f["components"].items()}, "neg_reml", round(nr(f["components"]), 4))
alt = RN.fit(perm, True)
print("numpy REML", alt["converged"], {k: round(v, 4) for k, v in alt["components"].items()}, "neg_reml", round(alt["neg_reml"], 4))
real = V.fit(df, True)
print("real statsmodels", {k: round(v, 4) for k, v in real["components"].items()})
for seed in (1, 2, 3):
    p2 = RN.permute_models(df, seed); f2 = V.fit(p2, True); c = f2["components"]
    print("other permutation seed", seed, "share", round((c["M"] + c["MD"] + c["MP"]) / sum(c.values()), 4))
g5 = {"M": 16.32673673807694, "MD": 2.603830939217106, "MP": 0.06996318843225441, "P": 130.12406950798848,
      "e": 23.888296733154043}
print("G5 rerun5 permuted solution: neg_reml", round(nr(g5), 4), "(lower is better)")
import os
print("threads", os.environ.get("OMP_NUM_THREADS"), os.cpu_count())
