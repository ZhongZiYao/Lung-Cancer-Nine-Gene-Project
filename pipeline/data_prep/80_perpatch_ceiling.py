# -*- coding: utf-8 -*-
"""80_perpatch_ceiling.py - 选出"有贡献的 patch" 这件事的天花板在哪

The proposal under test: "take more patches, then use some disambiguation mechanism to
pick the patches that actually contribute to training". That splits into two halves, and
they have very different evidence:

  more patches      -- already tested twice (§56 S_INST 512->2048: +0.004;
                       §79 fixed-2048: +0.0038, CI crosses 0). Not the bottleneck.
  better selection  -- gated attention IS already a learned instance selector, and every
                       explicit alternative lost (topk -0.012, max -0.031, lse -0.037,
                       mean -0.038). But we never measured the CEILING: if a single
                       patch carries almost no label information, no selector can help.

This script measures that ceiling directly and cheaply:
  1. fit a linear probe on INDIVIDUAL patches (train patches come from train slides only)
  2. score every patch of every test slide
  3. aggregate per slide four ways -> AUC

  max_score   : the single most label-informative patch in the slide (oracle-ish upper
                bound for "pick the best patch")
  topk_mean   : mean score of the top 1% / 5% / 10% of patches
  mean        : bag mean score, i.e. no selection at all

Comparing topk_mean against MIL tells us whether per-slide selection is adding anything
the current gated attention has not already extracted.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
N_PATCH = 512
FOLD_SEED = 101
MAX_TRAIN_PATCH = 60000

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
proj_all = man.set_index("case_id")["project"].to_dict()
idx = pd.read_csv(os.path.join(C, "NSCLC_gigap_sub512_index.csv"))
rm = dict(zip(idx.case_id, idx.mem_row))
cases = sorted(c for c in rm if proj_all.get(c) == "LUAD" and c in lab.index)
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
rows = np.array([rm[c] for c in cases])
K = len(GENES)
print("[data] %d 例 x %d patch, 特征 %d 维" % (len(cases), N_PATCH, 1536), flush=True)
X = np.load(os.path.join(C, "NSCLC_gigap_sub512.npy"), mmap_mode="r")
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = list(StratifiedKFold(5, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(cases)), tier))
print("[folds] fold_seed=%d, 5 折" % FOLD_SEED, flush=True)


def probe(tr_i, tr_y, j, seed=0):
    """Logistic regression on individual patches; returns a scoring function."""
    rng = np.random.RandomState(seed)
    Xt, yt = [], []
    for i in tr_i:
        v = np.asarray(X[rows[i]], dtype=np.float32)
        yv = tr_y[:, j]
        # class-balanced patch subsample: 40 pos + 40 neg patches per slide
        pos = np.flatnonzero(yv == 1); neg = np.flatnonzero(yv == 0)
        take_p = rng.choice(pos, min(40, len(pos)), replace=False) if len(pos) else np.array([], int)
        take_n = rng.choice(neg, min(40, len(neg)), replace=False) if len(neg) else np.array([], int)
        sel = np.concatenate([take_p, take_n])
        if len(sel):
            Xt.append(v[sel]); yt.append(yv[sel])
    if not Xt:
        return None
    Xt = np.concatenate(Xt); yt = np.concatenate(yt).astype(np.float32)
    if len(Xt) > MAX_TRAIN_PATCH:
        sel = rng.choice(len(Xt), MAX_TRAIN_PATCH, replace=False)
        Xt, yt = Xt[sel], yt[sel]
    mu = Xt.mean(0, keepdims=True); sd = Xt.std(0, keepdims=True) + 1e-6
    Z = np.concatenate([(Xt - mu) / sd, np.ones((len(Xt), 1), np.float32)], 1)
    w = np.zeros(Z.shape[1], np.float32)
    lr, wd, bs = 0.5, 1e-4, 4096
    for it in range(60):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - yt) / len(yt) + wd * np.r_[w[:-1], 0.0]
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(yt) + wd * np.diag(np.r_[np.ones(Z.shape[1] - 1), 0.0])
        try:
            step = np.linalg.solve(H + 1e-3 * np.eye(Z.shape[1]), g)
        except Exception:
            step = lr * g
        w -= step
    return (mu, sd, w)


def score(m, v):
    mu, sd, w = m
    Z = np.concatenate([(v - mu) / sd, np.ones((len(v), 1), np.float32)], 1)
    return Z @ w


AGG = ["max", "top1%", "top5%", "top10%", "mean"]
res = {g: {a: [] for a in AGG} for g in GENES}
T0 = time.time()
for f, (tri, tei) in enumerate(folds):
    acc = {a: np.zeros(len(tei)) for a in AGG}
    for j, g in enumerate(GENES):
        m = probe(tri, Y[tri], j, seed=f * 10 + j)
        if m is None:
            continue
        for t, i in enumerate(tei):
            s = score(m, np.asarray(X[rows[i]], dtype=np.float32))
            k = max(1, int(len(s) * 0.01)), max(1, int(len(s) * 0.05)), max(1, int(len(s) * 0.10))
            so = np.sort(s)[::-1]
            acc["max"][t] = so[0]
            acc["top1%"][t] = so[:k[0]].mean()
            acc["top5%"][t] = so[:k[1]].mean()
            acc["top10%"][t] = so[:k[2]].mean()
            acc["mean"][t] = s.mean()
    for a in AGG:
        for j, g in enumerate(GENES):
            res[g][a].append(roc_auc_score(Y[tei, j], acc[a]))
    print("  fold %d done [%.0fs]" % (f, time.time() - T0), flush=True)

print("\n" + "=" * 88, flush=True)
print("=== 单 patch 线性探针 + 逐切片聚合 (fold_seed=%d) ===" % FOLD_SEED, flush=True)
print("对照: 我们的 gated MIL OOF = 0.7379" + " " * 20, flush=True)
print("=" * 88, flush=True)
print("%-8s %8s %8s %8s %8s %8s" % ("gene", "max", "top1%", "top5%", "top10%", "mean"), flush=True)
means = {a: [] for a in AGG}
for g in GENES:
    print("%-8s %8.3f %8.3f %8.3f %8.3f %8.3f"
          % tuple([g] + [np.mean(res[g][a]) for a in AGG]), flush=True)
    for a in AGG:
        means[a].append(np.mean(res[g][a]))
print("%-8s %8.3f %8.3f %8.3f %8.3f %8.3f   <- mean over 6 genes"
      % tuple(means[a] for a in AGG), flush=True)
pd.DataFrame([dict(gene=g, **{a: float(np.mean(res[g][a])) for a in AGG}) for g in GENES]).to_csv(
    os.path.join(C, "perpatch_ceiling.csv"), index=False)
print("\n[saved] perpatch_ceiling.csv", flush=True)
print("DONE", flush=True)