# -*- coding: utf-8 -*-
"""58_bootstrap_ci.py — 对已有 test 预测做 bootstrap 置信区间 (纯 CPU, 不占 GPU)

和 57 号的重复抽样互补:
  57 问的是"换一套划分, test mean 会动多少" -> 划分层面的方差
  58 问的是"固定这一套划分, 只重采样 test 这 86 例, AUC 会动多少" -> 病例层面的方差

两者的区别很重要: 57 的 std 包含训练集变化, 58 的 CI 只包含评估集抽样。
58 更贴近"这个 AUC 点估计本身有多不确定"。
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
B = 4000
rng = np.random.RandomState(7)

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).set_index("case_id")
sp = pd.read_csv(os.path.join(C, "split_luad.csv"))
common = list(sp.case_id)
Y = lab.loc[common, GENES].values.astype(int)
split = sp.set_index("case_id").loc[common, "split"].values
te = split == "test"
y_te = Y[te]
print("test n=%d" % len(y_te), flush=True)

srcs = {}
for tag, fn in [("gigap_S2048", "inst_sweep_test_pred.npy"), ("ENS4_split6", "unified_test_pred.npy")]:
    f = os.path.join(C, fn)
    if os.path.exists(f):
        P = np.load(f)
        if P.shape[0] == len(y_te):
            srcs[tag] = P
            print("loaded %-14s %s" % (tag, P.shape), flush=True)

rows = []
for tag, P in srcs.items():
    print("\n=== %s ===" % tag, flush=True)
    for j, g in enumerate(GENES):
        y = y_te[:, j]; p = P[:, j]
        if not (10 <= y.sum() <= len(y) - 10):
            print("  %-6s n_pos=%2d  -> n/a (低于 AUC 门槛 10)" % (g, int(y.sum())), flush=True)
            continue
        pt = roc_auc_score(y, p)
        bs = np.empty(B)
        for b in range(B):
            s = rng.randint(0, len(y), len(y))
            if 0 < y[s].sum() < len(s):
                bs[b] = roc_auc_score(y[s], p[s])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(model=tag, gene=g, n_pos=int(y.sum()), auc=pt, ci_lo=lo, ci_hi=hi, ci_w=(hi - lo) / 2))
        print("  %-6s n_pos=%2d  AUC=%.3f  95%%CI [%.3f, %.3f]  halfwidth=%.3f"
              % (g, int(y.sum()), pt, lo, hi, (hi - lo) / 2), flush=True)
    # mean over genes, bootstrap 同一组重采样一起算
    use = [j for j in range(len(GENES)) if 10 <= y_te[:, j].sum() <= len(y_te) - 10]
    m = np.mean([roc_auc_score(y_te[:, j], P[:, j]) for j in use])
    bm = np.empty(B)
    for b in range(B):
        s = rng.randint(0, len(y_te), len(y_te))
        vals = []
        for j in use:
            yy, pp = y_te[s, j], P[s, j]
            if 0 < yy.sum() < len(s):
                vals.append(roc_auc_score(yy, pp))
        bm[b] = np.mean(vals)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    print("  %-6s MEAN     =%.4f 95%%CI [%.4f, %.4f] halfwidth=%.4f" % ("mean", m, lo, hi, (hi - lo) / 2), flush=True)
    rows.append(dict(model=tag, gene="MEAN", n_pos=-1, auc=m, ci_lo=lo, ci_hi=hi, ci_w=(hi - lo) / 2))

pd.DataFrame(rows).to_csv(os.path.join(C, "bootstrap_ci.csv"), index=False)
print("\n[saved] bootstrap_ci.csv", flush=True)
print("DONE", flush=True)