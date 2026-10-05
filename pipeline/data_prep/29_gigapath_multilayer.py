# -*- coding: utf-8 -*-
"""29_gigapath_multilayer.py — Prov-GigaPath 官方 slide embedding 的多层信息量

背景: 04_ab_feature_source.py 当年只取了 embedding[0](第 0 层), 14 层里 13 层没用过。
本脚本回答: 哪几层有信息? 多层拼接 / 平均能否超过单层?

注意: 这不是"新增数据集" —— parquet 早已在本地(444MB), 这里只是把已有资产的
第 1~13 层也用上。Case 集合与协议在本脚本内完全对齐, 内部可公平比较。
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
SEP = "=" * 92

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
coh = man[(man.project == "LUAD") & (man.has_label == 1)].dropna(subset=["TP53"]).copy()
coh = coh.set_index("case_id")
print(f"[LUAD cohort] n={len(coh)}")

# ---- 从 parquet 里抽本队列的 case (只读需要的 2 列, 再过滤, 避免 444MB 全量解码)
CACHE = os.path.join(C, "NSCLC_gigap_slide_L14.npy")
CIDX = os.path.join(C, "NSCLC_gigap_slide_L14_index.csv")

if os.path.exists(CACHE) and os.path.exists(CIDX):
    E = np.load(CACHE)
    ids = pd.read_csv(CIDX).case_id.tolist()
    print(f"[cache] 复用 {E.shape}")
else:
    want = {c.replace("TCGA-", "") for c in coh.index}
    t = pq.read_table(os.path.join(R, "data", "public_features", "prov_gigapath_tcga",
                                   "provgigapath_embeddings_with_metadata.parquet"),
                      columns=["_PATIENT", "embedding"])
    pat = t.column("_PATIENT").to_pylist()
    embs = t.column("embedding")
    acc = {}
    for i, p in enumerate(pat):
        if not p:
            continue
        key = p.replace("TCGA-", "")
        if key not in want or key in acc:
            continue
        arr = np.asarray(embs[i].as_py(), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            acc[key] = arr
    if not acc:
        raise SystemExit("parquet 未命中任何本队列 case")
    Lmin = min(a.shape[0] for a in acc.values())
    keys = [k for k, a in acc.items() if a.shape[0] == Lmin]
    E = np.stack([acc[k][:Lmin] for k in keys])
    ids = ["TCGA-" + k for k in keys]
    np.save(CACHE, E)
    pd.DataFrame({"case_id": ids}).to_csv(CIDX, index=False)
    print(f"[parquet] 11948 slide -> 命中 {len(ids)} case, L={Lmin}")

print(f"[embedding] shape={E.shape}  cases={len(ids)}")
sub = coh.loc[[c for c in ids if c in coh.index]]
ids = [c for c in ids if c in coh.index]
print(f"[交集] n={len(ids)}  LUAD 队列覆盖率 {len(ids)/len(coh):.1%}")
Y = sub[GENES].values.astype(int)
print("positive counts:", {g: int(Y[:, j].sum()) for j, g in enumerate(GENES)})

# ---- 统一评估协议: 5-fold OOF LR, 标准化 + PCA(降维防过拟合), 5 seed 配对
def oof_auc(X, y, seed, ncomp=128):
    if not (0 < int(y.sum()) < len(y)):
        return np.nan
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        nc = min(ncomp, X.shape[1], max(2, len(tr) - 1))
        pca = PCA(n_components=nc, random_state=0).fit(sc.transform(X[tr]))
        m = LogisticRegression(max_iter=3000, C=0.1,
                               class_weight="balanced").fit(pca.transform(sc.transform(X[tr])), y[tr])
        oof[te] = m.predict_proba(pca.transform(sc.transform(X[te])))[:, 1]
    return roc_auc_score(y, oof)

SEEDS = [0, 1, 2, 3, 4]
L = E.shape[1]
print("\n" + SEP)
print(f"=== 1. 逐层信息量 (PCA128 + LR, 5 seed 配对均值) ===")
print(f"{'layer':<8}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}{'std':>8}")
layer_auc = np.zeros((L, len(GENES)))
for l in range(L):
    X = E[:, l, :]
    per_seed = np.array([[oof_auc(X, Y[:, j], s) for j in range(len(GENES))] for s in SEEDS])
    layer_auc[l] = per_seed.mean(0)
    print(f"L{l:<7}" + "".join(f"{a:>9.3f}" for a in per_seed.mean(0))
          + f"{per_seed.mean(0).mean():>9.3f}{per_seed.mean(1).std():>8.4f}")

order = np.argsort(-layer_auc.mean(1))
print("\n信息量排序 (layer: mean AUC):")
print("  " + "  ".join(f"L{l}={layer_auc[l].mean():.3f}" for l in order[:6]))

# ---- 多层组合
print("\n" + SEP)
print("=== 2. 多层组合 vs 最佳单层 (5 seed 配对) ===")
variants = {}
variants["L0 (历史做法)"] = E[:, 0, :].copy()
variants["all 14 concat"] = E.reshape(len(ids), -1)
variants["14 层均值"] = E.mean(1)
top3 = sorted(order[:3])
variants[f"top3 concat {top3}"] = E[:, top3, :].reshape(len(ids), -1)
top5 = sorted(order[:5])
variants[f"top5 concat {top5}"] = E[:, top5, :].reshape(len(ids), -1)
variants[f"best single L{order[0]}"] = E[:, order[0], :].copy()
variants["L0+L13 (首末层)"] = E[:, [0, L - 1], :].reshape(len(ids), -1)

res = {}
print(f"{'variant':<24}{'dim':>7}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}{'std':>8}")
for name, X in variants.items():
    per_seed = np.array([[oof_auc(X, Y[:, j], s) for j in range(len(GENES))] for s in SEEDS])
    res[name] = per_seed
    print(f"{name:<24}{X.shape[1]:>7}" + "".join(f"{a:>9.3f}" for a in per_seed.mean(0))
          + f"{per_seed.mean(0).mean():>9.3f}{per_seed.mean(1).std():>8.4f}")

# ---- 与 L0 的配对差 + bootstrap 显著性
print("\n" + SEP)
print("=== 3. 相对 L0 的配对差 (per-gene, per-seed), 逐基因符号检验 ===")
base = res["L0 (历史做法)"].mean(0)
for name in variants:
    if name == "L0 (历史做法)":
        continue
    d = res[name].mean(0) - base
    win = int((d > 0).sum())
    print(f"{name:<24}" + "".join(f"{v:>+9.4f}" for v in d) + f"  胜 {win}/6")

out = []
for name in variants:
    per_seed = res[name]
    for si, s in enumerate(SEEDS):
        for j, g in enumerate(GENES):
            out.append(dict(variant=name, seed=s, gene=g, auc=float(per_seed[si, j])))
pd.DataFrame(out).to_csv(os.path.join(C, "gigap_multilayer.csv"), index=False)
print(f"\n[saved] {C}\\gigap_multilayer.csv")
print("DONE")