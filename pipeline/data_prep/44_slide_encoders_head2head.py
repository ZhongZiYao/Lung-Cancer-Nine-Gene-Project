# -*- coding: utf-8 -*-
"""44_slide_encoders_head2head.py — slide 级编码器正面对比 (决定是否值得做 patch+slide 融合)

背景: 我们做过 patch 级 MIL (0.743) 和 GigaPath slide 线性探针 (0.674),
但**从未量化过 TITAN 这个官方 slide encoder 的预测能力**。
TITAN 是多模态训练的(视觉自监督 + 病理报告视觉-语言对齐), 335k WSI,
与 Prov-GigaPath slide(纯视觉自监督, 长程注意力)是不同训练目标 —— 可能互补。

本脚本只回答一件事: 在同一批 LUAD case 上, 各种 slide 级表示能做多好?
如果 TITAN 显著强于 GigaPath slide, 才有理由做融合。

同时测: TITAN + GigaPath-slide 拼接, 以及两者的 score-level ensemble。
"""
import os, sys, re, pickle
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
tn = lambda b: "-".join(str(b).split("-")[:3]) if len(str(b).split("-")) >= 3 and str(b).startswith("TCGA") else None
SEEDS = [0, 1, 2, 3, 4]

# ---------- 标签 ----------
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
m = man[(man.has_label == 1) & (man.project.isin(["LUAD", "LUSC"]))]
base = m.set_index("case_id")

# ---------- TITAN ----------
tk = pickle.load(open(os.path.join(R, "models", "TITAN", "TCGA_TITAN_features.pkl"), "rb"))
T = {}
for f, e in zip(tk["filenames"], tk["embeddings"]):
    c = tn(os.path.basename(f))
    if c and c not in T:
        T[c] = np.asarray(e, np.float32)
print(f"[TITAN] 解析出 {len(T)} case, 维度 {len(next(iter(T.values())))}", flush=True)

# ---------- GigaPath slide (14 层) ----------
CACHE = os.path.join(C, "NSCLC_gigap_slide_L14.npy")
CIDX = os.path.join(C, "NSCLC_gigap_slide_L14_index.csv")
if not (os.path.exists(CACHE) and os.path.exists(CIDX)):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(R, "data", "public_features", "prov_gigapath_tcga",
                                    "provgigapath_embeddings_with_metadata.parquet"),
                      columns=["_PATIENT", "embedding"])
    pat = t.column("_PATIENT").to_pylist(); embs = t.column("embedding")
    acc = {}
    for i, p in enumerate(pat):
        if not p:
            continue
        k = p.replace("TCGA-", "")
        if k in acc:
            continue
        arr = np.asarray(embs[i].as_py(), dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            acc[k] = arr
    Lm = min(a.shape[0] for a in acc.values())
    keys = [k for k, a in acc.items() if a.shape[0] == Lm]
    E = np.stack([acc[k][:Lm] for k in keys])
    ids = ["TCGA-" + k for k in keys]
    np.save(CACHE, E)
    pd.DataFrame({"case_id": ids}).to_csv(CIDX, index=False)
    print(f"[GigaPath-slide] 缓存重建: {E.shape}", flush=True)
    G = {c: E[i] for i, c in enumerate(ids)}
else:
    E = np.load(CACHE)
    ids = pd.read_csv(CIDX).case_id.tolist()
    print(f"[GigaPath-slide] 复用缓存 {E.shape}", flush=True)
    G = {c: E[i] for i, c in enumerate(ids)}

# ---------- 对齐到有标签的肺癌 case ----------
cases = [c for c in (m.case_id.tolist()) if c in T and c in G]
proj = m.set_index("case_id").loc[cases, "project"]
Y = m.set_index("case_id").loc[cases, GENES].values.astype(int)
proj = proj.values
nLU = int((proj == "LUAD").sum())
print(f"\n[对齐] slide+patch 都有且有标签的肺癌 case: {len(cases)} (LUAD {nLU} / LUSC {len(cases)-nLU})", flush=True)

XT = np.stack([T[c] for c in cases])                      # (N, 768)
XG = np.stack([G[c].mean(0) for c in cases])              # (N, 768) 14 层均值
print(f"[特征] TITAN {XT.shape}  GigaPath-slide-mean {XG.shape}", flush=True)

mask = proj == "LUAD"
YLU = Y[mask]
XTlu, XGlu = XT[mask], XG[mask]


def probe(Xd, y, seed, ncomp=128):
    if not (10 <= y.sum() <= len(y) - 10):
        return np.nan
    o = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, y):
        sc = StandardScaler().fit(Xd[tr])
        nc = min(ncomp, Xd.shape[1], max(2, len(tr) - 1))
        pca = PCA(n_components=nc, random_state=0).fit(sc.transform(Xd[tr]))
        mdl = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(
            pca.transform(sc.transform(Xd[tr])), y[tr])
        o[te] = mdl.predict_proba(pca.transform(sc.transform(Xd[te])))[:, 1]
    return roc_auc_score(y, o)


VIEWS = [("TITAN 768", XTlu), ("GigaPath-slide 均值 768", XGlu), ("两者拼接 1536", np.hstack([XTlu, XGlu]))]
print("\n" + "=" * 96)
print("=== LUAD slide 级编码器正面对比 (5 seed, PCA128+LR) ===")
print("=" * 96)
print(f"{'表示':<26}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}")
store = {}
for name, M in VIEWS:
    per_seed = np.array([[probe(M, YLU[:, j], s) for j in range(len(GENES))] for s in SEEDS])
    store[name] = per_seed
    print(f"{name:<26}" + "".join(f"{v:>9.3f}" for v in per_seed.mean(0))
          + f"{per_seed.mean(0).mean():>9.4f}")
print(f"{'参照: patch MIL (41号)':<26}" + "".join(f"{v:>9.3f}" for v in
      [0.708, 0.775, 0.717, 0.626, 0.812, 0.748]) + f"{0.7313:>9.4f}")

# score-level ensemble: TITAN 与 GigaPath-slide 的 OOF 平均
print("\n=== TITAN + GigaPath-slide 的 score-level 集成 (0 额外训练) ===")
oofT = np.zeros((len(YLU), len(GENES))); oofG = np.zeros_like(oofT)
for j in range(len(GENES)):
    for seed in [0]:
        for nm, M, acc in [("T", XTlu, oofT), ("G", XGlu, oofG)]:
            pass
# 直接用每个 view 的 5 seed 平均 OOF (按 seed 平均) 做 ensemble
def oof_mean(M, y, j, seeds=SEEDS):
    ps = []
    for s in seeds:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(M, y):
            sc = StandardScaler().fit(M[tr])
            nc = min(128, M.shape[1], max(2, len(tr) - 1))
            pca = PCA(n_components=nc, random_state=0).fit(sc.transform(M[tr]))
            mdl = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(
                pca.transform(sc.transform(M[tr])), y[tr])
            o[te] = mdl.predict_proba(pca.transform(sc.transform(M[te])))[:, 1]
        ps.append(o)
    return np.mean(ps, 0)

ens = []
for j, g in enumerate(GENES):
    a = roc_auc_score(YLU[:, j], oof_mean(XTlu, YLU[:, j], j))
    b = roc_auc_score(YLU[:, j], oof_mean(XGlu, YLU[:, j], j))
    c = roc_auc_score(YLU[:, j], (oof_mean(XTlu, YLU[:, j], j) + oof_mean(XGlu, YLU[:, j], j)) / 2)
    ens.append((g, a, b, c))
print(f"{'gene':<8}{'TITAN':>10}{'GP-slide':>10}{'score集成':>12}")
for g, a, b, c in ens:
    print(f"{g:<8}{a:>10.3f}{b:>10.3f}{c:>12.3f}")
print(f"{'mean':<8}{np.mean([e[1] for e in ens]):>10.3f}{np.mean([e[2] for e in ens]):>10.3f}"
      f"{np.mean([e[3] for e in ens]):>12.3f}")

print("\n=== 融合可行性判据 ===")
best_slide = max(np.mean([e[1] for e in ens]), np.mean([e[3] for e in ens]))
print(f"  最佳 slide 级表示: {best_slide:.4f}")
print(f"  patch MIL (参照): 0.7313")
if best_slide > 0.72:
    print("  -> slide 级表示已接近 MIL, **融合有明确价值**, 值得投入")
elif best_slide > 0.68:
    print("  -> slide 级表示中等, 融合可能有小幅增益, 需实测")
else:
    print("  -> slide 级表示明显弱于 MIL, 融合收益预期低")
pd.DataFrame([dict(gene=e[0], titan=e[1], gp_slide=e[2], score_ens=e[3]) for e in ens]).to_csv(
    os.path.join(C, "slide_encoders_head2head.csv"), index=False)
print(f"\n[saved] {C}\\slide_encoders_head2head.csv")
print("DONE", flush=True)