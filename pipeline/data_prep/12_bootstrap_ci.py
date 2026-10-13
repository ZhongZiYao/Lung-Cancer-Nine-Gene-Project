# -*- coding: utf-8 -*-
"""12_bootstrap_ci.py — 对所有消融变体和基线做 bootstrap 置信区间 + 配对 Delta 检验
回答: 哪些提升是真的, 哪些在噪声范围内
"""
import os, sys, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
NB = 2000
rng = np.random.RandomState(0)

man = pd.read_csv(os.path.join(CACHE,"cases_NSCLC.csv"))
man = man.dropna(subset=["TP53"])
N = len(man); Y = man[GENES].values.astype(int); mLU = man.project.values=="LUAD"
strata = np.clip(Y.sum(1),0,3).astype(int)
print(f"[data] n={N}  genes={GENES}")

# ---- 载入各变体 OOF ----
P = {}
for f in sorted(glob.glob(os.path.join(CACHE,"oof_*_NSCLC.npy"))):
    tag = os.path.basename(f)[4:-len("_NSCLC.npy")]
    p = np.load(f)
    if p.shape[0]==N: P[tag]=p
print(f"[oof] 载入 {len(P)} 个变体: {list(P)}")

# ---- mean-pool + LR 基线 OOF ----
bl_path = os.path.join(CACHE,"oof_meanpool_NSCLC.npy")
if os.path.exists(bl_path):
    P["meanpool_LR"]=np.load(bl_path)
else:
    X = np.load(os.path.join(CACHE,"NSCLC_sub2048.npy"))
    RT = np.load(os.path.join(CACHE,"rows_int.npy")) if os.path.exists(os.path.join(CACHE,"rows_int.npy")) else None
    ti = pd.read_csv(os.path.join(CACHE,"NSCLC_sub2048_index_fixed.csv"))
    ti = ti.merge(man[["case_id"]], on="case_id", how="inner")
    ROW = ti.mem_row.values.astype(int)
    Z = X[ROW].astype(np.float32).mean(1)
    o = np.zeros((N,len(GENES)),np.float32)
    for trn,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Z,strata):
        for j in range(len(GENES)):
            if Y[trn,j].sum()<2: continue
            sc=StandardScaler().fit(Z[trn])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Z[trn]),Y[trn,j])
            o[te,j]=m.predict_proba(sc.transform(Z[te]))[:,1]
    np.save(bl_path,o); P["meanpool_LR"]=o
    del X, Z
    print("[oof] mean-pool 基线已计算并保存")

def auc(y,p): return float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan

# ---- 点估计 ----
print(f"\n=== 点估计 (mean AUC) ===")
print(f"{'variant':<24}{'all':>8}{'LUAD':>8}")
for t,p in P.items():
    a=np.mean([auc(Y[:,j],p[:,j]) for j in range(len(GENES))])
    al=np.nanmean([auc(Y[mLU,j],p[mLU,j]) for j in range(len(GENES))])
    print(f"{t:<24}{a:>8.3f}{al:>8.3f}")

# ---- bootstrap 索引(按 case 重采样, 所有变体共用同一组索引 -> 配对比较) ----
idxs = [rng.choice(N, N, replace=True) for _ in range(NB)]
def boot_auc(p, sel):
    out=np.full((NB,len(GENES)),np.nan)
    for b,ix in enumerate(idxs):
        for j in range(len(GENES)):
            y=Y[ix,j]
            if 0<y.sum()<N:
                out[b,j]=roc_auc_score(y,p[ix,j])
    return out          # (NB, K)

B = {t: boot_auc(p, None) for t,p in P.items()}
base = "base" if "base" in P else list(P)[0]

print(f"\n=== 逐基因 AUC ± 95%CI ({NB} 次 bootstrap) ===")
print(f"{'variant':<24}" + "".join(f"{g:>20}" for g in GENES))
for t in P:
    row=f"{t:<24}"
    for j in range(len(GENES)):
        v=B[t][:,j]; v=v[~np.isnan(v)]
        row+=f"{np.mean(v):>10.3f}±{1.96*np.std(v):>7.3f}"
    print(row)

print(f"\n=== 配对 Delta vs base ({NB} 次 bootstrap, * = 95%CI 不含 0) ===")
print(f"{'variant':<24}{'Δ mean(all)':>22}{'显著基因'}")
res=[]
for t in P:
    if t==base: continue
    d=B[t]-B[base]                     # (NB,K) 配对
    dm=np.nanmean(d,axis=1)
    lo,hi=np.nanpercentile(dm,[2.5,97.5])
    sig=[]
    for j,g in enumerate(GENES):
        v=d[:,j]; v=v[~np.isnan(v)]
        l,h=np.percentile(v,[2.5,97.5])
        if l*h>0: sig.append(f"{g}({np.mean(v):+.3f})")
    print(f"{t:<24}{np.nanmean(dm):>+10.4f} [{lo:+.3f},{hi:+.3f}]{'*' if lo*hi>0 else ' '}   {', '.join(sig) if sig else '无'}")
    res.append(dict(variant=t, delta_mean=float(np.nanmean(dm)), ci_lo=float(lo), ci_hi=float(hi), sig_genes=";".join(sig)))

# base vs meanpool
if "meanpool_LR" in P:
    d=B[base]-B["meanpool_LR"]; dm=np.nanmean(d,axis=1)
    lo,hi=np.nanpercentile(dm,[2.5,97.5])
    print(f"\n=== base vs mean-pool+LR (核心对照) ===")
    print(f"  Δ mean = {np.nanmean(dm):+.4f}  95%CI [{lo:+.3f}, {hi:+.3f}]  {'显著' if lo*hi>0 else '不显著'}")
    for j,g in enumerate(GENES):
        v=d[:,j]; v=v[~np.isnan(v)]; l,h=np.percentile(v,[2.5,97.5])
        print(f"    {g:<8}{np.mean(v):>+8.4f}  [{l:+.3f}, {h:+.3f}]  {'*' if l*h>0 else ''}")
    res.append(dict(variant="base-vs-meanpool", delta_mean=float(np.nanmean(dm)),
                    ci_lo=float(lo), ci_hi=float(hi),
                    sig_genes=";".join(g for j,g in enumerate(GENES)
                                       if np.percentile(d[:,j][~np.isnan(d[:,j])],[2.5,97.5]).prod()>0)))

pd.DataFrame(res).to_csv(os.path.join(CACHE,"bootstrap_ci.csv"),index=False)
print(f"\n[saved] {CACHE}\\bootstrap_ci.csv")
