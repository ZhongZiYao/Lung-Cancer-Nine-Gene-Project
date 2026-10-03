# -*- coding: utf-8 -*-
"""04_ab_feature_source.py  同一 cohort 下对比三种 slide 特征来源,排除混淆"""
import os, re, glob, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
tnorm = lambda s: (lambda m: f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})", str(s)))
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
coh = man[man.project.isin(["LUAD","LUSC"]) & (man.has_label==1)].dropna(subset=["TP53"])
coh = coh.set_index("case_id")
GENES = [g for g in ["EGFR","KRAS","TP53","BRAF","PIK3CA","ROS1","ALK"] if coh[g].sum()>=30]
print(f"[cohort] NSCLC n={len(coh)}  genes={GENES}")

def lr_auc(X,y):
    oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        sc=StandardScaler().fit(X[tr])
        m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr])
        oof[te]=m.predict_proba(sc.transform(X[te]))[:,1]
    return roc_auc_score(y,oof)

# A) 官方 TITAN pkl
tk=pickle.load(open(os.path.join(R,"models","TITAN","TCGA_TITAN_features.pkl"),"rb"))
lut={}; 
for i,f in enumerate(tk["filenames"]):
    c=tnorm(f)
    if c: lut.setdefault(c,i)
A_ids=[c for c in coh.index if c in lut]; A_X=np.asarray(tk["embeddings"])[[lut[c] for c in A_ids]]
# B) 本地算的 TITAN (deepgem_feat_v3_titan)
B={}
for p in glob.glob(os.path.join(R,"deepgem_feat_v3_titan","*.npz")):
    z=np.load(p,allow_pickle=True); v=np.asarray(z["features"]).reshape(-1,768)
    if v.shape[1]!=768: continue
    B["-".join(os.path.splitext(os.path.basename(p))[0].split("-")[:3])] = v.mean(0)
B_ids=[c for c in coh.index if c in B]; B_X=np.stack([B[c] for c in B_ids])
# C) GigaPath layer 0
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tnorm); C={}
for c,emb in zip(gp["case"],gp["embedding"]):
    if not c or c in C: continue
    a=np.asarray(emb)
    if a.dtype==object: a=np.stack(a.tolist())
    if a.ndim==2: a=a[0]
    if a.shape[0]==768: C[c]=a.astype(np.float32)
C_ids=[c for c in coh.index if c in C]; C_X=np.stack([C[c] for c in C_ids])

print(f"\n{'source':<26}{'n':>6}", "".join(f"{g:>9}" for g in GENES), f"{'mean':>8}")
res={}
for nm, ids, X in [("官方 TITAN pkl", A_ids, A_X), ("本地 TITAN(v3_titan)", B_ids, B_X), ("GigaPath layer0", C_ids, C_X)]:
    cl=coh.loc[ids]; aucs=[]
    for g in GENES:
        a=lr_auc(X, cl[g].values.astype(int)); aucs.append(a)
    res[nm]=aucs
    print(f"{nm:<26}{len(ids):>6}", "".join(f"{a:>9.3f}" for a in aucs), f"{np.mean(aucs):>8.3f}")

# 交集检验: 三个源都有的 case 上再比一次
com = sorted(set(A_ids) & set(B_ids) & set(C_ids))
print(f"\n=== 三源交集 n={len(com)} 上复测 ===")
print(f"{'source':<26}{'n':>6}", "".join(f"{g:>9}" for g in GENES), f"{'mean':>8}")
for nm, D in [("官方 TITAN pkl", A), ("本地 TITAN(v3_titan)", B), ("GigaPath layer0", C)]:
    X=np.stack([D[c] for c in com]); cl=coh.loc[com]
    aucs=[lr_auc(X, cl[g].values.astype(int)) for g in GENES]
    print(f"{nm:<26}{len(com):>6}", "".join(f"{a:>9.3f}" for a in aucs), f"{np.mean(aucs):>8.3f}")


