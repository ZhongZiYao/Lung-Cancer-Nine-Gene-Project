# -*- coding: utf-8 -*-
"""
03_slide_experiment.py
在统一 cohort 上做 slide 级特征质量验证 + 多 backbone 对比。
  - cohort: manifest 中有标签且属于 LUAD/LUSC 的 case
  - 特征: TITAN 768 维 / Prov-GigaPath 768 维 (取 layer 0)
  - 模型: LogisticRegression + 2 层 MLP, Stratified 5-fold, OOF AUC
  - 只跑阳性数 >= MIN_POS 的基因
输出: data/labels/slide_level_results.csv
"""
import os, re, pickle, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
GENES = ["EGFR","KRAS","ALK","ROS1","TP53","BRAF","PIK3CA","ERBB2","NRAS","RET","MET"]
MIN_POS = 30
tnorm = lambda s: (lambda m: f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(
    re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})", str(s)))

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", default="NSCLC", choices=["LUAD","LUSC","NSCLC"])
ap.add_argument("--min-pos", type=int, default=MIN_POS)
A = ap.parse_args()

man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
sel = {"LUAD": man.project=="LUAD", "LUSC": man.project=="LUSC",
       "NSCLC": man.project.isin(["LUAD","LUSC"])}[A.cohort]
coh = man[sel & (man.has_label==1)].copy()
coh = coh.dropna(subset=["TP53"])
print(f"[cohort] {A.cohort}  n={len(coh)}  (LUAD={int((coh.project=='LUAD').sum())}  LUSC={int((coh.project=='LUSC').sum())})")

# ---------- 特征装载 ----------
feats = {}
tk = pickle.load(open(os.path.join(R,"models","TITAN","TCGA_TITAN_features.pkl"),"rb"))
idx = {c:i for i,c in enumerate(map(tnorm, tk["filenames"]))}
E = np.asarray(tk["embeddings"]); ids = list(idx)
feats["TITAN_768"] = (ids, E)

gp = pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"] = gp["_PATIENT"].map(tnorm)
ids2, mats2 = [], []
for c, emb in zip(gp["case"], gp["embedding"]):
    if c and c in set(coh.case_id):
        a = np.asarray(emb)
        if a.dtype == object: a = np.stack(a.tolist())
        if a.ndim == 2: a = a[0]
        if a.shape[0] != 768: continue
        ids2.append(c); mats2.append(a.astype(np.float32))
feats["GigaPath_L0_768"] = (ids2, np.stack(mats2))
print(f"[feat] TITAN_768 = {E.shape}   GigaPath_L0_768 = {len(ids2)} slides")

# ---------- 模型 ----------
class MLP(nn.Module):
    def __init__(s,d,k):
        super().__init__(); s.f=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.3)); s.h=nn.Linear(128,k)
    def forward(s,x): return s.h(s.f(x))

def run(X, y, seed=0, epochs=120):
    oof = np.zeros(len(y))
    for tr,te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X,y):
        sc=StandardScaler().fit(X[tr]); xt,xe=sc.transform(X[tr]),sc.transform(X[te])
        w=torch.tensor(np.where(y[tr]==1,(len(y)-y[tr].sum())/max(y[tr].sum(),1),1.),dtype=torch.float32)
        net=MLP(X.shape[1],1); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
        lf=nn.BCEWithLogitsLoss(pos_weight=w)
        a=torch.tensor(xt,dtype=torch.float32); b=torch.tensor(y[tr],dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad(); l=lf(net(a).squeeze(-1),b); l.backward(); opt.step()
        with torch.no_grad(): oof[te]=net(torch.tensor(xe,dtype=torch.float32)).squeeze(-1).numpy()
    return roc_auc_score(y,oof)

def run_lr(X,y):
    oof=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        sc=StandardScaler().fit(X[tr])
        m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr])
        oof[te]=m.predict_proba(sc.transform(X[te]))[:,1]
    return roc_auc_score(y,oof)

genes_ok = [g for g in GENES if int(coh[g].sum())>=A.min_pos]
print(f"[genes] 阳性数>={A.min_pos}: {genes_ok}")
print(f"        剔除: {[g for g in GENES if g not in genes_ok]}")

rows=[]
for fname,(ids,M) in feats.items():
    lut = {c:i for i,c in enumerate(ids)}
    common = [c for c in coh.case_id if c in lut]
    X = M[[lut[c] for c in common]]
    cl = coh.set_index("case_id").loc[common]
    print(f"\n=== {fname}  n={len(common)}  dim={X.shape[1]} ===")
    print(f"  {'gene':<8}{'n_pos':>6}{'LR_AUC':>9}{'MLP_AUC':>10}{'delta':>8}")
    for g in genes_ok:
        y = cl[g].values.astype(int)
        a1 = run_lr(X,y); a2 = run(X,y)
        print(f"  {g:<8}{int(y.sum()):>6}{a1:>9.3f}{a2:>10.3f}{a2-a1:>+8.3f}")
        rows.append(dict(cohort=A.cohort, feature=fname, n=len(common), gene=g,
                         n_pos=int(y.sum()), auc_lr=round(a1,4), auc_mlp=round(a2,4)))
df=pd.DataFrame(rows)
out=os.path.join(R,"data","labels","slide_level_results.csv")
df.to_csv(out,index=False)
print(f"\n[wrote] {out}")
print("\n=== mean AUC by feature ===")
print(df.groupby(["feature","gene"]).auc_mlp.mean().unstack(0).round(3).to_string())

