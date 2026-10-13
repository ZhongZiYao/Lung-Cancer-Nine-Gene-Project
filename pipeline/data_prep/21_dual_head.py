# -*- coding: utf-8 -*-
"""21_dual_head.py — 共突变感知双头结构(修正: 两条分支都接 loss)
联合头 oc([bag, 共突变向量])  = 形态+基因组的全部可预测性
纯形态头 o(bag)              = 纯形态可预测性
两者在【单次前向传播内】分离, 无需事后校正
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
DIM, EP, S_INST, BATCH, LR, HID, FOLDS, SEED = 1536, 25, 512, 32, 3e-4, 256, 5, 0
LAM = float(sys.argv[1]) if len(sys.argv)>1 else 0.5
dev="cuda"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(CACHE,"NSCLC_gigap_sub512_index.csv"))
df  = idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
df  = df[df.project=="LUAD"].reset_index(drop=True)
N=len(df); Y=df[GENES].values.astype(int); RT=df.mem_row.values.astype(int)
X=np.load(os.path.join(CACHE,"NSCLC_gigap_sub512.npy"))
strata=np.clip(Y.sum(1),0,3).astype(int); K=len(GENES)
COMUT=np.stack([Y[:,k] for k in range(K)],1).astype(np.float32)

tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
STAGE=oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
AGE=np.nan_to_num(df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1),nan=0.0)

def geno_oof():
    o=np.zeros((N,K),np.float32)
    for j in range(K):
        Xf=np.column_stack([STAGE,AGE]+[COMUT[:,k] for k in range(K) if k!=j])
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,strata):
            if len(set(Y[tr,j]))<2: continue
            sc=StandardScaler().fit(Xf[tr])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),Y[tr,j])
            o[te,j]=m.predict_proba(sc.transform(Xf[te]))[:,1]
    return o

class DualHead(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.o_pure=nn.Linear(h,k)      # 纯形态
        s.o_joint=nn.Linear(h+k,k)   # 形态+共突变
    def forward(s,x,cm):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        bag=(z*a.unsqueeze(-1)).sum(1)
        return s.o_pure(bag), s.o_joint(torch.cat([bag,cm],1))

def train():
    op=np.zeros((N,K),np.float32); oj=np.zeros((N,K),np.float32)
    cm_t=torch.from_numpy(COMUT).to(dev)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=DualHead(DIM,HID,K).to(dev)
        opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EP,steps_per_epoch=max(1,len(tr)//BATCH))
        lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(SEED+f)
        for ep in range(EP):
            net.train(); perm=rng.permutation(tr)
            for i in range(0,len(perm)-BATCH+1,BATCH):
                rr=perm[i:i+BATCH]
                bt=np.empty((len(rr),S_INST,DIM),np.float16)
                for k2,rw in enumerate(rr):
                    v=X[RT[rw]]; bt[k2]=v[rng.choice(v.shape[0],S_INST,replace=v.shape[0]<S_INST)]
                bt=torch.from_numpy(bt).to(dev,torch.float32)
                yb=torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                p,jo=net(bt, cm_t[rr])
                # 两条分支都接 loss: 联合头学总量, 纯形态头被迫只用图像
                loss=(lf(jo,yb)*pw).mean() + LAM*(lf(p,yb)*pw).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            b=torch.from_numpy(X[np.ix_(RT[te],range(0,S_INST))]).to(dev,torch.float32)
            p,jo=net(b, cm_t[te])
            op[te]=p.float().cpu().numpy(); oj[te]=jo.float().cpu().numpy()
    return op,oj

auc=lambda y,p: float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan
gA=geno_oof()
op,oj=train()
print(f"\n[LUAD n={N}] 共突变感知双头 (lambda={LAM})")
print(f"{'gene':<8}{'基因组基线':>10}{'纯形态头':>10}{'形态净信号':>11}{'联合头':>9}{'联合净信号':>11}")
mp=[];mj=[];mn=[]
for j,g in enumerate(GENES):
    ga=auc(Y[:,j],gA[:,j]); p=auc(Y[:,j],op[:,j]); jo=auc(Y[:,j],oj[:,j])
    mp.append(p); mj.append(jo); mn.append(p-ga)
    print(f"{g:<8}{ga:>10.3f}{p:>10.3f}{p-ga:>+11.3f}{jo:>9.3f}{jo-ga:>+11.3f}")
print(f"{'mean':<8}{np.nanmean([auc(Y[:,j],gA[:,j]) for j in range(K)]):>10.3f}{np.nanmean(mp):>10.3f}{np.nanmean(mn):>+11.3f}{np.nanmean(mj):>9.3f}{np.nanmean([x-auc(Y[:,j],gA[:,j]) for j,x in enumerate(mj)]):>+11.3f}")
np.save(os.path.join(CACHE,"dual_pure_LUAD.npy"),op); np.save(os.path.join(CACHE,"dual_joint_LUAD.npy"),oj)
print("\nDONE")
