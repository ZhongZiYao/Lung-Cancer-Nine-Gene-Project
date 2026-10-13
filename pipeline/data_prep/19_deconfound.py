# -*- coding: utf-8 -*-
"""
19_deconfound.py  —  去混淆实验 (Dawood et al. Nat Biomed Eng 2026 议程第 3/5 条)

核心假设: LUAD 中 KRAS / NF1 / EGFR 的【负净信号】不是"没有形态信号",
          而是被基因间共突变混淆压制。

三个检验:
  T1 条件 AUC   : 固定主导混淆因子后, 图像 vs 基因组的净信号是否转正
  T2 重要性重加权: 让混淆因子在目标基因正/负组内分布一致后重训, 看净信号
  T3 去相关重采样: 平衡共突变模式后重训, 看净信号
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
dev="cuda"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(CACHE,"NSCLC_gigap_sub512_index.csv"))
df  = idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
df  = df[df.project=="LUAD"].reset_index(drop=True)
N=len(df); Y=df[GENES].values.astype(int); RT=df.mem_row.values.astype(int)
X=np.load(os.path.join(CACHE,"NSCLC_gigap_sub512.npy"))
strata=np.clip(Y.sum(1),0,3).astype(int)
print(f"[LUAD] n={N}  encoder=gigap  cache=512\n")

# ---- 临床变量(基因组基线用) ----
tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_?([0-9]{4})",str(s))) if False else (lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
STAGE=oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
AGE=np.nan_to_num(df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1),nan=0.0)
SUBT=(df.project.values=="LUSC").astype(float).reshape(-1,1)

# ---- T0: 共突变 phi 矩阵, 找主导混淆因子 ----
def phi(a,b):
    a=a.astype(bool); b=b.astype(bool)
    if a.sum()==0 or b.sum()==0: return 0.0
    n11=(a&b).sum(); n10=(a&~b).sum(); n01=(~a&b).sum(); n00=(~a&~b).sum()
    d=np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return float((n11*n00-n10*n01)/d) if d>0 else 0.0
print("=== 共突变 phi 相关矩阵 (LUAD) ===")
print("       " + "".join(f"{g:>8}" for g in GENES))
conf={}
for i,gi in enumerate(GENES):
    row=[phi(Y[:,i],Y[:,j]) for j in range(len(GENES))]
    conf[gi]=[(abs(v),GENES[j],v) for j,v in enumerate(row) if j!=i and Y[:,j].sum()>5]
    conf[gi].sort(reverse=True)
    print(f"{gi:<7}" + "".join(f"{v:>8.2f}" for v in row))
print()

class Net(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.o=nn.Linear(h,k)
    def forward(s,x):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        return s.o((z*a.unsqueeze(-1)).sum(1))

def geno_oof():
    o=np.zeros((N,len(GENES)),np.float32)
    for j in range(len(GENES)):
        Xf=np.column_stack([STAGE,AGE,SUBT]+[Y[:,k] for k in range(len(GENES)) if k!=j])
        if not (0<Y[:,j].sum()<N): continue
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,strata):
            if len(set(Y[tr,j]))<2: continue
            sc=StandardScaler().fit(Xf[tr])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),Y[tr,j])
            o[te,j]=m.predict_proba(sc.transform(Xf[te]))[:,1]
    return o

def train_image(sample_w=None):
    oof=np.zeros((N,len(GENES)),np.float32)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=Net(DIM,HID,len(GENES)).to(dev)
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
                p=net(bt); l=lf(p,yb)
                if sample_w is not None:
                    w=torch.from_numpy(sample_w[rr].astype(np.float32)).to(dev)
                    loss=(l*w.unsqueeze(1)).mean(0).mul(pw).mean()
                else:
                    loss=(l*pw).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            b=torch.from_numpy(X[np.ix_(RT[te],range(0,S_INST))]).to(dev,torch.float32)
            oof[te]=net(b).float().cpu().numpy()
    return oof

auc=lambda y,p: float(roc_auc_score(y,p)) if 0<p.sum()<len(p) and 0<(1-p).sum() else np.nan
gA=geno_oof()
iA=train_image()
print("=== 基线 (整体) ===")
print(f"{'gene':<8}{'基因组':>9}{'图像':>9}{'净信号':>10}{'主导混淆':>10}{'|phi|':>8}")
for j,g in enumerate(GENES):
    a=auc(Y[:,j],gA[:,j]); b=auc(Y[:,j],iA[:,j])
    top=conf[g][0] if conf[g] else ("-",0)
    print(f"{g:<8}{a:>9.3f}{b:>9.3f}{b-a:>+10.3f}{top[0]:>10}{top[2]:>+8.2f}")

print("\n=== T1 条件 AUC: 固定主导混淆因子后重算净信号 ===")
print(f"{'gene':<8}{'混淆':<8}{'层':<8}{'n':>5}{'基因组':>9}{'图像':>9}{'条件净信号':>11}")
cond_res={}
for j,g in enumerate(GENES):
    if not conf[g]: continue
    _,c,_=conf[g][0]; k=GENES.index(c); cv=Y[:,k]
    for lev in (0,1):
        m=(cv==lev) if lev else (cv==0)
        if m.sum()<30 or not (0<Y[m,j].sum()<m.sum()): continue
        a=auc(Y[m,j],gA[m,j]); b=auc(Y[m,j],iA[m,j])
        cond_res.setdefault(g,[]).append((c,lev,a,b))
        print(f"{g:<8}{c:<8}{'c='+str(lev):<8}{int(m.sum()):>5}{a:>9.3f}{b:>9.3f}{b-a:>+11.3f}")
np.save(os.path.join(CACHE,"oof_gigap512_LUAD.npy"), iA)
np.save(os.path.join(CACHE,"oof_genomic_LUAD.npy"), gA)
print("\nDONE")
