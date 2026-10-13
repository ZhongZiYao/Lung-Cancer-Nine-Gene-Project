# -*- coding: utf-8 -*-
"""23_seeds_aggregator.py — gated vs proto_loo 各 4 个种子的直接对照
gated 的跨种子基准(5 seed): mean 0.7392  std 0.0040  极差 0.0110
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
DIM, EP, S_INST, BATCH, LR, HID, FOLDS = 1536, 25, 512, 32, 3e-4, 256, 5
LAM = 0.5; dev="cuda"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(CACHE,"NSCLC_gigap_sub512_index.csv"))
df  = idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
df  = df[df.project=="LUAD"].reset_index(drop=True)
N=len(df); Y=df[GENES].values.astype(int); RT=df.mem_row.values.astype(int)
X=np.load(os.path.join(CACHE,"NSCLC_gigap_sub512.npy"))
strata=np.clip(Y.sum(1),0,3).astype(int); K=len(GENES)
CM=np.stack([Y[:,k] for k in range(K)],1).astype(np.float32)
auc=lambda y,p: float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan

class Net(nn.Module):
    def __init__(s,d,h,k,proto):
        super().__init__(); s.proto_mode=proto
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.od=nn.Linear(h,k)
        if proto:
            s.p=nn.Parameter(torch.randn(k,2,h)*0.1); s.ls=nn.Parameter(torch.full((k,),10.0))
    def forward(s,x):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        bag=(z*a.unsqueeze(-1)).sum(1)
        if not s.proto_mode: return s.od(bag), None
        zn=F.normalize(z,dim=-1); pn=F.normalize(s.p,dim=-1)
        sim=torch.einsum("bsh,gch->bsgc",zn,pn)
        sc4=s.ls.clamp(min=0.1).view(1,1,K,1); sc3=s.ls.clamp(min=0.1).view(1,1,K)
        il=(sc3*(sim[...,0]-sim[...,1])).permute(2,0,1)   # (K,B,S)
        return s.od(bag), il

def run(proto, seed):
    oof=np.zeros((N,K),np.float32)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=seed).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=Net(DIM,HID,K,proto).to(dev)
        opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EP,steps_per_epoch=max(1,len(tr)//BATCH))
        lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(seed*100+f)
        for ep in range(EP):
            net.train(); perm=rng.permutation(tr)
            for i in range(0,len(perm)-BATCH+1,BATCH):
                rr=perm[i:i+BATCH]
                bt=np.empty((len(rr),S_INST,DIM),np.float16)
                for k2,rw in enumerate(rr):
                    v=X[RT[rw]]; bt[k2]=v[rng.choice(v.shape[0],S_INST,replace=v.shape[0]<S_INST)]
                bt=torch.from_numpy(bt).to(dev,torch.float32)
                yb=torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                p,il=net(bt)
                loss=lf(p,yb).mul(pw).mean()
                if proto:
                    yb6=yb.t().unsqueeze(-1).expand(-1,-1,S_INST)
                    loss=loss+0.3*lf(il,yb6).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            b=torch.from_numpy(X[np.ix_(RT[te],range(0,S_INST))]).to(dev,torch.float32)
            oof[te]=net(b)[0].float().cpu().numpy()
    return [auc(Y[:,j],oof[:,j]) for j in range(K)]

res={}
for name,proto in [("gated",False),("proto_loo",True)]:
    ms=[]
    for sd in [0,1,2,3]:
        a=run(proto,sd); m=float(np.nanmean(a)); ms.append(m)
        print(f"  [{name}] seed={sd}  mean AUC={m:.4f}   " + " ".join(f"{g[:4]}:{v:.3f}" for g,v in zip(GENES,a)), flush=True)
    res[name]=ms
    print(f"[{name}] mean={np.mean(ms):.4f}  std={np.std(ms):.4f}")
d=np.array(res["proto_loo"])-np.array(res["gated"])
print(f"\n配对差(proto_loo - gated) = {d.mean():+.4f}  per-seed {np.round(d,4)}")
print("结论:", "有效" if d.mean()>2*np.std(res['gated']) else "疑似噪声")
