# -*- coding: utf-8 -*-
"""
07_nnmil_v2.py  —  nnMIL + 三个可插拔创新模块
  --mixup   : bag 级 Mixup (标签按有效 patch 数加权)
  --dsmil   : DSMIL 双流 (attention 流 / max 流), 流间差异逐基因惩罚
  --plscale : per-gene prototype scale + 退火 (标签消歧)
用法示例:
  python 07_nnmil_v2.py --epochs 60
  python 07_nnmil_v2.py --epochs 60 --mixup
  python 07_nnmil_v2.py --epochs 60 --mixup --dsmil
  python 07_nnmil_v2.py --epochs 60 --mixup --dsmil --plscale
"""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]   # LUAD 侧 n>=50, 且可对标 GAMIL/DeepGEM/Patho-Bench

ap = argparse.ArgumentParser()
ap.add_argument("--n-patch", type=int, default=2048)
ap.add_argument("--s-inst",   type=int, default=512)
ap.add_argument("--batch",    type=int, default=32)
ap.add_argument("--epochs",   type=int, default=60)
ap.add_argument("--lr",       type=float, default=3e-4)
ap.add_argument("--hid",      type=int, default=256)
ap.add_argument("--folds",    type=int, default=5)
ap.add_argument("--seed",     type=int, default=0)
ap.add_argument("--cohort",   default="NSCLC", choices=["NSCLC","LUAD","LUSC"])
ap.add_argument("--min-pos",   type=int, default=30, help="LUAD 侧阳性数低于此值的基因自动剔除")
ap.add_argument("--mixup",    action="store_true")
ap.add_argument("--dsmil",    action="store_true")
ap.add_argument("--plscale",  action="store_true")
ap.add_argument("--alpha",    type=float, default=0.3,  help="instance/prototype loss 权重")
ap.add_argument("--beta",     type=float, default=0.15, help="DSMIL 流间差异权重")
ap.add_argument("--mixup-alpha", type=float, default=0.4)
ap.add_argument("--topk",     type=int, default=32,   help="max 流取 top-k instance")
A = ap.parse_args()
dev = "cuda"
TAG = ("base" if not (A.mixup or A.dsmil or A.plscale) else
       "+".join([n for n,f in [("mixup",A.mixup),("dsmil",A.dsmil),("plscale",A.plscale)] if f]))

FIX = os.path.join(CACHE, f"NSCLC_sub{A.n_patch}_index_fixed.csv")
idx = pd.read_csv(FIX)
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
df = idx.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"])
if A.cohort == "LUAD": df = df[df.project=="LUAD"]
df = df.reset_index(drop=True)
N = len(df); X = np.load(os.path.join(CACHE, f"NSCLC_sub{A.n_patch}.npy"))  # RAM
ROW = df["mem_row"].values.astype(int)
Y = df[GENES].values.astype(np.float32); sub = df.project.values; mL = sub=="LUAD"
strata = np.clip(Y.sum(1),0,3).astype(int)
NPOS = Y.sum(0)
GENES = [g for g in GENES if NPOS[GENES.index(g)] >= A.min_pos]
K = len(GENES)
print(f"[genes] {GENES}")
print(f"[{TAG}] cohort={A.cohort} n={N} (LUAD={mL.sum()}) inst={A.s_inst} ep={A.epochs} lr={A.lr}")

def auc(y,p): return float(roc_auc_score(y,p)) if 0 < y.sum() < len(y) else float("nan")

# ---------- 基线 (只算一次, 缓存到 csv) ----------
sig = "-".join(GENES)
BL = os.path.join(CACHE, f"baseline_meanpool_lr_{sig}.csv")
if os.path.exists(BL):
    bl = pd.read_csv(BL).set_index("gene").loc[GENES]
    blm = bl.auc_all.mean()
else:
    Z = np.asarray(X[ROW], np.float32).mean(1); oof=np.zeros((N,K),np.float32)
    for tr,te in StratifiedKFold(A.folds,shuffle=True,random_state=A.seed).split(Z,strata):
        for j in range(K):
            y=Y[tr,j]
            if y.sum()<2: continue
            sc=StandardScaler().fit(Z[tr])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Z[tr]),y)
            oof[te,j]=m.predict_proba(sc.transform(Z[te]))[:,1]
    pd.DataFrame([dict(gene=GENES[j],auc_all=auc(Y[:,j],oof[:,j]),auc_LUAD=auc(Y[mL,j],oof[mL,j])) for j in range(K)]).to_csv(BL,index=False)
    bl = pd.read_csv(BL).set_index("gene").loc[GENES]; blm = bl.auc_all.mean()
print(f"[基线 mean-pool+LR] all={blm:.3f}  LUAD={bl.auc_LUAD.mean():.3f}")

# ---------- 模型 ----------
class V2(nn.Module):
    def __init__(s,d,h,k,use_max,use_proto):
        super().__init__()
        s.trunk=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.heads=nn.Linear(h,k)
        s.use_max=use_max
        if use_proto:
            s.proto=nn.Parameter(torch.randn(k,2,h)*0.1)
            s.logit_scale=nn.Parameter(torch.full((k,), 10.0))   # per-gene scale (可学)
    def attn(s,z):
        a=s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1)
        return torch.softmax(a,1)
    def forward(s,x,alpha=1.0):
        z=s.trunk(x)                       # (B,S,H)
        a=s.attn(z)
        bag=(z*a.unsqueeze(-1)).sum(1)     # (B,H)  clean 流
        out=s.heads(bag)
        aux={}
        if s.use_max:
            # DSMIL noisy 流: 取注意力 top-k instance, 用其权重(在 topk 内重归一化)加权求和
            topk=max(1,min(A.topk,z.shape[1]))
            idx=a.topk(topk,dim=1).indices                 # (B,k)
            mz=torch.gather(z,1,idx.unsqueeze(-1).expand(-1,-1,z.shape[-1]))
            ma=torch.gather(a,1,idx); ma=ma/ma.sum(1,keepdim=True).clamp_min(1e-6)
            out2=s.heads((mz*ma.unsqueeze(-1)).sum(1))
            out=torch.cat([out,out2],-1)  # (B,2K)
        if hasattr(s,"proto"):
            pzn=torch.nn.functional.normalize(z,dim=-1)
            ptn=torch.nn.functional.normalize(s.proto,dim=-1)
            sim=torch.einsum("bsh,gch->bsgc",pzn,ptn)
            sca=s.logit_scale.clamp(min=0.1).view(1,1,K,1)
            pr=torch.softmax(sca*sim,dim=-1)[...,1]      # (B,S,K) patch 软标签
            inst=(pr*a.unsqueeze(-1)).sum(1)              # (B,K) instance 级预测
            aux["inst"]=inst
        return out, aux


def sample_inst(rows, s, rng, lam_pairs=None):
    out=np.empty((len(rows),s,1024),dtype=np.float16)
    for i,r in enumerate(rows):
        v=X[ROW[r]]; out[i]=v[rng.choice(v.shape[0],s,replace=v.shape[0]<s)]
    return out

def infer(net,rows,reps=2,win=512):
    net.eval(); ps=[]
    with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
        for _ in range(reps):
            acc=None; nw=0
            for st in range(0,A.n_patch,win):
                b=np.asarray(X[ROW[rows],st:st+win])
                o,_=net(torch.from_numpy(b).to(dev,torch.float32))
                o=o.float(); acc=o if acc is None else acc+o; nw+=1
            ps.append((acc/nw).cpu().numpy())
    return np.mean(ps,0)

t0=time.time(); oof=np.zeros((N,K*(2 if A.dsmil else 1)),np.float32)
for f,(tr,te) in enumerate(StratifiedKFold(A.folds,shuffle=True,random_state=A.seed).split(np.zeros(N),strata)):
    pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.0),dtype=torch.float32,device=dev)
    net=V2(1024,A.hid,K,A.dsmil,A.plscale).to(dev)
    opt=torch.optim.AdamW(net.parameters(),A.lr,weight_decay=1e-3)
    lossf=nn.BCEWithLogitsLoss(reduction="none")
    rng=np.random.RandomState(A.seed+f)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,A.lr,epochs=A.epochs,steps_per_epoch=max(1,len(tr)//A.batch))
    for ep in range(A.epochs):
        net.train(); perm=rng.permutation(tr); tot=0.;nb=0
        # prototype 权重退火: 早期 bag 主导, 后期收紧
        aw = A.alpha if A.plscale else 0.0
        if A.plscale: aw = A.alpha * (0.3 + 0.7*ep/max(1,A.epochs-1))
        for i in range(0,len(perm)-A.batch+1,A.batch):
            rows=perm[i:i+A.batch]
            b=torch.from_numpy(sample_inst(rows,A.s_inst,rng)).to(dev,torch.float32)
            yb=torch.from_numpy(Y[rows]).to(dev)
            if A.mixup and len(rows)>1:
                perm2=torch.randperm(len(rows),device=dev)
                lam=float(np.random.beta(A.mixup_alpha,A.mixup_alpha))
                b=lam*b+(1-lam)*b[perm2]; yb=lam*yb+(1-lam)*yb[perm2]
            o,aux=net(b)
            o1=o[:,:K]
            loss=(lossf(o1,yb)*pw).mean()
            if A.dsmil:
                o2=o[:,K:]
                loss=loss+(lossf(o2,yb)*pw).mean()*0.5 + A.beta*((torch.sigmoid(o1)-torch.sigmoid(o2))**2).mean(0).sum()
            if A.plscale and "inst" in aux:
                loss=loss+aw*(lossf(aux["inst"],yb)*pw).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            try: sch.step()
            except Exception: pass
            tot+=loss.item(); nb+=1
        if (ep+1)%20==0: print(f"  fold{f} ep{ep+1}/{A.epochs} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)",flush=True)
    oof[te]=infer(net,te)
    print(f"[fold {f+1}/{A.folds}] {time.time()-t0:.0f}s",flush=True)

rows=[]
for j,g in enumerate(GENES):
    a=auc(Y[:,j],oof[:,j]); al=auc(Y[mL,j],oof[mL,j]); aL=auc(Y[~mL,j],oof[~mL,j])
    rows.append(dict(tag=TAG,gene=g,n_pos=int(Y[:,j].sum()),auc_all=round(a,4),auc_LUAD=round(al,4),auc_LUSC=round(aL,4)))
r=pd.DataFrame(rows)
print(f"\n[{TAG}] " + "  ".join(f"{q.gene}:{q.auc_all:.3f}" for q in r.itertuples()))
print(f"[{TAG}] mean_all={r.auc_all.mean():.3f}  mean_LUAD={r.auc_LUAD.mean():.3f}  (基线 {blm:.3f}/{bl.auc_LUAD.mean():.3f})")
out=os.path.join(CACHE, f"ablation_{A.cohort}.csv")
r.to_csv(out,index=False,mode="a" if os.path.exists(out) else "w",header=not os.path.exists(out))
np.save(os.path.join(CACHE,f"oof_{TAG}_{A.cohort}.npy"),oof)
df[["case_id","project"]+GENES].to_csv(os.path.join(CACHE,f"cases_{A.cohort}.csv"),index=False)
print(f"[saved] {out}   {time.time()-t0:.0f}s")






