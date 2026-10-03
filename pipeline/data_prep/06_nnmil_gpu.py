# -*- coding: utf-8 -*-
"""06_nnmil_gpu.py  -  GPU Gated-Attention MIL, nnMIL 训练配方 + mean-pool 线性基线"""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["EGFR","KRAS","TP53","BRAF","PIK3CA","ROS1"]

ap = argparse.ArgumentParser()
ap.add_argument("--n-patch", type=int, default=2048)
ap.add_argument("--s-inst", type=int, default=512)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--hid", type=int, default=256)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cohort", default="NSCLC")
A = ap.parse_args()
dev = "cuda"

FIX = os.path.join(CACHE, f"NSCLC_sub{A.n_patch}_index_fixed.csv")
idx = pd.read_csv(FIX if os.path.exists(FIX) else os.path.join(CACHE, f"NSCLC_sub{A.n_patch}_index.csv"))
assert "mem_row" in idx.columns, "需要 index_fixed(含正确 mem_row 列)"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
df = idx.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"])
if A.cohort == "LUAD": df = df[df.project=="LUAD"]
df = df.reset_index(drop=True)   # row 来自 index 的 mem_row 列(已修正错位)
N = len(df)
X = np.load(os.path.join(CACHE, f"NSCLC_sub{A.n_patch}.npy"), mmap_mode="r")
ROW = df["mem_row"].values.astype(int); assert len(ROW)==N
Y = df[GENES].values.astype(np.float32)
sub = df.project.values
mL = sub == "LUAD"
strata = np.clip(Y.sum(1),0,3).astype(int)
print(f"[cfg] {A.cohort} inst={A.s_inst} batch={A.batch} ep={A.epochs} lr={A.lr} dev={dev}")
print(f"[data] n={N} (LUAD={mL.sum()} LUSC={(~mL).sum()})  " + " ".join(f"{g}:{int(Y[:,j].sum())}" for j,g in enumerate(GENES)))

def auc(y,p):
    return float(roc_auc_score(y,p)) if 0 < y.sum() < len(y) else float("nan")

# ---------- 基线: mean-pool + LR ----------
Z = np.asarray(X[ROW], dtype=np.float32).mean(1)
bl = np.zeros((N,len(GENES)), np.float32)
for tr,te in StratifiedKFold(A.folds,shuffle=True,random_state=A.seed).split(Z,strata):
    for j in range(len(GENES)):
        y=Y[tr,j]
        if y.sum()<2: continue
        sc=StandardScaler().fit(Z[tr])
        m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Z[tr]),y)
        bl[te,j]=m.predict_proba(sc.transform(Z[te]))[:,1]
print("\n[基线 mean-pool + LR]")
print("  " + "gene".ljust(8) + "all".rjust(8) + "LUAD".rjust(8))
bl_rows=[]
for j,g in enumerate(GENES):
    a=auc(Y[:,j],bl[:,j]); al=auc(Y[mL,j],bl[mL,j])
    bl_rows.append((g,a,al)); print(f"  {g:<8}{a:>8.3f}{al:>8.3f}")
print(f"  MEAN    {np.mean([r[1] for r in bl_rows]):>7.3f}{np.mean([r[2] for r in bl_rows if r[2]==r[2]]):>8.3f}")

# ---------- GAMIL ----------
class GAMIL(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.trunk=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.heads=nn.Linear(h,k)
    def forward(s,x):
        z=s.trunk(x)
        a=s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1)
        a=torch.softmax(a,1).unsqueeze(-1)
        return s.heads((z*a).sum(1))

def sample_inst(rows,s_inst,rng):
    out=np.empty((len(rows),s_inst,1024),dtype=np.float16)
    for i,r in enumerate(rows):
        v=X[ROW[r]]; out[i]=v[rng.choice(v.shape[0],s_inst,replace=v.shape[0]<s_inst)]
    return out

def infer(net,rows,win=512,reps=2):
    net.eval(); ps=[]
    with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
        for _ in range(reps):
            acc=None; nw=0
            for st in range(0,A.n_patch,win):
                b=np.asarray(X[ROW[rows],st:st+win])
                p=net(torch.from_numpy(b).to(dev,torch.float32)).float()
                acc=p if acc is None else acc+p; nw+=1
            ps.append((acc/nw).cpu().numpy())
    return np.mean(ps,0)

t0=time.time(); oof=np.zeros((N,len(GENES)),np.float32)
for f,(tr,te) in enumerate(StratifiedKFold(A.folds,shuffle=True,random_state=A.seed).split(np.zeros(N),strata)):
    pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.0),
                     dtype=torch.float32,device=dev)
    net=GAMIL(1024,A.hid,len(GENES)).to(dev)
    opt=torch.optim.AdamW(net.parameters(),A.lr,weight_decay=1e-3)
    lossf=nn.BCEWithLogitsLoss(reduction="none")
    rng=np.random.RandomState(A.seed+f)
    nstep=max(1,len(tr)//A.batch)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,A.lr,epochs=A.epochs,steps_per_epoch=nstep)
    for ep in range(A.epochs):
        net.train(); perm=rng.permutation(tr); tot=0.0; nb=0
        for i in range(0,len(perm)-A.batch+1,A.batch):
            rows=perm[i:i+A.batch]
            b=torch.from_numpy(sample_inst(rows,A.s_inst,rng)).to(dev,torch.float32)
            p=net(b)
            loss=((lossf(p,torch.from_numpy(Y[rows]).to(dev))*pw).mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            try: sch.step()
            except Exception: pass
            tot+=loss.item(); nb+=1
        if (ep+1)%10==0: print(f"  fold{f} ep{ep+1}/{A.epochs} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)",flush=True)
    oof[te]=infer(net,te)
    print(f"[fold {f+1}/{A.folds}] {time.time()-t0:.0f}s",flush=True)

print("\n[GAMIL nnMIL]")
print("  " + "gene".ljust(8) + "all".rjust(8) + "LUAD".rjust(8) + "LUSC".rjust(8))
rows=[]
for j,g in enumerate(GENES):
    a=auc(Y[:,j],oof[:,j]); al=auc(Y[mL,j],oof[mL,j]); aL=auc(Y[~mL,j],oof[~mL,j])
    rows.append(dict(gene=g,n_pos=int(Y[:,j].sum()),auc_all=round(a,4),
                     auc_LUAD=round(al,4),auc_LUSC=round(aL,4)))
    print(f"  {g:<8}{a:>8.3f}{al:>8.3f}{aL:>8.3f}")
r=pd.DataFrame(rows); r.to_csv(os.path.join(CACHE,"nnmil_results.csv"),index=False)
np.save(os.path.join(CACHE,"nnmil_oof.npy"),oof); df[["case_id","project"]].to_csv(os.path.join(CACHE,"nnmil_cases.csv"),index=False)
bm=np.mean([x[1] for x in bl_rows]); gmn=r.auc_all.mean()
print(f"\n[mean] GAMIL={gmn:.3f}  基线={bm:.3f}  差={gmn-bm:+.3f}")
print(f"[LUAD] GAMIL={r.auc_LUAD.mean():.3f}  基线={np.mean([x[2] for x in bl_rows]):.3f}")
print(f"[wrote] {CACHE}\\nnmil_results.csv   {time.time()-t0:.0f}s")

