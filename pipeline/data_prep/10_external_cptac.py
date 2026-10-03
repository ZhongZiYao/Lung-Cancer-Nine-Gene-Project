# -*- coding: utf-8 -*-
"""
10_external_cptac.py  —  外部验证: TCGA-NSCLC 训练 -> CPTAC-NSCLC(115例) 零样本推理
外验集构建链: 本地 MAF file_id -> manifest.case_id -> submitter_id(C3L) -> cptac_nsclc patch 特征
"""
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
EXT   = os.path.join(R,"data","public_features","CPathPatchFeature","cptac_nsclc","uni","pt_files")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--s-inst", type=int, default=512)
ap.add_argument("--batch",  type=int, default=32)
ap.add_argument("--lr",     type=float, default=3e-4)
ap.add_argument("--hid",    type=int, default=256)
ap.add_argument("--folds",  type=int, default=5)
ap.add_argument("--seed",   type=int, default=0)
ap.add_argument("--mixup",  action="store_true")
ap.add_argument("--dsmil",  action="store_true")
ap.add_argument("--plscale",action="store_true")
A = ap.parse_args(); dev="cuda"
TAG = "base" if not (A.mixup or A.dsmil or A.plscale) else "+".join(n for n,f in [("mixup",A.mixup),("dsmil",A.dsmil),("plscale",A.plscale)] if f)

# ---------- 训练集 ----------
ti = pd.read_csv(os.path.join(CACHE,"NSCLC_sub2048_index_fixed.csv"))
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
tr = ti.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"]).reset_index(drop=True)
tr = tr[tr.project.isin(["LUAD","LUSC"])]
XT = np.load(os.path.join(CACHE,"NSCLC_sub2048.npy"))   # 全量载入 RAM, 消除随机读 IO 瓶颈
RT = tr.mem_row.values.astype(int); YT = tr[GENES].values.astype(np.float32)
st = np.clip(YT.sum(1),0,3).astype(int)

# ---------- 外验集 ----------
lab = pd.read_csv(os.path.join(R,"data","labels","cptac_gene_labels.csv"))
lab = lab[lab.submitter_id.str.startswith("C3L")].copy()
files = {}
for p in glob.glob(os.path.join(EXT,"*.pt")):
    b = os.path.basename(p).split("-"); files[f"{b[0]}-{b[1]}"] = p
ev_ids = [s for s in lab.submitter_id if s in files]
def pick(s, n, seed):
    t = torch.load(files[s], map_location="cpu", weights_only=False).numpy()
    r = np.random.RandomState(seed)
    return t[r.choice(t.shape[0], n, replace=t.shape[0]<n)]
EX = np.stack([pick(s, A.s_inst, 0) for s in ev_ids])
EY = lab.set_index("submitter_id").loc[ev_ids, GENES].values.astype(np.float32)
print(f"[外验集] CPTAC-NSCLC {len(ev_ids)} 例  patch={EX.shape[1:]}  " + " ".join(f"{g}:{int(EY[:,j].sum())}" for j,g in enumerate(GENES)))
print(f"[训练集] TCGA-NSCLC {len(tr)} 例")

# ---------- 模型 (与 07 同构) ----------
class V2(nn.Module):
    def __init__(s,d,h,k,use_max,use_proto):
        super().__init__()
        s.trunk=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.heads=nn.Linear(h,k); s.use_max=use_max
        if use_proto:
            s.proto=nn.Parameter(torch.randn(k,2,h)*0.1); s.logit_scale=nn.Parameter(torch.full((k,),10.0))
    def forward(s,x):
        z=s.trunk(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        out=s.heads((z*a.unsqueeze(-1)).sum(1))
        if s.use_max:
            k=min(32,x.shape[1]); idx=a.topk(k,1).indices
            mz=torch.gather(z,1,idx.unsqueeze(-1).expand(-1,-1,z.shape[-1]))
            ma=torch.gather(a,1,idx); ma=ma/ma.sum(1,keepdim=True).clamp_min(1e-6)
            out=torch.cat([out,s.heads((mz*ma.unsqueeze(-1)).sum(1))],-1)
        return out, z, a

K=len(GENES)
oof=np.zeros((len(tr),K),np.float32); extp=np.zeros((len(ev_ids),K*(2 if A.dsmil else 1)),np.float32)
for f,(trn,ten) in enumerate(StratifiedKFold(A.folds,shuffle=True,random_state=A.seed).split(np.zeros(len(tr)),st)):
    pw=torch.tensor(np.where(YT[trn].sum(0)>0,(len(trn)-YT[trn].sum(0))/np.maximum(YT[trn].sum(0),1),1.),dtype=torch.float32,device=dev)
    net=V2(1024,A.hid,K,A.dsmil,A.plscale).to(dev)
    opt=torch.optim.AdamW(net.parameters(),A.lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,A.lr,epochs=A.epochs,steps_per_epoch=max(1,len(trn)//A.batch))
    lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(A.seed+f)
    for ep in range(A.epochs):
        net.train(); perm=rng.permutation(trn)
        aw=(0.3*(0.3+0.7*ep/max(1,A.epochs-1))) if A.plscale else 0.
        for i in range(0,len(perm)-A.batch+1,A.batch):
            rows=perm[i:i+A.batch]
            b=np.empty((len(rows),A.s_inst,1024),np.float16)
            for k2,r in enumerate(rows):
                v=XT[RT[r]]; b[k2]=v[rng.choice(v.shape[0],A.s_inst,replace=v.shape[0]<A.s_inst)]
            b=torch.from_numpy(b).to(dev,torch.float32); yb=torch.from_numpy(YT[rows]).to(dev)
            if A.mixup and len(rows)>1:
                pm=torch.randperm(len(rows),device=dev); lam=float(np.random.beta(0.4,0.4))
                b=lam*b+(1-lam)*b[pm]; yb=lam*yb+(1-lam)*yb[pm]
            o,_,_=net(b); o1=o[:,:K]
            loss=(lf(o1,yb)*pw).mean()
            if A.dsmil:
                o2=o[:,K:]; loss=loss+(lf(o2,yb)*pw).mean()*0.5+0.15*((torch.sigmoid(o1)-torch.sigmoid(o2))**2).mean(0).sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
        for i in range(0,len(ten),32):
            rr=ten[i:i+32]
            b=np.stack([XT[RT[r]][rng.choice(XT[RT[r]].shape[0],A.s_inst,replace=False)] for r in rr])
            oof[rr]=net(torch.from_numpy(b).to(dev,torch.float32))[0].float().cpu().numpy()[:,:K]
        acc=None; nw=0
        for stt in range(0,2048,512):
            b=EX[:,stt:stt+512]
            o=net(torch.from_numpy(b).to(dev,torch.float32))[0].float()
            acc=o if acc is None else acc+o; nw+=1
        extp+= (acc/nw).cpu().numpy()
    print(f"[fold {f+1}/{A.folds}] done",flush=True)
extp/=A.folds

def auc(y,p): return float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else float("nan")
print(f"\n{'gene':<9}{'内部OOF':>9}{'外部CPTAC':>12}{'n_ext':>7}{'ext_prev':>10}")
rows=[]
for j,g in enumerate(GENES):
    a=auc(YT[:,j],oof[:,j]); ae=auc(EY[:,j],extp[:,j])
    rows.append(dict(tag=TAG,gene=g,auc_internal=round(a,4),auc_cptac=round(ae,4),
                     n_ext_pos=int(EY[:,j].sum()),n_ext=len(ev_ids)))
    print(f"{g:<9}{a:>9.3f}{ae:>12.3f}{int(EY[:,j].sum()):>7}{100*EY[:,j].mean():>9.1f}%")
df=pd.DataFrame(rows); out=os.path.join(CACHE,"external_cptac.csv")
df.to_csv(out,index=False,mode="a" if os.path.exists(out) else "w",header=not os.path.exists(out))
print(f"\n[mean] 内部={df.auc_internal.mean():.3f}  外部CPTAC={df.auc_cptac.mean():.3f}  掉={df.auc_cptac.mean()-df.auc_internal.mean():+.3f}")
print(f"[saved] {out}")



