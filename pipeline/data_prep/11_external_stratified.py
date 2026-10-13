# -*- coding: utf-8 -*-
"""11_external_stratified.py
外部验证: TCGA-NSCLC(925) 训练 -> CPTAC-NSCLC(115) 零样本推理
按癌种分层, 保存预测供 bootstrap CI
"""
import os, re, json, glob, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
EXT   = os.path.join(R,"data","public_features","CPathPatchFeature","cptac_nsclc","uni","pt_files")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
UU = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--s-inst", type=int, default=512)
ap.add_argument("--batch",  type=int, default=32)
ap.add_argument("--lr",     type=float, default=3e-4)
ap.add_argument("--hid",    type=int, default=256)
ap.add_argument("--folds",  type=int, default=5)
ap.add_argument("--seed",   type=int, default=0)
ap.add_argument("--mixup",   action="store_true")
ap.add_argument("--dsmil",   action="store_true")
ap.add_argument("--plscale", action="store_true")
A = ap.parse_args(); dev = "cuda"
TAG = "base" if not (A.mixup or A.dsmil or A.plscale) else "+".join(n for n,f in [("mixup",A.mixup),("dsmil",A.dsmil),("plscale",A.plscale)] if f)

# ---- 训练集 ----
ti = pd.read_csv(os.path.join(CACHE,"NSCLC_sub2048_index_fixed.csv"))
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
tr = ti.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"]).reset_index(drop=True)
tr = tr[tr.project.isin(["LUAD","LUSC"])]
XT = np.load(os.path.join(CACHE,"NSCLC_sub2048.npy"))            # 全量入 RAM
RT = tr.mem_row.values.astype(int); YT = tr[GENES].values.astype(np.float32)
st = np.clip(YT.sum(1),0,3).astype(int)
print(f"[train] TCGA-NSCLC {len(tr)}  LUAD={int((tr.project=='LUAD').sum())}  LUSC={int((tr.project=='LUSC').sum())}")

# ---- 外验集 ----
lab = pd.read_csv(os.path.join(R,"data","labels","cptac_gene_labels.csv"))
lab = lab[lab.submitter_id.str.startswith("C3L")].copy()
files = {}
for p in glob.glob(os.path.join(EXT,"*.pt")):
    b = os.path.basename(p).split("-"); files[f"{b[0]}-{b[1]}"] = p
ev = [s for s in lab.submitter_id if s in files]
def pick(s, n, seed=0):
    t = torch.load(files[s], map_location="cpu", weights_only=False).numpy()
    r = np.random.RandomState(seed)
    return t[r.choice(t.shape[0], n, replace=t.shape[0] < n)]
EX = np.stack([pick(s, A.s_inst) for s in ev])
EY = lab.set_index("submitter_id").loc[ev, GENES].values.astype(np.float32)

C = os.path.join(R,"data","CPTAC")
cm = {k.lower(): v for k, v in json.load(open(os.path.join(C,"cptac_case_mapping.json"), encoding="utf-8")).items()}
fid2 = {m["file_id"].lower(): m.get("case_ids",[]) for m in json.load(open(os.path.join(C,"cptac3_maf_manifest.json"), encoding="utf-8"))}
dt = {}
for p in glob.glob(os.path.join(C,"cptac_maf","*.maf.gz")):
    f = UU.search(os.path.basename(p)).group(1).lower()
    for u in fid2.get(f, []):
        uu = u.lower()
        if uu in cm and cm[uu].get("submitter_id"):
            dt[cm[uu]["submitter_id"]] = "LUAD" if "Adenocarcinoma" in (cm[uu].get("disease_type") or "") else "LUSC"
EP = np.array([dt.get(s,"LUSC") for s in ev])
print(f"[ext]   CPTAC-NSCLC {len(ev)}  LUAD={int((EP=='LUAD').sum())}  LUSC={int((EP=='LUSC').sum())}  "
      + " ".join(f"{g}:{int(EY[:,j].sum())}" for j,g in enumerate(GENES)))

# ---- 模型 ----
class V2(nn.Module):
    def __init__(s,d,h,k,um,up):
        super().__init__()
        s.trunk=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.heads=nn.Linear(h,k); s.um=um
        if up:
            s.proto=nn.Parameter(torch.randn(k,2,h)*0.1); s.logit_scale=nn.Parameter(torch.full((k,),10.0))
    def forward(s,x):
        z=s.trunk(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        out=s.heads((z*a.unsqueeze(-1)).sum(1))
        if s.um:
            kk=min(32,x.shape[1]); idx=a.topk(kk,1).indices
            mz=torch.gather(z,1,idx.unsqueeze(-1).expand(-1,-1,z.shape[-1]))
            ma=torch.gather(a,1,idx); ma=ma/ma.sum(1,keepdim=True).clamp_min(1e-6)
            out=torch.cat([out,s.heads((mz*ma.unsqueeze(-1)).sum(1))],-1)
        return out

K=len(GENES)
oof=np.zeros((len(tr),K),np.float32); extp=np.zeros((len(ev),K*(2 if A.dsmil else 1)),np.float32)
for f,(a,b) in enumerate(StratifiedKFold(A.folds,shuffle=True,random_state=A.seed).split(np.zeros(len(tr)),st)):
    pw=torch.tensor(np.where(YT[a].sum(0)>0,(len(a)-YT[a].sum(0))/np.maximum(YT[a].sum(0),1),1.),dtype=torch.float32,device=dev)
    net=V2(1024,A.hid,K,A.dsmil,A.plscale).to(dev)
    opt=torch.optim.AdamW(net.parameters(),A.lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,A.lr,epochs=A.epochs,steps_per_epoch=max(1,len(a)//A.batch))
    lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(A.seed+f)
    for ep in range(A.epochs):
        net.train(); perm=rng.permutation(a)
        for i in range(0,len(perm)-A.batch+1,A.batch):
            rr=perm[i:i+A.batch]
            bt=np.empty((len(rr),A.s_inst,1024),np.float16)
            for kk,row in enumerate(rr):
                v=XT[RT[row]]; bt[kk]=v[rng.choice(v.shape[0],A.s_inst,replace=v.shape[0]<A.s_inst)]
            bt=torch.from_numpy(bt).to(dev,torch.float32); yb=torch.from_numpy(YT[rr]).to(dev)
            if A.mixup and len(rr)>1:
                pm=torch.randperm(len(rr),device=dev); lam=float(np.random.beta(0.4,0.4))
                bt=lam*bt+(1-lam)*bt[pm]; yb=lam*yb+(1-lam)*yb[pm]
            o=net(bt); o1=o[:,:K]
            loss=(lf(o1,yb)*pw).mean()
            if A.dsmil:
                o2=o[:,K:]; loss=loss+(lf(o2,yb)*pw).mean()*0.5+0.15*((torch.sigmoid(o1)-torch.sigmoid(o2))**2).mean(0).sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
        for i in range(0,len(b),32):
            rr=b[i:i+32]
            bt=np.stack([XT[RT[r]][rng.choice(XT[RT[r]].shape[0],A.s_inst,replace=False)] for r in rr])
            oof[rr]=net(torch.from_numpy(bt).to(dev,torch.float32)).float().cpu().numpy()[:,:K]
        acc=None; nw=0
        for stt in range(0,2048,512):
            o=net(torch.from_numpy(EX[:,stt:stt+512]).to(dev,torch.float32)).float()
            acc=o if acc is None else acc+o; nw+=1
        extp+=(acc/nw).cpu().numpy()
    print(f"  fold {f+1}/{A.folds}",flush=True)
extp/=A.folds

def auc(y,p): return float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else float("nan")
mLU = tr.project.values=="LUAD"; eLU = EP=="LUAD"
rows=[]
print(f"\n{'gene':<9}{'TCGA内OOF':>11}{'CPTAC全部':>11}{'CPTAC-LUAD':>12}{'nLUAD+':>8}")
for j,g in enumerate(GENES):
    ai=auc(YT[:,j],oof[:,j]); ae=auc(EY[:,j],extp[:,j])
    al=auc(EY[eLU,j],extp[eLU,j]) if 0<EY[eLU,j].sum()<eLU.sum() else float("nan")
    rows.append(dict(tag=TAG,gene=g,n_pos=int(YT[:,j].sum()),auc_internal=round(ai,4),
                     auc_cptac=round(ae,4),auc_cptac_luad=round(al,4) if al==al else None,
                     n_ext_pos=int(EY[:,j].sum())))
    print(f"{g:<9}{ai:>11.3f}{ae:>11.3f}{al:>12.3f}{int(EY[eLU,j].sum()):>8}")
out=os.path.join(CACHE,"external_stratified.csv")
df=pd.DataFrame(rows)
if os.path.exists(out): df.to_csv(out,index=False,mode="a",header=False)
else: df.to_csv(out,index=False)
np.save(os.path.join(CACHE,f"extp_{TAG}.npy"),extp)
np.save(os.path.join(CACHE,f"oofint_{TAG}.npy"),oof)
pd.DataFrame(dict(case_id=tr.case_id,project=tr.project)).to_csv(os.path.join(CACHE,"cases_int.csv"),index=False)
pd.DataFrame(dict(submitter_id=ev,project=EP)).to_csv(os.path.join(CACHE,"cases_ext.csv"),index=False)
print(f"\n[mean] 内部={df.auc_internal.mean():.3f}  CPTAC全部={df.auc_cptac.mean():.3f}  CPTAC-LUAD={df.auc_cptac_luad.mean():.3f}")
print(f"[saved] {out}")


