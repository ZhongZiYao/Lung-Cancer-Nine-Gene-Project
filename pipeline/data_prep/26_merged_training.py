# -*- coding: utf-8 -*-
"""26_merged_training.py — TCGA 训练 vs TCGA+CPTAC 合并训练"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             confusion_matrix, average_precision_score)
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
DIM, EP, S_INST, BATCH, LR, HID, FOLDS = 1024, 25, 512, 32, 3e-4, 256, 5
dev="cuda"

# ---- 特征 ----
t = pd.read_csv(os.path.join(CACHE,"NSCLC_uni_sub512_index.csv")); t["cohort"]="TCGA"
c = pd.read_csv(os.path.join(CACHE,"CPTAC_uni_sub512_index.csv")); c["cohort"]="CPTAC"
meta = pd.concat([t[["case_id","mem_row","cohort"]], c[["case_id","mem_row","cohort"]]], ignore_index=True)
X = np.concatenate([np.load(os.path.join(CACHE,"NSCLC_uni_sub512.npy")),
                    np.load(os.path.join(CACHE,"CPTAC_uni_sub512.npy"))], axis=0)
print(f"[feat] {X.shape}")

# ---- 标签 ----
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
lab_t = man[["case_id","project"]+GENES].copy(); lab_t["cohort"]="TCGA"
lab_t = lab_t.dropna(subset=GENES)
cpl = pd.read_csv(os.path.join(R,"data","labels","cptac_gene_labels.csv"))
cpl = cpl[cpl.submitter_id.str.startswith("C3L")][["submitter_id"]+GENES].rename(columns={"submitter_id":"case_id"})
cpl["cohort"]="CPTAC"; cpl["project"]="LUAD"
labels = pd.concat([lab_t[["case_id","cohort"]+GENES], cpl[["case_id","cohort"]+GENES]], ignore_index=True)
labels = labels.drop_duplicates(subset=["case_id","cohort"])
m = meta.merge(labels, on=["case_id","cohort"], how="inner")
Y = m[GENES].values.astype(int); coh = m.cohort.values
ROW = m.mem_row.values.astype(int); N=len(m)
print(f"[cohort] n={N}  TCGA={int((coh=='TCGA').sum())}  CPTAC={int((coh=='CPTAC').sum())}")
for j,g in enumerate(GENES):
    a=Y[coh=='TCGA',j].mean(); b=Y[coh=='CPTAC',j].mean()
    print(f"          {g:<7} TCGA {int(Y[coh=='TCGA',j].sum()):3d}({100*a:4.1f}%)  CPTAC {int(Y[coh=='CPTAC',j].sum()):3d}({100*b:4.1f}%)")

def phi(a,b):
    a=a.astype(bool); b=b.astype(bool)
    if a.sum()==0 or b.sum()==0: return 0.0
    n11=(a&b).sum(); n10=(a&~b).sum(); n01=(~a&b).sum(); n00=(~a&~b).sum()
    d=np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return float((n11*n00-n10*n01)/d) if d>0 else 0.0
print(f"\n=== 共突变强度 mean|phi| ===")
for nm,msk in [("TCGA only",coh=="TCGA"),("CPTAC only",coh=="CPTAC"),("MERGED",np.ones(N,bool))]:
    ph=[abs(phi(Y[msk,i],Y[msk,j])) for i in range(6) for j in range(i+1,6)]
    print(f"  {nm:<12} {np.mean(ph):.4f}")

class Net(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.o=nn.Linear(h,k)
    def forward(s,x):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        return s.o((z*a.unsqueeze(-1)).sum(1))

def train(mask, seed=0):
    rws=ROW[mask]; Ym=Y[mask]; st=np.clip(Ym.sum(1),0,3).astype(int)
    oof=np.zeros((mask.sum(),6),np.float32)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=seed).split(np.zeros(mask.sum()),st)):
        pw=torch.tensor(np.where(Ym[tr].sum(0)>0,(len(tr)-Ym[tr].sum(0))/np.maximum(Ym[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=Net(DIM,HID,6).to(dev)
        opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EP,steps_per_epoch=max(1,len(tr)//BATCH))
        lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(seed*10+f)
        for ep in range(EP):
            net.train(); perm=rng.permutation(tr)
            for i in range(0,len(perm)-BATCH+1,BATCH):
                rr=perm[i:i+BATCH]
                bt=np.empty((len(rr),S_INST,DIM),np.float16)
                for k2,row in enumerate(rr):
                    v=X[rws[row]]; bt[k2]=v[rng.choice(v.shape[0],S_INST,replace=v.shape[0]<S_INST)]
                bt=torch.from_numpy(bt).to(dev,torch.float32)
                yb=torch.from_numpy(Ym[rr].astype(np.float32)).to(dev)
                p=net(bt); loss=(lf(p,yb)*pw).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            b=torch.from_numpy(X[np.ix_(rws[te],range(0,S_INST))]).to(dev,torch.float32)
            oof[te]=net(b).float().cpu().numpy()
    return oof

def youden(y,p):
    o=np.argsort(-p); ys=y[o]; ps=p[o]
    tpr=np.cumsum(ys)/max(ys.sum(),1); fpr=np.cumsum(1-ys)/max((1-ys).sum(),1)
    return float(ps[int(np.argmax(tpr-fpr))])

print(f"\n{'='*88}\n=== 训练集对比 (UNI 1024维, 5-fold) ===\n{'='*88}")
print(f"{'训练集':<14}{'gene':<8}{'AUC':>7}{'BA':>7}{'SEN':>7}{'SPEC':>7}{'AUPRC':>8}")
for nm,msk in [("TCGA-925",coh=="TCGA"),("MERGED-1041",np.ones(N,bool))]:
    oof=train(msk); Ym2=Y[msk]; st={}
    for j,g in enumerate(GENES):
        y=Ym2[:,j]; p=oof[:,j]
        if not (0<y.sum()<y.size): continue
        yh=(p>=youden(y,p)).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
        st[g]=(roc_auc_score(y,p),balanced_accuracy_score(y,yh),tp/(tp+fn),tn/(tn+fp),average_precision_score(y,p))
        print(f"{nm:<14}{g:<8}{st[g][0]:>7.3f}{st[g][1]:>7.3f}{st[g][2]:>7.3f}{st[g][3]:>7.3f}{st[g][4]:>8.3f}")
    a=np.array(list(st.values()))
    print(f"{nm:<14}{'mean':<8}{a[:,0].mean():>7.3f}{a[:,1].mean():>7.3f}{a[:,2].mean():>7.3f}{a[:,3].mean():>7.3f}{a[:,4].mean():>8.3f}\n")
    np.save(os.path.join(CACHE,f"merged_{nm}.npy"), oof)
print("DONE")
