# -*- coding: utf-8 -*-
"""18_sampling_ablation.py — 采样量消融 (可指定 encoder)
用法: python 18_sampling_ablation.py gigap 512 2048 4096
判定: 净信号随采样量上升 => 采样是瓶颈, 需改架构
      4096 与 2048 持平   => 采样不是瓶颈, 现有结论站得住
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
EPOCHS, BATCH, LR, HID, FOLDS, SEED = 25, 32, 3e-4, 256, 5, 0
S_INST = 512
dev="cuda"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
ENC = sys.argv[1] if len(sys.argv)>1 else "gigap"
DIM = {"uni":1024,"gigap":1536}[ENC]
PREFIX = "NSCLC" if ENC=="uni" else "NSCLC_gigap"
IDX = {"uni":{2048:"NSCLC_sub2048_index_fixed.csv"}, "gigap":{}}[ENC]

class Net(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.o=nn.Linear(h,k)
    def forward(s,x):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        return s.o((z*a.unsqueeze(-1)).sum(1))

def run(npatch, cohort):
    idxf = IDX.get(npatch, f"{PREFIX}_sub{npatch}_index.csv")
    idx=pd.read_csv(os.path.join(CACHE,idxf))
    df=idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
    if cohort!="NSCLC": df=df[df.project==cohort]
    df=df.reset_index(drop=True)
    N=len(df); Y=df[GENES].values.astype(int); RT=df.mem_row.values.astype(int)
    X=np.load(os.path.join(CACHE,f"{PREFIX}_sub{npatch}.npy"))
    strata=np.clip(Y.sum(1),0,3).astype(int); oof=np.zeros((N,len(GENES)),np.float32)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=Net(DIM,HID,len(GENES)).to(dev)
        opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EPOCHS,steps_per_epoch=max(1,len(tr)//BATCH))
        lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(SEED+f)
        for ep in range(EPOCHS):
            net.train(); perm=rng.permutation(tr)
            for i in range(0,len(perm)-BATCH+1,BATCH):
                rr=perm[i:i+BATCH]
                bt=np.empty((len(rr),S_INST,DIM),np.float16)
                for k2,rw in enumerate(rr):
                    v=X[RT[rw]]; bt[k2]=v[rng.choice(v.shape[0],S_INST,replace=v.shape[0]<S_INST)]
                bt=torch.from_numpy(bt).to(dev,torch.float32); yb=torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                p=net(bt); loss=(lf(p,yb)*pw).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            acc=None; nw=0
            for stt in range(0,npatch,S_INST):
                b=torch.from_numpy(X[np.ix_(RT[te],range(stt,stt+S_INST))]).to(dev,torch.float32)
                o=net(b).float(); acc=o if acc is None else acc+o; nw+=1
            oof[te]=(acc/nw).cpu().numpy()
    del X
    return [roc_auc_score(Y[:,j],oof[:,j]) if 0<Y[:,j].sum()<N else np.nan for j in range(len(GENES))], N, df

def genomic(df):
    import re
    tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
    gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
    gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
    oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
    Y=df[GENES].values.astype(int); N=len(df)
    st=oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
    ag=np.nan_to_num(df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1),nan=0.0)
    sb=(df.project.values=="LUSC").astype(float).reshape(-1,1)
    out=[]
    for j in range(len(GENES)):
        Xf=np.column_stack([st,ag,sb]+[Y[:,k] for k in range(len(GENES)) if k!=j])
        if not (0<Y[:,j].sum()<N): out.append(np.nan); continue
        o=np.zeros(N); strata=np.clip(Y.sum(1),0,3)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,strata):
            if len(set(Y[tr,j]))<2: continue
            sc=StandardScaler().fit(Xf[tr])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),Y[tr,j])
            o[te]=m.predict_proba(sc.transform(Xf[te]))[:,1]
        out.append(roc_auc_score(Y[:,j],o) if 0<Y[:,j].sum()<N else np.nan)
    return out

NPS=[int(x) for x in (sys.argv[2:] or [512,2048,4096])]
for cohort in ["LUAD","NSCLC"]:
    print(f"\n{'='*86}\n【{cohort}】采样量消融  encoder={ENC}({DIM}维)  训练每次采 {S_INST}\n{'='*86}")
    print(f"{'缓存patch数':<14}" + "".join(f"{g:>9}" for g in GENES) + f"{'mean':>9}{'净信号':>10}")
    gen=None
    for np_ in NPS:
        try: a,N,df = run(np_, cohort)
        except FileNotFoundError as e: print(f"{np_:<14} 缺缓存"); continue
        if gen is None: gen = genomic(df)
        net=[x-g for x,g in zip(a,gen)]
        print(f"{np_:<14}" + "".join(f"{x:>9.3f}" for x in a) + f"{np.nanmean(a):>9.3f}{np.nanmean(net):>+10.3f}")
    print(f"  基因组基线     " + "".join(f"{x:>9.3f}" for x in gen) if gen else "")
    print(f"  n={N}")
print("\nDONE")
