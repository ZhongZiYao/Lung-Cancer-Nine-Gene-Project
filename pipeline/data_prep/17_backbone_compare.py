# -*- coding: utf-8 -*-
"""17_backbone_compare.py — UNI v1 vs Prov-GigaPath 受控 backbone 对比"""
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
ENC = {"uni":1024, "gigap":1536}
PREFIX = {"uni":"NSCLC", "gigap":"NSCLC_gigap"}
IDXF = {"uni":"NSCLC_sub2048_index_fixed.csv", "gigap":"NSCLC_gigap_sub2048_index.csv"}
EPOCHS, S_INST, BATCH, LR, HID, FOLDS, SEED = 25, 512, 32, 3e-4, 256, 5, 0
dev = "cuda"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))

class Net(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.o=nn.Linear(h,k)
    def forward(s,x):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        return s.o((z*a.unsqueeze(-1)).sum(1))

def load_cohort(enc, cohort):
    idx=pd.read_csv(os.path.join(CACHE, IDXF[enc]))
    df=idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
    if cohort!="NSCLC": df=df[df.project==cohort]
    return df.reset_index(drop=True)

def run_gamil(enc, df):
    D=ENC[enc]; N=len(df); Y=df[GENES].values.astype(int); RT=df.mem_row.values.astype(int)
    X=np.load(os.path.join(CACHE,f"{PREFIX[enc]}_sub2048.npy"))
    strata=np.clip(Y.sum(1),0,3).astype(int); oof=np.zeros((N,len(GENES)),np.float32)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=Net(D,HID,len(GENES)).to(dev)
        opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EPOCHS,steps_per_epoch=max(1,len(tr)//BATCH))
        lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(SEED+f)
        for ep in range(EPOCHS):
            net.train(); perm=rng.permutation(tr)
            for i in range(0,len(perm)-BATCH+1,BATCH):
                rr=perm[i:i+BATCH]
                bt=np.empty((len(rr),S_INST,D),np.float16)
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
            for stt in range(0,2048,S_INST):
                b=torch.from_numpy(X[np.ix_(RT[te],range(stt,stt+S_INST))]).to(dev,torch.float32)
                o=net(b).float(); acc=o if acc is None else acc+o; nw+=1
            oof[te]=(acc/nw).cpu().numpy()
    del X
    return [roc_auc_score(Y[:,j],oof[:,j]) if 0<Y[:,j].sum()<N else np.nan for j in range(len(GENES))]

def genomic_baseline(df):
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

print("加载临床变量...", flush=True)
for cohort in ["LUAD","LUSC","NSCLC"]:
    print(f"\n{'='*88}\n【{cohort}】\n{'='*88}")
    d = load_cohort("uni", cohort)
    gen = genomic_baseline(d)
    print(f"{'模型':<26}" + "".join(f"{g:>10}" for g in GENES) + f"{'mean':>10}")
    print(f"{'  基因组基线(无图)':<26}" + "".join(f"{x:>10.3f}" for x in gen) + f"{np.nanmean(gen):>10.3f}")
    for enc,nm in [("uni","UNI v1 (1024)"),("gigap","GigaPath (1536)")]:
        a = run_gamil(enc, load_cohort(enc, cohort))
        print(f"{'  '+nm:<26}" + "".join(f"{x:>10.3f}" for x in a) + f"{np.nanmean(a):>10.3f}")
        print(f"{'    净信号':<26}" + "".join(f"{x-g:>+10.3f}" for x,g in zip(a,gen)) + f"{np.nanmean([x-g for x,g in zip(a,gen)]):>+10.3f}")
    print(f"  n={len(d)}")
print("\nDONE")

