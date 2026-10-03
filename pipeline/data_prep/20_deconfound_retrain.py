# -*- coding: utf-8 -*-
"""
20_deconfound_retrain.py  —  用【权重重训】坐实去混淆结论 + 分组异构建模

三个变体:
  A  baseline           图像模型(多任务 6 head)
  B  importance-weight  逐基因共突变去相关重要性加权
  C  group-heterogeneous 共突变主导基因显式注入共突变向量, 推理时可消融

评估: raw AUC / 基因组基线 AUC / 净信号 / 去混淆后净信号
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
K=len(GENES)
print(f"[LUAD] n={N}  encoder=gigap  cache=512\n")

# ---- 临床变量 ----
tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
STAGE=oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
AGE=np.nan_to_num(df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1),nan=0.0)
COMUT = np.stack([Y[:,k] for k in range(K)],1).astype(np.float32)   # 共突变矩阵 (N,K)

# ---- 逐基因重要性权重: 让【其他5基因的突变模式】层内的阳性率趋于均匀 ----
def iw_weights(j):
    others=[k for k in range(K) if k!=j]
    pat=np.array([hash(tuple(Y[i,others])) for i in range(N)])
    uniq,inv=np.unique(pat,return_inverse=True)
    pt=Y[:,j].mean(); w=np.ones(N)
    for c in range(len(uniq)):
        m=inv==c; pc=Y[m,j].mean()
        if 0<pc<1:
            w[m&(Y[:,j]==1)]=pt/pc
            w[m&(Y[:,j]==0)]=(1-pt)/(1-pc)
    return w/np.mean(w)
IW = np.stack([iw_weights(j) for j in range(K)],1)     # (N,K)
print("[IW] 权重范围(逐基因): " + ", ".join(f"{g}[{IW[:,j].min():.2f},{IW[:,j].max():.2f}]" for j,g in enumerate(GENES)))

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

class Net(nn.Module):
    def __init__(s,d,h,k,use_comut=False):
        super().__init__()
        s.use_comut=use_comut
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.o=nn.Linear(h,k)
        if use_comut: s.oc=nn.Linear(h+k,k)
    def forward(s,x,cm=None):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        bag=(z*a.unsqueeze(-1)).sum(1)                       # (B,H)
        if not s.use_comut:
            out = s.o(bag); return out, out
        B=bag.shape[0]
        joint=s.oc(torch.cat([bag,cm],1))                     # (B,K) 用了共突变
        pure=s.o(bag)                                        # (B,K) 纯图像
        return joint, pure

def train(use_iw=False, use_comut=False):
    oof=np.zeros((N,K),np.float32); oof_pure=np.zeros((N,K),np.float32)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=Net(DIM,HID,K,use_comut).to(dev)
        opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EP,steps_per_epoch=max(1,len(tr)//BATCH))
        lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(SEED+f)
        cm_t=torch.from_numpy(COMUT).to(dev)
        for ep in range(EP):
            net.train(); perm=rng.permutation(tr)
            for i in range(0,len(perm)-BATCH+1,BATCH):
                rr=perm[i:i+BATCH]
                bt=np.empty((len(rr),S_INST,DIM),np.float16)
                for k2,rw in enumerate(rr):
                    v=X[RT[rw]]; bt[k2]=v[rng.choice(v.shape[0],S_INST,replace=v.shape[0]<S_INST)]
                bt=torch.from_numpy(bt).to(dev,torch.float32)
                yb=torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
                o,pure=net(bt, cm_t[rr] if use_comut else None)
                l=lf(o,yb)                                   # (B,K)
                w=torch.from_numpy(IW[rr]).to(dev) if use_iw else torch.ones(len(rr),K,device=dev)
                loss=((l*w)*pw).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            b=torch.from_numpy(X[np.ix_(RT[te],range(0,S_INST))]).to(dev,torch.float32)
            o,pure=net(b, cm_t[te] if use_comut else None)
            oof[te]=o.float().cpu().numpy(); oof_pure[te]=pure.float().cpu().numpy()
    return oof, oof_pure

auc=lambda y,p: float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan
def wauc(y,p,w):
    o=np.argsort(-p); yy=y[o]; ww=w[o]
    tpr=np.cumsum(yy*ww)/max((yy*ww).sum(),1e-9); fpr=np.cumsum((1-yy)*ww)/max(((1-yy)*ww).sum(),1e-9)
    return float(np.trapezoid(tpr,fpr))

gA=geno_oof()
print(f"\n[基因组基线] mean AUC = {np.nanmean([auc(Y[:,j],gA[:,j]) for j in range(K)]):.3f}\n")

res={}
for name,kw in [("A baseline",{}),("B importance-weight",{"use_iw":True}),("C group-heterogeneous",{"use_comut":True})]:
    oof,pure = train(**kw)
    res[name]=(oof,pure)
    print(f"{'='*84}\n【{name}】\n{'='*84}")
    print(f"{'gene':<8}{'raw':>8}{'基因组':>8}{'净信号':>9}{'去混淆':>9}{'变化':>8}")
    for j,g in enumerate(GENES):
        raw=auc(Y[:,j],oof[:,j]); ga=auc(Y[:,j],gA[:,j])
        w=IW[:,j]
        gw=wauc(Y[:,j],gA[:,j],w); iw_=wauc(Y[:,j],oof[:,j],w)
        print(f"{g:<8}{raw:>8.3f}{ga:>8.3f}{raw-ga:>+9.3f}{iw_-gw:>+9.3f}{(iw_-gw)-(raw-ga):>+8.3f}")
    raws=[auc(Y[:,j],oof[:,j]) for j in range(K)]
    nets=[auc(Y[:,j],oof[:,j])-auc(Y[:,j],gA[:,j]) for j in range(K)]
    dn =[wauc(Y[:,j],oof[:,j],IW[:,j])-wauc(Y[:,j],gA[:,j],IW[:,j]) for j in range(K)]
    print(f"{'mean':<8}{np.nanmean(raws):>8.3f}{np.nanmean([auc(Y[:,j],gA[:,j]) for j in range(K)]):>8.3f}{np.nanmean(nets):>+9.3f}{np.nanmean(dn):>+9.3f}")
    if kw.get("use_comut"):
        print(f"\n  --- C 的推理消融: 去掉共突变注入后(纯图像头) ---")
        pr=[auc(Y[:,j],pure[:,j]) for j in range(K)]
        pn=[auc(Y[:,j],pure[:,j])-auc(Y[:,j],gA[:,j]) for j in range(K)]
        print("  " + "".join(f"{g}={p:.3f}(净{v:+.3f})  " for g,p,v in zip(GENES,pr,pn)))
        print(f"  mean 纯图像 AUC={np.nanmean(pr):.3f}  mean 净信号={np.nanmean(pn):+.3f}")

np.save(os.path.join(CACHE,"deconfound_results.npy"), np.stack([res[k][0] for k in res]))
print("\nDONE")


