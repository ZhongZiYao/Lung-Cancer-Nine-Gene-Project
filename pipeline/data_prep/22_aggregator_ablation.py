# -*- coding: utf-8 -*-
"""
22_aggregator_ablation.py
  (1) 修掉联合头的标签泄漏 -> leave-one-out
  (2) GAMIL 注意力聚合的 4 个变体消融
"""
import os, sys, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
DIM, EP, S_INST, BATCH, LR, HID, FOLDS, SEED = 1536, 25, 512, 32, 3e-4, 256, 5, 0
LAM = float(sys.argv[1]) if len(sys.argv)>1 else 0.5
dev="cuda"
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(CACHE,"NSCLC_gigap_sub512_index.csv"))
df  = idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
df  = df[df.project=="LUAD"].reset_index(drop=True)
N=len(df); Y=df[GENES].values.astype(int); RT=df.mem_row.values.astype(int)
X=np.load(os.path.join(CACHE,"NSCLC_gigap_sub512.npy"))
strata=np.clip(Y.sum(1),0,3).astype(int); K=len(GENES)
CM=np.stack([Y[:,k] for k in range(K)],1).astype(np.float32)     # (N,K) 全 6 基因

tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
STAGE=oh(df.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
AGE=np.nan_to_num(df.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1),nan=0.0)

def geno_oof():
    o=np.zeros((N,K),np.float32)
    for j in range(K):
        Xf=np.column_stack([STAGE,AGE]+[CM[:,k] for k in range(K) if k!=j])   # leave-one-out
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,strata):
            if len(set(Y[tr,j]))<2: continue
            sc=StandardScaler().fit(Xf[tr])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),Y[tr,j])
            o[te,j]=m.predict_proba(sc.transform(Xf[te]))[:,1]
    return o

class Trunk(nn.Module):
    def __init__(s,d,h,drop=0.25):
        super().__init__(); s.f=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(drop))
    def forward(s,x): return s.f(x)

def gated_att(V,U,w,z):
    return torch.softmax(w(torch.tanh(V(z))*torch.sigmoid(U(z))).squeeze(-1),1)

class Agg(nn.Module):
    def __init__(s,d,h,k,mode):
        super().__init__()
        s.mode=mode; s.K=k; s.H=h
        s.trunk=Trunk(d,h)
        if mode=="gated":
            s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
            s.od=nn.Linear(h,k)
        elif mode=="clam_sb":
            s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False)
            s.w_pos=nn.Linear(h,1,bias=False); s.w_neg=nn.Linear(h,1,bias=False)
            s.od=nn.Linear(h,k)
        elif mode=="abmilx":                     # 多头 + 全局相关
            s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False)
            s.w=nn.ModuleList([nn.Linear(h,1,bias=False) for _ in range(4)])
            s.od=nn.Linear(h,k)
        elif mode=="proto_loo":                  # 每基因 pos/neg 原型, per-patch 监督
            s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
            s.od=nn.Linear(h,k)
            s.proto=nn.Parameter(torch.randn(k,2,h)*0.1)
            s.logit_scale=nn.Parameter(torch.full((k,),10.0))
    def forward(s,x,cm):
        z=s.trunk(x); B,S,H=z.shape
        if s.mode=="gated":
            a=gated_att(s.V,s.U,s.w,z); bag=(z*a.unsqueeze(-1)).sum(1)
            return s.od(bag), {}, bag
        if s.mode=="clam_sb":
            ap=torch.softmax(s.w_pos(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
            an=torch.softmax(s.w_neg(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
            bag=(z*ap.unsqueeze(-1)).sum(1)-(z*an.unsqueeze(-1)).sum(1)
            return s.od(bag), {"ap":ap,"an":an,"z":z}, bag
        if s.mode=="abmilx":
            bags=[]; atts=[]
            for w_ in s.w:
                a=torch.softmax(w_(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
                bags.append((z*a.unsqueeze(-1)).sum(1)); atts.append(a)
            bag=torch.stack(bags,0).mean(0)                        # 多头平均
            return s.od(bag), {"atts":atts,"z":z}, bag
        # proto_loo
        a=gated_att(s.V,s.U,s.w,z); bag=(z*a.unsqueeze(-1)).sum(1)
        zn=F.normalize(z,dim=-1); pn=F.normalize(s.proto,dim=-1)
        sim=torch.einsum("bsh,gch->bsgc",zn,pn)                     # (B,S,K,2)
        sc4=s.logit_scale.clamp(min=0.1).view(1,1,K,1)   # 广播到 (B,S,K,2)
        sc3=s.logit_scale.clamp(min=0.1).view(1,1,K)     # 广播到 (B,S,K)
        soft=torch.softmax(sc4*sim,dim=-1)[...,1]                   # (B,S,K) patch 软标签
        inst_logit=(sc3*(sim[...,0]-sim[...,1])).permute(2,0,1)  # (1,1,K)*(B,S,K)->(K,B,S)
        inst_logit=(sc3*(sim[...,0]-sim[...,1])).permute(2,0,1)
        return s.od(bag), {"soft":soft,"inst_logit":inst_logit,"z":z}, bag
    def joint(s,bag,cm):
        """leave-one-out 联合头: 每个基因只喂其他 5 个基因"""
        B=bag.shape[0]; outs=[]
        for g in range(s.K):
            mask=[k for k in range(s.K) if k!=g]
            x=torch.cat([bag, cm[:,mask]],1)
            outs.append(s.oj[g](x).squeeze(-1))
        return torch.stack(outs,1)
class AggJ(Agg):
    def __init__(s,d,h,k,mode):
        super().__init__(d,h,k,mode)
        s.oj=nn.ModuleList([nn.Linear(h+k-1,1) for _ in range(k)])

def inst_prototypes(z, a, ybag):
    """CLAM instance clustering: 取注意力最大处的特征做类原型"""
    idx=a.argmax(1)                                        # (B,)
    e=z[torch.arange(z.shape[0],device=z.device), idx]     # (B,H)
    proto=torch.stack([e[ybag==1].mean(0) if (ybag==1).any() else e.mean(0),
                       e[ybag==0].mean(0) if (ybag==0).any() else e.mean(0)])
    return proto

def train(mode, use_joint=True):
    op=np.zeros((N,K),np.float32); oj=np.zeros((N,K),np.float32)
    cm_t=torch.from_numpy(CM).to(dev)
    for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
        pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
        net=AggJ(DIM,HID,K,mode).to(dev)
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
                p,aux,bag=net(bt, cm_t[rr])
                jo=net.joint(bag, cm_t[rr])
                loss=lf(p,yb).mul(pw).mean() + LAM*lf(jo,yb).mul(pw).mean()
                if mode=="clam_sb":
                    # CLAM-SB instance clustering: 逐基因
                    # 取正分支注意力最大处的 instance embedding, 拉近该基因的阳性原型
                    z_=aux["z"]; ap_=aux["ap"]
                    idx=ap_.argmax(1)
                    e_=z_[torch.arange(z_.shape[0],device=dev), idx]      # (B,H)
                    en=F.normalize(e_,dim=-1)
                    for g in range(K):
                        m1=yb[:,g]>0.5
                        if m1.sum()>0:
                            pr=F.normalize(en[m1].mean(0),dim=-1)
                            cos=(en*pr).sum(1)
                            loss=loss+0.3*((1.0-cos)*yb[:,g]).mean()
                if mode=="proto_loo":
                    il=aux["inst_logit"]                            # (K,B,S)
                    yb6=yb.t().unsqueeze(-1).expand(-1,-1,S_INST)   # (K,B,S)
                    loss=loss+0.3*lf(il,yb6).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
                try: sch.step()
                except Exception: pass
        net.eval()
        with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
            b=torch.from_numpy(X[np.ix_(RT[te],range(0,S_INST))]).to(dev,torch.float32)
            p,_,bag=net(b, cm_t[te]); jo=net.joint(bag, cm_t[te])
            op[te]=p.float().cpu().numpy(); oj[te]=jo.float().cpu().numpy()
    return op,oj

auc=lambda y,p: float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan
gA=geno_oof()
gA_auc=[auc(Y[:,j],gA[:,j]) for j in range(K)]
print(f"[LUAD n={N}]  基因组基线 mean AUC = {np.nanmean(gA_auc):.3f}")
print(f"{'聚合器':<14}{'纯形态':>9}{'形态净信号':>12}{'联合(LOO)':>11}{'联合净信号':>12}")
out={}
ALL4=["gated","clam_sb","abmilx","proto_loo"]
if "--only-proto" in sys.argv: MODES=["proto_loo"]
elif "--skip-gated" in sys.argv: MODES=["clam_sb","abmilx","proto_loo"]
else: MODES=ALL4
for mode in MODES:
    op,oj=train(mode)
    ap=[auc(Y[:,j],op[:,j]) for j in range(K)]
    an=[auc(Y[:,j],oj[:,j]) for j in range(K)]
    np_=[a-g for a,g in zip(ap,gA_auc)]; nn_=[a-g for a,g in zip(an,gA_auc)]
    out[mode]=(ap,an,np_,nn_)
    print(f"{mode:<14}{np.nanmean(ap):>9.3f}{np.nanmean(np_):>+12.3f}{np.nanmean(an):>11.3f}{np.nanmean(nn_):>+12.3f}")
    np.save(os.path.join(CACHE,f"agg_{mode}_pure.npy"),op); np.save(os.path.join(CACHE,f"agg_{mode}_joint.npy"),oj)

print(f"\n=== 逐基因纯形态 AUC ===")
print(f"{'gene':<8}" + "".join(f"{m:>12}" for m in out))
for j,g in enumerate(GENES):
    print(f"{g:<8}" + "".join(f"{out[m][0][j]:>12.3f}" for m in out))
print(f"\n=== 逐基因形态净信号 ===")
print(f"{'gene':<8}" + "".join(f"{m:>12}" for m in out))
for j,g in enumerate(GENES):
    print(f"{g:<8}" + "".join(f"{out[m][2][j]:>+12.3f}" for m in out))
print("\nDONE")
