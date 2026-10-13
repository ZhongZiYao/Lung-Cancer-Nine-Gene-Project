# -*- coding: utf-8 -*-
"""
15_residual_fusion.py  —  残差融合: 量化 H&E 形态的【真实净信号】

思路: 基因组基线(共突变+临床+癌种)先给出 p_gen,
      图像模型只学「基因组解释不了的那部分」:
          logit(p_final) = logit(p_gen) + f_image(x)
      报告:
        p_gen  单独 AUC   -> 基因组/共突变捷径
        p_img  单独 AUC   -> 图像裸预测
        p_fin  融合 AUC   -> 两者合并
        净信号 = AUC(p_fin) - AUC(p_gen)   <-- 这才是 H&E 的真实贡献
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
dev="cuda"
EP, S_INST, BATCH, LR, HID, FOLDS, SEED = 25, 512, 32, 3e-4, 256, 5, 0

man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
ti  = pd.read_csv(os.path.join(CACHE,"NSCLC_sub2048_index_fixed.csv"))
coh = ti.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"])
coh = coh[coh.project.isin(["LUAD","LUSC"])].reset_index(drop=True)
N=len(coh); Y=coh[GENES].values.astype(int)
X  = np.load(os.path.join(CACHE,"NSCLC_sub2048.npy"))
RT = coh.mem_row.values.astype(int)
sub=(coh.project.values=="LUSC").astype(float).reshape(-1,1)
strata=np.clip(Y.sum(1),0,3).astype(int)
print(f"[cohort] n={N} LUAD={int((sub==0).sum())} LUSC={int((sub==1).sum())}")

# ---------- 基因组基线特征 ----------
tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
stage=oh(coh.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values) if "ajcc_pathologic_tumor_stage" in gg.columns else np.zeros((N,1))
age=coh.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1)
age=np.nan_to_num(age,nan=0.0)
def geno_feat(j): return np.column_stack([stage, age, sub] + [Y[:,k] for k in range(len(GENES)) if k!=j])
def cv_gen(tr,te,j,seed=0):
    Xf=geno_feat(j); oof=np.zeros(len(te))
    itr,ite=next(iter(StratifiedKFold(5,shuffle=True,random_state=seed).split(Xf[tr],Y[tr,j])))
    sc=StandardScaler().fit(Xf[tr][itr])
    m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr][itr]),Y[tr][itr,j])
    oof=m.predict_proba(sc.transform(Xf[te]))[:,1]
    return np.clip(oof,1e-4,1-1e-4)

class Net(nn.Module):
    def __init__(s,d,h,k):
        super().__init__()
        s.t=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.25))
        s.V=nn.Linear(h,h,bias=False); s.U=nn.Linear(h,h,bias=False); s.w=nn.Linear(h,1,bias=False)
        s.od=nn.Linear(h,k); s.op=nn.Linear(h,k)
    def forward(s,x):
        z=s.t(x); a=torch.softmax(s.w(torch.tanh(s.V(z))*torch.sigmoid(s.U(z))).squeeze(-1),1)
        bag=(z*a.unsqueeze(-1)).sum(1)
        return s.od(bag), s.op(bag)      # (delta logit, image-only logit)

def samp(rows,rng):
    out=np.empty((len(rows),S_INST,1024),np.float16)
    for i,r in enumerate(rows):
        v=X[RT[r]]; out[i]=v[rng.choice(v.shape[0],S_INST,replace=v.shape[0]<S_INST)]
    return out

K=len(GENES)
p_img=np.zeros((N,K),np.float32); p_fin=np.zeros((N,K),np.float32); p_gen=np.zeros((N,K),np.float32)
for f,(tr,te) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(np.zeros(N),strata)):
    # 1) 基因组基线: 训练折内再切一层
    for j in range(K):
        if Y[tr,j].sum()<5: p_gen[tr,j]=Y[tr,j].mean(); p_gen[te,j]=Y[tr,j].mean(); continue
        Xf=geno_feat(j)
        pg_itr=np.zeros(len(tr)); rng_i=np.random.RandomState(SEED)
        for a,b in StratifiedKFold(4,shuffle=True,random_state=SEED).split(Xf[tr],Y[tr,j]):
            sc=StandardScaler().fit(Xf[tr][a])
            m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr][a]),Y[tr][a,j])
            pg_itr[b]=m.predict_proba(sc.transform(Xf[tr][b]))[:,1]
        sc=StandardScaler().fit(Xf[tr])
        m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),Y[tr,j])
        p_gen[tr,j]=np.clip(pg_itr,1e-4,1-1e-4)
        p_gen[te,j]=np.clip(m.predict_proba(sc.transform(Xf[te]))[:,1],1e-4,1-1e-4)
    # 2) 图像残差模型
    pw=torch.tensor(np.where(Y[tr].sum(0)>0,(len(tr)-Y[tr].sum(0))/np.maximum(Y[tr].sum(0),1),1.),dtype=torch.float32,device=dev)
    net=Net(1024,HID,K).to(dev)
    opt=torch.optim.AdamW(net.parameters(),LR,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,LR,epochs=EP,steps_per_epoch=max(1,len(tr)//BATCH))
    lf=nn.BCEWithLogitsLoss(reduction="none"); rng=np.random.RandomState(SEED+f)
    lg_tr=torch.log(torch.tensor(p_gen[tr],dtype=torch.float32))
    pos_of={v:i for i,v in enumerate(tr)}   # <-- 训练折内位置映射, 修复索引错位
    for ep in range(EP):
        net.train(); perm=rng.permutation(tr)
        for i in range(0,len(perm)-BATCH+1,BATCH):
            rr=perm[i:i+BATCH]
            b=torch.from_numpy(samp(rr,rng)).to(dev,torch.float32)
            yb=torch.from_numpy(Y[rr].astype(np.float32)).to(dev)
            d,po=net(b)
            lgl=lg_tr[[pos_of[r] for r in rr]].to(dev)
            lf_d=(lf(d+lgl,yb)*pw).mean()   # 残差目标
            lf_p=(lf(po,yb)*pw).mean()                            # 图像裸预测(辅助)
            loss=lf_d+0.3*lf_p
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda",torch.bfloat16):
        acc_d=acc_p=None; nw=0
        for st in range(0,2048,512):
            b=torch.from_numpy(X[np.ix_(RT[te],range(st,min(st+512,2048)))]).to(dev,torch.float32)
            d,po=net(b); d=d.float(); po=po.float()
            acc_d=d if acc_d is None else acc_d+d; acc_p=po if acc_p is None else acc_p+po; nw+=1
        d=(acc_d/nw).cpu().numpy(); po=(acc_p/nw).cpu().numpy()
    p_img[te]=1/(1+np.exp(-po))
    p_fin[te]=1/(1+np.exp(-d+np.log(p_gen[te]/(1-p_gen[te]))))
    print(f"  fold {f+1}/{FOLDS}",flush=True)

auc=lambda y,p: float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan
for tag,mask in [("LUAD+LUSC",np.ones(N,bool)),("LUAD",(coh.project.values=="LUAD")),("LUSC",coh.project.values=="LUSC")]:
    print(f"\n=== {tag} (n={int(mask.sum())}) ===")
    print(f"{'gene':<8}{'p_gen':>8}{'p_img':>8}{'p_fin':>8}{'净信号':>9}")
    g_,i_,f_,d_=[],[],[],[]
    for j,gn in enumerate(GENES):
        y=Y[mask,j]
        if not (0<y.sum()<mask.sum()): print(f"{gn:<8}  n+={int(y.sum())} 样本不足"); continue
        a=auc(y,p_gen[mask,j]); b=auc(y,p_img[mask,j]); c=auc(y,p_fin[mask,j])
        g_.append(a); i_.append(b); f_.append(c); d_.append(c-a)
        print(f"{gn:<8}{a:>8.3f}{b:>8.3f}{c:>8.3f}{c-a:>+9.3f}")
    print(f"{'mean':<8}{np.mean(g_):>8.3f}{np.mean(i_):>8.3f}{np.mean(f_):>8.3f}{np.mean(d_):>+9.3f}")
np.save(os.path.join(CACHE,"p_img_resid.npy"),p_img); np.save(os.path.join(CACHE,"p_gen.npy"),p_gen); np.save(os.path.join(CACHE,"p_fin.npy"),p_fin)
print(f"\n[saved] p_img_resid.npy / p_gen.npy / p_fin.npy")



