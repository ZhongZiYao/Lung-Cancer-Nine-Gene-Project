# -*- coding: utf-8 -*-
"""16_final_table.py — 论文主结果表
按癌种分列 x 三档模型(基因组基线/mean-pool图像基线/GAMIL图像) x 全指标(AUC/ACC/F1/SEN/SPEC)
阈值用训练折 Youden 指数, 避免 0.5 阈值在不平衡数据上的失真
"""
import os, sys, re, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
CACHE = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
ti  = pd.read_csv(os.path.join(CACHE,"NSCLC_sub2048_index_fixed.csv"))
coh = ti.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"])
coh = coh[coh.project.isin(["LUAD","LUSC"])].reset_index(drop=True)
X = np.load(os.path.join(CACHE,"NSCLC_sub2048.npy")); RT = coh.mem_row.values.astype(int)

# 临床变量
tn=lambda s:(lambda m:f"TCGA-{m.group(1)}-{m.group(2)}" if m else None)(re.search(r"TCGA[-_]?([0-9A-Z]{2})[-_]?([0-9]{4})",str(s)))
gp=pd.read_parquet(os.path.join(R,"data","public_features","prov_gigapath_tcga","provgigapath_embeddings_with_metadata.parquet"))
gp["case"]=gp["_PATIENT"].map(tn); gg=gp.dropna(subset=["case"]).groupby("case").first()
oh=lambda s: pd.get_dummies(pd.Series(s).fillna("__NA__").astype(str)).values.astype(float)
stage=oh(coh.case_id.map(gg.get("ajcc_pathologic_tumor_stage")).values)
age=np.nan_to_num(coh.case_id.map(gg.get("age_at_initial_pathologic_diagnosis")).values.astype(float).reshape(-1,1),nan=0.0)
sub=(coh.project.values=="LUSC").astype(float).reshape(-1,1)

def youden(y,p):
    """返回使 Youden 指数最大的【概率阈值】"""
    o=np.argsort(-p); ys=y[o]; ps=p[o]
    tpr=np.cumsum(ys)/max(ys.sum(),1); fpr=np.cumsum(1-ys)/max((1-ys).sum(),1)
    return float(ps[int(np.argmax(tpr-fpr))])

def full_metrics(y, p_in, thr_in):
    """thr_in: 在训练折上算出的 Youden 阈值; 测试折上评估"""
    yhat = (p_in >= thr_in).astype(int)
    return dict(AUC=roc_auc_score(y,p_in) if 0<y.sum()<len(y) else np.nan,
                ACC=accuracy_score(y,yhat), F1=f1_score(y,yhat,zero_division=0),
                SEN=recall_score(y,yhat,zero_division=0),
                SPEC=confusion_matrix(y,yhat).ravel()[1]/max(confusion_matrix(y,yhat).ravel()[1]+confusion_matrix(y,yhat).ravel()[3],1))

def geno_feat(mask, j):
    Yl = coh[GENES].values[mask]; jl = GENES.index(j)
    return np.column_stack([stage[mask], age[mask], sub[mask]] + [Yl[:,k] for k in range(len(GENES)) if k!=jl])

def run_cohort(name, mask, oof_img):
    Yl = coh[GENES].values[mask]; n=mask.sum()
    strata=np.clip(Yl.sum(1),0,3).astype(int)
    Z = X[RT[mask]].astype(np.float32).mean(1)
    print(f"\n{'='*100}\n【{name}】 n={n}\n{'='*100}")
    print(f"{'模型':<14}{'gene':<8}{'n+':>5}{'AUC':>8}{'ACC':>8}{'F1':>8}{'SEN':>8}{'SPEC':>8}")
    store={}
    for j,g in enumerate(GENES):
        y = Yl[:,j]
        if not (0<y.sum()<n):
            print(f"{'':<14}{g:<8}{int(y.sum()):>5}   样本不足, 跳过"); continue
        out={}
        for tag, Xf in [("基因组基线", geno_feat(mask,g)),
                        ("mean-pool图像", Z),
                        ("GAMIL图像", oof_img[:,j:j+1])]:
            oof=np.zeros(n)
            for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,strata):
                if len(set(y[tr]))<2: continue
                sc=StandardScaler().fit(Xf[tr])
                m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),y[tr])
                oof[te]=m.predict_proba(sc.transform(Xf[te]))[:,1]
            # Youden 阈值在训练折上定
            tr_pred=np.zeros(n)
            for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,strata):
                if len(set(y[tr]))<2: continue
                sc=StandardScaler().fit(Xf[tr])
                m=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xf[tr]),y[tr])
                tr_pred[tr]=m.predict_proba(sc.transform(Xf[tr]))[:,1]
            thr=youden(y,tr_pred)
            if not (0.05 < thr < 0.95): thr=0.5
            out[tag]=full_metrics(y,oof,thr)
        for tag in out:
            v=out[tag]
            print(f"{tag:<14}{g:<8}{int(y.sum()):>5}{v['AUC']:>8.3f}{v['ACC']:>8.3f}{v['F1']:>8.3f}{v['SEN']:>8.3f}{v['SPEC']:>8.3f}")
        store[g]=out
    print(f"\n--- {name} 均值 ---")
    print(f"{'模型':<14}{'AUC':>8}{'ACC':>8}{'F1':>8}{'SEN':>8}{'SPEC':>8}")
    for tag in ["基因组基线","mean-pool图像","GAMIL图像"]:
        ks=[g for g in store if tag in store[g]]
        if not ks: continue
        print(f"{tag:<14}" + "".join(f"{np.mean([store[g][tag][m] for g in ks]):>8.3f}" for m in ["AUC","ACC","F1","SEN","SPEC"]))
    return store

ALL={}
for coh_name, tag in [("LUAD","LUAD"),("LUSC","LUSC"),("LUAD+LUSC","NSCLC")]:
    f=os.path.join(CACHE,f"oof_base_{tag}.npy")
    if not os.path.exists(f): print(f"[skip] {coh_name} 缺 {os.path.basename(f)}"); continue
    mask = (coh.project.values=="LUAD") if tag=="LUAD" else ((coh.project.values=="LUSC") if tag=="LUSC" else np.ones(len(coh),bool))
    oo=np.load(f)
    if oo.shape[0]==len(coh): oo=oo[mask]     # NSCLC 存全队列 925 行需取子集; 分癌种已是子集
    ALL[coh_name]=run_cohort(coh_name, mask, oo)
import pickle
pickle.dump(ALL, open(os.path.join(CACHE,"final_table.pkl"),"wb"))
print(f"\n[saved] {CACHE}\\final_table.pkl")



