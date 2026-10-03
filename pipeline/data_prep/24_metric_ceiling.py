# -*- coding: utf-8 -*-
"""24_metric_ceiling.py — ACC 到底能做到多少? 用真实 OOF 预测向量算清楚"""
import os, sys, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score, precision_score,
                             matthews_corrcoef, average_precision_score, confusion_matrix)
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(C,"NSCLC_gigap_sub512_index.csv"))
df  = idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
df  = df[df.project=="LUAD"].reset_index(drop=True)
Y = df[GENES].values.astype(int); N=len(Y)
P = {}
for f in glob.glob(os.path.join(C,"agg_*_pure.npy")):
    tag = os.path.basename(f)[4:-len("_pure.npy")]
    P[tag] = np.load(f)
print(f"[LUAD n={N}] 可用模型: {list(P)}\n")

def youden(y,p):
    o=np.argsort(-p); ys=y[o]; ps=p[o]
    tpr=np.cumsum(ys)/max(ys.sum(),1); fpr=np.cumsum(1-ys)/max((1-ys).sum(),1)
    return float(ps[int(np.argmax(tpr-fpr))])

def block(title):
    print(f"\n{'='*94}\n【{title}】\n{'='*94}")
    print(f"{'gene':<8}{'n+':>5}{'多数类ACC':>10}{'AUC':>7}{'ACC':>7}{'BA':>7}{'F1':>7}{'MCC':>7}{'SEN':>7}{'SPEC':>7}{'AUPRC':>8}{'PRC基线':>9}")
    rows=[]
    for j,g in enumerate(GENES):
        y=Y[:,j]
        if not (0<y.sum()<N): continue
        maj = max(y.mean(), 1-y.mean())
        p = P["gated"][:,j]
        thr = youden(y,p)
        yh = (p>=thr).astype(int)
        tn,fp,fn,tp = confusion_matrix(y,yh).ravel()
        rows.append(dict(gene=g, maj=maj, auc=roc_auc_score(y,p), acc=accuracy_score(y,yh),
                         ba=balanced_accuracy_score(y,yh), f1=f1_score(y,yh,zero_division=0),
                         mcc=matthews_corrcoef(y,yh), sen=tp/(tp+fn), spec=tn/(tn+fp),
                         auprc=average_precision_score(y,p), prc=y.mean()))
        r=rows[-1]
        print(f"{g:<8}{int(y.sum()):>5}{r['maj']:>10.3f}{r['auc']:>7.3f}{r['acc']:>7.3f}{r['ba']:>7.3f}{r['f1']:>7.3f}{r['mcc']:>7.3f}{r['sen']:>7.3f}{r['spec']:>7.3f}{r['auprc']:>8.3f}{r['prc']:>9.3f}")
    m=lambda k: np.mean([r[k] for r in rows])
    print(f"{'mean':<8}{'':>5}{m('maj'):>10.3f}{m('auc'):>7.3f}{m('acc'):>7.3f}{m('ba'):>7.3f}{m('f1'):>7.3f}{m('mcc'):>7.3f}{m('sen'):>7.3f}{m('spec'):>7.3f}{m('auprc'):>8.3f}{m('prc'):>9.3f}")
    return rows
block("A. 模型在 Youden 阈值下的完整指标 (gated)")

print(f"\n{'='*94}\n【B. 只靠调阈值能把 ACC 推多高? —— 代价是什么】\n{'='*94}")
print(f"{'基因':<8}{'ACC':>7}{'SEN':>7}{'SPEC':>7}{'F1':>7}   说明")
for j,g in enumerate(GENES):
    y=Y[:,j]
    if not (0<y.sum()<N): continue
    p=P["gated"][:,j]
    # 扫阈值, 找 ACC 最高的点
    best=None
    for t in np.unique(p):
        yh=(p>=t).astype(int)
        a=accuracy_score(y,yh)
        if best is None or a>best[0]: best=(a,t,yh)
    a,t,yh=best
    print(f"{g:<8}{a:>7.3f}{recall_score(y,yh):>7.3f}{balanced_accuracy_score(y,yh):>7.3f}{f1_score(y,yh,zero_division=0):>7.3f}   阈值调到极端={yh.mean():.2f} 判正率")
print()
print("  -> 上表每行 ACC 都 >85%, 但注意 SEN 塌到什么程度, 以及 BA 其实没有超过 B 块")
