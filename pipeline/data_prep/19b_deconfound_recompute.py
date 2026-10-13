# -*- coding: utf-8 -*-
"""19b_deconfound_recompute.py — 用已保存的 OOF 重算(修 auc 函数的 bug)"""
import os, sys, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
iA = np.load(os.path.join(C,"oof_gigap512_LUAD.npy"))
gA = np.load(os.path.join(C,"oof_genomic_LUAD.npy"))
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(C,"NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id","project"]+GENES],on="case_id",how="inner").dropna(subset=["TP53"])
df = df[df.project=="LUAD"].reset_index(drop=True)
Y = df[GENES].values.astype(int); N=len(Y)
print(f"[LUAD] n={N}\n")

# 修好的 auc: 检查【标签】的类别分布
def auc(y,p):
    return float(roc_auc_score(y,p)) if 0 < y.sum() < len(y) else np.nan

def phi(a,b):
    a=a.astype(bool); b=b.astype(bool)
    if a.sum()==0 or b.sum()==0: return 0.0
    n11=(a&b).sum(); n10=(a&~b).sum(); n01=(~a&b).sum(); n00=(~a&~b).sum()
    d=np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return float((n11*n00-n10*n01)/d) if d>0 else 0.0

print("=== 共突变 phi 矩阵 (LUAD) ===")
print("        " + "".join(f"{g:>8}" for g in GENES))
conf={}
for i,gi in enumerate(GENES):
    row=[phi(Y[:,i],Y[:,j]) for j in range(len(GENES))]
    conf[gi]=sorted([(GENES[j],v) for j,v in enumerate(row) if j!=i], key=lambda t:-abs(t[1]))
    print(f"{gi:<8}" + "".join(f"{v:>8.2f}" for v in row))

print("\n=== 整体净信号 ===")
print(f"{'gene':<8}{'基因组':>9}{'图像':>9}{'净信号':>10}{'主导混淆':>10}{'phi':>8}")
overall={}
for j,g in enumerate(GENES):
    a=auc(Y[:,j],gA[:,j]); b=auc(Y[:,j],iA[:,j]); overall[g]=b-a
    c,v = conf[g][0]
    print(f"{g:<8}{a:>9.3f}{b:>9.3f}{b-a:>+10.3f}{c:>10}{v:>+8.2f}")

print("\n=== T1 条件 AUC: 固定主导混淆因子后重算 ===")
print(f"{'gene':<8}{'混淆':<8}{'层':<6}{'n':>5}{'基因组':>9}{'图像':>9}{'条件净信号':>11}{'vs整体':>9}")
for j,g in enumerate(GENES):
    c,_ = conf[g][0]; k=GENES.index(c); cv=Y[:,k]
    for lev in (0,1):
        m = (cv==lev)
        if m.sum()<30 or not (0<Y[m,j].sum()<m.sum()): continue
        a=auc(Y[m,j],gA[m,j]); b=auc(Y[m,j],iA[m,j])
        print(f"{g:<8}{c:<8}{'c='+str(lev):<6}{int(m.sum()):>5}{a:>9.3f}{b:>9.3f}{b-a:>+11.3f}{b-a-overall[g]:>+9.3f}")

print("\n=== T2 混淆调整(后验重加权)后的净信号 ===")
print("  思路: 把【其他5个基因的完整突变模式】作为条件, 在每个条件层内重加权,")
print("        使目标基因在该层内的阳性/阴性比例与总体一致 -> 消除共突变相关性")
for j,g in enumerate(GENES):
    if not (0<Y[:,j].sum()<N): continue
    others=[k for k in range(len(GENES)) if k!=j]
    pat = np.array([hash(tuple(Y[i,others])) for i in range(N)])
    uniq,inv = np.unique(pat,return_inverse=True)
    p_target = Y[:,j].mean(); w=np.ones(N)
    for c in range(len(uniq)):
        m = inv==c
        pc = Y[m,j].mean()
        if 0<pc<1:
            w[m & (Y[:,j]==1)] = p_target/pc
            w[m & (Y[:,j]==0)] = (1-p_target)/(1-pc)
    w = w/w.mean()
    # 用同一批 OOF 预测, 计算重加权后的 AUC(加权版)
    def wauc(y,p,ww):
        o=np.argsort(-p); yy=y[o]; ww=ww[o]
        tpr=np.cumsum(yy*ww)/max((yy*ww).sum(),1e-9); fpr=np.cumsum((1-yy)*ww)/max(((1-yy)*ww).sum(),1e-9)
        return float(np.trapz(tpr,fpr))
    ga_w = wauc(Y[:,j], gA[:,j], w); ia_w = wauc(Y[:,j], iA[:,j], w)
    print(f"  {g:<8} 基因组 {ga_w:.3f}  图像 {ia_w:.3f}  去混淆后净信号 {ia_w-ga_w:+.3f}  (原 {overall[g]:+.3f})")

print("\nDONE")
