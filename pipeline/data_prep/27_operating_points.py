# -*- coding: utf-8 -*-
"""27_operating_points.py — SPEC→SEN 操作点曲线
临床分诊真正关心的不是单一阈值, 而是"在多高的特异度下还能抓到多少阳性"。
输出: 每个基因在 SPEC = 50/60/.../99% 时的最大 SEN, 以及反向的 SEN→SPEC。
"""
import os, sys, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R,"data","patch_cache")
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
man = pd.read_csv(os.path.join(R,"data","labels","case_manifest.csv"))
idx = pd.read_csv(os.path.join(C,"NSCLC_gigap_sub512_index.csv"))
df = idx.merge(man[["case_id","project"]+GENES], on="case_id", how="inner").dropna(subset=["TP53"])
df = df[df.project=="LUAD"].reset_index(drop=True)
Y = df[GENES].values.astype(int)
P = np.load(os.path.join(C,"agg_gated_pure.npy"))
print(f"[LUAD n={len(Y)}] 特征=Prov-GigaPath 512patch  模型=gated\n")

def roc_points(y,p):
    o=np.argsort(-p); ys=y[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    tpr=tp/max(ys.sum(),1); fpr=fp/max((1-ys).sum(),1)
    return np.concatenate([[0],tpr]), np.concatenate([[0],fpr])

def sen_at_spec(y,p,target_spec):
    # target 是【特异度】。ROC 上 fpr=1-spec, 约束必须写成 fpr <= 1-target。
    tpr,fpr = roc_points(y,p)
    ok = fpr <= (1.0 - target_spec)
    return float(tpr[ok].max()) if ok.any() else 0.0
def spec_at_sen(y,p,target):
    tpr,fpr = roc_points(y,p)
    ok = tpr >= target
    return float(1-fpr[ok].min()) if ok.any() else 1.0

SPECS = [0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.98]
print("=== 给定特异度下的最大灵敏度 (SEN @ SPEC) ===")
print(f"{'gene':<8}{'n+':>5}{'prev%':>7}" + "".join((str(int(v*100))+"%").rjust(9) for v in SPECS))
rows=[]
for j,g in enumerate(GENES):
    y=Y[:,j]
    if not (0<y.sum()<len(y)): continue
    sens=[sen_at_spec(y,P[:,j],s) for s in SPECS]
    rows.append((g,int(y.sum()),y.mean(),sens))
    print(f"{g:<8}{int(y.sum()):>5}{100*y.mean():>6.1f}%" + "".join(f"{v:>9.3f}" for v in sens))
m=np.array([r[3] for r in rows])
print(f"{'mean':<8}{'':>5}{'':>7}" + "".join(f"{v:>9.3f}" for v in m.mean(0)))

SENS = [0.30,0.40,0.50,0.60,0.70,0.80]
print(f"\n=== 给定灵敏度下的最大特异度 (SPEC @ SEN) ===")
print(f"{'gene':<8}" + "".join((str(int(v*100))+"%").rjust(9) for v in SENS))
for j,g in enumerate(GENES):
    y=Y[:,j]
    if not (0<y.sum()<len(y)): continue
    sp=[spec_at_sen(y,P[:,j],s) for s in SENS]
    print(f"{g:<8}" + "".join(f"{v:>9.3f}" for v in sp))
ms=np.array([[spec_at_sen(Y[:,j],P[:,j],s) for s in SENS] for j in range(6) if 0<Y[:,j].sum()<len(Y)])
print(f"{'mean':<8}" + "".join(f"{v:>9.3f}" for v in ms.mean(0)))

# 分诊价值: 高特异度下的阳性捕获量
print(f"\n=== 分诊价值: 若设定 SPEC>=95% 阈值, 1000 例中能少做多少次测序 ===")
print(f"{'gene':<8}{'prev%':>8}{'SEN@95%':>10}{'每1000例漏诊':>14}{'每1000例可省':>14}")
for g,n,pr,sens in rows:
    s95 = sen_at_spec(Y[:,GENES.index(g)],P[:,GENES.index(g)],0.95)
    caught = pr*s95*1000
    print(f"{g:<8}{100*pr:>7.1f}%{s95:>10.3f}{pr*(1-s95)*1000:>14.0f}{pr*s95*1000:>14.0f}")
print("\n说明: '可省' = 模型高置信判阴性的病例中, 真实阴性者占比对应的测序节省上限(未计假阴)")

dfout = pd.DataFrame([dict(gene=r[0], n_pos=r[1], prev=r[2],
                          **{f"sen_at_spec{int(s*100)}": r[3][k] for k,s in enumerate(SPECS)})
                      for r in rows])
for _g, _n, _p, _sn in rows:
    _j = GENES.index(_g)
    for _s in SENS:
        dfout.loc[dfout.gene==_g, f"spec_at_sen{int(_s*100)}"] = spec_at_sen(Y[:,_j], P[:,_j], _s)
dfout.to_csv(os.path.join(C,"operating_points.csv"), index=False)
print(f"\n[saved] {C}\\operating_points.csv")
