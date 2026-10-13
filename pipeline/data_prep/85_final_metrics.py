import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.metrics import roc_auc_score, roc_curve
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache"
GENES = ["KRAS","TP53","KEAP1","NF1","STK11","EGFR"]
z = np.load(os.path.join(C, "mammoth_oof.npz"))
Y = z["Y"]; P0 = z["M0_linear"]           # (n_foldseed, n_case, 6)
print("oof array:", P0.shape, flush=True)
man = pd.read_csv(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\labels\case_manifest.csv")
lab = man[man.has_label==1].set_index("case_id")
idx = pd.read_csv(os.path.join(C,"NSCLC_gigap_sub4096_index.csv"))
rm = dict(zip(idx.case_id, idx.mem_row))
pj = man.set_index("case_id")["project"].to_dict()
cases = sorted(c for c in rm if pj.get(c)=="LUAD" and c in lab.index)
cases = [c for c in cases if any(abs(Y[i]-lab.loc[c,GENES].values.astype(int)).sum()<1e-6 for i,c2 in enumerate(cases) if c2==c)]
# 用行序对齐: 重建 Y 对应的 case 顺序
order = sorted(cases)
def row_for(c):
    for i, c2 in enumerate(order):
        if c2 == c: return i
    return None
Yv = Y; Pv = P0.mean(0)                  # 跨 3 套折划分平均

rows = []
for j,g in enumerate(GENES):
    y, p = Yv[:,j], Pv[:,j]
    auc = roc_auc_score(y,p)
    fpr,tpr,th = roc_curve(y,p); t = th[int(np.argmax(tpr-fpr))]
    yh = (p>=t).astype(int)
    rows.append(dict(gene=g, n_pos=int(y.sum()), n_neg=int((y==0).sum()), auc=auc,
                     acc=float((yh==y).mean()), sen=float(yh[y==1].mean()), spe=float(yh[y==0].mean())))
df = pd.DataFrame(rows)
print("\n=== 最终配置 逐基因指标 (OOF, 跨 3 套折划分平均, threshold=Youden) ===", flush=True)
print(df.to_string(index=False, float_format=lambda v: "%.4f" % v), flush=True)
print("\n  mean AUC = %.4f   mean ACC@Youden = %.4f   mean SEN = %.4f   mean SPE = %.4f"
      % (df.auc.mean(), df.acc.mean(), df.sen.mean(), df.spe.mean()), flush=True)
print("  balanced acc = %.4f" % ((df.sen.mean()+df.spe.mean())/2), flush=True)

# 单基因相对随机(0.5)的提升
print("\n  相对随机(0.5)的提升:", " ".join("%s +%.3f" % (r.gene, r.auc-0.5) for r in df.itertuples()), flush=True)

# 跨折划分 std
per = []
for i in range(P0.shape[0]):
    per.append(float(np.mean([roc_auc_score(Yv[:,j], P0[i][:,j]) for j in range(6)])))
print("\n  跨 3 套折划分的 mean AUC: %s  -> %.4f +- %.4f"
      % (" ".join("%.4f"%v for v in per), np.mean(per), np.std(per, ddof=1)), flush=True)

# 阴性 vs 阳性的可分性（描述性）
print("\n  参考: 阴性基线 AUC=0.5 ; 项目历史基因组基线 0.671", flush=True)