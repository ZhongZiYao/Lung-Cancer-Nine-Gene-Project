import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sklearn.metrics import roc_auc_score, roc_curve
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache"
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
z = np.load(os.path.join(C, "mammoth_oof.npz"))
Y = z["Y"]; P = z["M0_linear"]

def op_point(y, p):
    """Youden operating point taken straight from sklearn's roc_curve.
    Use the returned tpr / 1-fpr rather than re-thresholding the scores ourselves:
    roc_curve reports the operating point for the rule p > th, and re-deriving it with
    p >= th silently shifts the point by one distinct threshold."""
    fpr, tpr, th = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    sen = float(tpr[k]); spe = float(1.0 - fpr[k])
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    acc = (sen * npos + spe * nneg) / (npos + nneg)
    return sen, spe, acc, float(th[k]), float(tpr[k] - fpr[k])

yt = np.array([0, 0, 0, 1, 1, 1]); pt = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
print("self-check: SEN=%.2f SPE=%.2f ACC=%.2f J=%.2f  [expect 1 1 1 1]" % op_point(yt, pt)[:4], flush=True)

rows = []
for j, g in enumerate(GENES):
    A, SE, SP, AC, JJ = [], [], [], [], []
    for i in range(P.shape[0]):
        y, pp = Y[:, j], P[i][:, j]
        A.append(roc_auc_score(y, pp))
        se, sp, ac, t, v = op_point(y, pp)
        SE.append(se); SP.append(sp); AC.append(ac); JJ.append(v)
    rows.append(dict(gene=g, n_pos=int(Y[:, j].sum()), n_neg=int((Y[:, j] == 0).sum()),
                     AUC=np.mean(A), ACC=np.mean(AC), SEN=np.mean(SE), SPE=np.mean(SP),
                     BACC=(np.mean(SE) + np.mean(SP)) / 2, AUC_std=np.std(A, ddof=1),
                     YoudenJ=np.mean(JJ)))
df = pd.DataFrame(rows)
print("\n=== 最终模型逐基因指标 (3 套折划分各自取 Youden 操作点再平均) ===", flush=True)
print(df.to_string(index=False, float_format=lambda v: "%.4f" % v), flush=True)
print("\n  MEAN  AUC=%.4f (跨折 std %.4f)  ACC=%.4f  SEN=%.4f  SPE=%.4f  BACC=%.4f  Youden J=%.4f"
      % (df.AUC.mean(), df.AUC_std.mean(), df.ACC.mean(), df.SEN.mean(), df.SPE.mean(),
         df.BACC.mean(), df.YoudenJ.mean()), flush=True)
# 固定敏感度下的特异度
print("\n=== SEN 固定时的工作点 (mean over 3 fold seeds) ===", flush=True)
print("%-6s %7s %7s %7s" % ("gene", "SEN80%", "SPE80%", "AUC@80"), flush=True)
for j, g in enumerate(GENES):
    se80, sp80, auc80 = [], [], []
    for i in range(P.shape[0]):
        y, pp = Y[:, j], P[i][:, j]
        fpr, tpr, th = roc_curve(y, pp)
        def at(target, arr):
            ok = np.where(arr >= target)[0]
            return int(ok[0]) if len(ok) else len(arr) - 1
        k80 = at(0.80, tpr)
        se80.append(tpr[k80]); sp80.append(1 - fpr[k80]); auc80.append(roc_auc_score(y, pp))
    print("%-6s %7.3f %7.3f %7.3f" % (g, np.mean(se80), np.mean(sp80), np.mean(auc80)), flush=True)
df.to_csv(os.path.join(C, "final_metrics.csv"), index=False)
print("\n[saved] final_metrics.csv", flush=True)