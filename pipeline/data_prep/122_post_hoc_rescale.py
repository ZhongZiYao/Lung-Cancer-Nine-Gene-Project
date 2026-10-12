"""122_post_hoc_rescale.py - post-hoc rescale S48 logits using training fold pos_weight

No retraining. Use S117b 3fs OOF (434, 6) logits. For each gene:
  new_logit = original_logit + log(pos_weight)

This is the simplest cost-sensitive post-hoc adjustment, equivalent to the
inside of CB loss but applied to pre-trained model logits.

Then evaluate mean AUC + per-gene AUC + sens/spec/F1/bacc @ Youden.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
FOLD_SEED = 101
NEG = [298, 206, 356, 379, 371, 378]   # full-dataset neg count per gene
POS = [136, 228, 78, 55, 63, 56]      # full-dataset pos count per gene
POS_WEIGHT = np.array([n / max(p, 1) for n, p in zip(NEG, POS)])
LOG_PW = np.log(POS_WEIGHT)            # additive offset in logit space
print(f"POS_WEIGHT per gene: {POS_WEIGHT.round(2).tolist()}")
print(f"LOG_POS_WEIGHT per gene: {LOG_PW.round(2).tolist()}")

# Load S117b 3fs OOF
data = np.load(r'E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\s117b_3fold_oof.npz', allow_pickle=True)
Y = data['Y']
common = data['common']
oof_orig = (data['oof_fs101'] + data['oof_fs202'] + data['oof_fs303']) / 3
oof_rescaled = oof_orig + LOG_PW  # broadcast (434, 6) + (6,)

print(f"\n=== Original vs Rescaled logit stats per gene ===")
print(f"{'gene':<8}{'Y_pos':>7}{'orig_AUC':>10}{'re_AUC':>10}{'orig_sens@0.5':>15}{'re_sens@0.5':>15}{'orig_spec':>11}{'re_spec':>11}")
for i, g in enumerate(GENES):
    y = Y[:, i]
    p_orig = 1.0 / (1.0 + np.exp(-oof_orig[:, i]))
    p_re = 1.0 / (1.0 + np.exp(-oof_rescaled[:, i]))
    yp_o = (p_orig >= 0.5).astype(int)
    yp_r = (p_re >= 0.5).astype(int)
    auc_o = roc_auc_score(y, p_orig)
    auc_r = roc_auc_score(y, p_re)
    sens_o = recall_score(y, yp_o, zero_division=0)
    sens_r = recall_score(y, yp_r, zero_division=0)
    spec_o = recall_score(1-y, 1-yp_o, zero_division=0)
    spec_r = recall_score(1-y, 1-yp_r, zero_division=0)
    print(f"{g:<8}{int(y.sum()):>7}{auc_o:>10.4f}{auc_r:>10.4f}{sens_o:>15.4f}{sens_r:>15.4f}{spec_o:>11.4f}{spec_r:>11.4f}")

# Mean AUC across genes
def mean_auc(Y, P):
    aucs = []
    for j in range(len(GENES)):
        npos = int(Y[:, j].sum())
        if npos >= MIN_POS and npos <= len(Y) - MIN_POS:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")

orig_mean = mean_auc(Y, oof_orig)
re_mean = mean_auc(Y, oof_rescaled)
print(f"\nmean AUC (6 genes):  original = {orig_mean:.4f}  rescaled = {re_mean:.4f}")

# Per-gene Youden optimal threshold (rescaled)
print(f"\n=== Rescaled + Youden optimal threshold ===")
print(f"{'gene':<8}{'opt_thr':>10}{'sens':>10}{'spec':>10}{'f1':>10}{'bacc':>10}{'mcc':>10}{'auc':>10}")
y_re_total = 1.0 / (1.0 + np.exp(-oof_rescaled))
total_f1, total_bacc, total_mcc, total_sens, total_spec = [], [], [], [], []
for i, g in enumerate(GENES):
    y = Y[:, i]
    p = y_re_total[:, i]
    if y.sum() < MIN_POS:
        continue
    thrs = np.linspace(0.05, 0.95, 91)
    best_j, best_t, best_f1 = -1, 0.5, 0
    for t in thrs:
        yp = (p >= t).astype(int)
        sens = yp[y==1].mean()
        spec = (1-yp)[y==0].mean()
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    yp = (p >= best_t).astype(int)
    sens = yp[y==1].mean()
    spec = (1-yp)[y==0].mean()
    prec = (yp & (y==1)).sum() / max(yp.sum(), 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    bacc = (sens + spec) / 2
    mcc = matthews_corrcoef(y, yp)
    auc = roc_auc_score(y, p)
    total_sens.append(sens); total_spec.append(spec)
    total_f1.append(f1); total_bacc.append(bacc); total_mcc.append(mcc)
    print(f"{g:<8}{best_t:>10.3f}{sens:>10.4f}{spec:>10.4f}{f1:>10.4f}{bacc:>10.4f}{mcc:>10.4f}{auc:>10.4f}")

print(f"\nmean sens@Youden = {np.mean(total_sens):.4f}")
print(f"mean spec@Youden = {np.mean(total_spec):.4f}")
print(f"mean F1@Youden   = {np.mean(total_f1):.4f}")
print(f"mean BAcc@Youden = {np.mean(total_bacc):.4f}")
print(f"mean MCC@Youden  = {np.mean(total_mcc):.4f}")

# Save results
import json
res = {
    "method": "S48 + log(pos_weight) post-hoc",
    "mean_auc_orig": orig_mean,
    "mean_auc_rescaled": re_mean,
    "per_gene": {},
    "mean_sens_youden": float(np.mean(total_sens)),
    "mean_spec_youden": float(np.mean(total_spec)),
    "mean_f1_youden": float(np.mean(total_f1)),
    "mean_bacc_youden": float(np.mean(total_bacc)),
    "mean_mcc_youden": float(np.mean(total_mcc)),
}
for i, g in enumerate(GENES):
    res["per_gene"][g] = {"pos_weight_full": round(float(POS_WEIGHT[i]), 2)}
print(f"\n=== RESULT SUMMARY ===")
print(json.dumps(res, indent=2))