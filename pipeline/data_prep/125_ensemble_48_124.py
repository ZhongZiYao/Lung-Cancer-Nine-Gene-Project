"""125_ensemble_48_124.py - Ensemble ABMIL + Transformer MIL (5 min runtime, no training)

Simple post-hoc ensemble: oof = (oof_48 + oof_124) / 2
Test if Transformer diversity gives +1% over either alone.

Loads S117b 3fs OOF (ABMIL) + S124 OOF (Transformer), evaluates mean AUC + per-gene.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, balanced_accuracy_score, matthews_corrcoef
)

GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8

# Load ABMIL OOF (§117b 3 fs mean)
abmil_data = np.load(r'E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\s117b_3fold_oof.npz', allow_pickle=True)
Y = abmil_data['Y']
oof_abmil = (abmil_data['oof_fs101'] + abmil_data['oof_fs202'] + abmil_data['oof_fs303']) / 3

# Load Transformer OOF (§124)
tf_data = np.load(r'E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache\s124_transformer_mil.npz', allow_pickle=True)
oof_tf = tf_data['oof']
common_tf = tf_data['common']

# Align: if common arrays differ, need to remap. ABMIL common is just cases order in data/patch_cache
# Both OOF are indexed by same case_id order in Y
# Verify: compare abmil_data['common'] and tf_data['common']
if not np.array_equal(abmil_data['common'], common_tf):
    # find index remapping
    print("WARNING: common differs, remapping")
    tf_idx_in_abmil = [list(abmil_data['common']).index(c) for c in common_tf]
    oof_tf_remap = oof_tf[np.array(tf_idx_in_abmil)]
    oof_tf = oof_tf_remap
    Y_remap = Y[np.array(tf_idx_in_abmil)]
else:
    Y_remap = Y

print(f"data: ABMIL={oof_abmil.shape} TF={oof_tf.shape} Y={Y.shape}")


def mean_auc_safe(Y, P):
    aucs = []
    for j in range(len(GENES)):
        npos = int(Y[:, j].sum())
        if npos >= MIN_POS and npos <= len(Y) - MIN_POS:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


# Compare 4 strategies
strategies = {
    "ABMIL §48 (baseline)": oof_abmil,
    "Transformer §124": oof_tf,
    "Ensemble (mean)": (oof_abmil + oof_tf) / 2,
    "Ensemble (weighted 0.6*ABMIL + 0.4*TF)": 0.6 * oof_abmil + 0.4 * oof_tf,
}

print(f"\n{'strategy':<35}{'mean AUC':>10}{'TP53_sens':>12}{'TP53_spec':>12}{'TP53_bacc':>12}{'STK11_AUC':>12}")
results = {}
for name, oof in strategies.items():
    auc = mean_auc_safe(Y, oof)
    per_gene = {g: float(roc_auc_score(Y[:, gi], oof[:, gi])) for gi, g in enumerate(GENES)}
    y = Y[:, GENES.index('TP53')]
    p = 1.0 / (1.0 + np.exp(-oof[:, GENES.index('TP53')]))
    yp = (p >= 0.5).astype(int)
    tp53_sens = recall_score(y, yp, zero_division=0)
    tp53_spec = recall_score(1-y, 1-yp, zero_division=0)
    tp53_bacc = balanced_accuracy_score(y, yp)
    stk11_auc = per_gene['STK11']
    results[name] = (auc, per_gene, tp53_sens, tp53_spec, tp53_bacc)
    print(f"{name:<35}{auc:>10.4f}{tp53_sens:>12.4f}{tp53_spec:>12.4f}{tp53_bacc:>12.4f}{stk11_auc:>12.4f}")

# Per-gene AUC table
print(f"\n=== Per-gene AUC comparison ===")
print(f"{'gene':<8}{'abmil':>10}{'trans':>10}{'mean':>10}{'weighted':>11}{'best':>11}")
for gi, g in enumerate(GENES):
    a = results["ABMIL §48 (baseline)"][1][g]
    t = results["Transformer §124"][1][g]
    m = roc_auc_score(Y[:, gi], strategies["Ensemble (mean)"][:, gi])
    w = roc_auc_score(Y[:, gi], strategies["Ensemble (weighted 0.6*ABMIL + 0.4*TF)"][:, gi])
    best = max(a, t, m, w)
    print(f"{g:<8}{a:>10.4f}{t:>10.4f}{m:>10.4f}{w:>11.4f}{best:>11.4f}")

# Save best ensemble OOF
best_strategy = max(results, key=lambda k: results[k][0])
best_oof = strategies[best_strategy]
print(f"\nBest strategy: {best_strategy}  mean AUC = {results[best_strategy][0]:.4f}")
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s125_ensemble_best.npz",
                    oof=best_oof, Y=Y, common=abmil_data['common'])
print(f"[saved] s125_ensemble_best.npz")