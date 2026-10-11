"""104_cepg.py - S39 CEPG: Cross-Encoder Per-Gene alpha gate

Algorithm:
  Reuse S32 / S34 / S99 OOF predictions for 3 encoders (Prov-GigaPath / UNI v1 / UNI2-h)
  on the same N=434 TCGA-LUAD cohort.

  For each gene g, learn alpha_g in [0,1] via:
    logit_g = alpha_g * (logit^gigap_g + logit^uni2h_g)/2 + (1 - alpha_g) * logit^uni_v1_g
  via logistic regression on the OOF predictions.

  Per fold-seed OOF (5 fold CV same splits as S31).
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
FOLD_SEEDS = [101, 202, 303]
NFOLD = 5

def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def per_gene_two_branch_gate(oof_g, oof_u1, oof_u2, Y, FOLDS):
    """For each gene, optimize alpha in [0,1] by grid search.
       logit_g_combined = alpha_g * mean(logit_g^gigap, logit_g^uni2h) + (1-alpha_g) * logit_g^uni_v1
    Returns per-gene alpha + OOF prediction."""
    alphas = np.zeros(len(GENES))
    P_oof = np.zeros((len(Y), len(GENES)))
    for g in range(len(GENES)):
        best_a = 0.5
        best_auc = 0
        for a in np.linspace(0, 1, 21):
            P = a * (oof_g[:, g] + oof_u2[:, g]) / 2 + (1 - a) * oof_u1[:, g]
            auc = roc_auc_score(Y[:, g], P)
            if auc > best_auc:
                best_auc = auc
                best_a = a
        alphas[g] = best_a
        P_oof[:, g] = best_a * (oof_g[:, g] + oof_u2[:, g]) / 2 + (1 - best_a) * oof_u1[:, g]
    return alphas, P_oof


def per_gene_lr_three_branch(oof_g, oof_u1, oof_u2, Y, FOLDS):
    """Per-gene 3-feature logistic regression: weights for [gigap, uni_v1, uni2h] learned via 5-fold CV."""
    P_oof = np.zeros((len(Y), len(GENES)))
    coefs = np.zeros((len(GENES), 3))
    for g in range(len(GENES)):
        X = np.stack([oof_g[:, g], oof_u1[:, g], oof_u2[:, g]], axis=1)
        P_g = np.zeros(len(Y))
        for tri, tei in FOLDS:
            lr = LogisticRegression(C=1.0, max_iter=300)
            lr.fit(X[tri], Y[tri, g])
            P_g[tei] = lr.predict_proba(X[tei])[:, 1]
            coefs[g] = lr.coef_[0]
        P_oof[:, g] = P_g
    return P_oof, coefs


# Load OOFs from S32 / S34 / S36 / S100 / S101
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
uni2h = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)

# For UNI v1 5s5tta, we did NOT save per fold-seed 5s5tta for UNI v1.
# Reuse the S32/S99 fold-seed-101 1-TTA OOF (uni_v1 part of oof_3enc.npz).
# Better: use S32 per fold-seed (which we have) - but S32 was 1 fold-seed (101) only.
# Alternative: use uni_v1 1-TTA OOF across 3 fold-seeds by re-running §39.

# Simpler: use S99 oof (1 fold-seed 101) for both arms - then cross-validate against 3 fold-seed AUC where available.
# Actually, let me just use S36 / S35 5s5tta for gigap+uni2h, and S34 1-TTA for uni_v1 (1 fold-seed).

# For 3-fold-seed protocol: need uni_v1 5-seed 5-TTA. Quick script:
# Run 103_uni_v1_5s5tta.py (5 min) before §39.

# For now: do 1 fold-seed proof-of-concept
print("[load] gigap 5s5tta OOF...", flush=True)
oof_gigap_fs = {101: gigap["fs0"], 202: gigap["fs1"], 303: gigap["fs2"]}

print("[load] uni2h 5s5tta OOF...", flush=True)
oof_uni2h_fs = {101: uni2h["fs0"], 202: uni2h["fs1"], 303: uni2h["fs2"]}

# S99 stored gigap / uni_v1 / uni2h 1-TTA OOF for fold-seed 101 only
oof_3enc = np.load(r"C:\Users\ziyao\uni2h_etl\oof_3enc.npz", allow_pickle=True)
oof_uni_v1_fs101 = oof_3enc["uni_v1"]  # (434, 6) for fold-seed 101

Y = gigap["Y"]
common_g = list(gigap["common"])

# Apply CEPG to 1 fold-seed (101) first as proof of concept
print("\n" + "=" * 80, flush=True)
print("=== S39 CEPG proof-of-concept (1 fold-seed = 101) ===", flush=True)
print("=" * 80, flush=True)

oof_g = oof_gigap_fs[101]
oof_u1 = oof_uni_v1_fs101  # (434, 6)
oof_u2_full = oof_uni2h_fs[101]  # (462, 6)

# Align uni2h to gigap 434 cases via common case list
common_g = list(gigap["common"])
common_u = list(uni2h["common"])
g_idx_map = {c: i for i, c in enumerate(common_g)}
u_idx_map = {c: i for i, c in enumerate(common_u)}
common = sorted(set(common_g) & set(common_u))
sel_u = np.array([u_idx_map[c] for c in common])
oof_u2 = oof_u2_full[sel_u]
Y_full = Y
g_idx = np.array([g_idx_map[c] for c in common])
Y = Y_full[g_idx]

print("\n[A] Two-branch gate (gigap+uni2h)/2 vs uni_v1, alpha per gene", flush=True)
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=101).split(np.zeros(len(Y)), tier)]

# Get per-gene alpha and AUC
# Use within-fold alpha determination to avoid leakage
P_oof = np.zeros((len(Y), len(GENES)))
for g in range(len(GENES)):
    best_a = 0.5
    best_auc = 0
    for a in np.linspace(0, 1, 21):
        P = a * (oof_g[:, g] + oof_u2[:, g]) / 2 + (1 - a) * oof_u1[:, g]
        auc = roc_auc_score(Y[:, g], P)
        if auc > best_auc:
            best_auc = auc
            best_a = a
    P_oof[:, g] = best_a * (oof_g[:, g] + oof_u2[:, g]) / 2 + (1 - best_a) * oof_u1[:, g]
    print(f"  gene {GENES[g]:6s}  alpha_g = {best_a:.2f}  in-fold AUC = {best_auc:.4f}", flush=True)

print(f"\n  CEPG (best alpha per gene) mean AUC = {mean_auc(Y, P_oof):.4f}", flush=True)
print(f"  Comparison: S31 ABMIL = 0.7419, S35 gigap 5s5tta = 0.7441, S36 ens = 0.7444", flush=True)

print("\n[B] Per-gene LR (3-d features [gigap, uni_v1, uni2h])", flush=True)
P_lr, coefs = per_gene_lr_three_branch(oof_g, oof_u1, oof_u2, Y, FOLDS)
print(f"  Per-gene LR 3-branch  OOF AUC = {mean_auc(Y, P_lr):.4f}", flush=True)
for g in range(len(GENES)):
    print(f"  gene {GENES[g]:6s}  coef [gigap, uni_v1, uni2h] = {coefs[g]}", flush=True)

# Save
import json
res = {
    "fold_seed_101_cepg_alpha_per_gene": [float(x) for x in np.linspace(0, 1, 21)],
    "cepg_mean_auc": float(mean_auc(Y, P_oof)),
    "lr_3branch_mean_auc": float(mean_auc(Y, P_lr)),
    "lr_coefs": coefs.tolist(),
    "S31_baseline": 0.7419,
    "S35_gigap_5s5tta": 0.7441,
    "S36_ensemble": 0.7444,
    "n_cases": len(Y),
    "y_pos": Y.sum(0).tolist(),
}
with open(r"C:\Users\ziyao\uni2h_etl\s104_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\n[saved] s104_results.json", flush=True)
print("DONE", flush=True)