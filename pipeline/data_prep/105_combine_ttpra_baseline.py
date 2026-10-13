"""105_combine_ttpra_baseline.py - combine TTPRA with ABMIL (gigap 5s5tta) ensemble
fold-seed 101 only

Search alpha in [0, 1] for alpha*TTPRA + (1-alpha)*ABMIL.
Expect: at some alpha between 0.3-0.5, mean AUC peaks (above 0.7444 baseline).
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from sklearn.metrics import roc_auc_score

GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8

def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


# Load gigap 5s5tta OOF (S35 S36 100b) - has 3 fold-seed OOF
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
oof_g_fs101 = gigap["fs0"]  # (434, 6) - this is the fold-seed 101 OOF
Y = gigap["Y"]
common_g = list(gigap["common"])

# Load TTPRA OOF (1 fold-seed 101)
ttpra = np.load(r"C:\Users\ziyao\uni2h_etl\ttpra_oof_fs101.npz", allow_pickle=True)
oof_t = ttpra["fs101"]  # (434, 6)

# Align case order
common_t = list(ttpra["common"])
g_idx = {c: i for i, c in enumerate(common_g)}
t_idx = {c: i for i, c in enumerate(common_t)}
common = sorted(set(common_g) & set(common_t))
sel_g = np.array([g_idx[c] for c in common])
sel_t = np.array([t_idx[c] for c in common])

P_g = oof_g_fs101[sel_g]
P_t = oof_t[sel_t]
Y = Y[sel_g]

# First, individual baselines
auc_g = mean_auc(Y, P_g)
auc_t = mean_auc(Y, P_t)
print(f"[A] ABMIL (gigap 5s5tta)  = {auc_g:.4f}", flush=True)
print(f"[B] TTPRA only (S=128 K=30) = {auc_t:.4f}", flush=True)

# Per-gene AUC
print(f"\n  per-gene AUCs:", flush=True)
for g in range(len(GENES)):
    auc_g_g = roc_auc_score(Y[:, g], P_g[:, g])
    auc_t_g = roc_auc_score(Y[:, g], P_t[:, g])
    print(f"    {GENES[g]:6s}  ABMIL={auc_g_g:.4f}  TTPRA={auc_t_g:.4f}  diff={auc_t_g-auc_g_g:+.4f}", flush=True)

# Linear combination: alpha * TTPRA + (1-alpha) * ABMIL
print(f"\n=== Linear combination: alpha*TTPRA + (1-alpha)*ABMIL ===", flush=True)
best_alpha = 0
best_auc = auc_g
for a in np.linspace(0, 1, 21):
    P = a * P_t + (1 - a) * P_g
    auc = mean_auc(Y, P)
    if auc > best_auc:
        best_auc = auc
        best_alpha = a

# Per-gene optimal alpha (since each gene has different signal)
print(f"\n  Best global alpha = {best_alpha:.2f}  AUC = {best_auc:.4f}  (delta = {best_auc - auc_g:+.4f})", flush=True)

# Per-gene optimal alpha (use whatever alpha maximizes each gene)
print(f"\n=== Per-gene optimal alpha ===", flush=True)
P_per_gene = np.zeros_like(P_g)
alphas = np.zeros(len(GENES))
for g in range(len(GENES)):
    best_a = 0
    best_a_g_auc = roc_auc_score(Y[:, g], P_g[:, g])
    for a in np.linspace(0, 1, 21):
        P = a * P_t[:, g] + (1 - a) * P_g[:, g]
        auc = roc_auc_score(Y[:, g], P)
        if auc > best_a_g_auc:
            best_a_g_auc = auc
            best_a = a
    alphas[g] = best_a
    P_per_gene[:, g] = best_a * P_t[:, g] + (1 - best_a) * P_g[:, g]
    print(f"  {GENES[g]:6s}  best_alpha={best_a:.2f}  best_auc={best_a_g_auc:.4f}  vs ABMIL={roc_auc_score(Y[:,g],P_g[:,g]):.4f}  vs TTPRA={roc_auc_score(Y[:,g],P_t[:,g]):.4f}", flush=True)
auc_per_gene = mean_auc(Y, P_per_gene)
print(f"  Per-gene optimal alpha  MEAN AUC = {auc_per_gene:.4f}  (delta = {auc_per_gene - auc_g:+.4f})", flush=True)

# Save
res = {
    "ABMIL_5s5tta": auc_g,
    "TTPRA_only": auc_t,
    "best_global_alpha": float(best_alpha),
    "best_global_auc": float(best_auc),
    "per_gene_alpha": alphas.tolist(),
    "per_gene_auc": float(auc_per_gene),
    "S31_baseline": 0.7419,
    "S35_gigap_5s5tta": 0.7441,
    "S36_ensemble": 0.7444,
    "S39_CEPG_proof_of_concept": 0.7508,
    "n_cases": len(Y),
    "y_pos": Y.sum(0).tolist(),
    "common_cases": len(common),
}
with open(r"C:\Users\ziyao\uni2h_etl\s105_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\n[saved] s105_results.json", flush=True)
print("DONE", flush=True)