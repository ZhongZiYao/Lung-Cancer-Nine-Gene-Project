"""112_combine_cea_cepg.py - S44: Combine §43c CEA v3 OOF + §39 CEPG OOF

Hypothesis: CEA v3 captures different signal than CEPG (cross-encoder attention vs per-gene alpha),
so combining may give +Δ over either alone.
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


# Load §43c CEA v3 OOF
cea = np.load(r"C:\Users\ziyao\uni2h_etl\cea_v3_oof.npz", allow_pickle=True)
P_cea = cea["oof"]  # (434, 6)
Y = cea["Y"]
common = list(cea["common"])

# Load §39 CEPG OOF (from §104b per fold-seed 101)
with open(r"C:\Users\ziyao\uni2h_etl\s104b_results.json", "r", encoding="utf-8") as f:
    cepg_data = json.load(f)
alphas = np.array(cepg_data["per_fold_seed"]["101"]["alphas"])

# Need §35 gigap 5s5tta + §36 uni2h 5s5tta OOF for fold-seed 101
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
uni2h = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)

# Align to common
common_g = list(gigap["common"])
common_u = list(uni2h["common"])
g_idx = {c: i for i, c in enumerate(common_g)}
u_idx = {c: i for i, c in enumerate(common_u)}
sel_g = np.array([g_idx[c] for c in common])
sel_u = np.array([u_idx[c] for c in common])

P_g = gigap["fs0"][sel_g]
P_u2 = uni2h["fs0"][sel_u]
P_cepg = alphas * P_u2 + (1 - alphas) * P_g  # (434, 6)

# Combine CEA + CEPG
print("Per-source OOF AUC (fs101):", flush=True)
print(f"  CEA v3 = {mean_auc(Y, P_cea):.4f}", flush=True)
print(f"  CEPG    = {mean_auc(Y, P_cepg):.4f}", flush=True)

# Grid search per-gene weight between CEA and CEPG
best_oof = np.zeros_like(P_cea)
alphas_combine = np.zeros(len(GENES))
for g in range(len(GENES)):
    best_a = 0.5
    best_auc = roc_auc_score(Y[:, g], 0.5 * P_cea[:, g] + 0.5 * P_cepg[:, g])
    for a in np.linspace(0, 1, 21):
        P = a * P_cea[:, g] + (1 - a) * P_cepg[:, g]
        auc = roc_auc_score(Y[:, g], P)
        if auc > best_auc:
            best_auc = auc
            best_a = a
    alphas_combine[g] = best_a
    best_oof[:, g] = best_a * P_cea[:, g] + (1 - best_a) * P_cepg[:, g]
    print(f"  {GENES[g]:6s}: alpha_cea={best_a:.2f}  AUC={best_auc:.4f}  (CEA={roc_auc_score(Y[:,g],P_cea[:,g]):.4f}  CEPG={roc_auc_score(Y[:,g],P_cepg[:,g]):.4f})", flush=True)

auc_combine = mean_auc(Y, best_oof)
print(f"\n=== Combined CEA + CEPG (per-gene alpha) ===", flush=True)
print(f"  OOF AUC = {auc_combine:.4f}", flush=True)
print(f"  Δ vs CEPG alone = {auc_combine - mean_auc(Y, P_cepg):+.4f}", flush=True)
print(f"  Δ vs S31 baseline 0.7419 = {auc_combine - 0.7419:+.4f}", flush=True)

# Also try uniform average
P_uniform = (P_cea + P_cepg) / 2
auc_uniform = mean_auc(Y, P_uniform)
print(f"\n  Uniform average: {auc_uniform:.4f}  (Δ vs CEPG = {auc_uniform - mean_auc(Y, P_cepg):+.4f})", flush=True)

# Save
import json
res_out = {
    "fs101_cea_v3": float(mean_auc(Y, P_cea)),
    "fs101_cepg": float(mean_auc(Y, P_cepg)),
    "fs101_combine_uniform": float(auc_uniform),
    "fs101_combine_per_gene_alpha": float(auc_combine),
    "combine_per_gene_alpha": alphas_combine.tolist(),
    "S31_baseline": 0.7419,
    "S39_CEPG_3fs_mean": 0.7506,
}
with open(r"C:\Users\ziyao\uni2h_etl\s112_results.json", "w", encoding="utf-8") as f:
    json.dump(res_out, f, indent=2, ensure_ascii=False)
print(f"\n[saved] s112_results.json", flush=True)
print("DONE", flush=True)