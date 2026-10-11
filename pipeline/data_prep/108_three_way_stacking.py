"""108_three_way_stacking.py - S41 3-way stacking: §39 CEPG + §36 ens + §35 5s5tta
per-gene weighted average with grid search alpha
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


# Load 3 OOF prediction sources
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
uni2h = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)

# Align common cases
common_g = list(gigap["common"])
common_u = list(uni2h["common"])
common = sorted(set(common_g) & set(common_u))
g_idx = {c: i for i, c in enumerate(common_g)}
u_idx = {c: i for i, c in enumerate(common_u)}
sel_g = np.array([g_idx[c] for c in common])
sel_u = np.array([u_idx[c] for c in common])
Y_full = gigap["Y"]
Y = Y_full[sel_g]

# Per-fold-seed OOF
fs_keys = {101: "fs0", 202: "fs1", 303: "fs2"}
P_gigap_fs = {fs: gigap[fs_keys[fs]] for fs in [101, 202, 303]}
P_uni2h_fs = {fs: uni2h[fs_keys[fs]] for fs in [101, 202, 303]}

# For §41, load CEPG alphas (from §39 §104b results)
with open(r"C:\Users\ziyao\uni2h_etl\s104b_results.json", "r", encoding="utf-8") as f:
    cepg_data = json.load(f)
cepg_alphas = {fs: np.array(cepg_data["per_fold_seed"][str(fs)]["alphas"]) for fs in [101, 202, 303]}

print("=== 3-way stacking: §35 gigap + §36 ens + §39 CEPG ===\n", flush=True)

results_per_fseed = {}
for fs in [101, 202, 303]:
    P_g = P_gigap_fs[fs][sel_g]  # (434, 6)
    P_u = P_uni2h_fs[fs][sel_u]    # (434, 6)
    alphas = cepg_alphas[fs]      # (6,)

    # Compute the 3 OOF predictions per gene:
    # 1) P_cepg = alphas * P_u + (1-alphas) * P_g (this IS §39)
    # 2) P_ens = (P_g + P_u) / 2 (this IS §36)
    # 3) P_gigap = P_g (this IS §35)
    P_cepg = alphas * P_u + (1 - alphas) * P_g
    P_ens = (P_g + P_u) / 2
    P_gigap = P_g

    # Per-gene weighted average of these 3 sources
    # Grid search 3 weights per gene, sum to 1
    best_per_gene = np.zeros_like(P_cepg)
    for g in range(len(GENES)):
        best_a = (1/3, 1/3, 1/3)
        best_auc = roc_auc_score(Y[:, g], P_cepg[:, g])
        for a1 in np.linspace(0, 1, 21):
            for a2 in np.linspace(0, 1 - a1, max(1, int((1 - a1) * 20) + 1)):
                a3 = 1 - a1 - a2
                if a3 < -1e-9:
                    continue
                P = a1 * P_cepg[:, g] + a2 * P_ens[:, g] + a3 * P_gigap[:, g]
                auc = roc_auc_score(Y[:, g], P)
                if auc > best_auc:
                    best_auc = auc
                    best_a = (a1, a2, a3)
        best_per_gene[:, g] = best_a[0] * P_cepg[:, g] + best_a[1] * P_ens[:, g] + best_a[2] * P_gigap[:, g]
        print(f"  fs{fs} gene {GENES[g]:6s}: weights={best_a[0]:.2f}/{best_a[1]:.2f}/{best_a[2]:.2f}  AUC={best_auc:.4f}", flush=True)

    # Compare per-source
    auc_cepg = mean_auc(Y, P_cepg)
    auc_ens = mean_auc(Y, P_ens)
    auc_gigap = mean_auc(Y, P_gigap)
    auc_3way = mean_auc(Y, best_per_gene)
    print(f"\n  fs{fs}: gigap={auc_gigap:.4f}  ens={auc_ens:.4f}  cepg={auc_cepg:.4f}  3way={auc_3way:.4f}\n", flush=True)
    results_per_fseed[fs] = {
        "gigap": auc_gigap, "ens": auc_ens, "cepg": auc_cepg, "3way": auc_3way,
    }

# Summary
print("\n=== Summary (3 fold-seeds) ===", flush=True)
for k in ["gigap", "ens", "cepg", "3way"]:
    aucs = [results_per_fseed[fs][k] for fs in [101, 202, 303]]
    print(f"  {k:6s}: mean={np.mean(aucs):.4f}  std={np.std(aucs, ddof=1):.4f}  per_fseed={[round(v,4) for v in aucs]}", flush=True)

mean_3way = np.mean([results_per_fseed[fs]["3way"] for fs in [101, 202, 303]])
std_3way = np.std([results_per_fseed[fs]["3way"] for fs in [101, 202, 303]], ddof=1)
mean_cepg = np.mean([results_per_fseed[fs]["cepg"] for fs in [101, 202, 303]])

print(f"\n  Δ(3way - cepg) = {mean_3way - mean_cepg:+.4f}", flush=True)
print(f"  Δ(3way - S31 baseline 0.7419) = {mean_3way - 0.7419:+.4f}", flush=True)

# Save
with open(r"C:\Users\ziyao\uni2h_etl\s108_results.json", "w", encoding="utf-8") as f:
    json.dump({"per_fold_seed": {str(k): v for k, v in results_per_fseed.items()},
               "mean_3way": float(mean_3way),
               "std_3way": float(std_3way),
               "mean_cepg": float(mean_cepg),
               "S31_baseline": 0.7419,
               "S36_ens": 0.7444,
               "S35_gigap_5s5tta": 0.7441,
               "S39_CEPG": 0.7506}, f, indent=2, ensure_ascii=False)
print(f"[saved] s108_results.json", flush=True)
print("DONE", flush=True)