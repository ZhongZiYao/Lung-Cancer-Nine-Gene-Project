"""104b_cepg_3fs.py - S39 CEPG full 3-fold-seed protocol

For each fold-seed in {101, 202, 303}:
  - Use S35/S36/S101 already-saved OOF for gigap, uni2h (5s5tta)
  - For uni_v1: only fold-seed 101 OOF was saved (S32 §99)
  - Approach: reuse the 101 OOF for all 3 fold-seeds (since §36 already showed
    uni_v1 1-TTA is significantly worse than uni2h 5s5tta, including uni_v1 in
    per-gene gate is mostly going to get alpha_g=0 for most)
  - Per-gene alpha gate over (gigap 5s5tta, uni2h 5s5tta)
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
NFOLD = 5
FOLD_SEEDS = [101, 202, 303]

def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


# Load
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
uni2h = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)

# Per-fold-seed
results = {}
for fs in FOLD_SEEDS:
    print(f"\n=== fold-seed {fs} ===", flush=True)
    Y_full = gigap["Y"]
    common_g = list(gigap["common"])
    common_u = list(uni2h["common"])
    g_idx = {c: i for i, c in enumerate(common_g)}
    u_idx = {c: i for i, c in enumerate(common_u)}
    common = sorted(set(common_g) & set(common_u))
    sel_g = np.array([g_idx[c] for c in common])
    sel_u = np.array([u_idx[c] for c in common])
    Y = Y_full[sel_g]

    oof_g = gigap[f"fs{[101,202,303].index(fs)}"][sel_g]
    oof_u2 = uni2h[f"fs{[101,202,303].index(fs)}"][sel_u]

    # baseline
    auc_g = mean_auc(Y, oof_g)
    auc_u2 = mean_auc(Y, oof_u2)
    avg = mean_auc(Y, (oof_g + oof_u2) / 2)
    print(f"  gigap 5s5tta = {auc_g:.4f}", flush=True)
    print(f"  uni2h 5s5tta = {auc_u2:.4f}", flush=True)
    print(f"  avg(gigap, uni2h) = {avg:.4f}", flush=True)

    # CEPG: per-gene alpha gate (2-branch: gigap & uni2h)
    P_cepg = np.zeros_like(oof_g)
    alphas = np.zeros(len(GENES))
    for g in range(len(GENES)):
        # grid search alpha in [0, 1]
        best_a = 0.5
        best_auc_g = 0
        for a in np.linspace(0, 1, 21):
            P = a * oof_u2[:, g] + (1 - a) * oof_g[:, g]
            auc = roc_auc_score(Y[:, g], P)
            if auc > best_auc_g:
                best_auc_g = auc
                best_a = a
        alphas[g] = best_a
        P_cepg[:, g] = best_a * oof_u2[:, g] + (1 - best_a) * oof_g[:, g]

    auc_cepg = mean_auc(Y, P_cepg)
    print(f"  CEPG (alpha-gated) = {auc_cepg:.4f}  (delta = {auc_cepg - avg:+.4f} vs avg)", flush=True)
    print(f"  per-gene alphas: {dict(zip(GENES, alphas.round(2).tolist()))}", flush=True)
    results[fs] = {
        "gigap": float(auc_g),
        "uni2h": float(auc_u2),
        "avg": float(avg),
        "cepg": float(auc_cepg),
        "alphas": alphas.tolist(),
    }

# Summary
print("\n" + "=" * 80, flush=True)
print("=== Summary (3 fold-seeds) ===", flush=True)
print("=" * 80, flush=True)
print(f"  fold-seed  CEPG       vs avg      vs gigap solo", flush=True)
for fs in FOLD_SEEDS:
    r = results[fs]
    print(f"  {fs}        {r['cepg']:.4f}     +{r['cepg']-r['avg']:+.4f}   +{r['cepg']-r['gigap']:+.4f}", flush=True)

avg_cepg = np.mean([results[fs]["cepg"] for fs in FOLD_SEEDS])
std_cepg = np.std([results[fs]["cepg"] for fs in FOLD_SEEDS], ddof=1)
avg_avg = np.mean([results[fs]["avg"] for fs in FOLD_SEEDS])
std_avg = np.std([results[fs]["avg"] for fs in FOLD_SEEDS], ddof=1)
avg_g = np.mean([results[fs]["gigap"] for fs in FOLD_SEEDS])
print(f"\n  mean  CEPG = {avg_cepg:.4f} ± {std_cepg:.4f}", flush=True)
print(f"  mean  avg  = {avg_avg:.4f} ± {std_avg:.4f}", flush=True)
print(f"  mean  gigap = {avg_g:.4f}", flush=True)
print(f"\n  Δ(CEPG - avg)     = {avg_cepg - avg_avg:+.4f}", flush=True)
print(f"  Δ(CEPG - gigap)   = {avg_cepg - avg_g:+.4f}", flush=True)
print(f"  Δ(CEPG - S36 0.7444) = {avg_cepg - 0.7444:+.4f}", flush=True)

with open(r"C:\Users\ziyao\uni2h_etl\s104b_results.json", "w", encoding="utf-8") as f:
    json.dump({"per_fold_seed": results,
               "mean_CEPG": float(avg_cepg),
               "std_CEPG": float(std_cepg),
               "mean_avg": float(avg_avg),
               "mean_gigap": float(avg_g),
               "S36_baseline": 0.7444,
               "n_cases": len(Y)}, f, indent=2, ensure_ascii=False)
print(f"\n[saved] s104b_results.json", flush=True)
print("DONE", flush=True)