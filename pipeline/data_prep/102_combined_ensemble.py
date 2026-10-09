"""102_combined_ensemble.py - S36: combine §35 gigap 5s5tta + §36 uni2h 5s5tta via logit avg + LR meta

Both npz files have fs0/1/2 OOF prediction per fold-seed.
Common case set: 434 (from §32 intersection).

Outputs:
  - gigap 5s5tta alone (sanity reproduce §35)
  - uni2h 5s5tta alone (sanity reproduce §36, but only on common 434)
  - logit avg (gigap+uni2h)/2
  - per-gene LR meta (input = 2 features per gene from each model)
  - 3-encoder (gigap+uni2h+uni_v1) 5-fold OOF avg from §99 (1 fold-seed only)
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8

def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))

# Load §35 gigap 5s5tta OOF
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
Y_g = gigap["Y"]
common_g = list(gigap["common"])
oof_gigap_fs = [gigap["fs0"], gigap["fs1"], gigap["fs2"]]

# Load §36 uni2h 5s5tta OOF
uni2h = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)
Y_u = uni2h["Y"]
common_u = list(uni2h["common"])
oof_uni2h_fs = [uni2h["fs0"], uni2h["fs1"], uni2h["fs2"]]

# Find common cases (intersection)
common = sorted(set(common_g) & set(common_u))
print(f"[intersection] {len(common)} common cases (gigap: {len(common_g)}, uni2h: {len(common_u)})", flush=True)

# Index maps
g_idx = {c: i for i, c in enumerate(common_g)}
u_idx = {c: i for i, c in enumerate(common_u)}
sel = np.array([g_idx[c] for c in common])  # for gigap
sel_u = np.array([u_idx[c] for c in common])  # for uni2h

# Get Y
Y = np.array([Y_g[g_idx[c]] for c in common])
print(f"[Y] pos = {Y.sum(0).tolist()}", flush=True)

# Per-fold-seed: extract common-case subset
g_fs = [oof_gigap_fs[i][sel] for i in range(3)]
u_fs = [oof_uni2h_fs[i][sel_u] for i in range(3)]

# Per-fold-seed AUC
print("\n=== Per-fold-seed single-encoder (on 434 common) ===", flush=True)
for i in range(3):
    a_g = mean_auc(Y, g_fs[i])
    a_u = mean_auc(Y, u_fs[i])
    print(f"  foldseed {i}: gigap={a_g:.4f}  uni2h={a_u:.4f}", flush=True)

# Per-fold-seed ensemble
print("\n=== Per-fold-seed logit average (gigap+uni2h)/2 ===", flush=True)
ens_fs = [(g + u) / 2 for g, u in zip(g_fs, u_fs)]
ens_aucs = [mean_auc(Y, p) for p in ens_fs]
for i, a in enumerate(ens_aucs):
    print(f"  foldseed {i}: ensemble = {a:.4f}  (vs gigap {mean_auc(Y, g_fs[i]):.4f}, vs uni2h {mean_auc(Y, u_fs[i]):.4f})", flush=True)

# Summary
print("\n=== OOF mean (3 fold-seeds) ===", flush=True)
g_aucs = [mean_auc(Y, g) for g in g_fs]
u_aucs = [mean_auc(Y, u) for u in u_fs]
print(f"  gigap (on common 434)  mean = {np.mean(g_aucs):.4f}  std = {np.std(g_aucs, ddof=1):.4f}  per_fseed = {[round(v,4) for v in g_aucs]}", flush=True)
print(f"  uni2h (on common 434) mean = {np.mean(u_aucs):.4f}  std = {np.std(u_aucs, ddof=1):.4f}  per_fseed = {[round(v,4) for v in u_aucs]}", flush=True)
print(f"  ensemble (gigap+uni2h)/2 mean = {np.mean(ens_aucs):.4f}  std = {np.std(ens_aucs, ddof=1):.4f}  per_fseed = {[round(v,4) for v in ens_aucs]}", flush=True)
print(f"  S31 baseline (gigap 3s1tta, 434) = 0.7419", flush=True)
print(f"  S35 §35 (gigap 5s5tta, 434)   = 0.7441", flush=True)

# Delta analysis
print("\n=== Paired Delta ===", flush=True)
deltas = [e - g for e, g in zip(ens_aucs, g_aucs)]
print(f"  ens - gigap    mean = {np.mean(deltas):+.4f}  per_fseed = {[round(v,4) for v in deltas]}  same_sign = {sum(1 for v in deltas if v>0)}/3", flush=True)

# Per-gene LR meta
print("\n=== Per-gene LR meta (gigap + uni2h, 5-fold CV) ===", flush=True)
# Use the same FOLDS (101,202,303) for OOF
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=101).split(np.zeros(len(common)), tier)]

# Use ensemble predictions as input, train a small LR per gene
P_meta = np.zeros((len(common), len(GENES)))
for g in range(len(GENES)):
    X_meta = np.stack([g_fs[0][:, g], u_fs[0][:, g]], axis=1)  # use fs0 only for the meta split
    P_g = np.zeros(len(common))
    for tri, tei in FOLDS:
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(X_meta[tri], Y[tri, g])
        P_g[tei] = lr.predict_proba(X_meta[tei])[:, 1]
    P_meta[:, g] = P_g
print(f"  LR meta (gigap+uni2h)  OOF AUC = {mean_auc(Y, P_meta):.4f}", flush=True)

# Save
import json
res = {
    "common_n": len(common),
    "gigap_5s5tta_per_fseed": g_aucs,
    "uni2h_5s5tta_per_fseed": u_aucs,
    "ensemble_per_fseed": ens_aucs,
    "gigap_mean": float(np.mean(g_aucs)),
    "uni2h_mean": float(np.mean(u_aucs)),
    "ensemble_mean": float(np.mean(ens_aucs)),
    "ensemble_delta_vs_gigap": float(np.mean(deltas)),
    "lr_meta_auc": float(mean_auc(Y, P_meta)),
    "y_pos": Y.sum(0).tolist(),
}
with open(r"C:\Users\ziyao\uni2h_etl\s102_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"[saved] s102_results.json", flush=True)
print("DONE", flush=True)
