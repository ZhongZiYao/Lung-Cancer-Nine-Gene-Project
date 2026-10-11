"""106_ttpra_case_level.py - Case-level TTPRA (mean feature per case, KNN to find similar cases)

Hypothesis: patch-level KNN failed because patches within the same WSI are too similar,
causing K=30 neighbors to be ~95% from the same case (which is not in train but cases with
similar morphology share the same gene labels). With case-level mean features, KNN genuinely
finds cross-patient similar cases.

Method:
  - Per case: compute mean patch feature (case-level representation)
  - For test case: KNN over train case mean features (K=10, 20)
  - Aggregate neighbor case labels (with optional distance weighting)
  - Compare to patch-level TTPRA
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
NFOLD = 5
N_PATCH_PER_CASE = 512  # subsample for speed
K_NN = 20

arr = np.load(os.path.join(C, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
DIM = arr.shape[2]
ix = pd.read_csv(os.path.join(C, "NSCLC_gigap_luad_sub2048_index.csv"))
ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
cases = sorted(c for c in ix.case_id if c in lab.index)
rm = dict(zip(ix.case_id, ix.mem_row))
rows_idx = np.array([rm[c] for c in cases])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
print(f"[data] N={len(cases)}  dim={DIM}  Y_pos={Y.sum(0).tolist()}", flush=True)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def case_mean_feature(case_row_idx, n_take=N_PATCH_PER_CASE, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    v = np.asarray(arr[case_row_idx], dtype=np.float32)
    M = v.shape[0]
    if M > n_take:
        idx = rng.choice(M, n_take, replace=False)
        v = v[idx]
    return v.mean(axis=0)  # (D,)


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=101).split(np.zeros(len(cases)), tier)]

print(f"\n=== Case-level TTPRA (N={N_PATCH_PER_CASE}, K={K_NN}) fold-seed 101 ===", flush=True)
T0 = time.time()

oof = np.zeros((len(cases), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    print(f"\n--- fold {fold_idx}/5  train_n={len(tri)} test_n={len(tei)} ---", flush=True)
    rng = np.random.RandomState(7777)

    # Case mean features
    train_case_feats = np.stack([case_mean_feature(rows_idx[ci], n_take=N_PATCH_PER_CASE, rng=rng) for ci in tri])
    test_case_feats = np.stack([case_mean_feature(rows_idx[ci], n_take=N_PATCH_PER_CASE, rng=np.random.RandomState(ci)) for ci in tei])
    print(f"  case features: train {train_case_feats.shape}, test {test_case_feats.shape}", flush=True)

    scaler = StandardScaler().fit(train_case_feats)
    train_feats_s = scaler.transform(train_case_feats)
    test_feats_s = scaler.transform(test_case_feats)

    # KNN over case mean features
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=K_NN, n_jobs=1).fit(train_feats_s)
    distances, neighbors = nn.kneighbors(test_feats_s, n_neighbors=K_NN, return_distance=True)
    print(f"  NN query: {time.time()-t0:.1f}s", flush=True)

    # Aggregate neighbor case labels (with optional distance weighting)
    for i, ci in enumerate(tei):
        neighbor_case_idx = np.array(tri)[neighbors[i]]
        dist = distances[i]
        # Distance-based: w = 1 / (dist + 1e-6), then normalize
        w = 1.0 / (dist + 1e-6)
        w /= w.sum()
        # Weighted neighbor label frequency
        case_pred = np.zeros(len(GENES))
        for k in range(K_NN):
            case_pred += w[k] * Y[neighbor_case_idx[k]]
        oof[ci] = case_pred

print("\n=== Case-level TTPRA OOF (uniform weight) ===", flush=True)
auc_u = mean_auc(Y, oof)
print(f"  uniform weight  = {auc_u:.4f}", flush=True)

# Also try simple unweighted average
oof2 = np.zeros((len(cases), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    rng = np.random.RandomState(7777)
    train_case_feats = np.stack([case_mean_feature(rows_idx[ci], n_take=N_PATCH_PER_CASE, rng=rng) for ci in tri])
    test_case_feats = np.stack([case_mean_feature(rows_idx[ci], n_take=N_PATCH_PER_CASE, rng=np.random.RandomState(ci)) for ci in tei])
    scaler = StandardScaler().fit(train_case_feats)
    train_feats_s = scaler.transform(train_case_feats)
    test_feats_s = scaler.transform(test_case_feats)
    nn = NearestNeighbors(n_neighbors=K_NN, n_jobs=1).fit(train_feats_s)
    _, neighbors = nn.kneighbors(test_case_feats, n_neighbors=K_NN)
    for i, ci in enumerate(tei):
        neighbor_case_idx = np.array(tri)[neighbors[i]]
        oof2[ci] = Y[neighbor_case_idx].mean(axis=0)

auc_simple = mean_auc(Y, oof2)
print(f"  simple (1/K) avg = {auc_simple:.4f}", flush=True)

# Per-gene AUC
print("\n=== Per-gene (uniform weight) ===", flush=True)
for g in range(len(GENES)):
    print(f"  {GENES[g]:6s}: AUC={roc_auc_score(Y[:, g], oof[:, g]):.4f}", flush=True)

# Save
import json
res = {
    "case_level_uniform": float(auc_u),
    "case_level_simple": float(auc_simple),
    "patch_level_103b": 0.6123,
    "ABMIL_baseline_103b": 0.7419,
    "S36_ensemble": 0.7444,
    "S39_CEPG": 0.7506,
    "n_cases": len(Y),
    "y_pos": Y.sum(0).tolist(),
    "K_NN": K_NN,
    "N_PATCH_PER_CASE": N_PATCH_PER_CASE,
}
with open(r"C:\Users\ziyao\uni2h_etl\s106_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\n[saved] s106_results.json", flush=True)
print(f"total elapsed {(time.time()-T0)/60:.1f}min", flush=True)
print("DONE", flush=True)