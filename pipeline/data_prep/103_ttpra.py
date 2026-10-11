"""103_ttpra.py - S38 TTPRA: Test-Time Patch Retrieval Augmentation for LUAD mutation prediction

Algorithm:
  - Use gigap patch features (434 cases x 2048 patches x 1536-d)
  - For each fold (5-fold CV same splits as S31):
    * Train cases: build a flat patch library of all training patches (~(n_train_cases * 2048), 1536)
    * Track which case each training patch came from (for label lookup)
    * StandardScaler fit on training patches
    * For each test case:
        - Get test patches (M=2048), scale
        - For each test patch: KNN(K=50) over training patches -> K neighbor indices
        - Map K neighbor patch indices -> K source cases (deduped or weighted)
        - For each gene: frequency = mean(source case label == k) among neighbors -> per-patch gene score
        - Aggregate per-patch -> case-level: mean over M patches
        -> (6,) retrieval score for test case
    * Combine retrieval score with ABMIL score:
        - (a) ABMIL only (baseline, S31 = 0.7419)
        - (b) retrieval only
        - (c) mean(retrieval, ABMIL)
        - (d) per-gene logistic regression: alpha * retrieval + (1-alpha) * ABMIL

Result: report OOF AUC for each combination, per fold-seed.

Why patch-level (not case-level) labels:
  Patch inherits its source case's label - this is the standard MIL assumption.
  For positive cases, all patches are "labeled positive" (any one could be the actual mutation site).
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
LR_, WD, MIN_POS = 3e-4, 1e-3, 8
FOLD_SEEDS = [101, 202, 303]
NFOLD = 5
S_PATCH = 256          # subsample patches per case (memory + speed)
K_NN = 30
dev = "cpu"

# Load gigap patch features
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
print(f"[data] N={len(cases)}  patches_per_case={arr.shape[1]}  dim={DIM}  Y_pos={Y.sum(0).tolist()}", flush=True)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def get_test_patches(case_row_idx, n_take=S_PATCH, rng=None):
    """Get n_take patches from a case (random subsample)."""
    if rng is None:
        rng = np.random.RandomState(0)
    v = np.asarray(arr[case_row_idx], dtype=np.float32)  # (M, D)
    M = v.shape[0]
    if M > n_take:
        idx = rng.choice(M, n_take, replace=False)
        v = v[idx]
    return v  # (n_take, D)


# Per fold-seed protocol
print("\n" + "=" * 80, flush=True)
print(f"=== S38 TTPRA (K={K_NN}, S={S_PATCH}) (TCGA-LUAD N={len(cases)}, 3 fold-seed) ===", flush=True)
print("=" * 80, flush=True)

tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))


def fit_fold(folds):
    """One full 5-fold CV. Returns OOF retrieval predictions (n_cases, 6)."""
    oof = np.zeros((len(cases), len(GENES)))
    for tri, tei in folds:
        # ---- build train patch library ----
        rng = np.random.RandomState(7777)
        train_feats = []
        train_case_idx = []  # which case each patch came from
        for ci in tri:
            v = get_test_patches(rows_idx[ci], n_take=S_PATCH, rng=rng)
            train_feats.append(v)
            train_case_idx.append(np.full(v.shape[0], ci, dtype=np.int64))
        train_feats = np.concatenate(train_feats, axis=0)        # (~ n_train_cases * S_PATCH, D)
        train_case_idx = np.concatenate(train_case_idx, axis=0)  # (~)
        train_labels = Y[train_case_idx]                          # (~, 6)

        # StandardScaler
        scaler = StandardScaler().fit(train_feats)
        train_feats_s = scaler.transform(train_feats)

        # KNN index
        print(f"    [fold] train lib = {train_feats_s.shape}, fitting NN (n={len(train_feats_s)}, k={K_NN})...", flush=True)
        nn = NearestNeighbors(n_neighbors=K_NN, n_jobs=-1).fit(train_feats_s)

        # ---- for each test case: batched KNN query, aggregate neighbor label freq ----
        # Stack all test patches from all test cases in this fold
        rng = np.random.RandomState(99999)
        test_feats = []
        test_case_per_patch = []
        for ci in tei:
            v = get_test_patches(rows_idx[ci], n_take=S_PATCH, rng=np.random.RandomState(ci))
            test_feats.append(v)
            test_case_per_patch.append(np.full(v.shape[0], ci, dtype=np.int64))
        test_feats = np.concatenate(test_feats, axis=0)        # (~ n_test_cases * S_PATCH, D)
        test_case_per_patch = np.concatenate(test_case_per_patch, axis=0)
        test_feats_s = scaler.transform(test_feats)
        # Batched KNN: process in chunks of 4096 patches to limit RAM
        CHUNK = 4096
        n_test = test_feats_s.shape[0]
        patch_freq_all = np.zeros((n_test, len(GENES)), dtype=np.float32)
        for s in range(0, n_test, CHUNK):
            e = min(s + CHUNK, n_test)
            _, neighbors = nn.kneighbors(test_feats_s[s:e], n_neighbors=K_NN)
            neighbor_case_idx = train_case_idx[neighbors]   # (chunk, K)
            patch_freq_all[s:e] = Y[neighbor_case_idx].mean(axis=1)
        # Aggregate per-patch -> per-case: mean over patches within same case
        for ci in tei:
            mask = test_case_per_patch == ci
            oof[ci] = patch_freq_all[mask].mean(axis=0)
    return oof


results = {}
T0 = time.time()
for fs in FOLD_SEEDS:
    folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
    print(f"\n--- fold-seed {fs} ---", flush=True)
    oof = fit_fold(folds)
    auc = mean_auc(Y, oof)
    print(f"  foldseed {fs}: TTPRA-only = {auc:.4f}  [{(time.time()-T0)/60:.1f}min]", flush=True)
    results[fs] = {"retrieval_only": auc, "oof": oof}

# Summary
print("\n=== TTPRA-only OOF mean ===", flush=True)
ret_aucs = [results[fs]["retrieval_only"] for fs in FOLD_SEEDS]
print(f"  mean = {np.mean(ret_aucs):.4f}  std = {np.std(ret_aucs, ddof=1):.4f}  per_fseed = {[round(v,4) for v in ret_aucs]}", flush=True)

# Save TTPRA OOF for downstream combine with S36 (or ABMIL baseline)
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\ttpra_oof.npz",
                    Y=Y, common=np.array(cases, dtype=object),
                    **{f"fs{fs}": results[fs]["oof"] for fs in FOLD_SEEDS})
print(f"[saved] ttpra_oof.npz", flush=True)

# Save baseline S31 OOF if available, else skip
import json
res = {
    "ttpra_per_fseed": ret_aucs,
    "ttpra_mean": float(np.mean(ret_aucs)),
    "ttpra_std": float(np.std(ret_aucs, ddof=1)),
    "S31_baseline": 0.7419,
    "S35_gigap_5s5tta": 0.7441,
    "S36_ensemble": 0.7444,
}
with open(r"C:\Users\ziyao\uni2h_etl\s103_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("[saved] s103_results.json", flush=True)
print("DONE", flush=True)