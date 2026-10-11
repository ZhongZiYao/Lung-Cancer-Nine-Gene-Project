"""103b_ttpra_fs101.py - quick TTPRA on fold-seed 101 only, smaller S=128"""
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
S_PATCH = 128
K_NN = 30

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

def get_test_patches(case_row_idx, n_take=S_PATCH, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    v = np.asarray(arr[case_row_idx], dtype=np.float32)
    M = v.shape[0]
    if M > n_take:
        idx = rng.choice(M, n_take, replace=False)
        v = v[idx]
    return v

tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=101).split(np.zeros(len(cases)), tier)]

print(f"\n=== S38b TTPRA (S={S_PATCH}, K={K_NN}) fold-seed 101 only ===", flush=True)
T0 = time.time()

oof = np.zeros((len(cases), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    print(f"\n--- fold {fold_idx}/5  train_n={len(tri)} test_n={len(tei)}  elapsed {(time.time()-T0)/60:.1f}min ---", flush=True)
    rng = np.random.RandomState(7777)
    train_feats = []
    train_case_idx = []
    for ci in tri:
        v = get_test_patches(rows_idx[ci], n_take=S_PATCH, rng=rng)
        train_feats.append(v)
        train_case_idx.append(np.full(v.shape[0], ci, dtype=np.int64))
    train_feats = np.concatenate(train_feats, axis=0)
    train_case_idx = np.concatenate(train_case_idx, axis=0)
    train_labels = Y[train_case_idx]
    print(f"  train lib = {train_feats.shape}", flush=True)

    scaler = StandardScaler().fit(train_feats)
    train_feats_s = scaler.transform(train_feats)

    t0_nn = time.time()
    nn = NearestNeighbors(n_neighbors=K_NN, n_jobs=1).fit(train_feats_s)
    print(f"  NN build: {time.time()-t0_nn:.1f}s", flush=True)

    test_feats = []
    test_case_per_patch = []
    for ci in tei:
        v = get_test_patches(rows_idx[ci], n_take=S_PATCH, rng=np.random.RandomState(ci))
        test_feats.append(v)
        test_case_per_patch.append(np.full(v.shape[0], ci, dtype=np.int64))
    test_feats = np.concatenate(test_feats, axis=0)
    test_case_per_patch = np.concatenate(test_case_per_patch, axis=0)
    test_feats_s = scaler.transform(test_feats)

    t0_q = time.time()
    CHUNK = 1024
    n_test = test_feats_s.shape[0]
    patch_freq_all = np.zeros((n_test, len(GENES)), dtype=np.float32)
    for s in range(0, n_test, CHUNK):
        e = min(s + CHUNK, n_test)
        _, neighbors = nn.kneighbors(test_feats_s[s:e], n_neighbors=K_NN)
        neighbor_case_idx = train_case_idx[neighbors]
        patch_freq_all[s:e] = Y[neighbor_case_idx].mean(axis=1)
    print(f"  NN query {n_test} patches: {time.time()-t0_q:.1f}s", flush=True)

    for ci in tei:
        mask = test_case_per_patch == ci
        oof[ci] = patch_freq_all[mask].mean(axis=0)

auc = mean_auc(Y, oof)
print(f"\n=== TTPRA fold-seed 101 OOF AUC = {auc:.4f}  (total elapsed {(time.time()-T0)/60:.1f}min) ===", flush=True)

# Per-gene
for g in range(len(GENES)):
    print(f"  {GENES[g]:6s}: AUC={roc_auc_score(Y[:, g], oof[:, g]):.4f}")

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\ttpra_oof_fs101.npz",
                    fs101=Oof if False else oof, Y=Y, common=np.array(cases, dtype=object))
print(f"[saved]\nOOF saved to ttpra_oof_fs101.npz", flush=True)
print("DONE", flush=True)