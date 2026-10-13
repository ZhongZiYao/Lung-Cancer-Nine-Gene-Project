"""107c_proto_v3.py - Prototype MIL with K-means init + T=1.0 (single fold-seed proof of concept)

Fix the bad T=0.1 + random init by:
1. K-means init prototypes from train patches (per gene: K=8 prototypes = cluster centers)
   Wait - K-means on patches across all genes won't give gene-specific prototypes.
   Use gene-positive-patches K-means instead.
2. T=1.0 (less sharp)
3. Larger initial lr warmup
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C_dir = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
FOLD_SEED = 101
NFOLD = 5
N_PATCH = 256
K_PROTO = 8
DIM_H = 256
LR_, WD = 1e-3, 1e-3
EPOCH = 50
BATCH = 32
T = 1.0

dev = "cuda" if torch.cuda.is_available() else "cpu"

arr = np.load(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
DIM = arr.shape[2]
ix = pd.read_csv(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048_index.csv"))
ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
cases = sorted(c for c in ix.case_id if c in lab.index)
rm = dict(zip(ix.case_id, ix.mem_row))
rows_idx = np.array([rm[c] for c in cases])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
print(f"[data] N={len(cases)}  dim={DIM}  Y_pos={Y.sum(0).tolist()}", flush=True)

print("[load] reading all patches into memory...", flush=True)
case_patches = np.zeros((len(cases), 2048, DIM), dtype=np.float32)
for i, ci in enumerate(cases):
    case_patches[i] = arr[rows_idx[i]][:2048]
print(f"[load] done: {case_patches.nbytes/1e9:.1f} GB", flush=True)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def init_prototypes_per_gene(tri, Y, K=K_PROTO, max_per_gene=200, rng=None):
    """K-means per gene on positive training patches (truncated to manageable size)."""
    if rng is None:
        rng = np.random.RandomState(42)
    protos = np.zeros((len(GENES), K, DIM_H), dtype=np.float32)
    trunk = nn.Sequential(nn.Linear(DIM, DIM_H), nn.ReLU()).to(dev)
    # Build (case_idx, patch_idx) for every train patch in z
    case_id_arr = []
    patch_id_arr = []
    for ci_idx, ci in enumerate(tri):
        n_p = case_patches[ci].shape[0]
        n_take = min(N_PATCH, n_p)
        patch_id_arr.append(rng.choice(n_p, n_take, replace=False))
        case_id_arr.append(np.full(n_take, ci_idx, dtype=np.int64))
    case_id_arr = np.concatenate(case_id_arr)  # (N_tr,) index into Y[tri]
    patch_id_arr = np.concatenate(patch_id_arr)
    train_patch_flat = np.stack([case_patches[ci][patch_id_arr[r]] for r, ci in enumerate(tri[case_id_arr])])
    print(f"    init: {train_patch_flat.shape[0]} train patches loaded for prototype init", flush=True)
    with torch.no_grad():
        z = trunk(torch.from_numpy(train_patch_flat).to(dev)).cpu().numpy()
    print(f"    init: z.shape={z.shape}, case_id_arr.shape={case_id_arr.shape}", flush=True)
    for g in range(len(GENES)):
        # take patches from positive training cases
        pos_mask = (Y[tri][case_id_arr, g] == 1)
        pos_z = z[pos_mask]
        if pos_z.shape[0] >= K:
            km = KMeans(n_clusters=K, random_state=0, n_init=3).fit(pos_z[:max_per_gene])
            protos[g] = km.cluster_centers_
        else:
            protos[g] = z[rng.choice(z.shape[0], K, replace=True)]
    return protos, trunk


class ProtoNet(nn.Module):
    def __init__(s, d_in, n_genes, k, h, drop):
        super().__init__()
        s.k = k
        s.n_genes = n_genes
        s.trunk = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop))
        s.prototypes = nn.Parameter(torch.randn(n_genes, k, h) * 0.05)
        s.gene_bias = nn.Parameter(torch.zeros(n_genes))

    def forward(s, x):
        B, N, _ = x.shape
        z = s.trunk(x.reshape(B * N, -1)).reshape(B, N, -1)
        z_norm = (z ** 2).sum(-1, keepdim=True)
        p_norm = (s.prototypes ** 2).sum(-1).reshape(1, 1, s.n_genes, s.k)
        cross = torch.einsum("bnh,gkh->bngk", z, s.prototypes)
        dist = z_norm.unsqueeze(-1) + p_norm - 2 * cross
        log_p_gk = -dist / T
        log_p_gk = log_p_gk - log_p_gk.max(dim=-1, keepdim=True).values
        log_p_gx = log_p_gk.logsumexp(dim=-1)
        log_p_gX = log_p_gx.mean(dim=1)
        return log_p_gX + s.gene_bias


def fit(tr_rows, tr_y, seed, init_p):
    torch.manual_seed(seed)
    net = ProtoNet(DIM, len(GENES), K_PROTO, DIM_H, 0.25).to(dev)
    with torch.no_grad():
        net.prototypes.data = torch.from_numpy(init_p).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=EPOCH, steps_per_epoch=max(1, len(tr_rows) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(EPOCH):
        net.train()
        perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            r = perm[i:i + BATCH]
            xb = torch.from_numpy(np.stack([case_patches[tr_rows[ri]][rng.choice(min(N_PATCH, case_patches[tr_rows[ri]].shape[0]), N_PATCH, replace=case_patches[tr_rows[ri]].shape[0] < N_PATCH)] for ri in r])).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, te_rows, seed):
    net.eval()
    rng = np.random.RandomState(seed * 1000)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = torch.from_numpy(np.stack([case_patches[te_rows[ri]][rng.choice(min(N_PATCH, case_patches[te_rows[ri]].shape[0]), N_PATCH, replace=case_patches[te_rows[ri]].shape[0] < N_PATCH)] for ri in range(len(te_rows))])).to(dev, torch.float32)
        return net(xb).cpu().numpy()


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(cases)), tier)]

print(f"\n=== S40v3 Prototype MIL (K={K_PROTO}, T={T}, K-means init, EPOCH={EPOCH}) fold-seed {FOLD_SEED} ===", flush=True)
T0 = time.time()

oof = np.zeros((len(cases), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    print(f"\n--- fold {fold_idx}/5 ---", flush=True)
    print("  init prototypes via K-means...", flush=True)
    init_p, _ = init_prototypes_per_gene(tri, Y, K=K_PROTO)

    preds = []
    for s in [0, 1, 2]:
        net = fit(tri, Y[tri], s, init_p)
        net.eval()
        rng = np.random.RandomState(s * 1000)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            xb = torch.from_numpy(np.stack([case_patches[ci][rng.choice(min(N_PATCH, case_patches[ci].shape[0]), N_PATCH, replace=case_patches[ci].shape[0] < N_PATCH)] for ci in tei])).to(dev, torch.float32)
            preds.append(net(xb).cpu().numpy())
    oof[tei] = np.mean(preds, axis=0)
    fold_auc = mean_auc(Y[tei], oof[tei])
    print(f"  fold {fold_idx} OOF AUC = {fold_auc:.4f}  ({(time.time()-T0)/60:.1f}min)", flush=True)

auc = mean_auc(Y, oof)
print(f"\n=== Final fold-seed {FOLD_SEED} OOF AUC = {auc:.4f} ===", flush=True)

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\proto_v3_oof.npz", oof=oof, Y=Y, common=np.array(cases, dtype=object))
print(f"[saved] proto_v3_oof.npz", flush=True)
print("DONE", flush=True)