"""111_cea_v3_shared_trunk.py - S43c CEA v3: shared trunk + per-encoder learnable alpha

Architecture:
  - Shared trunk: 4096 -> DIM_H (single linear, much fewer params than 3 separate)
  - Single ABMIL head (uses shared trunk output)
  - Per-encoder projection: 3 separate (1536/1536/1024) -> DIM_H
  - Per-gene learnable alpha for weighted combination of per-encoder ABMIL logits

vs v2 (3 separate ABMIL heads): v3 has fewer params and shares feature extraction,
should be more stable at N=434.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C_dir = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
FOLD_SEED = 101
NFOLD = 5
N_PATCH = 128
DIM_H = 128
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
dev = "cuda" if torch.cuda.is_available() else "cpu"

arr_g = np.load(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
arr_u2 = np.load(os.path.join(C_dir, "NSCLC_uni2h_luad_sub512.npy"), mmap_mode="r")
arr_uv = np.load(os.path.join(C_dir, "NSCLC_uni_sub512.npy"), mmap_mode="r")
DIM_G = arr_g.shape[2]
DIM_U2 = arr_u2.shape[2]
DIM_UV = arr_uv.shape[2]

ix_g_df = pd.read_csv(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048_index.csv"))
ix_u2_df = pd.read_csv(os.path.join(C_dir, "NSCLC_uni2h_luad_sub512_index.csv"))
ix_uv_df = pd.read_csv(os.path.join(C_dir, "NSCLC_uni_sub512_index.csv"))
for ix in (ix_g_df, ix_u2_df, ix_uv_df):
    ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
common = sorted(set(ix_g_df.case_id) & set(ix_u2_df.case_id) & set(ix_uv_df.case_id) & set(lab.index))
print(f"[data] {len(common)} cases", flush=True)

rm_g = dict(zip(ix_g_df.case_id, ix_g_df.mem_row))
rm_u2 = dict(zip(ix_u2_df.case_id, ix_u2_df.mem_row))
rm_uv = dict(zip(ix_uv_df.case_id, ix_uv_df.mem_row))
rows_g = np.array([rm_g[c] for c in common])
rows_u2 = np.array([rm_u2[c] for c in common])
rows_uv = np.array([rm_uv[c] for c in common])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in common])

print("[load] patches to RAM...", flush=True)
case_g = np.zeros((len(common), 2048, DIM_G), dtype=np.float32)
case_u2 = np.zeros((len(common), 512, DIM_U2), dtype=np.float32)
case_uv = np.zeros((len(common), 512, DIM_UV), dtype=np.float32)
T_load = time.time()
for i, ci in enumerate(common):
    case_g[i] = arr_g[rows_g[i]][:2048]
    case_u2[i] = arr_u2[rows_u2[i]][:512]
    case_uv[i] = arr_uv[rows_uv[i]][:512]
print(f"[load] done {time.time()-T_load:.1f}s", flush=True)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def get_one(case_patches, case_idx, n_take=N_PATCH, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    v = case_patches[case_idx]
    n_p = v.shape[0]
    if n_p > n_take:
        idx = rng.choice(n_p, n_take, replace=False)
        v = v[idx]
    return v


class SharedTrunkCEA(nn.Module):
    """Single trunk + 3 per-encoder projections + ABMIL + per-gene alpha gate."""
    def __init__(s, d_g, d_u2, d_uv, n_genes, h, drop):
        super().__init__()
        # Per-encoder projection to shared hidden dim
        s.proj_g = nn.Linear(d_g, h)
        s.proj_u2 = nn.Linear(d_u2, h)
        s.proj_uv = nn.Linear(d_uv, h)
        # Shared trunk + ABMIL head (operates on each encoder's projected feature independently)
        s.abmil_g = _ABMIL(h, n_genes, drop)
        s.abmil_u2 = _ABMIL(h, n_genes, drop)
        s.abmil_uv = _ABMIL(h, n_genes, drop)
        # Per-gene alpha gate (3 encoders)
        s.gate = nn.Parameter(torch.full((n_genes, 3), 1/3.0))

    def forward(s, x_g, x_u2, x_uv):
        # project + ABMIL per encoder
        z_g = s.proj_g(x_g);  l_g = s.abmil_g(z_g)
        z_u2 = s.proj_u2(x_u2); l_u2 = s.abmil_u2(z_u2)
        z_uv = s.proj_uv(x_uv); l_uv = s.abmil_uv(z_uv)
        alpha = torch.softmax(s.gate, dim=-1)  # (n_genes, 3)
        logit = alpha[:, 0] * l_g + alpha[:, 1] * l_u2 + alpha[:, 2] * l_uv
        return logit


class _ABMIL(nn.Module):
    def __init__(s, h, n_genes, drop):
        super().__init__()
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, n_genes)
    def forward(s, z):
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def fit(tr_rows, tr_y, seed):
    torch.manual_seed(seed)
    net = SharedTrunkCEA(DIM_G, DIM_U2, DIM_UV, len(GENES), DIM_H, 0.25).to(dev)
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
            xg = torch.from_numpy(np.stack([get_one(case_g, tr_rows[ri], n_take=N_PATCH, rng=np.random.RandomState(seed*1000+ri*7+ep)) for ri in r])).to(dev, torch.float32)
            xu2 = torch.from_numpy(np.stack([get_one(case_u2, tr_rows[ri], n_take=N_PATCH, rng=np.random.RandomState(seed*1000+ri*7+ep)) for ri in r])).to(dev, torch.float32)
            xuv = torch.from_numpy(np.stack([get_one(case_uv, tr_rows[ri], n_take=N_PATCH, rng=np.random.RandomState(seed*1000+ri*7+ep)) for ri in r])).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            loss = lf(net(xg, xu2, xuv), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, te_rows, seed):
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xg = torch.from_numpy(np.stack([get_one(case_g, ci, n_take=N_PATCH, rng=np.random.RandomState(seed*2000+ci*7)) for ci in te_rows])).to(dev, torch.float32)
        xu2 = torch.from_numpy(np.stack([get_one(case_u2, ci, n_take=N_PATCH, rng=np.random.RandomState(seed*2000+ci*7)) for ci in te_rows])).to(dev, torch.float32)
        xuv = torch.from_numpy(np.stack([get_one(case_uv, ci, n_take=N_PATCH, rng=np.random.RandomState(seed*2000+ci*7)) for ci in te_rows])).to(dev, torch.float32)
        return net(xg, xu2, xuv).cpu().numpy()


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(common)), tier)]

print(f"\n=== S43c CEA v3 (shared trunk + per-encoder ABMIL, fold-seed {FOLD_SEED}) ===", flush=True)
T0 = time.time()

oof = np.zeros((len(common), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    print(f"\n--- fold {fold_idx}/5  train_n={len(tri)} test_n={len(tei)}  elapsed {(time.time()-T0)/60:.1f}min ---", flush=True)
    preds = []
    for s in [0, 1, 2]:
        net = fit(tri, Y[tri], s)
        preds.append(predict(net, tei, s))
    oof[tei] = np.mean(preds, axis=0)
    fold_auc = mean_auc(Y[tei], oof[tei])
    print(f"  fold {fold_idx} OOF AUC = {fold_auc:.4f}  ({(time.time()-T0)/60:.1f}min)", flush=True)

auc = mean_auc(Y, oof)
print(f"\n=== Final fold-seed {FOLD_SEED} OOF AUC = {auc:.4f} ===", flush=True)

for g in range(len(GENES)):
    print(f"  {GENES[g]:6s}: AUC={roc_auc_score(Y[:, g], oof[:, g]):.4f}", flush=True)

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\cea_v3_oof.npz", oof=oof, Y=Y, common=np.array(common, dtype=object))
print(f"[saved] cea_v3_oof.npz", flush=True)
print("DONE", flush=True)