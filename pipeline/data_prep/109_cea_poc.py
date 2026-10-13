"""109_cea_poc.py - S43 CEA Cross-Encoder Attention proof-of-concept (fold-seed 101)

Algorithm:
  Each patch is encoded by K encoders (gigap/uni2h/uni_v1) -> concat feature (1536+1536+1024 = 4096)
  Per-gene learnable query Q_g
  Cross-attention:
    K = patch feature projector (4096 -> d_k)
    V = patch feature projector (4096 -> d_v)
    attention_score = Q_g @ K^T / sqrt(d_k)
    attn = softmax(score) over patches
    out = attn @ V  (d_v,)
  logit_gene = sum_encoder-specific head weight(attn @ V_e)

  Key difference from §39 CEPG (per-gene alpha on logits):
    CEA does patch-level cross-encoder attention (information flows BETWEEN encoders at patch level),
    while CEPG is post-hoc per-gene weighting (no cross-encoder info flow at the model level).
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
D_HID = 256  # hidden dim for Q/K/V
LR_, WD = 1e-3, 1e-3
EPOCH = 25
BATCH = 32
T = 0.5
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Load 3 encoders
arr_gigap = np.load(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
arr_uni2h = np.load(os.path.join(C_dir, "NSCLC_uni2h_luad_sub512.npy"), mmap_mode="r")
arr_uni_v1 = np.load(os.path.join(C_dir, "NSCLC_uni_sub512.npy"), mmap_mode="r")

DIM_G = arr_gigap.shape[2]  # 1536
DIM_U2 = arr_uni2h.shape[2]  # 1536
DIM_UV = arr_uni_v1.shape[2]  # 1024
DIM_TOTAL = DIM_G + DIM_U2 + DIM_UV  # 4096
print(f"[data] gigap dim {DIM_G}, uni2h dim {DIM_U2}, uni_v1 dim {DIM_UV}, total {DIM_TOTAL}", flush=True)

# Index 434 common cases
ix_g = pd.read_csv(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048_index.csv"))
ix_u2 = pd.read_csv(os.path.join(C_dir, "NSCLC_uni2h_luad_sub512_index.csv"))
ix_uv = pd.read_csv(os.path.join(C_dir, "NSCLC_uni_sub512_index.csv"))
for ix in (ix_g, ix_u2, ix_uv):
    ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[man.has_label == 1].set_index("case_id")

cases_g = set(ix_g.case_id) & set(lab.index)
cases_u2 = set(ix_u2.case_id) & set(lab.index)
cases_uv = set(ix_uv.case_id) & set(lab.index)
common = sorted(cases_g & cases_u2 & cases_uv)
print(f"[data] {len(common)} common cases across 3 encoders", flush=True)

# Build row indices
rm_g = dict(zip(ix_g.case_id, ix_g.mem_row))
rm_u2 = dict(zip(ix_u2.case_id, ix_u2.mem_row))
rm_uv = dict(zip(ix_uv.case_id, ix_uv.mem_row))
rows_g = np.array([rm_g[c] for c in common])
rows_u2 = np.array([rm_u2[c] for c in common])
rows_uv = np.array([rm_uv[c] for c in common])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in common])
print(f"[data] Y_pos = {Y.sum(0).tolist()}", flush=True)

# Load all patches into RAM (5.5 GB + ~700 MB + 1 GB = ~7 GB total OK)
print("[load] reading all encoder patches into memory...", flush=True)
case_patches_g = np.zeros((len(common), 2048, DIM_G), dtype=np.float32)
case_patches_u2 = np.zeros((len(common), 512, DIM_U2), dtype=np.float32)
case_patches_uv = np.zeros((len(common), 512, DIM_UV), dtype=np.float32)
T_load = time.time()
for i, ci in enumerate(common):
    case_patches_g[i] = arr_gigap[rows_g[i]][:2048]
    case_patches_u2[i] = arr_uni2h[rows_u2[i]][:512]
    case_patches_uv[i] = arr_uni_v1[rows_uv[i]][:512]
print(f"[load] done: {case_patches_g.nbytes/1e9:.2f} + {case_patches_u2.nbytes/1e9:.2f} + {case_patches_uv.nbytes/1e9:.2f} GB in {time.time()-T_load:.1f}s", flush=True)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def get_patches_concat(case_idx, n_take=N_PATCH, rng=None):
    """Get (n_take, 4096) concat patches for a case across 3 encoders."""
    if rng is None:
        rng = np.random.RandomState(0)
    p_g = case_patches_g[case_idx]
    p_u2 = case_patches_u2[case_idx]
    p_uv = case_patches_uv[case_idx]
    n_p = min(p_g.shape[0], p_u2.shape[0], p_uv.shape[0])
    if n_p > n_take:
        idx = rng.choice(n_p, n_take, replace=False)
    else:
        idx = np.arange(n_p)
    return np.concatenate([p_g[idx], p_u2[idx], p_uv[idx]], axis=1)  # (n_take, 4096)


class CEANet(nn.Module):
    """Cross-Encoder Attention MIL.

    For each gene g:
      Q_g: (d_q,) learnable
      K = patch_feature @ W_K   -> (N, d_q)
      V = patch_feature @ W_V_e -> (N, d_v) for each encoder e (or shared?)
      attn_g = softmax(Q_g @ K^T / sqrt(d_q)) over patches  -> (N,)
      out_g = sum_p attn_g[p] * V[p]   -> (d_v,)
      logits[g] = W_out[g] @ out_g  (or per-encoder head output then avg)
    """
    def __init__(s, d_total, n_genes, d_hid, n_encoders=3, drop=0.25):
        super().__init__()
        s.n_encoders = n_encoders
        s.n_genes = n_genes
        s.d_hid = d_hid
        # Encoder-specific projection: d_total -> d_hid
        s.enc_proj = nn.ModuleList([nn.Linear(d_total // n_encoders, d_hid) for _ in range(n_encoders)])
        # K projection (shared across encoders)
        s.W_K = nn.Linear(d_hid, d_hid, bias=False)
        # Per-gene per-encoder V projection (so each gene can attend differently per encoder)
        s.W_V = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_hid, d_hid, bias=False) for _ in range(n_encoders)])
            for _ in range(n_genes)
        ])
        # Per-gene query
        s.Q = nn.Parameter(torch.randn(n_genes, d_hid) * 0.1)
        # Per-gene per-encoder classifier (V aggregated -> logit_g_e)
        s.od = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_hid, 1) for _ in range(n_encoders)])
            for _ in range(n_genes)
        ])
        # Per-gene encoder weights (learned or default 1/3)
        s.enc_alpha = nn.Parameter(torch.ones(n_genes, n_encoders) / n_encoders)

    def forward(s, x):
        """x: (B, N, d_total) concatenated multi-encoder features"""
        B, N, _ = x.shape
        d_per_enc = x.shape[-1] // s.n_encoders
        # Split per encoder
        enc_feats = [x[..., e * d_per_enc:(e + 1) * d_per_enc] for e in range(s.n_encoders)]
        # Project per encoder: (B, N, d_hid)
        z = [s.enc_proj[e](enc_feats[e]) for e in range(s.n_encoders)]
        # K shared
        K = s.W_K(sum(z) / s.n_encoders)  # (B, N, d_hid)  -- mean over encoders for K
        # Per-gene V per-encoder
        out_per_enc_per_gene = []  # (n_genes, n_encoders) of (B, d_hid)
        for g in range(s.n_genes):
            oes = []
            for e in range(s.n_encoders):
                # V from encoder e using gene g's V projection
                V_g_e = s.W_V[g][e](z[e])  # (B, N, d_hid)
                # attention
                attn = torch.einsum("bnh,h->bn", K, s.Q[g]) / (s.d_hid ** 0.5)  # (B, N)
                attn = torch.softmax(attn, dim=-1)  # (B, N)
                # weighted sum
                pooled = torch.einsum("bn,bnd->bd", attn, V_g_e)  # (B, d_hid)
                # per-encoder logits
                oes.append(s.pooled(g, e, pooled))
            out_per_enc_per_gene.append(oes)
        # Stack: out_per_enc_per_gene[g][e] = (B,)
        # Concat per gene: out_g = stack of (B, n_encoders) logits
        # logit = sum_e alpha[g, e] * out[g][e]
        # Actually simpler: weighted sum via softmax(alpha) or use raw alpha
        alpha = torch.softmax(s.enc_alpha, dim=-1)  # (n_genes, n_encoders)
        logits = torch.zeros(B, s.n_genes, device=x.device)
        for g in range(s.n_genes):
            for e in range(s.n_encoders):
                logits[:, g] = logits[:, g] + alpha[g, e] * out_per_enc_per_gene[g][e][:, 0]
        return logits

    def pooled(s, g, e, x):
        return s.od[g][e](x)


def fit(tr_rows, tr_y, seed):
    torch.manual_seed(seed)
    net = CEANet(DIM_TOTAL, len(GENES), D_HID, n_encoders=3, drop=0.25).to(dev)
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
            xb = torch.from_numpy(np.stack([get_patches_concat(ci, n_take=N_PATCH, rng=np.random.RandomState(seed*1000 + ci*7 + ep)) for ci in tr_rows[r]])).to(dev, torch.float32)
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
        xb = torch.from_numpy(np.stack([get_patches_concat(ci, n_take=N_PATCH, rng=np.random.RandomState(seed*2000+ci*7)) for ci in te_rows])).to(dev, torch.float32)
        return net(xb).cpu().numpy()


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(common)), tier)]

print(f"\n=== S43 CEA Cross-Encoder Attention (fold-seed {FOLD_SEED}, N_PATCH={N_PATCH}, D_HID={D_HID}) ===", flush=True)
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

# Per-gene
for g in range(len(GENES)):
    print(f"  {GENES[g]:6s}: AUC={roc_auc_score(Y[:, g], oof[:, g]):.4f}")

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\cea_oof.npz", oof=oof, Y=Y, common=np.array(common, dtype=object))
print(f"[saved] cea_oof.npz", flush=True)
print("DONE", flush=True)