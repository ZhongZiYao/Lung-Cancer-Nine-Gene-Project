"""107_prototype_mil.py - S40 Prototype-Enhanced MIL (TPMIL-style)

Algorithm:
  Per-gene K prototype vectors P_g (k=8) on patch feature space.
  For each patch x_i, for each gene g, compute soft assignment to K prototypes via
  softmax over -||x_i - P_g^k||^2.
  bag-level score_g = mean over patches of log p(g|x) (log of mean softmax).
  Loss: BCE per gene + L2 prototype regularization.

Inspiration:
  - TPMIL (Yang 2023) - small cohort SOTA
  - DTFD-MIL (Zhang 2022) - pseudo-bag augmentation

Output: 3 fold-seeds 脳 5-fold OOF AUC for 9-gene TCGA-LUAD N=434.
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
FOLD_SEEDS = [101, 202, 303]
NFOLD = 5
N_PATCH = 256
K_PROTO = 8  # per gene
DIM_H = 256
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
T = 0.1  # softmax temperature

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


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def get_patches(case_row_idx, n_take=N_PATCH, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    v = np.asarray(arr[case_row_idx], dtype=np.float32)
    M = v.shape[0]
    if M > n_take:
        idx = rng.choice(M, n_take, replace=False)
        v = v[idx]
    return v


class ProtoNet(nn.Module):
    """Per-gene K prototypes in patch-feature space (after optional trunk).

    Forward:
      X: (B, N, D_in)
      prototypes_g: (K, D_h) for each gene g
      -> per-gene logit: mean over patches of log p(g | x_i)
    """
    def __init__(s, d_in, n_genes, k, h, drop):
        super().__init__()
        s.k = k
        s.n_genes = n_genes
        s.trunk = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop))
        # K prototypes per gene, learnable
        s.prototypes = nn.Parameter(torch.randn(n_genes, k, h) * 0.1)
        # gene bias
        s.gene_bias = nn.Parameter(torch.zeros(n_genes))

    def forward(s, x):
        # x: (B, N, D_in)
        B, N, _ = x.shape
        z = s.trunk(x.reshape(B * N, -1)).reshape(B, N, -1)  # (B, N, h)
        # dist to all prototypes: (B, N, n_genes, k)
        # ||z - P||^2 = ||z||^2 - 2 z P + |P|^2
        z_norm = (z ** 2).sum(-1, keepdim=True)                              # (B, N, 1)
        p_norm = (s.prototypes ** 2).sum(-1).reshape(1, 1, s.n_genes, s.k)  # (1,1,n_genes,k)
        cross = torch.einsum("bnh,gkh->bngk", z, s.prototypes)              # (B, N, n_genes, k)
        dist = z_norm.unsqueeze(-1) + p_norm - 2 * cross                       # (B, N, n_genes, k)
        # log p(g|x) for each (B, N) gene:
        # g 鈭?R^(n_genes, k, h). patch embedding z_bh.
        # log p_gk(x) = -||z - P_gk||^2 / T, then log-sum-exp over k = log p(g|x)
        log_p_gk = -dist / T                                          # (B, N, n_genes, k)
        log_p_gk = log_p_gk - log_p_gk.max(dim=-1, keepdim=True).values  # stable
        log_p_gx = log_p_gk.logsumexp(dim=-1)                            # (B, N, n_genes)
        log_p_gx = log_p_gx - log_p_gx.logsumexp(dim=-1, keepdim=True)
        # bag-level: log p(g | X) 鈮?logsumexp_i (log p(g|x_i)) - log N
        log_p_gX = log_p_gx.logsumexp(dim=1) - np.log(N) if False else log_p_gx.mean(dim=1)
        # add bias
        return log_p_gX + s.gene_bias


def fit(tr_rows, tr_y, seed):
    torch.manual_seed(seed)
    net = ProtoNet(DIM, len(GENES), K_PROTO, DIM_H, 0.25).to(dev)
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
            xb = torch.from_numpy(np.stack([get_patches(rows_idx[ci], n_take=N_PATCH, rng=np.random.RandomState(seed)) for ci in tr_rows[r]])).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, te_rows, seed):
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = torch.from_numpy(np.stack([get_patches(rows_idx[ci], n_take=N_PATCH, rng=np.random.RandomState(seed + 9999)) for ci in te_rows])).to(dev, torch.float32)
        return net(xb).cpu().numpy()


print("\n" + "=" * 80, flush=True)
print(f"=== S40 Prototype MIL (K={K_PROTO}, T={T}) (TCGA-LUAD N={len(cases)}, 3 fold-seed) ===", flush=True)
print("=" * 80, flush=True)

tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
all_res = {}
T0 = time.time()
for fs in FOLD_SEEDS:
    folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
    oof = np.zeros((len(cases), len(GENES)))
    for tri, tei in folds:
        # 3 model seeds for stability
        preds = []
        for s in [0, 1, 2]:
            net = fit(rows_idx, Y, s) if False else None
            # fit per-fold
            torch.manual_seed(s)
            net = ProtoNet(DIM, len(GENES), K_PROTO, DIM_H, 0.25).to(dev)
            opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=EPOCH, steps_per_epoch=max(1, len(tri) // BATCH))
            lf = nn.BCEWithLogitsLoss(reduction="none")
            pw = torch.tensor(np.where(Y[tri].sum(0) > 0, (len(tri) - Y[tri].sum(0)) / np.maximum(Y[tri].sum(0), 1), 1.),
                              dtype=torch.float32, device=dev)
            rng = np.random.RandomState(s + 12345)
            for ep in range(EPOCH):
                net.train()
                perm = rng.permutation(len(tri))
                for i in range(0, len(perm) - BATCH + 1, BATCH):
                    r = perm[i:i + BATCH]
                    xb = torch.from_numpy(np.stack([get_patches(rows_idx[ci], n_take=N_PATCH, rng=np.random.RandomState(s*1000 + ci)) for ci in tri[r]])).to(dev, torch.float32)
                    yb = torch.from_numpy(Y[tri][r].astype(np.float32)).to(dev)
                    loss = lf(net(xb), yb).mul(pw).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    try: sch.step()
                    except Exception: pass
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                xb = torch.from_numpy(np.stack([get_patches(rows_idx[ci], n_take=N_PATCH, rng=np.random.RandomState(s*2000+ci)) for ci in tei])).to(dev, torch.float32)
                preds.append(net(xb).cpu().numpy())
        oof[tei] = np.mean(preds, axis=0)
    auc = mean_auc(Y, oof)
    elapsed = (time.time()-T0)/60
    print(f"  fold-seed {fs}: Prototype MIL OOF = {auc:.4f}  [{elapsed:.1f}min]", flush=True)
    with open(r"C:\Users\ziyao\uni2h_etl\s107_progress.log", "a", encoding="utf-8") as pf:
        pf.write(f"[{elapsed:5.1f}min] fold-seed {fs}: {auc:.4f}\n")
    all_res[fs] = auc
    np.savez_compressed(rf"C:\Users\ziyao\uni2h_etl\proto_oof_fs{fs}.npz",
                        oof=oof, Y=Y, common=np.array(cases, dtype=object))

mean_proto = float(np.mean(list(all_res.values())))
std_proto = float(np.std(list(all_res.values()), ddof=1))
print(f"\n=== Summary ===", flush=True)
print(f"  mean = {mean_proto:.4f}  std = {std_proto:.4f}  per_fseed = {[round(v,4) for v in all_res.values()]}", flush=True)
print(f"  S31 baseline 0.7419, S35 5s5tta 0.7441, S36 ens 0.7444, S39 CEPG 0.7506", flush=True)
print(f"  螖 vs 搂39 CEPG = {mean_proto - 0.7506:+.4f}", flush=True)

import json
with open(r"C:\Users\ziyao\uni2h_etl\s107_results.json", "w", encoding="utf-8") as f:
    json.dump({"per_fold_seed": {str(k): float(v) for k, v in all_res.items()},
               "mean": mean_proto,
               "std": std_proto,
               "S39_CEPG": 0.7506,
               "S36_ens": 0.7444,
               "S31_baseline": 0.7419,
               "K_proto": K_PROTO,
               "T": T}, f, indent=2, ensure_ascii=False)
print(f"\n[saved] s107_results.json", flush=True)
print("DONE", flush=True)
