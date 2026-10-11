"""107b_prototype_v2.py - S40 Prototype MIL with all patches cached in memory
- Loads all 434 cases × 2048 patches into RAM once (12.3 GB → OK with 32 GB)
- Fits fast on cached data
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
K_PROTO = 8
DIM_H = 256
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
T = 0.1
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

# Cache all patches in memory (fast)
print("[load] reading all patches into memory...", flush=True)
T_load = time.time()
case_patches = np.zeros((len(cases), 2048, DIM), dtype=np.float32)
for i, ci in enumerate(cases):
    case_patches[i] = arr[rows_idx[i]][:2048]
print(f"[load] done: case_patches.shape={case_patches.shape} ({case_patches.nbytes/1e9:.1f} GB) in {time.time()-T_load:.1f}s", flush=True)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


def get_patches(case_idx, n_take=N_PATCH, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    v = case_patches[case_idx]
    M = v.shape[0]
    if M > n_take:
        idx = rng.choice(M, n_take, replace=False)
        v = v[idx]
    return v


class ProtoNet(nn.Module):
    def __init__(s, d_in, n_genes, k, h, drop):
        super().__init__()
        s.k = k
        s.n_genes = n_genes
        s.trunk = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop))
        s.prototypes = nn.Parameter(torch.randn(n_genes, k, h) * 0.1)
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


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
print("\n" + "=" * 80, flush=True)
print(f"=== S40 Prototype MIL v2 (K={K_PROTO}, T={T}, N_PATCH={N_PATCH}) (TCGA-LUAD N={len(cases)}, 3 fold-seed) ===", flush=True)
print("=" * 80, flush=True)

results = {}
T_total = time.time()
prog_path = r"C:\Users\ziyao\uni2h_etl\s107b_progress.log"
import json
for fs in FOLD_SEEDS:
    T_fs = time.time()
    folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
    oof = np.zeros((len(cases), len(GENES)))
    for fold_idx, (tri, tei) in enumerate(folds):
        T_fold = time.time()
        preds = []
        for s in [0, 1, 2]:
            torch.manual_seed(s + fs * 100)
            net = ProtoNet(DIM, len(GENES), K_PROTO, DIM_H, 0.25).to(dev)
            opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=EPOCH, steps_per_epoch=max(1, len(tri) // BATCH))
            lf = nn.BCEWithLogitsLoss(reduction="none")
            pw = torch.tensor(np.where(Y[tri].sum(0) > 0, (len(tri) - Y[tri].sum(0)) / np.maximum(Y[tri].sum(0), 1), 1.),
                              dtype=torch.float32, device=dev)
            rng = np.random.RandomState(s + fs * 100)
            for ep in range(EPOCH):
                net.train()
                perm = rng.permutation(len(tri))
                for i in range(0, len(perm) - BATCH + 1, BATCH):
                    r = perm[i:i + BATCH]
                    xb = torch.from_numpy(np.stack([get_patches(ci, n_take=N_PATCH, rng=np.random.RandomState(s*1000 + ci*7 + ep)) for ci in tri[r]])).to(dev, torch.float32)
                    yb = torch.from_numpy(Y[tri][r].astype(np.float32)).to(dev)
                    loss = lf(net(xb), yb).mul(pw).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                    try: sch.step()
                    except Exception: pass
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                xb = torch.from_numpy(np.stack([get_patches(ci, n_take=N_PATCH, rng=np.random.RandomState(s*2000+ci*7)) for ci in tei])).to(dev, torch.float32)
                preds.append(net(xb).cpu().numpy())
        oof[tei] = np.mean(preds, axis=0)
        print(f"  fs{fs} fold{fold_idx}: AUC={mean_auc(Y[tei], oof[tei]):.4f}  fold_time={(time.time()-T_fold):.1f}s", flush=True)
    auc = mean_auc(Y, oof)
    print(f"\n  fold-seed {fs}: Prototype MIL = {auc:.4f}  [{(time.time()-T_fs)/60:.1f}min]", flush=True)
    results[fs] = auc
    np.savez_compressed(rf"C:\Users\ziyao\uni2h_etl\proto_v2_oof_fs{fs}.npz",
                        oof=oof, Y=Y, common=np.array(cases, dtype=object))
    with open(prog_path, "a", encoding="utf-8") as pf:
        pf.write(f"[{(time.time()-T_total)/60:5.1f}min] fs{fs}: {auc:.4f}\n")

mean_proto = float(np.mean(list(results.values())))
std_proto = float(np.std(list(results.values()), ddof=1))
print(f"\n=== Summary ===", flush=True)
print(f"  mean = {mean_proto:.4f}  std = {std_proto:.4f}  per_fseed = {[round(v,4) for v in results.values()]}", flush=True)
print(f"  S31 baseline 0.7419, S35 5s5tta 0.7441, S36 ens 0.7444, S39 CEPG 0.7506", flush=True)
print(f"  Δ vs §39 CEPG = {mean_proto - 0.7506:+.4f}", flush=True)

with open(r"C:\Users\ziyao\uni2h_etl\s107b_results.json", "w", encoding="utf-8") as f:
    json.dump({"per_fold_seed": {str(k): float(v) for k, v in results.items()},
               "mean": mean_proto, "std": std_proto,
               "S39_CEPG": 0.7506, "S36_ens": 0.7444, "S31_baseline": 0.7419,
               "K_proto": K_PROTO, "T": T}, f, indent=2, ensure_ascii=False)
print(f"[saved] s107b_results.json", flush=True)
print("DONE", flush=True)