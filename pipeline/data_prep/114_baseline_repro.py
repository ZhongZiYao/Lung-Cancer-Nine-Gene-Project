"""114_baseline_repro.py - S47 verification: baseline reproduce + alpha ablation

Two purposes:
  1. Reproduce §31 baseline with the EXACT same folds / seeds / architecture as §47,
     but WITHOUT distillation. If baseline reproduces 0.7419, then S47's +0.081 is real.
  2. Ablate alpha (distillation weight) to check the gain scales monotonically.
     alpha=0.0 -> pure hard-label BCE (baseline)
     alpha=1.0 -> pure KL (pure dark knowledge)
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C_dir = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
FOLD_SEED = 101
NFOLD = 5
N_PATCH = 256
DIM_H = 256
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
T = 2.0
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

gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
common_g = list(gigap["common"])
g_idx = {c: i for i, c in enumerate(common_g)}
sel_g = np.array([g_idx[c] for c in cases])
P_teacher = gigap["fs0"][sel_g]


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


class ABMIL(nn.Module):
    def __init__(s, d_in, n_genes, h, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, n_genes)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def fit(tr_rows, tr_y, tr_teacher, seed, lam):
    torch.manual_seed(seed)
    net = ABMIL(DIM, len(GENES), DIM_H, 0.25).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=EPOCH, steps_per_epoch=max(1, len(tr_rows) // BATCH))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    teacher_t = torch.from_numpy(tr_teacher).to(dev).float()
    rng = np.random.RandomState(seed)
    for ep in range(EPOCH):
        net.train()
        perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            r = perm[i:i + BATCH]
            xb = torch.from_numpy(np.stack([get_patches(rows_idx[tr_rows[ri]], n_take=N_PATCH,
                                                       rng=np.random.RandomState(seed*1000+ri*7+ep)) for ri in r])).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            tb = teacher_t[r]
            sl = net(xb)
            hard_loss = bce(sl, yb).mul(pw).mean()
            if lam > 0:
                soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                     F.softmax(tb / T, dim=-1),
                                     reduction="batchmean") * (T ** 2)
                loss = (1 - lam) * hard_loss + lam * soft_loss
            else:
                loss = hard_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, te_rows, seed):
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = torch.from_numpy(np.stack([get_patches(rows_idx[ci], n_take=N_PATCH,
                                                   rng=np.random.RandomState(seed*2000+ci*7)) for ci in te_rows])).to(dev, torch.float32)
        return net(xb).float().cpu().numpy()


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(cases)), tier)]

print(f"[data] N={len(cases)}  teacher mean AUC = {mean_auc(Y, P_teacher):.4f}", flush=True)
print(f"\n=== Alpha ablation (T={T}, fs={FOLD_SEED}) ===", flush=True)

results = {}
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    T0 = time.time()
    oof = np.zeros((len(cases), len(GENES)))
    for fold_idx, (tri, tei) in enumerate(folds):
        preds = []
        for s in [0, 1, 2]:
            net = fit(tri, Y[tri], P_teacher[tri], s, lam)
            preds.append(predict(net, tei, s))
        oof[tei] = np.mean(preds, axis=0)
    auc = mean_auc(Y, oof)
    results[lam] = auc
    tag = "baseline (no distill)" if lam == 0 else ("pure dark knowledge" if lam == 1.0 else "blend")
    print(f"  alpha={lam:.2f}  OOF AUC = {auc:.4f}   [{tag}, {(time.time()-T0)/60:.1f}min]", flush=True)
    np.savez_compressed(rf"C:\Users\ziyao\uni2h_etl\s114_oof_lam{lam:.2f}.npz", oof=oof, Y=Y, common=np.array(cases, dtype=object))

print("\n=== Summary ===", flush=True)
for lam, auc in results.items():
    print(f"  alpha={lam:.2f}: {auc:.4f}  (Δ vs baseline {auc - results[0.0]:+.4f})", flush=True)
print("DONE", flush=True)