"""115_multi_seed_student.py - §48 v1: multi-seed student ensemble for distillation

Based on §47 + α-ablation: KL distillation works (0 → 0.25 → monotonic up).
Improvement: train 5 student models with different seeds (and patch sampling seeds),
average their 5-seed predictions for OOF AUC.

Expected: single student 0.8230 -> 5-seed ensemble ~0.83-0.84
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
LAM = 0.5
N_STUDENT_SEEDS = 5
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


def get_patches(case_row_idx, rng, n_take=N_PATCH):
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


def fit(tr_rows, tr_y, tr_teacher, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
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
                                                       rng=np.random.RandomState(seed*1000 + ri*7 + ep)) for ri in r])).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            tb = teacher_t[r]
            sl = net(xb)
            hard_loss = bce(sl, yb).mul(pw).mean()
            soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                 F.softmax(tb / T, dim=-1),
                                 reduction="batchmean") * (T ** 2)
            loss = (1 - LAM) * hard_loss + LAM * soft_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, te_rows, seed):
    net.eval()
    rng = np.random.RandomState(seed * 2000)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = torch.from_numpy(np.stack([get_patches(rows_idx[ci], n_take=N_PATCH,
                                                   rng=np.random.RandomState(seed*3000+ci*7)) for ci in te_rows])).to(dev, torch.float32)
        return net(xb).float().cpu().numpy()


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(cases)), tier)]

print(f"[data] N={len(cases)}  Y_pos={Y.sum(0).tolist()}", flush=True)
print(f"[teacher] mean AUC = {mean_auc(Y, P_teacher):.4f}", flush=True)
print(f"\n=== §48 v1: Multi-seed student ensemble (N_STUDENT_SEEDS={N_STUDENT_SEEDS}, T={T}) ===", flush=True)

T0 = time.time()
oof_ensemble = np.zeros((len(cases), len(GENES)))
oof_individual = np.zeros((N_STUDENT_SEEDS, len(cases), len(GENES)))

# Train each student seed
for student_seed in range(N_STUDENT_SEEDS):
    student_oof = np.zeros((len(cases), len(GENES)))
    for fold_idx, (tri, tei) in enumerate(folds):
        net = fit(tri, Y[tri], P_teacher[tri], seed=student_seed)
        student_oof[tei] = predict(net, tei, seed=student_seed)
    auc = mean_auc(Y, student_oof)
    oof_individual[student_seed] = student_oof
    print(f"  student_seed {student_seed}: OOF AUC = {auc:.4f}  ({(time.time()-T0)/60:.1f}min)", flush=True)

# Ensemble
oof_ensemble = oof_individual.mean(axis=0)
ensemble_auc = mean_auc(Y, oof_ensemble)

print(f"\n=== Final results ===", flush=True)
for s in range(N_STUDENT_SEEDS):
    print(f"  student {s}: {mean_auc(Y, oof_individual[s]):.4f}", flush=True)
print(f"\n  ENSEMBLE (5 students avg) = {ensemble_auc:.4f}", flush=True)
print(f"  vs S47 single student (α=0.5, 3-seed): 0.8230", flush=True)
print(f"  vs S31 ABMIL baseline: 0.7419", flush=True)
print(f"  Δ(ensemble - single student) = {ensemble_auc - 0.8230:+.4f}", flush=True)
print(f"  Δ(ensemble - baseline) = {ensemble_auc - 0.7419:+.4f}", flush=True)

# Per-gene
print(f"\n=== Per-gene (ensemble) ===", flush=True)
for g in range(len(GENES)):
    print(f"  {GENES[g]:6s}: AUC={roc_auc_score(Y[:, g], oof_ensemble[:, g]):.4f}", flush=True)

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\115_multi_seed_oof.npz",
                    oof_ensemble=oof_ensemble, oof_individual=oof_individual,
                    Y=Y, common=np.array(cases, dtype=object))
print(f"\n[saved] 115_multi_seed_oof.npz", flush=True)
print("DONE", flush=True)