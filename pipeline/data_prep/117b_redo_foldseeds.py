"""117b_redo_foldseeds.py - redo S117 fs202 + fs303 (fs101 already in s114b), save npz per fold-seed

S117 last crashed on `mean()` undefined -> npz never saved. fs101 OOF is identical to S114b's
s114_oof_lam1.00.npz (we keep that), only fs202 + fs303 missing. This script:
  - runs fs202 + fs303 (5 fold x 3 model-seed each = 30 fits)
  - save oof_fs202.npz and oof_fs303.npz after each fold-seed finishes
  - even if crash midway, at least fs202 saved before fs303 starts
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
NFOLD = 5
N_PATCH = 256
DIM_H = 256
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
T = 2.0
LAMBDA_KD = 1.0
MODEL_SEEDS = [0, 1, 2]
# fs101 already covered by s114_oof_lam1.00.npz; redo only fs202, fs303
FOLD_SEEDS = [202, 303]

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


def fit(tr_rows, tr_y, tr_teacher, seed):
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
            soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                 F.softmax(tb / T, dim=-1),
                                 reduction="batchmean") * (T ** 2)
            loss = soft_loss  # alpha=1.0 -> pure KD, no hard BCE
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


dev = "cuda" if torch.cuda.is_available() else "cpu"
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))

print(f"[data] N={len(cases)}  Y_pos={Y.sum(0).tolist()}", flush=True)
print(f"\n=== S117 REDO: fs={FOLD_SEEDS} (fs101 already in s114_oof_lam1.00.npz) ===", flush=True)

T0 = time.time()
oof_per_fs = {}
results_per_fs = {}

for fs_idx, fs in enumerate(FOLD_SEEDS):
    # fs_idx is 0 for fs202, 1 for fs303 - matching gigap fs0/fs1/fs2 indexing:
    # gigap["fs0"] = teacher fs101, ["fs1"] = fs202, ["fs2"] = fs303
    P_teacher = gigap[f"fs{fs_idx+1}"][sel_g] if fs_idx < 2 else gigap["fs2"][sel_g]
    # safer: just use fs_idx+1
    if fs == 202:
        P_teacher = gigap["fs1"][sel_g]
    elif fs == 303:
        P_teacher = gigap["fs2"][sel_g]
    print(f"\n--- fold-seed {fs} (teacher mean AUC = {mean_auc(Y, P_teacher):.4f}) ---", flush=True)
    folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
    oof = np.zeros((len(cases), len(GENES)))
    for fold_idx, (tri, tei) in enumerate(folds):
        preds = []
        for s in MODEL_SEEDS:
            net = fit(tri, Y[tri], P_teacher[tri], s)
            preds.append(predict(net, tei, s))
        oof[tei] = np.mean(preds, axis=0)
    auc = mean_auc(Y, oof)
    print(f"  fold-seed {fs} OOF AUC = {auc:.4f}  ({(time.time()-T0)/60:.1f}min)", flush=True)
    # SAVE per fs immediately, in case crash later
    np.savez_compressed(rf"C:\Users\ziyao\uni2h_etl\s117b_oof_fs{fs}.npz",
                        oof=oof, Y=Y, common=np.array(cases, dtype=object))
    print(f"  [saved] s117b_oof_fs{fs}.npz", flush=True)
    oof_per_fs[fs] = oof
    results_per_fs[fs] = auc
    # per-gene AUC for this fs
    per_gene = {GENES[g]: float(roc_auc_score(Y[:, g], oof[:, g])) for g in range(len(GENES))}
    print(f"  per-gene {fs}: {per_gene}", flush=True)

# Aggregate (load fs101 from s114b)
fs101_oof = np.load(r"C:\Users\ziyao\uni2h_etl\s114_oof_lam1.00.npz")["oof"]
oof_per_fs_full = {101: fs101_oof, **oof_per_fs}
results_per_fs_full = {101: 0.8662, **results_per_fs}

oof_avg = np.mean([oof_per_fs_full[f] for f in [101, 202, 303]], axis=0)
auc_avg = mean_auc(Y, oof_avg)
mean_fs = float(np.mean(list(results_per_fs_full.values())))
std_fs = float(np.std(list(results_per_fs_full.values())))

print(f"\n=== S117 REDO FINAL ===", flush=True)
print(f"  per-fs AUC = {results_per_fs_full}", flush=True)
print(f"  mean +/- std across fs = {mean_fs:.4f} +/- {std_fs:.4f}", flush=True)
print(f"  ensemble (mean of 3 OOFs) = {auc_avg:.4f}", flush=True)
print(f"  vs baseline 0.7419 = {auc_avg - 0.7419:+.4f}", flush=True)
print(f"  Per-gene overall (3 fs mean OOF):", flush=True)
for g in range(len(GENES)):
    pgs = [roc_auc_score(Y[:, g], oof_per_fs_full[f][:, g]) for f in [101, 202, 303]]
    print(f"  {GENES[g]:6s}: {float(np.mean(pgs)):.4f} +/- {float(np.std(pgs)):.4f}", flush=True)

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s117b_3fold_oof.npz",
                    **{f"oof_fs{f}": oof_per_fs_full[f] for f in [101, 202, 303]},
                    oof_avg=oof_avg, Y=Y, common=np.array(cases, dtype=object))
print(f"\n[saved] s117b_3fold_oof.npz", flush=True)
print(f"DONE total = {(time.time()-T0)/60:.1f}min", flush=True)