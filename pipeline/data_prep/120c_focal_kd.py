"""120c_focal_kd.py - S56 Focal Loss + KD (replace print with file write to avoid stdout buffer bug)

Each gamma result written to s120b.progress.log via file write + flush + os.fsync
"""
import os, sys, time, csv, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C_dir = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
FOLD_SEED = 101
MODEL_SEEDS = [0, 1, 2]
NFOLD = 5
N_PATCH = 256
DIM_H = 256
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
T = 2.0
ALPHA_KD = 1.0
GAMMAS = [0.0, 0.5, 1.0, 2.0]
dev = "cuda" if torch.cuda.is_available() else "cpu"

PROGRESS_LOG = r"C:\Users\ziyao\uni2h_etl\s120c_progress.log"
CSV_PATH = r"C:\Users\ziyao\uni2h_etl\s120c_focal_kd.csv"


def plog(msg):
    """Write to progress log with flush + fsync (bypass stdout buffer)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# Init progress log
plog(f"[startup] dev={dev}")
plog(f"[startup] GAMMAS={GAMMAS}  EPOCH={EPOCH}  BATCH={BATCH}  T={T}")

# Data
arr = np.load(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
ix = pd.read_csv(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048_index.csv"))
ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[man.has_label == 1].set_index("case_id")
cases = sorted([c for c in ix.case_id if c in lab.index])
rm = dict(zip(ix.case_id, ix.mem_row))
rows_idx = np.array([rm[c] for c in cases])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
plog(f"[data] N={len(cases)} Y_pos={Y.sum(0).tolist()}")

gigap_oof = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
common_t = list(gigap_oof["common"])
g_idx = {c: i for i, c in enumerate(common_t)}
P_teacher = gigap_oof["fs0"][[g_idx[c] for c in cases]]
plog(f"[teacher] mean AUC = {roc_auc_score(Y, P_teacher):.4f}")


def mean_auc_safe(Y, P):
    aucs = []
    for j in range(len(GENES)):
        npos = int(Y[:, j].sum())
        if npos >= MIN_POS and npos <= len(Y) - MIN_POS:
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def get_patches(idx, n_take=N_PATCH, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    v = np.asarray(arr[idx], dtype=np.float32)
    M = v.shape[0]
    if M > n_take: v = v[rng.choice(M, n_take, replace=False)]
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


def focal_bce_per_gene(logits, y, gamma):
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    focal_factor = (1 - pt) ** gamma
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    return (focal_factor * bce).mean()


def fit_focal_kd(tr_rows, tr_y, tr_teacher, seed, gamma):
    torch.manual_seed(seed)
    net = ABMIL(arr.shape[2], len(GENES), DIM_H, 0.25).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=EPOCH,
                                              steps_per_epoch=max(1, len(tr_rows) // BATCH))
    teacher_t = torch.from_numpy(tr_teacher).to(dev).float()
    rng = np.random.RandomState(seed)
    for ep in range(EPOCH):
        net.train()
        perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            r = perm[i:i + BATCH]
            xb = torch.from_numpy(np.stack([get_patches(rows_idx[tr_rows[ri]],
                                                          n_take=N_PATCH,
                                                          rng=np.random.RandomState(seed * 1000 + ri * 7 + ep))
                                          for ri in r])).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            tb = teacher_t[r]
            sl = net(xb)
            hard_loss = focal_bce_per_gene(sl, yb, gamma)
            soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                 F.softmax(tb / T, dim=-1),
                                 reduction='batchmean') * (T ** 2)
            loss = (1 - ALPHA_KD) * hard_loss + ALPHA_KD * soft_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, te_rows, seed):
    net.eval()
    rng = np.random.RandomState(seed + 999)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        xb = torch.from_numpy(np.stack([get_patches(rows_idx[ri], n_take=N_PATCH, rng=rng)
                                          for ri in te_rows])).to(dev, torch.float32)
        return net(xb).float().cpu().numpy()


tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED)
                        .split(np.zeros(len(cases)), tier)]
plog(f"[folds] {len(folds)} stratified on TP53+NF1")

# Init CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["gamma", "fold_seed", "mean_auc", "tp53_sens", "tp53_spec", "tp53_f1", "tp53_bacc",
                "stk11_auc", "kras_sens", "kras_spec", "duration_min", "timestamp"])

results = {}
T0 = time.time()
for gamma in GAMMAS:
    t_g = time.time()
    plog(f"=== gamma={gamma} START ===")
    oof = np.zeros((len(cases), len(GENES)))
    for fold_idx, (tri, tei) in enumerate(folds):
        preds = []
        for s in MODEL_SEEDS:
            net = fit_focal_kd(tri, Y[tri], P_teacher[tri], s, gamma)
            preds.append(predict(net, tei, s))
        oof[tei] = np.mean(preds, axis=0)
        plog(f"  gamma={gamma} fold {fold_idx}/5 done  ({(time.time()-t_g)/60:.1f}min)")
    auc = mean_auc_safe(Y, oof)
    per_gene = {GENES[g]: float(roc_auc_score(Y[:, g], oof[:, g])) for g in range(len(GENES))}
    # TP53 sens/spec @0.5 + Youden
    y = Y[:, GENES.index('TP53')]
    p = 1.0 / (1.0 + np.exp(-oof[:, GENES.index('TP53')]))
    yp = (p >= 0.5).astype(int)
    tp53_sens = recall_score(y, yp, zero_division=0)
    tp53_spec = recall_score(1-y, 1-yp, zero_division=0)
    tp53_f1 = f1_score(y, yp, zero_division=0)
    tp53_bacc = balanced_accuracy_score(y, yp)
    # KRAS
    y = Y[:, GENES.index('KRAS')]
    p = 1.0 / (1.0 + np.exp(-oof[:, GENES.index('KRAS')]))
    yp = (p >= 0.5).astype(int)
    kras_sens = recall_score(y, yp, zero_division=0)
    kras_spec = recall_score(1-y, 1-yp, zero_division=0)
    stk11_auc = per_gene['STK11']
    results[gamma] = (oof, auc, per_gene)
    dur = (time.time()-t_g) / 60
    plog(f"=== gamma={gamma} mean AUC = {auc:.4f}  TP53(b=0.5 sens={tp53_sens:.3f} spec={tp53_spec:.3f})  STK11={stk11_auc:.4f}  ({dur:.1f}min)")
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([gamma, FOLD_SEED, auc, tp53_sens, tp53_spec, tp53_f1, tp53_bacc,
                    stk11_auc, kras_sens, kras_spec, f"{dur:.1f}", time.strftime("%Y-%m-%d %H:%M:%S")])
    np.savez_compressed(rf"C:\Users\ziyao\uni2h_etl\s120c_oof_gamma{gamma}.npz",
                        oof=oof, Y=Y, common=np.array(cases, dtype=object))
    plog(f"[saved] s120c_oof_gamma{gamma}.npz")

plog(f"DONE total = {(time.time()-T0)/60:.1f}min")