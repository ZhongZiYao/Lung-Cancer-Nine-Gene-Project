"""121_posweight_kd.py - §56 Class-Balanced Self-Distillation (alternative to focal, more stable)

L = pos_weight_g * BCE(student, hard)  +  alpha_KD * KL(student || teacher)
  pos_weight_g = neg_count_g / pos_count_g (per gene, fixed for fold)

This is the simplest class-imbalance solution (Cui et al. 2019 / standard nn.BCE pos_weight).
For majority class (TP53 pos=53%), pos_weight ~0.9 (minority up-weight for hard examples? actually this down-weights majority).
"""
import os, sys, time, csv
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, recall_score, f1_score, balanced_accuracy_score
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
LR_, WD = 3e-4, 1e-3   # back to §48 baseline (more stable)
EPOCH = 25
BATCH = 32
T = 2.0
ALPHA_KD = 1.0
dev = "cuda" if torch.cuda.is_available() else "cpu"

PROGRESS_LOG = r"C:\Users\ziyao\uni2h_etl\s121_progress.log"
CSV_PATH = r"C:\Users\ziyao\uni2h_etl\s121_posweight_kd.csv"


def plog(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


plog(f"[startup] dev={dev} pos_weight per gene + KD alpha=1.0")

arr = np.load(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
ix = pd.read_csv(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048_index.csv"))
ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[man.has_label == 1].set_index("case_id")
cases = sorted([c for c in ix.case_id if c in lab.index])
rm = dict(zip(ix.case_id, ix.mem_row))
rows_idx = np.array([rm[c] for c in cases])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])

# Compute per-gene pos_weight = neg_count / pos_count per train fold
gigap_oof = np.load(r"C:\Users\ziyao\uni2h_etl\oof_gigap_5s5tta.npz", allow_pickle=True)
common_t = list(gigap_oof["common"])
g_idx = {c: i for i, c in enumerate(common_t)}
P_teacher = gigap_oof["fs0"][[g_idx[c] for c in cases]]


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


def fit_posweight_kd(tr_rows, tr_y, tr_teacher, seed, pos_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ABMIL(arr.shape[2], len(GENES), DIM_H, 0.25).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=EPOCH,
                                              steps_per_epoch=max(1, len(tr_rows) // BATCH))
    teacher_t = torch.from_numpy(tr_teacher).to(dev).float()
    pw_t = torch.from_numpy(pos_weight.astype(np.float32)).to(dev)
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
            # weighted BCE: per-sample loss = BCE(student, hard) * pos_weight[y]
            bce = F.binary_cross_entropy_with_logits(sl, yb, reduction='none')  # (B, 6)
            hard_loss = (bce * pw_t.unsqueeze(0)).mean()
            soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                 F.softmax(tb / T, dim=-1),
                                 reduction='batchmean') * (T ** 2)
            loss = (1 - ALPHA_KD) * hard_loss + ALPHA_KD * soft_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
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
folds = [(tri, tei) for tri, tei in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED)
                                    .split(np.zeros(len(cases)), tier)]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["fold_seed", "mean_auc", "tp53_sens", "tp53_spec", "tp53_f1", "tp53_bacc",
                "stk11_auc", "kras_sens", "kras_spec", "duration_min", "timestamp"])

T0 = time.time()
oof = np.zeros((len(cases), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    t_fold = time.time()
    # pos_weight per gene from TRAIN fold only
    Ytr = Y[tri]
    pos_weight = np.where(Ytr.sum(0) > 0,
                          (len(Ytr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1),
                          1.0).astype(np.float32)
    pos_weight = np.clip(pos_weight, 0.5, 10.0)  # cap to prevent extreme
    plog(f"fold {fold_idx}/5 START  pos_weight={pos_weight.round(2).tolist()}")
    preds = []
    for s in MODEL_SEEDS:
        net = fit_posweight_kd(tri, Ytr, P_teacher[tri], s, pos_weight)
        preds.append(predict(net, tei, s))
    oof[tei] = np.mean(preds, axis=0)
    plog(f"  fold {fold_idx}/5 done  ({(time.time()-t_fold)/60:.1f}min)")

auc = mean_auc_safe(Y, oof)
per_gene = {GENES[g]: float(roc_auc_score(Y[:, g], oof[:, g])) for g in range(len(GENES))}
y = Y[:, GENES.index('TP53')]
p = 1.0 / (1.0 + np.exp(-oof[:, GENES.index('TP53')]))
yp = (p >= 0.5).astype(int)
tp53_sens = recall_score(y, yp, zero_division=0)
tp53_spec = recall_score(1-y, 1-yp, zero_division=0)
tp53_f1 = f1_score(y, yp, zero_division=0)
tp53_bacc = balanced_accuracy_score(y, yp)
y = Y[:, GENES.index('KRAS')]
p = 1.0 / (1.0 + np.exp(-oof[:, GENES.index('KRAS')]))
yp = (p >= 0.5).astype(int)
kras_sens = recall_score(y, yp, zero_division=0)
kras_spec = recall_score(1-y, 1-yp, zero_division=0)
stk11_auc = per_gene['STK11']

plog(f"=== DONE  mean AUC = {auc:.4f}  TP53(s={tp53_sens:.3f} sp={tp53_spec:.3f} f1={tp53_f1:.3f} bacc={tp53_bacc:.3f})  STK11={stk11_auc:.4f}  KRAS(s={kras_sens:.3f} sp={kras_spec:.3f})  total={(time.time()-T0)/60:.1f}min")

with open(CSV_PATH, "a", newline="", encoding="utf-8") as fcsv:
    w = csv.writer(fcsv)
    w.writerow([FOLD_SEED, auc, tp53_sens, tp53_spec, tp53_f1, tp53_bacc,
                stk11_auc, kras_sens, kras_spec, f"{(time.time()-T0)/60:.1f}", time.strftime("%Y-%m-%d %H:%M:%S")])
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s121_posweight_oof.npz",
                    oof=oof, Y=Y, common=np.array(cases, dtype=object), pos_weight=pos_weight)
plog(f"[saved] s121_posweight_oof.npz  DONE total = {(time.time()-T0)/60:.1f}min")