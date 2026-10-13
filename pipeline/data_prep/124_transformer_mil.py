"""124_transformer_mil.py - §57 Transformer-Encoder MIL Aggregator

Architecture innovation: 2-layer Transformer encoder (LayerNorm + MultiheadAttention + FFN + [CLS] pooling)
replaces §48 single-layer ABMIL gated attention. Deep stacked transformer + KD α=1.0 was never
tested on this dataset (§42 single-layer Trans-MIL didn't help, but we never tried deep + [CLS]).

Loss: same as §48 alpha=1.0 KD against §35 gigap 5s5tta teacher.
Protocol: 1 fold-seed 101, 5 fold, 3 model seed = 15 fits
Time estimate: ~15 min/fold, ~75 min total.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
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
LR_, WD = 3e-4, 1e-3
EPOCH = 25
BATCH = 32
T = 2.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
PROGRESS_LOG = r"C:\Users\ziyao\uni2h_etl\s124_progress.log"


def plog(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


plog(f"[startup] dev={dev}  TransformerMIL: 2-layer encoder, 4 heads, dim=256")

arr = np.load(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
ix = pd.read_csv(os.path.join(C_dir, "NSCLC_gigap_luad_sub2048_index.csv"))
ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
lab = man[man.has_label == 1].set_index("case_id")
cases = sorted([c for c in ix.case_id if c in lab.index])
rm = dict(zip(ix.case_id, ix.mem_row))
rows_idx = np.array([rm[c] for c in cases])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])

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


class TransformerMIL(nn.Module):
    """2-layer Transformer encoder MIL aggregator with [CLS] pooling."""
    def __init__(s, d_in, n_genes, h, nhead=4, n_layers=2, drop=0.25):
        super().__init__()
        s.proj = nn.Linear(d_in, h)
        s.cls = nn.Parameter(torch.zeros(1, 1, h))
        nn.init.trunc_normal_(s.cls, std=0.02)
        s.pos_drop = nn.Dropout(drop)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=nhead, dim_feedforward=h * 4,
            dropout=drop, activation='gelu', batch_first=True, norm_first=True
        )
        s.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        s.norm = nn.LayerNorm(h)
        s.od = nn.Linear(h, n_genes)

    def forward(s, x):
        B = x.shape[0]
        x = s.proj(x)
        cls = s.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = s.pos_drop(x)
        x = s.encoder(x)
        cls_out = s.norm(x[:, 0, :])
        return s.od(cls_out)


def fit_transformer(tr_rows, tr_y, tr_teacher, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = TransformerMIL(arr.shape[2], len(GENES), DIM_H).to(dev)
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
            tb = teacher_t[r]
            sl = net(xb)
            soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                 F.softmax(tb / T, dim=-1),
                                 reduction='batchmean') * (T ** 2)
            loss = soft_loss
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

T0 = time.time()
oof = np.zeros((len(cases), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    t_f = time.time()
    preds = []
    for s in MODEL_SEEDS:
        net = fit_transformer(tri, Y[tri], P_teacher[tri], s)
        preds.append(predict(net, tei, s))
    oof[tei] = np.mean(preds, axis=0)
    plog(f"fold {fold_idx}/5 done  ({(time.time()-t_f)/60:.1f}min)")

auc = mean_auc_safe(Y, oof)
per_gene = {GENES[g]: float(roc_auc_score(Y[:, g], oof[:, g])) for g in range(len(GENES))}
y = Y[:, GENES.index('TP53')]
p = 1.0 / (1.0 + np.exp(-oof[:, GENES.index('TP53')]))
yp = (p >= 0.5).astype(int)
tp53_sens = recall_score(y, yp, zero_division=0)
tp53_spec = recall_score(1-y, 1-yp, zero_division=0)
tp53_f1 = f1_score(y, yp, zero_division=0)
tp53_bacc = balanced_accuracy_score(y, yp)
stk11_auc = per_gene['STK11']

plog(f"=== §124 DONE  mean AUC = {auc:.4f}  TP53(b=0.5 s={tp53_sens:.3f} sp={tp53_spec:.3f})  STK11={stk11_auc:.4f}  total={(time.time()-T0)/60:.1f}min")
plog(f"per_gene: {per_gene}")

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s124_transformer_mil.npz",
                    oof=oof, Y=Y, common=np.array(cases, dtype=object))
plog(f"[saved] s124_transformer_mil.npz")