"""101_uni2h_5s5tta.py - S36: UNI2-h 5-seed ensemble + 5-TTA (TCGA-LUAD N=434, 3 fold-seed)

Mirror of 100 but on UNI2-h (1536-d) instead of Prov-GigaPath.
Combined with 100 results, logit-average (P_gigap_5s5tta + P_uni2h_5s5tta) / 2
to test whether the §34 observation (gigap+uni2h helps) generalises to
5-seed + 5-TTA regime.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
LR_, WD, MIN_POS = 3e-4, 1e-3, 8
FOLD_SEEDS, MODEL_SEEDS, NFOLD = [101, 202, 303], [0, 1, 2, 3, 4], 5
CFG = dict(h=256, ep=25, drop=0.25)
S_INST = 512
N_TTA = 5
dev = "cuda" if torch.cuda.is_available() else "cpu"

# UNI2-h TCGA-LUAD
arr = np.load(os.path.join(C, "NSCLC_uni2h_luad_sub512.npy"), mmap_mode="r")
DIM = arr.shape[2]
ix = pd.read_csv(os.path.join(C, "NSCLC_uni2h_luad_sub512_index.csv"))
ix["case_id"] = ix.case_id.astype(str)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[man.has_label == 1].set_index("case_id")
cases = sorted(c for c in ix.case_id if c in lab.index)
rm = dict(zip(ix.case_id, ix.mem_row))
rows = np.array([rm[c] for c in cases])
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in cases])
print(f"[data] N={len(cases)}  dim={DIM}  Y_pos={Y.sum(0).tolist()}", flush=True)

tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(cases)), tier)]
         for fs in FOLD_SEEDS]


class Net(nn.Module):
    def __init__(s, d, h, k, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, k)
    def forward(s, x):
        z = s.trunk(x)
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)
        return s.od((z * a.unsqueeze(-1)).sum(1))


def get_batch(rows, rng, S=S_INST):
    bt = np.empty((len(rows), S, DIM), np.float16)
    for i, r in enumerate(rows):
        v = np.asarray(arr[r], dtype=np.float16)
        if v.shape[0] == 0: bt[i] = 0; continue
        bt[i] = v[rng.choice(v.shape[0], S, replace=v.shape[0] < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tr_rows, tr_y, seed):
    torch.manual_seed(seed)
    net = Net(DIM, CFG["h"], len(GENES), CFG["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR_, weight_decay=WD)
    BATCH = 32
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR_, epochs=CFG["ep"], steps_per_epoch=max(1, len(tr_rows) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(CFG["ep"]):
        net.train(); perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            xb = get_batch(tr_rows[rr], rng)
            yb = torch.from_numpy(tr_y[rr].astype(np.float32)).to(dev)
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict_tta(net, te_rows, n_tta=N_TTA):
    net.eval()
    preds = []
    for t in range(n_tta):
        rng = np.random.RandomState(7919 + t * 113)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            xb = get_batch(te_rows, rng)
            preds.append(net(xb).float().cpu().numpy())
    return np.mean(preds, axis=0)


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


print("[bench] single fit...")
t0 = time.time()
fit(rows[:100], Y[:100], 999)
print(f"  warmup fit = {time.time()-t0:.1f}s", flush=True)

print("\n" + "=" * 100, flush=True)
print(f"=== S36 UNI2-h 5-seed ensemble + {N_TTA}-TTA (TCGA-LUAD N={len(cases)}, 3 fold-seed) ===", flush=True)
print("=" * 100, flush=True)

# Per fold-seed OOF (5x seed × 5x TTA averaged)
oof_5tta_per_fseed = []  # 3x (434, 6)
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    oof_5tta = np.zeros((len(cases), len(GENES)))
    for tri, tei in folds:
        for s in MODEL_SEEDS:
            net = fit(rows[tri], Y[tri], s)
            p5tta = predict_tta(net, rows[tei], n_tta=N_TTA)
            oof_5tta[tei] += p5tta / len(MODEL_SEEDS)
    a = mean_auc(Y, oof_5tta)
    oof_5tta_per_fseed.append(oof_5tta)
    elapsed = (time.time()-T0)/60
    print(f"  foldseed {fi} [elapsed {elapsed:.1f}min]  5-seed 5-TTA = {a:.4f}", flush=True)
    with open(r"C:\Users\ziyao\uni2h_etl\s101_progress.log", "a", encoding="utf-8") as pf:
        pf.write(f"[{elapsed:5.1f}min] foldseed {fi}: {a:.4f}\n")

auc_per_fseed = [mean_auc(Y, o) for o in oof_5tta_per_fseed]
print("\n=== UNI2-h 5-seed 5-TTA OOF mean (3 fold-seeds) ===", flush=True)
print(f"  mean = {np.mean(auc_per_fseed):.4f}  std = {np.std(auc_per_fseed, ddof=1):.4f}  per_fseed = {[round(v,4) for v in auc_per_fseed]}", flush=True)

# Save OOF for downstream ensemble
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz",
                    fs0=oof_5tta_per_fseed[0],
                    fs1=oof_5tta_per_fseed[1],
                    fs2=oof_5tta_per_fseed[2],
                    Y=Y, common=np.array(cases, dtype=object))
print(f"[saved] oof_uni2h_5s5tta.npz", flush=True)
print("DONE", flush=True)
