"""100_seed_ensemble_tta.py - S35: 5-seed ensemble + TTA on Prov-GigaPath TCGA-LUAD

Strategy:
  - 5 model seeds x 5 fold x 3 fold-seed = 75 fits per encoder
  - 1 fold = (5 seeds x 5 TTA samples) = 25 test predictions averaged
  - Compare: single seed / 5-seed avg / 5-seed + TTA / 5-seed + TTA + ensemble with uni2h

Goal: validate whether pure model-seed ensemble (within same architecture) helps.
The OOF stacking A showed gigap+uni2h avg got +0.002, in the right direction.
Now test pure model-seed averaging as a complementary cheap optimization.
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

# Use Prov-GigaPath (baseline encoder)
arr = np.load(os.path.join(C, "NSCLC_gigap_luad_sub2048.npy"), mmap_mode="r")
DIM = arr.shape[2]
ix = pd.read_csv(os.path.join(C, "NSCLC_gigap_luad_sub2048_index.csv"))
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
    """Run test prediction n_tta times with different patch sample, average logits."""
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


# Single-fit timing
print("[bench] single fit...")
t0 = time.time()
fit(rows[:100], Y[:100], 999)  # warmup
print(f"  warmup fit = {time.time()-t0:.1f}s", flush=True)

print("\n" + "=" * 100, flush=True)
print(f"=== S35 5-seed ensemble + {N_TTA}-sample TTA (TCGA-LUAD N={len(cases)}, 3 fold-seed) ===", flush=True)
print("=" * 100, flush=True)

# Per-seed AUCs and ensembles
res_per_seed = {s: [] for s in MODEL_SEEDS}
res_5seed = []     # logit-avg over 5 model seeds, no TTA (use single TTA sample)
res_5seed_tta = []  # logit-avg over 5 model seeds × 5 TTA samples
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    oof_per_seed = {s: np.zeros((len(cases), len(GENES))) for s in MODEL_SEEDS}
    oof_5seed = np.zeros((len(cases), len(GENES)))
    oof_5seed_tta = np.zeros((len(cases), len(GENES)))
    for tri, tei in folds:
        for s in MODEL_SEEDS:
            net = fit(rows[tri], Y[tri], s)
            # 1 TTA
            rng = np.random.RandomState(s * 1000 + 7919)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                xb = get_batch(rows[tei], rng)
                p1 = net(xb).float().cpu().numpy()
            oof_per_seed[s][tei] = p1
            oof_5seed[tei] += p1 / len(MODEL_SEEDS)
            # 5 TTA
            oof_5seed_tta[tei] += predict_tta(net, rows[tei], n_tta=N_TTA) / len(MODEL_SEEDS)
    for s in MODEL_SEEDS:
        a = mean_auc(Y, oof_per_seed[s])
        res_per_seed[s].append(a)
    res_5seed.append(mean_auc(Y, oof_5seed))
    res_5seed_tta.append(mean_auc(Y, oof_5seed_tta))
    elapsed = (time.time()-T0)/60
    print(f"  foldseed {fi} [elapsed {elapsed:.1f}min]", flush=True)
    print(f"    per-seed AUCs:  " + "  ".join(f"s={s}={res_per_seed[s][-1]:.4f}" for s in MODEL_SEEDS), flush=True)
    print(f"    5-seed avg (1 TTA)  = {res_5seed[-1]:.4f}", flush=True)
    print(f"    5-seed avg (5 TTA)  = {res_5seed_tta[-1]:.4f}", flush=True)
    with open(r"C:\Users\ziyao\uni2h_etl\s100_progress.log", "a", encoding="utf-8") as pf:
        pf.write(f"[{elapsed:5.1f}min] foldseed {fi}: " + " ".join(f"s{s}={res_per_seed[s][-1]:.4f}" for s in MODEL_SEEDS) + f" ens1tta={res_5seed[-1]:.4f} ens5tta={res_5seed_tta[-1]:.4f}\n")

print("\n=== OOF mean (3 fold-seeds) ===", flush=True)
for s in MODEL_SEEDS:
    print(f"  s={s}  mean={np.mean(res_per_seed[s]):.4f}  std={np.std(res_per_seed[s], ddof=1):.4f}", flush=True)
print(f"  5-seed avg (1 TTA)  mean={np.mean(res_5seed):.4f}  std={np.std(res_5seed, ddof=1):.4f}  per_fseed={[round(v,4) for v in res_5seed]}", flush=True)
print(f"  5-seed avg (5 TTA)  mean={np.mean(res_5seed_tta):.4f}  std={np.std(res_5seed_tta, ddof=1):.4f}  per_fseed={[round(v,4) for v in res_5seed_tta]}", flush=True)
print(f"  S31 baseline (3 seed, no TTA): 0.7419", flush=True)

import json
res = {
    "per_seed_mean": {f"s{s}": float(np.mean(res_per_seed[s])) for s in MODEL_SEEDS},
    "per_seed_per_fold": {f"s{s}": [float(v) for v in res_per_seed[s]] for s in MODEL_SEEDS},
    "5seed_1tta_per_fold": [float(v) for v in res_5seed],
    "5seed_5tta_per_fold": [float(v) for v in res_5seed_tta],
    "5seed_1tta_mean": float(np.mean(res_5seed)),
    "5seed_5tta_mean": float(np.mean(res_5seed_tta)),
    "baseline_s31": 0.7419,
}
with open(r"C:\Users\ziyao\uni2h_etl\s100_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"[saved] s100_results.json", flush=True)
print("DONE", flush=True)
