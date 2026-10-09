"""98_pool_ensemble.py - §33: per-gene multi-pool head ensemble (gated + max + mean -> average logits)

Hypothesis:
  Current baseline (§31) uses a single gated attention pooler shared across all 6 genes.
  Two independent innovations are combined:
    (a) Per-gene attention pooler: each gene gets its own (V_g, U_g, w_g, od_g) so the
        model learns gene-specific tissue patterns (e.g. KRAS favours mucinous areas,
        TP53 favours high-grade nuclei) instead of one global mix.
    (b) Multi-pool ensemble inside each gene head: gated + max + mean average of logits.
        max/mean are parameter-free but capture different signal: max for sharp
        "is-there-any-positive-patch", mean for diffuse distribution, gated for soft
        attention. Averaging logits reduces variance.

  Both ideas are tested against the §31 baseline on the same N=434 TCGA-LUAD cohort,
  same fold split, same 5-fold x 3 fold-seed x 3 model-seed protocol.

Cost: ~1.5x parameters and ~1.2x compute vs single-pool gated.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
LR, WD, MIN_POS = 3e-4, 1e-3, 8
FOLD_SEEDS, MODEL_SEEDS, NFOLD = [101, 202, 303], [0, 1, 2], 5
CFG = dict(h=256, ep=25, drop=0.25)
S_INST = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Data: Prov-GigaPath TCGA-LUAD only (matches §31 baseline)
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


class MultiPoolNet(nn.Module):
    """Shared trunk -> per-gene (gated + max + mean) -> average logits.

    Parameters:
      trunk: 1536 -> 256
      per gene (k=6): V_g (256,256), U_g (256,256), w_g (256,1), od_g (256,1)
      Total extra vs baseline: 6 * (256*256 + 256*256 + 256 + 256 + 1) ~= 786k params
    """
    def __init__(s, d, h, k, drop):
        super().__init__()
        s.k = k
        s.trunk = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(k)])
        s.U = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(k)])
        s.w = nn.ModuleList([nn.Linear(h, 1, bias=False) for _ in range(k)])
        s.od = nn.ModuleList([nn.Linear(h, 1) for _ in range(k)])

    def forward(s, x):
        z = s.trunk(x)
        logits = []
        for g in range(s.k):
            a = torch.softmax(s.w[g](torch.tanh(s.V[g](z)) * torch.sigmoid(s.U[g](z))).squeeze(-1), 1)
            h_g = (z * a.unsqueeze(-1)).sum(1)            # gated attention
            h_mx = z.max(1).values                          # max pool
            h_mn = z.mean(1)                                 # mean pool
            lg = (s.od[g](h_g) + s.od[g](h_mx) + s.od[g](h_mn)) / 3.0
            logits.append(lg)
        return torch.cat(logits, dim=-1)


def get_batch(rows, rng, S=S_INST, D=DIM):
    bt = np.empty((len(rows), S, D), np.float16)
    for i, r in enumerate(rows):
        v = np.asarray(arr[r], dtype=np.float16)
        if v.shape[0] == 0:
            bt[i] = 0; continue
        bt[i] = v[rng.choice(v.shape[0], S, replace=v.shape[0] < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(tr_rows, tr_y, te_rows, seed):
    torch.manual_seed(seed)
    net = MultiPoolNet(DIM, CFG["h"], len(GENES), CFG["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    BATCH = 32
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=CFG["ep"], steps_per_epoch=max(1, len(tr_rows) // BATCH))
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
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = get_batch(te_rows, np.random.RandomState(seed + 999))
        return net(xb).float().cpu().numpy()


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


print("\n" + "=" * 100, flush=True)
print(f"=== S33 per-gene multi-pool ensemble (TCGA-LUAD N={len(cases)}, 3 fold-seed x 3 model-seed) ===", flush=True)
print("=" * 100, flush=True)

# ---- arm A: per-gene multi-pool ensemble ----
res = []
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    oof = np.zeros((len(cases), len(GENES)))
    for tri, tei in folds:
        oof[tei] = np.mean([fit(rows[tri], Y[tri], rows[tei], s) for s in MODEL_SEEDS], 0)
    res.append(mean_auc(Y, oof))
    print(f"  foldseed {fi}: multi-pool = {res[-1]:.4f}  [{(time.time()-T0)/60:.1f}min]", flush=True)
    with open(r"C:\Users\ziyao\uni2h_etl\s98_progress.log", "a", encoding="utf-8") as pf:
        pf.write(f"[{(time.time()-T0)/60:5.1f}min] foldseed {fi}: {res[-1]:.4f}\n")

print("\n=== OOF mean (3 fold-seeds) ===", flush=True)
print(f"  multi-pool  mean={np.mean(res):.4f}  std={np.std(res, ddof=1):.4f}  per_fseed={[round(v,4) for v in res]}", flush=True)
print(f"  S31 baseline (gated single-pool) for comparison: 0.7419 +/- 0.0039", flush=True)
print(f"  Delta = {np.mean(res)-0.7419:+.4f}", flush=True)

pd.DataFrame(dict(fold_seed=FOLD_SEEDS, mean_auc=res)).to_csv(
    os.path.join(C, "multi_pool_per_gene.csv"), index=False)
print("[saved] multi_pool_per_gene.csv", flush=True)
print("DONE", flush=True)
