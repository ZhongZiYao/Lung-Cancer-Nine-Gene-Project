"""97_encoder_swap.py - §32-1: same TCGA-LUAD cohort, swap encoder (Prov-GigaPath vs UNI v1 vs UNI2-h).

Pairing: 434 cases where all three encoders have features (intersection of three indices).
5 fold-seeds x 3 model-seeds x 3 encoders (fully paired per fold-seed/model-seed).
OOF mean AUC + paired bootstrap delta vs Prov-GigaPath baseline.
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
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- (1) load feature + index per encoder, take intersection of cases ----
ENC = {
    "gigap":  ("NSCLC_gigap_luad_sub2048.npy",  "NSCLC_gigap_luad_sub2048_index.csv",  1536),
    "uni_v1": ("NSCLC_uni_sub512.npy",          "NSCLC_uni_sub512_index.csv",          1024),
    "uni2h":  ("NSCLC_uni2h_luad_sub512.npy",   "NSCLC_uni2h_luad_sub512_index.csv",   1536),
}
ix = {k: pd.read_csv(os.path.join(C, v[1])) for k, v in ENC.items()}
for k, df in ix.items():
    df["case_id"] = df.case_id.astype(str)
common = sorted(set.intersection(*(set(df.case_id[df.case_id.str.startswith("TCGA-")]) for df in ix.values())))
print(f"[intersection] TCGA-LUAD cases shared by all 3 encoders: {len(common)}", flush=True)

# ---- (2) labels from case_manifest ----
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[(man.has_label == 1) & (man.case_id.isin(common))].set_index("case_id")
common = sorted(c for c in common if c in lab.index)
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in common])
print(f"[data] N={len(common)}  Y_pos={Y.sum(0).tolist()}", flush=True)

# ---- (3) concatenate all features into one global array ----
feats = {}
OFF = 0
g_rows = {}
for k, (npy, csv, dim) in ENC.items():
    arr = np.load(os.path.join(C, npy), mmap_mode="r")
    feats[k] = arr
    rm = dict(zip(ix[k].case_id, ix[k].mem_row))
    g_rows[k] = np.array([rm[c] for c in common])
    print(f"[{k}] npy={arr.shape}  dim={dim}  rows_used={len(g_rows[k])}", flush=True)

# ---- (4) fold splits based on (TP53, NF1) status ----
tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [[(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(common)), tier)]
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

def get_batch(enc, rows, rng, S=512, D=None):
    arr = feats[enc]
    bt = np.empty((len(rows), S, D), np.float16)
    for i, r in enumerate(rows):
        v = np.asarray(arr[r], dtype=np.float16)
        if v.shape[0] == 0:
            bt[i] = 0; continue
        bt[i] = v[rng.choice(v.shape[0], S, replace=v.shape[0] < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)

S_INST = 512

def fit(enc, tr_rows, tr_y, te_rows, seed):
    torch.manual_seed(seed)
    D = ENC[enc][2]
    net = Net(D, CFG["h"], len(GENES), CFG["drop"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    BATCH = 32
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=CFG["ep"],
                                              steps_per_epoch=max(1, len(tr_rows) // BATCH))
    lf = nn.BCEWithLogitsLoss(reduction="none")
    pw = torch.tensor(np.where(tr_y.sum(0) > 0, (len(tr_y) - tr_y.sum(0)) / np.maximum(tr_y.sum(0), 1), 1.),
                      dtype=torch.float32, device=dev)
    rng = np.random.RandomState(seed)
    for ep in range(CFG["ep"]):
        net.train(); perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            rr = perm[i:i + BATCH]
            xb = get_batch(enc, tr_rows[rr], rng, S=S_INST, D=D)
            yb = torch.from_numpy(tr_y[rr].astype(np.float32)).to(dev)
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = get_batch(enc, te_rows, np.random.RandomState(seed + 999), S=S_INST, D=D)
        return net(xb).float().cpu().numpy()

def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))

print("\n" + "=" * 100, flush=True)
print(f"=== §32-1 同队列换编码器 (TCGA-LUAD N={len(common)}, 3 fold-seeds x 3 model-seeds) ===", flush=True)
print("=" * 100, flush=True)
res = {k: [] for k in ENC}
T0 = time.time()
for fi, folds in enumerate(FOLDS):
    oof = {k: np.zeros((len(common), len(GENES))) for k in ENC}
    for fi_idx, (tri, tei) in enumerate(folds):
        for enc in ENC:
            rows = g_rows[enc][tri]
            fold_t = time.time()
            for s in MODEL_SEEDS:
                fit(enc, rows, Y[tri], g_rows[enc][tei], s)
                print(f"    [{time.time()-T0:6.0f}s] foldseed {fi} fold {fi_idx} enc={enc} seed={s} done", flush=True)
                with open(r"C:\Users\ziyao\uni2h_etl\s97_progress.log", "a", encoding="utf-8") as pf:
                    pf.write(f"[{time.time()-T0:6.0f}s] foldseed {fi} fold {fi_idx} enc={enc} seed={s} done\n")
                    pf.flush()
            oof[enc][tei] = np.mean([fit(enc, rows, Y[tri], g_rows[enc][tei], s) for s in MODEL_SEEDS], 0)
    for enc in ENC:
        res[enc].append(mean_auc(Y, oof[enc]))
    line = "  ".join("%s=%.4f" % (k, res[k][-1]) for k in ENC)
    print(f"  foldseed {fi}: {line}  [{time.time()-T0:.0f}s]", flush=True)

base = "gigap"
deltas = {k: [b - a for a, b in zip(res[base], res[k])] for k in ENC if k != base}
print("\n=== OOF mean (per encoder, 3 fold-seeds) ===", flush=True)
for k in ENC:
    print(f"  {k:8s}  mean={np.mean(res[k]):.4f}  std={np.std(res[k], ddof=1):.4f}  per_fseed={[round(v,4) for v in res[k]]}", flush=True)

print("\n=== Paired Δ vs gigap baseline ===", flush=True)
for k, d in deltas.items():
    same = sum(1 for x in d if x > 0)
    print(f"  {k:8s}  Δ={np.mean(d):+.4f}  per_fseed={[round(v,4) for v in d]}  same_sign={same}/3", flush=True)

print(f"\n  判定阈值 2σ=0.014  ->  ", end="", flush=True)
results = []
for k, d in deltas.items():
    verdict = "胜出 ✓" if np.mean(d) > 0.014 else ("劣于 ✗" if np.mean(d) < -0.014 else "落在噪声内")
    results.append((k, np.mean(d), verdict))
for k, m, v in results:
    print(f"{k} {v} ", end="", flush=True)
print(flush=True)

rows_out = []
for fs_i, fs in enumerate(FOLD_SEEDS):
    for k in ENC:
        rows_out.append(dict(fold_seed=fs, encoder=k, mean_auc=res[k][fs_i]))
pd.DataFrame(rows_out).to_csv(os.path.join(C, "encoder_swap.csv"), index=False)
print(f"[saved] encoder_swap.csv", flush=True)
print("DONE", flush=True)
