"""99_oof_stacking.py - S34-A: OOF stacking across 3 encoders (gigap/uni_v1/uni2h)

Strategy:
  1. Run 1 fold-seed (101) x 5 fold x 3 encoders, **save OOF prediction (434,6) per encoder**
  2. Logit-average ensemble: P_ens = (P_gigap + P_uni_v1 + P_uni2h) / 3
  3. Per-pair averages: gigap+uni2h, gigap+uni_v1, uni2h+uni_v1
  4. LR meta-learner stacking: features = concat(3 OOFs, axis=1) -> 434 x 18
  5. Per-gene logistic regression on (P_gigap_gene, P_uni_v1_gene, P_uni2h_gene)

Compare to baselines (S31 main result 0.7419, S32 per-encoder means).
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
LR_, WD, MIN_POS = 3e-4, 1e-3, 8
FOLD_SEED, MODEL_SEEDS, NFOLD = 101, [0, 1, 2], 5
CFG = dict(h=256, ep=25, drop=0.25)
S_INST = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

ENC = {
    "gigap":  ("NSCLC_gigap_luad_sub2048.npy",  "NSCLC_gigap_luad_sub2048_index.csv",  1536),
    "uni_v1": ("NSCLC_uni_sub512.npy",          "NSCLC_uni_sub512_index.csv",          1024),
    "uni2h":  ("NSCLC_uni2h_luad_sub512.npy",   "NSCLC_uni2h_luad_sub512_index.csv",   1536),
}

# Build common case set (intersection of 3 encoders' TCGA-LUAD cases with has_label=1)
man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["case_id"] = man.case_id.astype(str)
lab = man[man.has_label == 1].set_index("case_id")

case_sets = {}
for enc, (npy, csv, dim) in ENC.items():
    ix = pd.read_csv(os.path.join(C, csv))
    ix["case_id"] = ix.case_id.astype(str)
    case_sets[enc] = set(ix.case_id.tolist()) & set(lab.index.tolist())

common = sorted(set.intersection(*case_sets.values()))
print(f"[data] common {len(common)} cases across 3 encoders", flush=True)
Y = np.array([[int(lab.loc[c, g]) for g in GENES] for c in common])
print(f"[data] Y_pos = {Y.sum(0).tolist()}", flush=True)

# Load feature arrays (mmap)
feats = {enc: np.load(os.path.join(C, npy), mmap_mode="r") for enc, (npy, _, _) in ENC.items()}
rows_idx = {}
for enc, (_, csv, _) in ENC.items():
    ix = pd.read_csv(os.path.join(C, csv))
    ix["case_id"] = ix.case_id.astype(str)
    rm = dict(zip(ix.case_id, ix.mem_row))
    rows_idx[enc] = np.array([rm[c] for c in common])

tier = (Y[:, GENES.index("TP53")].astype(str) + "_" + Y[:, GENES.index("NF1")].astype(str))
FOLDS = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(common)), tier)]


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


def get_batch(enc, rows, rng, S=S_INST):
    arr = feats[enc]
    D = ENC[enc][2]
    bt = np.empty((len(rows), S, D), np.float16)
    for i, r in enumerate(rows):
        v = np.asarray(arr[r], dtype=np.float16)
        if v.shape[0] == 0: bt[i] = 0; continue
        bt[i] = v[rng.choice(v.shape[0], S, replace=v.shape[0] < S)]
    return torch.from_numpy(bt).to(dev, torch.float32)


def fit(enc, tr_rows, tr_y, te_rows, seed):
    torch.manual_seed(seed)
    D = ENC[enc][2]
    net = Net(D, CFG["h"], len(GENES), CFG["drop"]).to(dev)
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
            xb = get_batch(enc, tr_rows[rr], rng)
            yb = torch.from_numpy(tr_y[rr].astype(np.float32)).to(dev)
            loss = lf(net(xb), yb).mul(pw).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        xb = get_batch(enc, te_rows, np.random.RandomState(seed + 999))
        return net(xb).float().cpu().numpy()


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


# ---- (1) Run 3 encoders, save OOF ----
oof = {enc: np.zeros((len(common), len(GENES))) for enc in ENC}
T0 = time.time()
for tri, tei in FOLDS:
    for enc in ENC:
        rows = rows_idx[enc][tri]
        oof[enc][tei] = np.mean([fit(enc, rows, Y[tri], rows_idx[enc][tei], s) for s in MODEL_SEEDS], 0)
    print(f"  fold done [{(time.time()-T0)/60:.1f}min]", flush=True)

for enc in ENC:
    auc = mean_auc(Y, oof[enc])
    print(f"  single-encoder {enc:8s}  OOF AUC = {auc:.4f}", flush=True)

# Save raw OOF logits for downstream
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\oof_3enc.npz",
                    gigap=oof["gigap"], uni_v1=oof["uni_v1"], uni2h=oof["uni2h"],
                    Y=Y, common=np.array(common, dtype=object))
print(f"[saved] oof_3enc.npz", flush=True)

# ---- (2) Logit average ----
print("\n=== Logit-average ensemble ===", flush=True)
combos = [
    ("gigap+uni2h",          ["gigap", "uni2h"]),
    ("gigap+uni_v1",          ["gigap", "uni_v1"]),
    ("uni2h+uni_v1",          ["uni2h", "uni_v1"]),
    ("gigap+uni2h+uni_v1",    ["gigap", "uni2h", "uni_v1"]),
    ("gigap+uni2h (50/50)",   ["gigap", "uni2h"]),  # same as first but explicit
]
for name, encs in combos:
    P = np.mean([oof[e] for e in encs], axis=0)
    auc = mean_auc(Y, P)
    print(f"  {name:30s}  OOF AUC = {auc:.4f}  (vs gigap solo {mean_auc(Y, oof['gigap']):.4f})", flush=True)

# ---- (3) LR meta-learner stacking (per-gene LR on 3 encoder logits) ----
print("\n=== Per-gene LR meta-learner (3 features per gene) ===", flush=True)
P_meta = np.zeros((len(common), len(GENES)))
for g in range(len(GENES)):
    X_meta = np.stack([oof["gigap"][:, g], oof["uni_v1"][:, g], oof["uni2h"][:, g]], axis=1)  # (n, 3)
    # 5-fold CV on the meta-learner (use FOLDS from main to keep parity)
    P_g = np.zeros(len(common))
    for tri, tei in FOLDS:
        lr = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        lr.fit(X_meta[tri], Y[tri, g])
        P_g[tei] = lr.predict_proba(X_meta[tei])[:, 1]
    P_meta[:, g] = P_g
auc = mean_auc(Y, P_meta)
print(f"  LR meta-learner (per-gene)  OOF AUC = {auc:.4f}", flush=True)

# Also a single LR across all genes (less likely to help)
print("\n=== Single LR on all-encoder concatenation (18-d features) ===", flush=True)
P_all = np.zeros((len(common), len(GENES)))
for g in range(len(GENES)):
    X = np.hstack([oof["gigap"], oof["uni_v1"], oof["uni2h"]])  # (n, 18)
    P_g = np.zeros(len(common))
    for tri, tei in FOLDS:
        lr = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        lr.fit(X[tri], Y[tri, g])
        P_g[tei] = lr.predict_proba(X[tei])[:, 1]
    P_all[:, g] = P_g
auc = mean_auc(Y, P_all)
print(f"  LR(18-d)  OOF AUC = {auc:.4f}", flush=True)

# Save results CSV
import json
res = {
    "single_encoder": {e: mean_auc(Y, oof[e]) for e in ENC},
    "logit_avg": {
        "gigap+uni2h": mean_auc(Y, (oof["gigap"]+oof["uni2h"])/2),
        "gigap+uni_v1": mean_auc(Y, (oof["gigap"]+oof["uni_v1"])/2),
        "uni2h+uni_v1": mean_auc(Y, (oof["uni2h"]+oof["uni_v1"])/2),
        "3encoder": mean_auc(Y, (oof["gigap"]+oof["uni2h"]+oof["uni_v1"])/3),
    },
    "lr_meta_per_gene": mean_auc(Y, P_meta),
    "lr_18d_all": mean_auc(Y, P_all),
    "y_pos": Y.sum(0).tolist(),
    "n_cases": len(common),
}
with open(r"C:\Users\ziyao\uni2h_etl\s99_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"[saved] s99_results.json", flush=True)
print("DONE", flush=True)
