"""118_kd_uni_cptac_external.py - S48 self-distillation in UNI v1 encoder + CPTAC external validation

Why UNI v1 not Prov-GigaPath:
  - CPTAC-LUAD patch features only exist locally as UNI v1 1024-d (data/patch_cache/CPTAC_uni_sub512.npy)
  - Prov-GigaPath 1536-d features for CPTAC not extracted (would take ~ 30 hours)
  - Switch encoder to UNI v1, replicate S48 architecture + KD alpha=1.0 on TCGA, evaluate on CPTAC 127 cases

Protocol:
  - TCGA-LUAD N=434 UNI v1 1024-d patches (NSCLC_uni_sub512.npy)
  - teacher = oof_uni2h_5s5tta.npz (TCGA 5s5tta OOF, 6-dim soft logits)
  - student trained with KD alpha=1.0 (pure distillation, no hard BCE)
  - external eval on CPTAC-LUAD 127 cases (CPTAC_uni_sub512.npy + cptac_luad_usable.csv)
  - 5-fold OOF on TCGA, evaluate on CPTAC once per fold-seed for ABMIL avg

Time: 1 fold-seed * 5 fold * 3 model-seed = 15 fits on TCGA (~ 15 min) + 1 fit per seed on CPTAC (~ 3 min) = 18 min
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
DIM, BATCH, S_INST, LR, WD = 1024, 32, 512, 3e-4, 1e-3
EPOCH = 25
T = 2.0
LAMBDA_KD = 1.0
H_DIM = 256
DROP = 0.25
FOLD_SEEDS = [101]  # first run; can extend
MODEL_SEEDS = [0, 1, 2]
NFOLD = 5
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Load TCGA UNI v1
Xt = np.load(os.path.join(C, "NSCLC_uni_sub512.npy"))
Xc = np.load(os.path.join(C, "CPTAC_uni_sub512.npy"))
OFF = len(Xt)
X = np.concatenate([Xt, Xc], axis=0)
del Xt, Xc

ui = pd.read_csv(os.path.join(C, "NSCLC_uni_sub512_index.csv"))
ci = pd.read_csv(os.path.join(C, "CPTAC_uni_sub512_index.csv"))
rm_t = dict(zip(ui.case_id, ui.mem_row))
rm_c = dict(zip(ci.case_id, ci.mem_row))

man = pd.read_csv(os.path.join(R, "data", "labels", "case_manifest.csv"))
man["project"] = man.project.astype(str)
lab = man[man.has_label == 1].set_index("case_id")

t_cases = sorted(c for c in rm_t if c in lab.index and man.set_index("case_id").project[c] == "LUAD")
cp = pd.read_csv(os.path.join(C, "cptac_luad_usable.csv"))
c_cases = list(cp.case_id)
assert all(c in rm_c for c in c_cases)

Y_t = np.array([[int(lab.loc[c, g]) for g in GENES] for c in t_cases])
Y_c = cp[GENES].values.astype(int)
g_t = np.array([rm_t[c] for c in t_cases])
g_c = OFF + np.array([rm_c[c] for c in c_cases])

print(f"[data] TCGA-LUAD N={len(t_cases)}  CPTAC-LUAD N={len(c_cases)}", flush=True)
print(f"[data] TCGA  Y_pos = {Y_t.sum(0).tolist()}", flush=True)
print(f"[data] CPTAC Y_pos = {Y_c.sum(0).tolist()}", flush=True)

# Teacher: oof_uni2h_5s5tta (TCGA UNI2-h 5-seed 5-TTA OOF)
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)
common_t = list(gigap["common"])
# t_cases 都是 TCGA- 开头的, 与 common 匹配
g_idx = {c: i for i, c in enumerate(common_t)}
P_teacher_fs = {fs_idx: gigap[f"fs{fs_idx}"] for fs_idx in range(3)}


def mean_auc(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j])
                          for j in range(len(GENES)) if MIN_POS <= Y[:, j].sum() <= len(Y) - MIN_POS]))


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
    net = ABMIL(DIM, len(GENES), H_DIM, DROP).to(dev)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCH,
                                              steps_per_epoch=max(1, len(tr_rows) // BATCH))
    teacher_t = torch.from_numpy(tr_teacher).to(dev).float()
    rng = np.random.RandomState(seed)
    for ep in range(EPOCH):
        net.train(); perm = rng.permutation(len(tr_rows))
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            r = perm[i:i + BATCH]
            bt = np.empty((BATCH, S_INST, DIM), np.float16)
            for j, rr in enumerate(r):
                v = X[tr_rows[rr]]; m = v.shape[0]
                bt[j] = v[rng.choice(m, S_INST, replace=m < S_INST)]
            xb = torch.from_numpy(bt).to(dev, torch.float32)
            yb = torch.from_numpy(tr_y[r].astype(np.float32)).to(dev)
            tb = teacher_t[r]
            sl = net(xb)
            soft_loss = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                 F.softmax(tb / T, dim=-1),
                                 reduction="batchmean") * (T ** 2)
            loss = soft_loss  # alpha=1.0 -> pure KD
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, rows, seed):
    net.eval()
    rng = np.random.RandomState(seed + 999)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        bt = np.empty((len(rows), S_INST, DIM), np.float16)
        for j, r in enumerate(rows):
            v = X[r]; m = v.shape[0]
            bt[j] = v[rng.choice(m, S_INST, replace=m < S_INST)]
        return net(torch.from_numpy(bt).to(dev, torch.float32)).float().cpu().numpy()


# Map TCGA cases to teacher OOF indices
t_idx_in_teacher = np.array([g_idx[c] for c in t_cases if c in g_idx])
t_cases_in_teacher = [c for c in t_cases if c in g_idx]
Y_t = np.array([[int(lab.loc[c, g]) for g in GENES] for c in t_cases_in_teacher])
g_t = np.array([rm_t[c] for c in t_cases_in_teacher])
print(f"[teacher] TCGA in teacher: {len(t_cases_in_teacher)}/{len(t_cases)}", flush=True)

# Run fold-seed 101
fs = 101
fs_idx = FOLD_SEEDS.index(fs)
print(f"\n=== S118 TCGA UNI v1 + KD alpha=1.0, fold-seed {fs} ===", flush=True)
P_teacher = P_teacher_fs[fs_idx][t_idx_in_teacher]
print(f"[teacher] mean AUC = {mean_auc(Y_t, P_teacher):.4f}", flush=True)

tier = (Y_t[:, GENES.index("TP53")].astype(str) + "_" + Y_t[:, GENES.index("NF1")].astype(str))
folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(t_cases_in_teacher)), tier)]

T0 = time.time()
oof = np.zeros((len(t_cases_in_teacher), len(GENES)))
for fold_idx, (tri, tei) in enumerate(folds):
    preds = []
    for s in MODEL_SEEDS:
        net = fit(g_t[tri], Y_t[tri], P_teacher[tri], s)
        preds.append(predict(net, g_t[tei], s))
    oof[tei] = np.mean(preds, axis=0)
    fold_auc = mean_auc(Y_t[tei], oof[tei])
    print(f"  fold {fold_idx}/5 fold_AUC={fold_auc:.4f} ({(time.time()-T0)/60:.1f}min)", flush=True)

auc_tcga = mean_auc(Y_t, oof)
print(f"\n=== TCGA OOF AUC = {auc_tcga:.4f} ({(time.time()-T0)/60:.1f}min) ===", flush=True)
print(f"  per-gene TCGA:", flush=True)
for g in range(len(GENES)):
    print(f"    {GENES[g]:6s}: {roc_auc_score(Y_t[:, g], oof[:, g]):.4f}", flush=True)

# External evaluation on CPTAC
print(f"\n=== CPTAC-LUAD 127 cases external ===", flush=True)
preds_ext = []
for s in MODEL_SEEDS:
    # Train on full TCGA, evaluate on CPTAC
    net = fit(g_t, Y_t, P_teacher, s)
    P_ext = predict(net, g_c, s)
    preds_ext.append(P_ext)
P_ext_avg = np.mean(preds_ext, axis=0)
auc_cptac = mean_auc(Y_c, P_ext_avg)
print(f"  CPTAC mean AUC = {auc_cptac:.4f}", flush=True)
print(f"  per-gene CPTAC:", flush=True)
for g in range(len(GENES)):
    if MIN_POS <= Y_c[:, g].sum() <= len(Y_c) - MIN_POS:
        print(f"    {GENES[g]:6s}: {roc_auc_score(Y_c[:, g], P_ext_avg[:, g]):.4f} (n_pos={int(Y_c[:, g].sum())})", flush=True)
    else:
        print(f"    {GENES[g]:6s}: n/a (n_pos={int(Y_c[:, g].sum())})", flush=True)

# Save
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s118_kd_uni_external.npz",
                    oof_tcga=oof, Y_t=Y_t, common_t=np.array(t_cases_in_teacher, dtype=object),
                    P_ext=P_ext_avg, Y_c=Y_c, common_c=np.array(c_cases, dtype=object))
print(f"\n[saved] s118_kd_uni_external.npz", flush=True)

print(f"\n=== SUMMARY ===", flush=True)
print(f"  TCGA-LUAD 5-fold OOF AUC = {auc_tcga:.4f}", flush=True)
print(f"  CPTAC-LUAD external  AUC = {auc_cptac:.4f}", flush=True)
print(f"  Drop CPTAC-TCGA = {auc_cptac - auc_tcga:+.4f}", flush=True)
print(f"  vs baseline §31 (TCGA 0.7419, CPTAC 0.653, drop = {0.653 - 0.7419:+.4f})", flush=True)
print(f"DONE total = {(time.time()-T0)/60:.1f}min", flush=True)