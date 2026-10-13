"""119_kd_uni_3fs_external.py - full protocol for §118-like architecture, 3 fold-seed, save student weights

Goal: rigorous external validation of S48 architecture (KD alpha=1.0) on UNI v1 encoder, with
TCGA-LUAD 3-fold-seed x 5-fold x 3-model-seed internal OOF, plus CPTAC external 127 cases.
Each student model state_dict saved so the "best model" is reproducible.

Time: 3 fold-seed * 5 fold * 3 model-seed = 45 fits on TCGA (~15 min) + 3 fold-seed * 3 model-seed
      CPTAC external inference (~5 min) = ~20 min total
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
H_DIM, DROP = 256, 0.25
FOLD_SEEDS = [101, 202, 303]
MODEL_SEEDS = [0, 1, 2]
NFOLD = 5
dev = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DIR = r"C:\Users\ziyao\uni2h_etl\s119_weights"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Load features
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
proj_all = man.set_index("case_id")["project"].to_dict()

cp = pd.read_csv(os.path.join(C, "cptac_luad_usable.csv"))
c_cases = list(cp.case_id)
assert all(c in rm_c for c in c_cases)
Y_c = cp[GENES].values.astype(int)
g_c = OFF + np.array([rm_c[c] for c in c_cases])

# Teacher OOF
gigap = np.load(r"C:\Users\ziyao\uni2h_etl\oof_uni2h_5s5tta.npz", allow_pickle=True)
common_t = list(gigap["common"])
g_idx = {c: i for i, c in enumerate(common_t)}


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


def fit(tr_rows, tr_y, tr_teacher, seed, ckpt_path=None):
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
            loss = soft_loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            try: sch.step()
            except Exception: pass
    if ckpt_path:
        torch.save(net.state_dict(), ckpt_path)
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


# Match TCGA cases in teacher
t_cases = sorted(c for c in rm_t if c in lab.index and proj_all.get(c) == "LUAD" and c in g_idx)
Y_t = np.array([[int(lab.loc[c, g]) for g in GENES] for c in t_cases])
g_t = np.array([rm_t[c] for c in t_cases])
print(f"[data] TCGA {len(t_cases)}  CPTAC {len(c_cases)}", flush=True)

oof_per_fs = {}
ext_per_fs = {}

T0 = time.time()
for fs_idx, fs in enumerate(FOLD_SEEDS):
    P_teacher = gigap[f"fs{fs_idx}"][np.array([g_idx[c] for c in t_cases])]
    teacher_auc = mean_auc(Y_t, P_teacher)
    print(f"\n=== fold-seed {fs} (teacher AUC = {teacher_auc:.4f}) ===", flush=True)

    tier = (Y_t[:, GENES.index("TP53")].astype(str) + "_" + Y_t[:, GENES.index("NF1")].astype(str))
    folds = [(a, b) for a, b in StratifiedKFold(NFOLD, shuffle=True, random_state=fs).split(np.zeros(len(t_cases)), tier)]
    oof = np.zeros((len(t_cases), len(GENES)))
    for fold_idx, (tri, tei) in enumerate(folds):
        preds = []
        for s in MODEL_SEEDS:
            ckpt = os.path.join(WEIGHT_DIR, f"fs{fs}_fold{fold_idx}_seed{s}.pt")
            net = fit(g_t[tri], Y_t[tri], P_teacher[tri], s, ckpt_path=ckpt)
            preds.append(predict(net, g_t[tei], s))
        oof[tei] = np.mean(preds, axis=0)
    auc_tcga = mean_auc(Y_t, oof)
    print(f"  TCGA OOF fs{fs} = {auc_tcga:.4f}  ({(time.time()-T0)/60:.1f}min)", flush=True)
    oof_per_fs[fs] = oof

    # CPTAC external: train on full TCGA, predict CPTAC
    ext_preds = []
    for s in MODEL_SEEDS:
        ckpt = os.path.join(WEIGHT_DIR, f"fs{fs}_full_seed{s}.pt")
        net = fit(g_t, Y_t, P_teacher, s, ckpt_path=ckpt)
        ext_preds.append(predict(net, g_c, s))
    P_ext_avg = np.mean(ext_preds, axis=0)
    auc_ext = mean_auc(Y_c, P_ext_avg)
    print(f"  CPTAC ext fs{fs} = {auc_ext:.4f}", flush=True)
    ext_per_fs[fs] = (P_ext_avg, auc_ext)

# Aggregate
mean_oof_aucs = []
mean_ext_aucs = []
for fs in FOLD_SEEDS:
    P_ext_avg, auc_ext = ext_per_fs[fs]
    auc_tcga = mean_auc(Y_t, oof_per_fs[fs])
    mean_oof_aucs.append(auc_tcga)
    mean_ext_aucs.append(auc_ext)
    print(f"  fs{fs}: TCGA={auc_tcga:.4f}  CPTAC={auc_ext:.4f}  drop={auc_ext-auc_tcga:+.4f}", flush=True)

print(f"\n=== §119 FINAL (3 fold-seed) ===", flush=True)
print(f"  TCGA OOF  mean +/- std = {np.mean(mean_oof_aucs):.4f} +/- {np.std(mean_oof_aucs):.4f}", flush=True)
print(f"  CPTAC ext mean +/- std = {np.mean(mean_ext_aucs):.4f} +/- {np.std(mean_ext_aucs):.4f}", flush=True)
print(f"  drop mean = {np.mean(mean_ext_aucs) - np.mean(mean_oof_aucs):+.4f}", flush=True)

# Per-gene
oof_avg = np.mean([oof_per_fs[f] for f in FOLD_SEEDS], axis=0)
P_ext_avg_total = np.mean([ext_per_fs[f][0] for f in FOLD_SEEDS], axis=0)
print(f"  Per-gene (mean of 3 fs OOF):", flush=True)
for g in range(len(GENES)):
    auc = roc_auc_score(Y_t[:, g], oof_avg[:, g])
    print(f"    TCGA {GENES[g]:6s}: {auc:.4f}", flush=True)
print(f"  Per-gene (mean of 3 fs ext pred):", flush=True)
for g in range(len(GENES)):
    if MIN_POS <= Y_c[:, g].sum() <= len(Y_c) - MIN_POS:
        auc = roc_auc_score(Y_c[:, g], P_ext_avg_total[:, g])
        print(f"    CPTAC {GENES[g]:6s}: {auc:.4f}  (n_pos={int(Y_c[:, g].sum())})", flush=True)
    else:
        print(f"    CPTAC {GENES[g]:6s}: n/a  (n_pos={int(Y_c[:, g].sum())})", flush=True)

# Save
np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s119_3fs_external.npz",
                    **{f"oof_fs{f}": oof_per_fs[f] for f in FOLD_SEEDS},
                    **{f"ext_fs{f}": ext_per_fs[f][0] for f in FOLD_SEEDS},
                    Y_t=Y_t, common_t=np.array(t_cases, dtype=object),
                    Y_c=Y_c, common_c=np.array(c_cases, dtype=object))
print(f"\n[saved] s119_3fs_external.npz", flush=True)
print(f"DONE total = {(time.time()-T0)/60:.1f}min", flush=True)