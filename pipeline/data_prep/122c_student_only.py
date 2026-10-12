"""122c_student_only.py - only student training + LOOCV on saved features (skipping feature extraction)

Features already saved as s122b_feats_<case_id>.npy
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 2  # only 5 cases, original 8 is too strict
LABEL_CSV = os.path.join(R, "data", "patch_cache", "cptac_luad_usable.csv")
N_PATCH = 256
DIM = 1536
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Load features
patients = ["C3L-00009", "C3L-00093", "C3L-00094", "C3L-00263", "C3L-00279"]
all_feats = {}
for pid in patients:
    p = rf"C:\Users\ziyao\uni2h_etl\s122b_feats_{pid}.npy"
    all_feats[pid] = np.load(p)
print(f"[data] {len(all_feats)} cases, each {all_feats[patients[0]].shape}", flush=True)

lab_df = pd.read_csv(LABEL_CSV).set_index("case_id")
common = [c for c in patients if c in lab_df.index]
Y = np.array([[int(lab_df.loc[c, g]) for g in GENES] for c in common])
print(f"[labels] {len(common)} cases, Y_pos = {Y.sum(0).tolist()}", flush=True)


class ABMIL(nn.Module):
    def __init__(s, d_in, n_genes, h, drop):
        super().__init__()
        s.trunk = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop))
        s.V = nn.Linear(h, h, bias=False); s.U = nn.Linear(h, h, bias=False)
        s.w = nn.Linear(h, 1, bias=False); s.od = nn.Linear(h, n_genes)
    def forward(s, x):
        # x: (B, 256, 1536) - batched case
        # Always work in batched mode for simplicity
        if x.dim() == 2:
            x = x.unsqueeze(0)
        z = s.trunk(x)  # (B, 256, 256)
        # attention weights per patch
        a = torch.softmax(s.w(torch.tanh(s.V(z)) * torch.sigmoid(s.U(z))).squeeze(-1), 1)  # (B, 256)
        z_pool = (z * a.unsqueeze(-1)).sum(1)  # (B, 256)
        return s.od(z_pool)  # (B, n_genes)


def fit(Xtr, ytr):
    """Xtr: (N_cases * 256, 1536), ytr: (N_cases * 256, 6) — per-case label broadcast."""
    torch.manual_seed(0)
    net = ABMIL(DIM, len(GENES), 256, 0.25).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-3)
    n_cases = len(Xtr) // 256
    epochs = 25
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, epochs=epochs, steps_per_epoch=n_cases)
    bce = nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr).to(dev)  # (N_cases*256, 1536)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(dev)  # (N_cases*256, 6)
    rng = np.random.RandomState(0)
    for ep in range(epochs):
        case_order = rng.permutation(n_cases)
        for ci in case_order:
            start = ci * 256
            end = start + 256
            xb = Xtr_t[start:end].unsqueeze(0)  # (1, 256, 1536)
            yb = ytr_t[start]  # (6,) - all 256 patches have same label
            opt.zero_grad()
            out = net(xb)  # (1, 6)
            loss = bce(out.squeeze(0), yb)
            loss.backward(); opt.step()
            try: sch.step()
            except Exception: pass
    return net


def predict(net, Xte):
    net.eval()
    Xte_t = torch.from_numpy(Xte).to(dev)
    Xte_t = Xte_t.unsqueeze(0)  # add batch dim
    with torch.no_grad():
        return net(Xte_t).squeeze(0).cpu().numpy()


print(f"\n=== §122c LOOCV on 5 cases ===", flush=True)
preds = np.zeros((len(common), len(GENES)))
T0 = time.time()
for hold in range(len(common)):
    train_cases = [c for i, c in enumerate(common) if i != hold]
    Xtr = np.concatenate([all_feats[c] for c in train_cases], axis=0)
    ytr = np.repeat(Y[[i for i, c in enumerate(common) if c in train_cases]], 256, axis=0)
    Xte = all_feats[common[hold]]
    net = fit(Xtr, ytr)
    P = predict(net, Xte)
    preds[hold] = P
    print(f"  LOOCV fold {hold}: {common[hold]} done ({(time.time()-T0)/60:.1f}min)", flush=True)

# AUC per gene
aucs = []
print(f"\n=== §122c POC result (5 CPTAC cases, midnight encoder, ABMIL BCE) ===", flush=True)
for g in range(len(GENES)):
    if MIN_POS <= Y[:, g].sum() <= len(Y) - MIN_POS:
        auc = roc_auc_score(Y[:, g], preds[:, g])
        print(f"  {GENES[g]:6s}: AUC={auc:.4f}  (n_pos={int(Y[:, g].sum())})")
        aucs.append(auc)
    else:
        print(f"  {GENES[g]:6s}: n/a  (n_pos={int(Y[:, g].sum())})")
mean_auc = float(np.mean(aucs)) if aucs else float("nan")
print(f"\nmean AUC = {mean_auc:.4f} (across valid genes)", flush=True)
print(f"  vs §31 baseline UNI v1 CPTAC = 0.653  (delta {mean_auc-0.653:+.4f})", flush=True)

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s122c_poc_oof.npz",
                    preds=preds, Y=Y, common=np.array(common, dtype=object))
print(f"\n[saved] s122c_poc_oof.npz", flush=True)
print(f"DONE total = {(time.time()-T0)/60:.1f}min", flush=True)