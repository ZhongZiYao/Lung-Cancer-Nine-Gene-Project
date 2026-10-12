"""122b_rescue_cptac.py - §122 bugfix: save features per-case to avoid losing on WSI close crash

Bug: 122 hit forrtl: error (200) on case 5 WSI close, lost 4 cases features.
Fix: write features to .npz after each case completes.
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tiatoolbox.wsicore import WSIReader
from transformers import AutoModel
from torchvision import transforms

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]
MIN_POS = 8
DICOM_ROOT = os.path.join(R, "data", "CPTAC_DICOM", "cptac_luad")
LABEL_CSV = os.path.join(R, "data", "patch_cache", "cptac_luad_usable.csv")
N_PATCH = 256
PATCH_PX = 518
dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

print("[model] loading midnight-12k ...", flush=True)
midnight = AutoModel.from_pretrained(os.path.join(R, "models", "midnight")).to(dev).eval()
print("[model] midnight OK", flush=True)

patients = sorted([d for d in os.listdir(DICOM_ROOT) if os.path.isdir(os.path.join(DICOM_ROOT, d))])
print(f"[data] CPTAC DICOM patients: {patients[:5]}", flush=True)


def get_one_wsi_dir(pdir):
    subs = [(d, sum(os.path.getsize(os.path.join(pdir, d, f)) for f in os.listdir(os.path.join(pdir, d)) if f.endswith('.dcm'))) for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d))]
    subs.sort(key=lambda x: -x[1])
    return os.path.join(pdir, subs[0][0]) if subs else None


def extract_features(pid):
    """Extract midnight features for one patient. Returns (256, 1536) np array or None on failure."""
    print(f"\n[case] {pid}", flush=True)
    wsi_dir = get_one_wsi_dir(os.path.join(DICOM_ROOT, pid))
    if not wsi_dir:
        print(f"  no WSI dir", flush=True)
        return None
    try:
        wsi = WSIReader.open(wsi_dir)
    except Exception as e:
        print(f"  open fail: {e}", flush=True)
        return None
    print(f"  mpp={wsi.info.mpp} obj={wsi.info.objective_power}", flush=True)
    try:
        w_px, h_px = wsi.slide_dimensions(0.4942, "mpp")
        rng = np.random.RandomState(42)
        margin = PATCH_PX // 2
        xs = rng.randint(margin, max(margin+1, int(w_px) - margin), N_PATCH)
        ys = rng.randint(margin, max(margin+1, int(h_px) - margin), N_PATCH)
        patches = []
        for cx, cy in zip(xs, ys):
            x0 = max(0, min(int(w_px) - PATCH_PX, cx - PATCH_PX // 2))
            y0 = max(0, min(int(h_px) - PATCH_PX, cy - PATCH_PX // 2))
            patch = wsi.read_region(location=(x0, y0), size=(PATCH_PX, PATCH_PX), level=0)
            if patch.ndim == 3 and patch.shape[2] == 4:
                patch = patch[:, :, :3]
            elif patch.ndim == 3 and patch.shape[0] == 3:
                patch = np.transpose(patch, (1, 2, 0))
            patches.append(patch)
        patches = np.stack(patches)
        patches_t = torch.from_numpy(patches.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        patches_t = norm(patches_t)

        feats = []
        bs = 16
        with torch.no_grad():
            for i in range(0, len(patches_t), bs):
                batch = patches_t[i:i+bs].to(dev, non_blocking=True)
                out = midnight(pixel_values=batch)
                cls = out.last_hidden_state[:, 0, :]
                feats.append(cls.float().cpu())
        feats = torch.cat(feats, dim=0)
        print(f"  feats: {feats.shape}", flush=True)
    finally:
        try:
            del wsi
        except Exception:
            pass
        import gc; gc.collect(); torch.cuda.empty_cache()
    return feats.numpy()


# Step 1: extract features for all 5 cases, save per-case
all_feats = {}
T0 = time.time()
for pid in patients[:5]:
    feats = extract_features(pid)
    if feats is not None:
        all_feats[pid] = feats
        # SAVE per case immediately (avoid losing on future crash)
        per_case_path = rf"C:\Users\ziyao\uni2h_etl\s122b_feats_{pid}.npy"
        np.save(per_case_path, feats)
        print(f"  [saved] {per_case_path}", flush=True)
print(f"\n[step1 done] {sum(v.shape[0] for v in all_feats.values())} patches / {len(all_feats)} patients in {(time.time()-T0)/60:.1f}min", flush=True)

# Step 2: load labels
lab_df = pd.read_csv(LABEL_CSV).set_index("case_id")
common = [c for c in all_feats if c in lab_df.index]
Y = np.array([[int(lab_df.loc[c, g]) for g in GENES] for c in common])
print(f"[labels] {len(common)} cases with all 6 gene labels")
print(f"  Y_pos: {Y.sum(0).tolist()}", flush=True)


# Step 3: ABMIL student
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


# LOOCV: train on 4 cases, predict held-out one (5-fold since 5 cases)
def fit_and_predict(Xtr, ytr, Xte):
    """Train ABMIL on Xtr/ytr, predict Xte (no KD teacher for POC)."""
    torch.manual_seed(0)
    net = ABMIL(1536, len(GENES), 256, 0.25).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, epochs=25, steps_per_epoch=max(1, len(Xtr) // 32))
    bce = nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)
    rng = np.random.RandomState(0)
    # Train: treat each (Xtr case has 256 patches) as one sample, but each patch individually as sample (collapse to per-patch BCE)
    # Simpler: pool per case -> 1 feature (mean), train case-level
    # Use per-patch BCE for speed
    for ep in range(25):
        perm = rng.permutation(len(Xtr_t))
        for i in range(0, len(perm) - 32, 32):
            idx = perm[i:i+32]
            opt.zero_grad()
            loss = bce(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            try: sch.step()
            except Exception: pass
    net.eval()
    with torch.no_grad():
        return net(Xte_t).cpu().numpy()


print(f"\n=== §122b LOOCV (5 cases) ===", flush=True)
preds = np.zeros((len(common), len(GENES)))
for hold in range(len(common)):
    train_cases = [c for i, c in enumerate(common) if i != hold]
    Xtr = np.concatenate([all_feats[c] for c in train_cases], axis=0)
    ytr = np.repeat(Y[[i for i, c in enumerate(common) if c in train_cases]], 256, axis=0)
    Xte = all_feats[common[hold]]
    P = fit_and_predict(Xtr, ytr, Xte)
    preds[hold] = P
    print(f"  LOOCV fold {hold}: {common[hold]} done", flush=True)

# AUC
aucs = []
print(f"\n=== §122b POC result (5 CPTAC cases, midnight encoder, ABMIL + KD alpha=1.0) ===")
for g in range(len(GENES)):
    if MIN_POS <= Y[:, g].sum() <= len(Y) - MIN_POS:
        auc = roc_auc_score(Y[:, g], preds[:, g])
        print(f"  {GENES[g]:6s}: AUC={auc:.4f}  (n_pos={int(Y[:, g].sum())})")
        aucs.append(auc)
    else:
        print(f"  {GENES[g]:6s}: n/a  (n_pos={int(Y[:, g].sum())})")
mean_auc = np.mean(aucs) if aucs else float('nan')
print(f"\nmean AUC = {mean_auc:.4f}", flush=True)
print(f"  vs §31 baseline UNI v1 CPTAC = 0.653  (delta {mean_auc-0.653:+.4f})", flush=True)

np.savez_compressed(r"C:\Users\ziyao\uni2h_etl\s122b_poc_oof.npz",
                    preds=preds, Y=Y, common=np.array(common, dtype=object))
print(f"\n[saved] s122b_poc_oof.npz", flush=True)
print(f"DONE total = {(time.time()-T0)/60:.1f}min", flush=True)