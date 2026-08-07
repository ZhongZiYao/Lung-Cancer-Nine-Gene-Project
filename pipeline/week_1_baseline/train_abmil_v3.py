"""
train_abmil_v3.py
─────────────────
ABMIL on deepgem_feat_v3 (7000 patch/case avg, 36 case)

diff vs feat_416:
  • 36 case (vs 415) — small N
  • 7000 patch/case (vs 640) — large N
  • 5 file corrupt (skipped)
  • subsample to MAX_PATCH per case (random per epoch)

Pipeline:
  1. Load v3 via manual zipfile
  2. 5-fold OOF on KRAS
  3. Random subsample MAX_PATCH per case per epoch (avoid OOM + noise)
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(line_buffering=True)
import zipfile
import io
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as _roc_auc

sys.path.insert(0, str(Path(__file__).parent))
from abmil_model import ABMIL, abmil_bce_loss

REPO = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
V3_DIR = REPO / "deepgem_feat_v3"
PANEL = REPO / "data" / "MC3" / "9gene_panel_LUAD.csv"
OUT_DIR = REPO / "pipeline" / "week_1_baseline" / "outputs_abmil_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN = 256
ATTN_DIM = 128
DROPOUT = 0.25
LR = 2e-4
WEIGHT_DECAY = 1e-4
BATCH_CASES = 2          # smaller batch (more patches per case)
N_FOLDS = 5
N_EPOCHS = 30
PATIENCE = 8
MAX_PATCH = 2000         # subsample per epoch
GENE = "KRAS"            # start with KRAS


def load_v3_features(npz_path: Path) -> np.ndarray | None:
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            with zf.open("features.npy") as fp:
                return np.load(io.BytesIO(fp.read())).astype(np.float32)
    except Exception:
        return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("loading v3 features ...", flush=True)
    t0 = time.time()
    feat_cache = {}
    for p in sorted(V3_DIR.glob("*.npz")):
        feats = load_v3_features(p)
        if feats is not None and feats.ndim == 2 and feats.shape[-1] == 1024:
            feat_cache[p.stem] = feats
    print(f"  loaded {len(feat_cache)} cases ({time.time()-t0:.1f}s)", flush=True)
    ns = np.array([v.shape[0] for v in feat_cache.values()])
    print(f"  patch: min={ns.min()} max={ns.max()} mean={ns.mean():.0f} median={np.median(ns):.0f}", flush=True)

    labels = {}
    with open(PANEL) as f:
        rd = csv.DictReader(f)
        for r in rd:
            labels[r["case_id"]] = int(r[GENE])
    case_ids = sorted([c for c in feat_cache.keys() if c in labels])
    y = np.array([labels[c] for c in case_ids])
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    print(f"\n{GENE}: {len(case_ids)} cases, {n_pos} pos, {n_neg} neg ({100*n_pos/len(y):.1f}%)", flush=True)
    print(f"Config: hidden={HIDDEN} dropout={DROPOUT} LR={LR} batch={BATCH_CASES}cases "
          f"MAX_PATCH={MAX_PATCH}  {N_FOLDS}-fold", flush=True)

    rng = np.random.default_rng(42)
    oof = np.zeros(len(case_ids), dtype=np.float32)
    fold_aucs = []

    for fold_i, (tr, te) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42).split(np.zeros(len(y)), y), 1):
        tr_cases = [case_ids[i] for i in tr]
        te_cases = [case_ids[i] for i in te]
        tr_y = y[tr]; te_y = y[te]
        n_val = max(2, int(len(tr) * 0.15))
        perm = rng.permutation(len(tr))
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        tr_cases_ = [tr_cases[i] for i in tr_idx]
        val_cases = [tr_cases[i] for i in val_idx]
        tr_y_ = tr_y[tr_idx]; val_y_ = tr_y[val_idx]
        n_pos_tr = int(tr_y_.sum())
        pos_w = torch.tensor([min((len(tr_y_) - n_pos_tr) / max(n_pos_tr, 1), 10.0)], device=device)
        print(f"\n[fold {fold_i}/{N_FOLDS}] train={len(tr_cases_)} val={len(val_cases)} test={len(te_cases)} pos_w={pos_w.item():.2f}", flush=True)

        model = ABMIL(in_dim=1024, hidden=HIDDEN, attn_dim=ATTN_DIM, dropout=DROPOUT).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        print(f"  model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

        best_val_loss = float("inf")
        best_state = None
        best_val_auc = 0.0
        best_val_auc_ep = 0
        epochs_no_improve = 0

        for ep in range(N_EPOCHS):
            t_ep = time.time()
            model.train()
            ep_rng = np.random.default_rng(ep + fold_i * 1000)
            perm = ep_rng.permutation(len(tr_cases_))
            losses = []
            for b_start in range(0, len(perm), BATCH_CASES):
                batch = perm[b_start: b_start + BATCH_CASES]
                opt.zero_grad()
                b_loss = torch.zeros(1, device=device)
                for ti in batch:
                    pf_full = feat_cache[tr_cases_[ti]]
                    # subsample MAX_PATCH
                    if pf_full.shape[0] > MAX_PATCH:
                        sel = ep_rng.choice(pf_full.shape[0], MAX_PATCH, replace=False)
                        pf = torch.from_numpy(pf_full[sel]).to(device)
                    else:
                        pf = torch.from_numpy(pf_full).to(device)
                    logit, _ = model(pf)
                    b_loss = b_loss + abmil_bce_loss(
                        logit, torch.tensor([float(tr_y_[ti])], device=device), pos_weight=pos_w
                    )
                b_loss = b_loss / len(batch)
                b_loss.backward()
                opt.step()
                losses.append(b_loss.item())
            avg_loss = float(np.mean(losses))

            # val
            model.eval()
            with torch.no_grad():
                v_preds = []
                for c in val_cases:
                    pf = torch.from_numpy(feat_cache[c]).to(device)
                    logit, _ = model(pf)
                    v_preds.append(float(torch.sigmoid(logit).cpu().item()))
            v_p = np.array(v_preds)
            if 0 < val_y_.sum() < len(val_y_):
                v_auc = float(_roc_auc(val_y_, v_p))
            else:
                v_auc = float("nan")
            improved_loss = avg_loss < best_val_loss
            if improved_loss:
                best_val_loss = avg_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not np.isnan(v_auc) and v_auc > best_val_auc:
                best_val_auc = v_auc
                best_val_auc_ep = ep + 1
            epochs_no_improve = 0 if improved_loss else epochs_no_improve + 1
            marker = ""
            if improved_loss: marker += " *L"
            if not np.isnan(v_auc) and v_auc == best_val_auc: marker += " *V"
            print(f"  ep{ep+1:>2}/{N_EPOCHS}  loss={avg_loss:.3f}  val_auc={v_auc:.3f}  "
                  f"({time.time()-t_ep:.1f}s){marker}", flush=True)
            if epochs_no_improve >= PATIENCE:
                print(f"  [early stop] ep{ep+1}  best_val_loss={best_val_loss:.3f}  "
                      f"best_val_auc={best_val_auc:.3f} @ ep{best_val_auc_ep}", flush=True)
                break

        # test
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            t_preds = []
            for c in te_cases:
                pf = torch.from_numpy(feat_cache[c]).to(device)
                logit, _ = model(pf)
                t_preds.append(float(torch.sigmoid(logit).cpu().item()))
        oof[te] = np.array(t_preds)
        t_auc = _roc_auc(te_y, t_preds) if 0 < te_y.sum() < len(te_y) else float("nan")
        fold_aucs.append(t_auc)
        print(f"  >>> [fold {fold_i} done] test_AUC={t_auc:.3f}  best_val_AUC={best_val_auc:.3f}\n", flush=True)

    # final OOF
    oof_auc = _roc_auc(y, oof) if 0 < y.sum() < len(y) else float("nan")
    print(f"\n{'='*60}\n  {GENE}  v3 汇总\n{'='*60}")
    print(f"  per-fold test AUC: {fold_aucs}")
    print(f"  OOF AUC: {oof_auc:.3f}")

    # save
    with open(OUT_DIR / "abmil_v3_kras_oof.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", f"true_{GENE}", "pred_prob"])
        for i, c in enumerate(case_ids):
            w.writerow([c, int(y[i]), f"{oof[i]:.6f}"])
    with open(OUT_DIR / "abmil_v3_kras_summary.txt", "w") as f:
        f.write(f"ABMIL on v3 — {GENE}\n")
        f.write(f"cases: {len(case_ids)}  patch mean: {ns.mean():.0f}  MAX_PATCH: {MAX_PATCH}\n")
        f.write(f"per-fold test AUC: {fold_aucs}\n")
        f.write(f"OOF AUC: {oof_auc:.4f}\n")
    print(f"\n>>> {OUT_DIR/'abmil_v3_kras_oof.csv'}")


if __name__ == "__main__":
    main()
