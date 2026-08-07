"""
train_abmil_v3_9gene.py
──────────────────────
ABMIL on deepgem_feat_v3 — 9 gene 5-fold OOF
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

GENES = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF", "PIK3CA", "ERBB2", "NRAS"]
HIDDEN = 256
ATTN_DIM = 128
DROPOUT = 0.25
LR = 2e-4
WEIGHT_DECAY = 1e-4
BATCH_CASES = 2
N_FOLDS = 5
N_EPOCHS = 30
PATIENCE = 8
MAX_PATCH = 2000


def load_v3_features(npz_path: Path) -> np.ndarray | None:
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            with zf.open("features.npy") as fp:
                return np.load(io.BytesIO(fp.read())).astype(np.float32)
    except Exception:
        return None


def train_one_gene(gene: str, case_ids: list[str], y: np.ndarray,
                    feat_cache: dict[str, np.ndarray], device: torch.device) -> dict:
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    print(f"\n{'='*60}\n  {gene}  (n={len(case_ids)}, pos={n_pos}, neg={n_neg})\n{'='*60}", flush=True)

    # 5-fold OOF
    oof = np.zeros(len(case_ids), dtype=np.float32)
    fold_aucs = []
    rng = np.random.default_rng(42)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
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
        if n_pos_tr == 0 or n_pos_tr == len(tr_y_):
            print(f"  [fold {fold_i}] skip (no pos/neg in train)", flush=True)
            continue
        pos_w = torch.tensor([min((len(tr_y_) - n_pos_tr) / max(n_pos_tr, 1), 10.0)], device=device)
        print(f"  [fold {fold_i}/{N_FOLDS}] train={len(tr_cases_)} val={len(val_cases)} test={len(te_cases)} pos_w={pos_w.item():.2f}", flush=True)

        model = ABMIL(in_dim=1024, hidden=HIDDEN, attn_dim=ATTN_DIM, dropout=DROPOUT).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_state = None
        best_val_auc = 0.0
        best_val_auc_ep = 0
        epochs_no_improve = 0
        last_state = None

        for ep in range(N_EPOCHS):
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
            last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
            if epochs_no_improve >= PATIENCE:
                break

        # pick best state (fallback to last if best is None)
        if best_state is None:
            best_state = last_state
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
        print(f"  >>> [fold {fold_i} done] test_AUC={t_auc:.3f}  best_val_AUC={best_val_auc:.3f}", flush=True)

    if 0 < y.sum() < len(y):
        oof_auc = _roc_auc(y, oof)
    else:
        oof_auc = float("nan")
    print(f"  >>> {gene}  OOF AUC = {oof_auc:.3f}  per-fold={fold_aucs}", flush=True)

    # save per-gene OOF
    with open(OUT_DIR / f"abmil_v3_{gene.lower()}_oof.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", f"true_{gene}", "pred_prob"])
        for i, c in enumerate(case_ids):
            w.writerow([c, int(y[i]), f"{oof[i]:.6f}"])
    return {"gene": gene, "oof_auc": oof_auc, "fold_aucs": fold_aucs,
            "n_pos": n_pos, "n_neg": n_neg}


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

    labels = {}
    with open(PANEL) as f:
        rd = csv.DictReader(f)
        for r in rd:
            labels[r["case_id"]] = r
    case_ids = sorted([c for c in feat_cache.keys() if c in labels])
    print(f"  v3 cases with labels: {len(case_ids)}", flush=True)

    results = []
    for g in GENES:
        y = np.array([int(labels[c][g]) for c in case_ids])
        if y.sum() < 5 or (1 - y).sum() < 5:
            print(f"\n=== {g}: skip (pos={int(y.sum())}, neg={int((1-y).sum())}) ===", flush=True)
            continue
        res = train_one_gene(g, case_ids, y, feat_cache, device)
        results.append(res)

    # final summary
    print(f"\n{'='*70}")
    print(f"  v3 ABMIL 9-gene OOF AUC summary")
    print(f"{'='*70}")
    print(f"{'Gene':<8} {'N_pos':>5} {'N_neg':>5} {'OOF_AUC':>9} {'folds':<40}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        f_str = "[" + ", ".join(f"{a:.3f}" for a in r["fold_aucs"]) + "]"
        print(f"  {r['gene']:<6} {r['n_pos']:>5} {r['n_neg']:>5} {r['oof_auc']:>7.3f} {f_str}", flush=True)

    with open(OUT_DIR / "abmil_v3_9gene_summary.txt", "w") as f:
        f.write("ABMIL on v3 — 9-gene OOF AUC\n")
        f.write(f"cases: {len(case_ids)}  MAX_PATCH: {MAX_PATCH}\n\n")
        f.write(f"{'Gene':<8} {'N_pos':>5} {'N_neg':>5} {'OOF_AUC':>9} {'folds':<40}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f_str = "[" + ", ".join(f"{a:.3f}" for a in r["fold_aucs"]) + "]"
            f.write(f"  {r['gene']:<6} {r['n_pos']:>5} {r['n_neg']:>5} {r['oof_auc']:>7.3f} {f_str}\n")
    print(f"\n>>> {OUT_DIR/'abmil_v3_9gene_summary.txt'}")


if __name__ == "__main__":
    main()
