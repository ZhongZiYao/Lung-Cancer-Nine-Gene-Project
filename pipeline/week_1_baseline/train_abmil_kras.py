"""
train_abmil_kras.py
───────────────────
ABMIL baseline 训练 — 验证"AUC 0.5 是模型问题不是数据问题"

数据:  feat_416/  (416 TCGA-LUAD case × UNI 1024-dim patch features)
标签:  data/MC3/9gene_panel_LUAD.csv  (KRAS = 150+, 266-)
模型:  ABMIL (CLAM 原文风格, single-branch, 1024→256→1)
训练:  5-fold CV × 1 seed, 50 epoch, patience 10, AdamW LR 2e-4
       mini-batch = 4 cases per step (避免 per-case SGD 梯度噪声)

输出:  pipeline/week_1_baseline/outputs_abmil_baseline/
        abmil_kras_oof.csv   (每个 case 5-fold OOF 预测)
        abmil_kras_summary.txt
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as _roc_auc

sys.path.insert(0, str(Path(__file__).parent))
from abmil_model import ABMIL, abmil_bce_loss

# ────── Config ──────
REPO = Path(r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer")
FEAT_DIR = REPO / "feat_416"
PANEL = REPO / "data" / "MC3" / "9gene_panel_LUAD.csv"
OUT_DIR = REPO / "pipeline" / "week_1_baseline" / "outputs_abmil_baseline"

HIDDEN = 256
ATTN_DIM = 128
DROPOUT = 0.25
LR = 2e-4
WEIGHT_DECAY = 1e-4
BATCH_CASES = 4          # mini-batch = 4 cases per optimizer step
N_FOLDS = 5
N_SEEDS = 1
N_EPOCHS = 50
PATIENCE = 10

GENE = "KRAS"            # 先跑 KRAS — 之前一直卡 KRAS

# 预加载 case → features 到内存 (416 case × 640 patch × 1024-dim float32 ≈ 1.0 GB)
# 一次性 load 比每次 np.load 快 10x
def load_all_features(feat_dir: Path) -> dict[str, np.ndarray]:
    cache = {}
    for p in sorted(feat_dir.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        cache[p.stem] = d["features"].astype(np.float32)
        d.close()
    return cache


def load_labels(panel_path: Path, gene: str) -> dict[str, int]:
    labels = {}
    with open(panel_path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            labels[r["case_id"]] = int(r[gene])
    return labels


def train_one_fold(
    train_cases: list[str], train_y: np.ndarray,
    val_cases: list[str], val_y: np.ndarray,
    test_cases: list[str],
    device: torch.device,
    pos_w: torch.Tensor,
    feat_cache: dict[str, np.ndarray],
    fold_i: int, seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """返回 (test_preds, val_preds_each_epoch, val_auc_each_epoch)."""
    model = ABMIL(in_dim=1024, hidden=HIDDEN, attn_dim=ATTN_DIM, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = 0.0
    best_val_auc_ep = 0
    best_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0
    val_auc_hist: list[float] = []
    val_loss_hist: list[float] = []

    n_train = len(train_cases)

    for ep in range(N_EPOCHS):
        ep_t = time.time()
        model.train()
        # shuffle
        rng = np.random.default_rng(ep + seed * 1000)
        perm = rng.permutation(n_train)
        ep_losses = []

        # mini-batch loop
        for batch_start in range(0, n_train, BATCH_CASES):
            batch_idx = perm[batch_start: batch_start + BATCH_CASES]
            opt.zero_grad()
            batch_loss = torch.zeros(1, device=device)
            for ti in batch_idx:
                cid = train_cases[ti]
                pf = torch.from_numpy(feat_cache[cid]).to(device)  # [N, 1024]
                logit, _ = model(pf)
                loss_i = abmil_bce_loss(logit, torch.tensor([float(train_y[ti])], device=device),
                                        pos_weight=pos_w)
                batch_loss = batch_loss + loss_i
            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            opt.step()
            ep_losses.append(batch_loss.item())

        avg_loss = float(np.mean(ep_losses))

        # val
        model.eval()
        with torch.no_grad():
            val_logits = []
            for cid in val_cases:
                pf = torch.from_numpy(feat_cache[cid]).to(device)
                logit, _ = model(pf)
                val_logits.append(float(torch.sigmoid(logit).cpu().item()))
        val_p = np.array(val_logits)
        if 0 < val_y.sum() < len(val_y):
            val_auc = float(_roc_auc(val_y, val_p))
        else:
            val_auc = float("nan")
        val_auc_hist.append(val_auc)
        val_loss_hist.append(avg_loss)

        improved_loss = avg_loss < best_val_loss
        if improved_loss:
            best_val_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_auc_ep = ep + 1

        epochs_no_improve = 0 if improved_loss else epochs_no_improve + 1
        marker = ""
        if improved_loss: marker += " *L"
        if not np.isnan(val_auc) and val_auc == best_val_auc: marker += " *V"

        print(f"    ep{ep+1:>2}/{N_EPOCHS}  loss={avg_loss:.3f}  val_auc={val_auc:.3f}  "
              f"({time.time()-ep_t:.1f}s){marker}", flush=True)

        if epochs_no_improve >= PATIENCE:
            print(f"    [early stop] ep{ep+1}  best_val_loss={best_val_loss:.3f}  "
                  f"best_val_auc={best_val_auc:.3f} @ ep{best_val_auc_ep}", flush=True)
            break

    # 加载 best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = []
        for cid in test_cases:
            pf = torch.from_numpy(feat_cache[cid]).to(device)
            logit, _ = model(pf)
            test_preds.append(float(torch.sigmoid(logit).cpu().item()))

    return test_preds, val_auc_hist, val_loss_hist, best_val_auc, best_val_auc_ep


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\nLoading features from {FEAT_DIR} ...", flush=True)
    t0 = time.time()
    feat_cache = load_all_features(FEAT_DIR)
    print(f"  loaded {len(feat_cache)} cases in {time.time()-t0:.1f}s", flush=True)
    print(f"  memory: {sum(v.nbytes for v in feat_cache.values())/1024/1024/1024:.2f} GB", flush=True)

    labels = load_labels(PANEL, GENE)
    # filter to cases with both feat & label
    case_ids = [c for c in feat_cache.keys() if c in labels]
    case_ids.sort()
    y = np.array([labels[c] for c in case_ids])
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\n{GENE}: {len(case_ids)} cases, {n_pos} pos, {n_neg} neg ({100*n_pos/len(y):.1f}%)", flush=True)
    print(f"Config: hidden={HIDDEN} dropout={DROPOUT} LR={LR} "
          f"batch={BATCH_CASES}cases  {N_FOLDS}-fold × {N_SEEDS} seed", flush=True)
    print(f"        {N_EPOCHS} epoch max, patience={PATIENCE}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_oof = np.zeros(len(case_ids), dtype=np.float32)
    fold_aucs = []
    seed_aucs = []

    for seed in [42, 7, 123][:N_SEEDS]:
        print(f"\n{'='*60}\n  seed={seed}\n{'='*60}", flush=True)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(case_ids), dtype=np.float32)
        per_fold = []
        for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            tr_cases = [case_ids[i] for i in tr]
            te_cases = [case_ids[i] for i in te]
            tr_y = y[tr]
            te_y = y[te]
            # 80/15/5 → train/val  (5% val from train)
            n_val = max(1, int(len(tr) * 0.15))
            rng = np.random.default_rng(seed * 100 + fold_i)
            perm = rng.permutation(len(tr))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            tr_cases_ = [tr_cases[i] for i in tr_idx]
            val_cases = [tr_cases[i] for i in val_idx]
            tr_y_ = tr_y[tr_idx]
            val_y_ = tr_y[val_idx]
            n_pos_tr = int(tr_y_.sum())
            pos_w = torch.tensor(
                [min((len(tr_y_) - n_pos_tr) / max(n_pos_tr, 1), 10.0)],
                device=device,
            )
            print(f"\n  [fold {fold_i}/{N_FOLDS}]  train={len(tr_cases_)} val={len(val_cases)} test={len(te_cases)}  "
                  f"pos_w={pos_w.item():.2f}", flush=True)

            t_fold = time.time()
            test_preds, val_auc_hist, val_loss_hist, best_auc, best_ep = train_one_fold(
                tr_cases_, tr_y_, val_cases, val_y_, te_cases, device, pos_w, feat_cache,
                fold_i, seed,
            )
            oof[te] = test_preds
            test_auc = _roc_auc(te_y, test_preds) if 0 < te_y.sum() < len(te_y) else float("nan")
            per_fold.append(test_auc)
            print(f"  >>> [fold {fold_i} done] test_AUC={test_auc:.3f}  "
                  f"best_val_AUC={best_auc:.3f} @ ep{best_ep}  "
                  f"({(time.time()-t_fold)/60:.1f}min)\n", flush=True)

        seed_auc = _roc_auc(y, oof) if 0 < y.sum() < len(y) else float("nan")
        seed_aucs.append(seed_auc)
        all_oof += oof
        print(f"\n  seed {seed}: OOF AUC = {seed_auc:.3f}\n", flush=True)
        for i, a in enumerate(per_fold, 1):
            print(f"    fold {i}: test_AUC = {a:.3f}", flush=True)
        fold_aucs.append(per_fold)

    all_oof /= N_SEEDS
    final_auc = _roc_auc(y, all_oof) if 0 < y.sum() < len(y) else float("nan")
    mean_seed = float(np.mean(seed_aucs))
    std_seed = float(np.std(seed_aucs))
    print(f"\n{'='*60}\n  {GENE} 汇总\n{'='*60}")
    print(f"  per-seed OOF AUC : {seed_aucs}")
    print(f"  mean ± std       : {mean_seed:.3f} ± {std_seed:.3f}")
    print(f"  averaged OOF AUC : {final_auc:.3f}")

    # save
    with open(OUT_DIR / "abmil_kras_oof.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", f"true_{GENE}", "pred_prob"])
        for i, c in enumerate(case_ids):
            w.writerow([c, int(y[i]), f"{all_oof[i]:.6f}"])

    with open(OUT_DIR / "abmil_kras_summary.txt", "w") as f:
        f.write(f"ABMIL baseline — {GENE} — feat_416 (UNI 1024-dim)\n")
        f.write(f"Config: hidden={HIDDEN} attn={ATTN_DIM} dropout={DROPOUT} "
                f"LR={LR} bs={BATCH_CASES} epoch={N_EPOCHS} patience={PATIENCE}\n")
        f.write(f"Folds: {N_FOLDS}  Seeds: {N_SEEDS}\n\n")
        f.write(f"per-seed OOF AUC : {seed_aucs}\n")
        f.write(f"mean ± std       : {mean_seed:.4f} ± {std_seed:.4f}\n")
        f.write(f"averaged OOF AUC : {final_auc:.4f}\n")

    print(f"\n>>> OOF: {OUT_DIR/'abmil_kras_oof.csv'}")
    print(f">>> Summary: {OUT_DIR/'abmil_kras_summary.txt'}")


if __name__ == "__main__":
    main()
