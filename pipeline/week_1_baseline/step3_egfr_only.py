"""
step3_egfr_only.py
──────────────────
专门为 EGFR 单基因训练的 5-fold CV 脚本。

输出:
  outputs_egfr_only/egfr_only_predictions.csv
  outputs_egfr_only/fold_summary.csv
  outputs_egfr_only/summary.txt
  outputs_egfr_only/figures/roc_egfr.png

用法:
  python step3_egfr_only.py
  python step3_egfr_only.py --hidden-dim 32 --epochs 50 --patience 8

模型:EGFROnlyModel(hidden_dim=64, dropout=0.5)
Loss:BCELoss with pos_weight(~5.4 from 70/13)
CV:5-fold StratifiedKFold(by any_positive)
"""
from __future__ import annotations
import argparse
import csv
import functools
import sys
import time
import warnings
from pathlib import Path

# 强制 unbuffered(双保险;即使外层没用 python -u,也能逐行 flush)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from egfr_only_model import EGFROnlyModel, compute_egfr_loss, PartialLoss


# ─────────────── 数据 ───────────────
class CaseDataset(torch.utils.data.Dataset):
    def __init__(self, cases, bags_dir, labels):
        self.cases = cases
        self.bags_dir = bags_dir
        self.labels = labels

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid = self.cases[i]
        npz_path = self.bags_dir / f"{cid}.npz"
        # 鲁棒:文件损坏时跳过(等后台写完)
        if not npz_path.exists():
            raise FileNotFoundError(f"{cid}: npz not found")
        for attempt in range(3):
            try:
                d = np.load(npz_path, allow_pickle=True)
                break
            except Exception:
                if attempt == 2:
                    raise RuntimeError(f"{cid}: npz corrupt after 3 retries")
                import time as _time
                _time.sleep(0.5)
        # 兼容两种 npz 字段:
        #   旧 (bags_v3, bags_per_case): 'patch_features'
        #   新 (deepgem_feat, deepgem_feat_v2): 'features'
        if "patch_features" in d.files:
            feats = d["patch_features"]
        elif "features" in d.files:
            feats = d["features"]
        else:
            raise KeyError(f"{cid}: npz 无 'patch_features'/'features' 字段, keys={d.files}")
        # 兼容 slice_ids: 新版只有 1 个 SM 时也是 int32
        sm = d["slice_ids"] if "slice_ids" in d.files else np.zeros(len(feats), dtype=np.int32)
        d.close()
        return {
            "case_id": cid,
            "patch_feats": torch.from_numpy(feats).float(),
            "sm_indices": torch.from_numpy(sm).long(),
            "label": torch.tensor(self.labels[cid], dtype=torch.float32),
        }


def collate(batch):
    return batch[0]


# ─────────────── 训练 ───────────────
def train_one_fold(train_cases, val_cases, bags_dir, labels,
                   hidden_dim, epochs, patience, device, lr, weight_decay,
                   dropout, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CaseDataset(train_cases, bags_dir, labels)
    val_ds = CaseDataset(val_cases, bags_dir, labels)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )

    # dim 是 hidden_dim 别名 (兼容旧命令行参数)
    model = EGFROnlyModel(in_dim=1024, dim=hidden_dim, depth=3, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # PartialLoss 在 train_one_fold 内部维护(每个 fold 重新初始化)
    # 因为 case 不同,patch 数也不同

    # pos_weight
    train_labels = np.array([labels[c] for c in train_cases])
    n_pos = train_labels.sum()
    n_neg = len(train_cases) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 10.0)
    pos_weight_t = torch.tensor([pos_weight], device=device)
    print(f"  pos_weight = {pos_weight:.3f}  (train EGFR+: {n_pos}, -: {n_neg})")

    best_auc = -1.0
    best_preds = None
    best_case_ids = None
    best_labels = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            patch_feats = batch["patch_feats"].to(device)
            sm_idx = batch["sm_indices"].to(device)
            label = batch["label"].to(device).unsqueeze(0)  # [1]
            n_sm = int(sm_idx.max().item()) + 1
            logit, instance_logits, _ = model(patch_feats, sm_idx, n_sm)
            # DeepGEM 风格: bag_loss + λ * PartialLoss
            N = patch_feats.shape[0]
            # 初始化或复用 partial_loss (每个 fold 一个)
            if not hasattr(train_one_fold, '_partial_loss') or \
               train_one_fold._partial_loss.confidence.shape[0] != N:
                train_one_fold._partial_loss = PartialLoss(
                    num_patches=N, num_classes=2,
                ).to(device)
            partial_loss = train_one_fold._partial_loss
            batch_indices = torch.arange(N, device=device)
            loss, bag_loss, part_loss = compute_egfr_loss(
                logit, instance_logits, label,
                partial_loss, batch_indices,
                pos_weight_bag=float(pos_weight_t.item()),
                pos_weight_patch=1.0,
                lambda_partial=0.3,
                conf_ema_m=0.95 - 0.15 * (epoch - 1) / max(epochs - 1, 1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / max(n, 1)

        # eval
        model.eval()
        case_ids, preds, lbls = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                cid = batch["case_id"]
                patch_feats = batch["patch_feats"].to(device)
                sm_idx = batch["sm_indices"].to(device)
                label = batch["label"]
                n_sm = int(sm_idx.max().item()) + 1
                logit, _, _ = model(patch_feats, sm_idx, n_sm)
                p = torch.sigmoid(logit).cpu().numpy()
                case_ids.append(cid)
                preds.append(float(p))
                lbls.append(int(label.item()))
        from sklearn.metrics import roc_auc_score
        y = np.array(lbls)
        p = np.array(preds)
        n_pos_fold = int(y.sum())
        if 0 < n_pos_fold < len(y):
            val_auc = float(roc_auc_score(y, p))
        else:
            val_auc = 0.0  # fallback

        elapsed = time.time() - t0
        print(f"    ep{epoch:>2}: loss={train_loss:.3f} val_auc={val_auc:.3f} ({elapsed:.1f}s)", end="")
        if val_auc > best_auc:
            best_auc = val_auc
            best_preds = preds
            best_case_ids = case_ids
            best_labels = lbls
            epochs_no_improve = 0
            print("  [best]")
        else:
            epochs_no_improve += 1
            print(f"  ({epochs_no_improve}/{patience})")
        if epochs_no_improve >= patience:
            print(f"    early stop at epoch {epoch}")
            break

    if best_preds is None:
        # fallback
        best_preds = preds
        best_case_ids = case_ids
        best_labels = lbls

    return {
        "best_auc": best_auc,
        "preds": best_preds,
        "labels": best_labels,
        "case_ids": best_case_ids,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags-dir",
                    default="pipeline/week_1_baseline/outputs/bags_per_case")
    ap.add_argument("--manifest",
                    default="pipeline/week_1_baseline/case_manifest.csv")
    ap.add_argument("--out-dir",
                    default="pipeline/week_1_baseline/outputs_egfr_only")
    ap.add_argument("--hidden-dim", type=int, default=128,
                    help="默认 128(v2 太弱导致学不到)")
    ap.add_argument("--epochs", type=int, default=80,
                    help="默认 80(v2 早停太快)")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="默认 1e-4(v2 2e-4 太大震荡)")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="默认 1e-4(v2 1e-3 强正则过强)")
    ap.add_argument("--dropout", type=float, default=0.25,
                    help="默认 0.25(v2 0.5 太强)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step3-EGFR] device={device}, hidden_dim={args.hidden_dim}, folds={args.folds}")

    # 读 manifest,只看 EGFR label
    cases, labels = [], {}
    with open(args.manifest) as f:
        rd = csv.DictReader(f)
        for r in rd:
            cid = r["case_id"]
            cases.append(cid)
            labels[cid] = int(r["EGFR"])  # 只用 EGFR
    print(f"[Step3-EGFR] 共 {len(cases)} case, EGFR+={sum(labels.values())}, -={len(cases)-sum(labels.values())}")

    bags_dir = Path(args.bags_dir)
    n_avail = len(list(bags_dir.glob("*.npz")))
    print(f"[Step3-EGFR] bags 目录下有 {n_avail} 个 case 的 .npz")
    # 过滤:文件存在 + 可读(避免并发写入的损坏 npz)
    available = []
    skipped_corrupt = []
    for p in bags_dir.glob("*.npz"):
        try:
            d = np.load(p, allow_pickle=True)
            _ = d["features"].shape
            d.close()
            available.append(p.stem)
        except Exception:
            skipped_corrupt.append(p.stem)
    if skipped_corrupt:
        print(f"[Step3-EGFR] 跳过 {len(skipped_corrupt)} 个损坏 npz: {skipped_corrupt[:5]}{'...' if len(skipped_corrupt)>5 else ''}")
    cases = [c for c in cases if c in available]
    print(f"[Step3-EGFR] 实际可训: {len(cases)} case, EGFR+={sum(labels[c] for c in cases)}")

    # 5-fold StratifiedKFold by EGFR label
    y_egfr = np.array([labels[c] for c in cases])
    # 但 NRAS+TP53+... 的分层 by any_positive
    # 读 any_positive
    any_pos = {}
    with open(args.manifest) as f:
        rd = csv.DictReader(f)
        for r in rd:
            any_pos[r["case_id"]] = int(r["any_9gene_positive"])
    y_any = np.array([any_pos[c] for c in cases])

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    oof_preds = np.zeros(len(cases), dtype=np.float32)
    fold_results = []

    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(cases)), y_any), 1):
        train_cases = [cases[i] for i in tr]
        test_cases = [cases[i] for i in te]
        egfr_test_pos = sum(labels[c] for c in test_cases)
        print(f"\n========== Fold {fold_i}/{args.folds} ==========")
        print(f"  train={len(train_cases)} test={len(test_cases)} EGFR+={egfr_test_pos}")
        res = train_one_fold(
            train_cases, test_cases, bags_dir, labels,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs, patience=args.patience,
            device=device, lr=args.lr, weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed + fold_i,
        )
        for cid, p in zip(res["case_ids"], res["preds"]):
            idx = cases.index(cid)
            oof_preds[idx] = p
        fold_results.append({"fold": fold_i, "best_auc": res["best_auc"]})
        print(f"  >> Fold {fold_i} best val AUC = {res['best_auc']:.3f}")

    # predictions.csv
    with open(out_dir / "egfr_only_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "true_EGFR", "pred_prob_EGFR"])
        for i, c in enumerate(cases):
            w.writerow([c, labels[c], float(oof_preds[i])])

    # fold_summary.csv
    with open(out_dir / "fold_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_val_auc"])
        for r in fold_results:
            w.writerow([r["fold"], round(r["best_auc"], 4)])

    # Overall AUC
    from sklearn.metrics import roc_auc_score
    y = np.array([labels[c] for c in cases])
    p = oof_preds
    overall_auc = float(roc_auc_score(y, p))
    print(f"\n[Step3-EGFR] OOF AUC = {overall_auc:.3f}")

    # summary.txt
    with open(out_dir / "summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EGFR-only 报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"Case 数: {len(cases)}\n")
        f.write(f"EGFR 阳性: {sum(labels.values())}, 阴性: {len(cases)-sum(labels.values())}\n")
        f.write(f"CV: {args.folds}-fold StratifiedKFold (by any_positive)\n")
        f.write(f"Backbone: UNI (1024-dim, frozen)\n")
        f.write(f"模型: EGFROnlyModel hidden_dim={args.hidden_dim}, dropout=0.5, wd={args.weight_decay}\n")
        f.write(f"\n── OOF AUC ──\n")
        f.write(f"  EGFR: {overall_auc:.3f}\n")
        f.write("\n── 各 fold val best AUC ──\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']}: {r['best_auc']:.3f}\n")
    print(f"\n[Step3-EGFR] predictions -> {out_dir}/egfr_only_predictions.csv")
    print(f"[Step3-EGFR] summary -> {out_dir}/summary.txt")
    print("\n[Step3-EGFR] done")


if __name__ == "__main__":
    main()