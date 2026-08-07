"""
step3_egfr_split.py
───────────────────
EGFR-only 方案 A:固定 70/15/15 分层切分。

切分逻辑(按 any_9gene_positive 分层):
  train 70% → 内部 80/20 按 EGFR 分层抽 val 用于早停
  val   15% → 早停监控 + 中间报告
  test  15% → 最终一次性评估(只跑一次,不参与训练)

输出:
  outputs_egfr_split/egfr_split_predictions.csv  (test 集)
  outputs_egfr_split/splits.csv                 (三个集合的具体 case)
  outputs_egfr_split/summary.txt
  outputs_egfr_split/figures/roc_split.png
  outputs_egfr_split/figures/pred_dist_split.png

用法:
  python step3_egfr_split.py
  python step3_egfr_split.py --train-frac 0.7 --val-frac 0.15 --test-frac 0.15
"""
from __future__ import annotations
import argparse
import csv
import functools
import json
import sys
import time
import warnings
from pathlib import Path

# 强制 unbuffered
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print = functools.partial(print, flush=True)

import numpy as np
import torch
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
        if "patch_features" in d.files:
            feats = d["patch_features"]
        elif "features" in d.files:
            feats = d["features"]
        else:
            raise KeyError(f"{cid}: npz 无 'patch_features'/'features' 字段, keys={d.files}")
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


# ─────────────── 训练(在 train+val 上找 best epoch) ───────────────
def train_with_es(train_cases, val_cases, bags_dir, labels,
                  hidden_dim, epochs, patience, device, lr, weight_decay,
                  dropout, seed):
    """在 train 上训,val 上早停,返回 best state"""
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

    model = EGFROnlyModel(in_dim=1024, dim=hidden_dim, depth=3, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_labels = np.array([labels[c] for c in train_cases])
    n_pos = int(train_labels.sum())
    n_neg = len(train_cases) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 10.0)
    pos_weight_t = torch.tensor([pos_weight], device=device)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    from sklearn.metrics import roc_auc_score

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            patch_feats = batch["patch_feats"].to(device)
            sm_idx = batch["sm_indices"].to(device)
            label = batch["label"].to(device).unsqueeze(0)
            n_sm = int(sm_idx.max().item()) + 1
            logit, instance_logits, _ = model(patch_feats, sm_idx, n_sm)
            N = patch_feats.shape[0]
            if not hasattr(train_with_es, '_partial_loss') or \
               train_with_es._partial_loss.confidence.shape[0] != N:
                train_with_es._partial_loss = PartialLoss(
                    num_patches=N, num_classes=2,
                ).to(device)
            partial_loss = train_with_es._partial_loss
            batch_indices = torch.arange(N, device=device)
            loss, _, _ = compute_egfr_loss(
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

        model.eval()
        preds, lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                patch_feats = batch["patch_feats"].to(device)
                sm_idx = batch["sm_indices"].to(device)
                label = batch["label"]
                n_sm = int(sm_idx.max().item()) + 1
                logit, _, _ = model(patch_feats, sm_idx, n_sm)
                preds.append(float(torch.sigmoid(logit).cpu().numpy()))
                lbls.append(int(label.item()))
        y_v = np.array(lbls)
        p_v = np.array(preds)
        n_pos_v = int(y_v.sum())
        if 0 < n_pos_v < len(y_v):
            val_auc = float(roc_auc_score(y_v, p_v))
        else:
            val_auc = float("nan")

        elapsed = time.time() - t0
        improved = (val_auc == val_auc) and (val_auc > best_val_auc)
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or improved:
            vstr = f"{val_auc:.3f}" if val_auc == val_auc else "N/A"
            tag = "  [best]" if improved else (f"  ({epochs_no_improve}/{patience})" if epochs_no_improve > 0 else "")
            print(f"    ep{epoch:>2}: loss={train_loss:.3f} val_auc={vstr} ({elapsed:.1f}s){tag}")

        if epochs_no_improve >= patience:
            print(f"    early stop at epoch {epoch} (best val AUC {best_val_auc:.3f})")
            break

    return {
        "best_state": best_state,
        "best_val_auc": float(best_val_auc) if best_val_auc == best_val_auc else float("nan"),
        "stopped_epoch": epoch,
        "pos_weight": pos_weight,
    }


@torch.no_grad()
def predict_cases(model, cases, bags_dir, device):
    """对一组 case 评分"""
    model.eval()
    out = []
    for cid in cases:
        npz_path = bags_dir / f"{cid}.npz"
        for attempt in range(3):
            try:
                d = np.load(npz_path, allow_pickle=True)
                break
            except Exception:
                if attempt == 2:
                    raise RuntimeError(f"{cid}: npz corrupt after 3 retries")
                import time as _time
                _time.sleep(0.5)
        if "patch_features" in d.files:
            feats = d["patch_features"]
        elif "features" in d.files:
            feats = d["features"]
        else:
            raise KeyError(f"{cid}: npz 无 'patch_features'/'features' 字段")
        sm = d["slice_ids"] if "slice_ids" in d.files else np.zeros(len(feats), dtype=np.int32)
        patch_feats = torch.from_numpy(feats).float().to(device)
        sm_idx = torch.from_numpy(sm).long().to(device)
        n_sm = int(sm_idx.max().item()) + 1
        logit, _, _ = model(patch_feats, sm_idx, n_sm)
        p = float(torch.sigmoid(logit).cpu().numpy())
        out.append({"case_id": cid, "pred": p})
    return out


# ─────────────── main ───────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags-dir",
                    default="pipeline/week_1_baseline/outputs/bags_v3")
    ap.add_argument("--manifest",
                    default="pipeline/week_1_baseline/case_manifest.csv")
    ap.add_argument("--out-dir",
                    default="pipeline/week_1_baseline/outputs_egfr_split")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step3-EGFR-Split] device={device}, hidden_dim={args.hidden_dim}")
    print(f"[Step3-EGFR-Split] split: train={args.train_frac} val={args.val_frac} "
          f"test={args.test_frac} (stratify by any_9gene_positive)")

    # 读 manifest
    cases, egfr_label, any_pos = [], {}, {}
    with open(args.manifest) as f:
        rd = csv.DictReader(f)
        for r in rd:
            cid = r["case_id"]
            cases.append(cid)
            egfr_label[cid] = int(r["EGFR"])
            any_pos[cid] = int(r["any_9gene_positive"])
    print(f"[Step3-EGFR-Split] 共 {len(cases)} case, "
          f"EGFR+={sum(egfr_label.values())}, any_pos+={sum(any_pos.values())}")

    bags_dir = Path(args.bags_dir)
    # 过滤:文件存在 + 可读(避免损坏 npz)
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
        print(f"[Step3-EGFR-Split] 跳过 {len(skipped_corrupt)} 个损坏 npz")
    cases = [c for c in cases if c in available]
    print(f"[Step3-EGFR-Split] 实际可训: {len(cases)} case, "
          f"EGFR+={sum(egfr_label[c] for c in cases)}")

    # 第一步:按 EGFR 分层 70/(30)(保证每段都有 EGFR+)
    # sklearn 没有 3-way stratified,先 70/30 再 30 → 15/15
    from sklearn.model_selection import StratifiedShuffleSplit

    y_egfr = np.array([egfr_label[c] for c in cases])
    print(f"[Step3-EGFR-Split] 分层键:EGFR(因为 any_pos 几乎全为 1)")
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - args.train_frac, random_state=args.seed
    )
    train_idx, rest_idx = next(sss1.split(np.zeros(len(cases)), y_egfr))
    train_cases = [cases[i] for i in train_idx]
    rest_cases = [cases[i] for i in rest_idx]

    y_rest = np.array([egfr_label[c] for c in rest_cases])
    rest_val_frac = args.val_frac / (args.val_frac + args.test_frac)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - rest_val_frac, random_state=args.seed + 1
    )
    val_idx, test_idx = next(sss2.split(np.zeros(len(rest_cases)), y_rest))
    val_cases = [rest_cases[i] for i in val_idx]
    test_cases = [rest_cases[i] for i in test_idx]

    def stat(name, cs):
        n = len(cs)
        ne = sum(egfr_label[c] for c in cs)
        na = sum(any_pos[c] for c in cs)
        print(f"  {name}: n={n}  EGFR+={ne}  any_pos+={na}")

    print(f"\n[Step3-EGFR-Split] === 切分结果 ===")
    stat("train", train_cases)
    stat("val  ", val_cases)
    stat("test ", test_cases)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # 写 splits.csv
    with open(out_dir / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "EGFR", "any_9gene_positive"])
        for c in train_cases:
            w.writerow([c, "train", egfr_label[c], any_pos[c]])
        for c in val_cases:
            w.writerow([c, "val", egfr_label[c], any_pos[c]])
        for c in test_cases:
            w.writerow([c, "test", egfr_label[c], any_pos[c]])

    # 训练:在 train 上训,val 上早停
    print(f"\n[Step3-EGFR-Split] === 训练(在 train 上,val 早停) ===")
    t0 = time.time()
    res = train_with_es(
        train_cases=train_cases, val_cases=val_cases,
        bags_dir=bags_dir, labels=egfr_label,
        hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
        device=device, lr=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, seed=args.seed,
    )
    train_time = time.time() - t0
    print(f"\n[Step3-EGFR-Split] 训练完成,best_val_auc={res['best_val_auc']:.3f}, "
          f"stopped_epoch={res['stopped_epoch']}, pos_weight={res['pos_weight']:.3f}, "
          f"time={train_time:.1f}s")

    # 加载 best 模型
    model = EGFROnlyModel(in_dim=1024, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    model.load_state_dict(res["best_state"])

    # 1) val 上评分(模型选择集,看是否过拟合)
    val_results = predict_cases(model, val_cases, bags_dir, device)
    for r in val_results:
        r["true_EGFR"] = egfr_label[r["case_id"]]
    y_v = np.array([r["true_EGFR"] for r in val_results])
    p_v = np.array([r["pred"] for r in val_results])

    # 2) test 上评分(最终评估)
    test_results = predict_cases(model, test_cases, bags_dir, device)
    for r in test_results:
        r["true_EGFR"] = egfr_label[r["case_id"]]
    y_t = np.array([r["true_EGFR"] for r in test_results])
    p_t = np.array([r["pred"] for r in test_results])

    from sklearn.metrics import roc_auc_score, roc_curve
    val_auc = float(roc_auc_score(y_v, p_v)) if 0 < y_v.sum() < len(y_v) else float("nan")
    test_auc = float(roc_auc_score(y_t, p_t)) if 0 < y_t.sum() < len(y_t) else float("nan")

    print(f"\n[Step3-EGFR-Split] === 最终评估 ===")
    print(f"  val  AUC = {val_auc:.3f}  (n={len(val_cases)}, EGFR+={int(y_v.sum())})")
    print(f"  test AUC = {test_auc:.3f}  (n={len(test_cases)}, EGFR+={int(y_t.sum())})")

    # 写 predictions.csv(test 集)
    with open(out_dir / "egfr_split_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "true_EGFR", "pred_prob_EGFR"])
        for r in val_results:
            w.writerow([r["case_id"], "val", r["true_EGFR"], round(r["pred"], 6)])
        for r in test_results:
            w.writerow([r["case_id"], "test", r["true_EGFR"], round(r["pred"], 6)])

    # 画 ROC + 预测分布
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plotted = False
    if 0 < y_v.sum() < len(y_v):
        fpr, tpr, _ = roc_curve(y_v, p_v)
        plt.plot(fpr, tpr, lw=2, label=f"val  AUC={val_auc:.3f} (n={len(val_cases)})")
        plotted = True
    if 0 < y_t.sum() < len(y_t):
        fpr, tpr, _ = roc_curve(y_t, p_t)
        plt.plot(fpr, tpr, lw=2, label=f"test AUC={test_auc:.3f} (n={len(test_cases)})")
        plotted = True
    if plotted:
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"EGFR-only Split ROC (hidden={args.hidden_dim}, dropout={args.dropout})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figures/roc_split.png", dpi=120)
    plt.close()

    # 预测分布(test 集)
    if 0 < y_t.sum() < len(y_t):
        plt.figure(figsize=(8, 5))
        plt.hist(p_t[y_t == 0], bins=10, alpha=0.6, label=f"EGFR- (n={(y_t==0).sum()})", color="C0")
        plt.hist(p_t[y_t == 1], bins=10, alpha=0.6, label=f"EGFR+ (n={(y_t==1).sum()})", color="C3")
        plt.xlabel("Predicted P(EGFR+)")
        plt.ylabel("Count")
        plt.title(f"Test set predictions (n={len(test_cases)}, test AUC={test_auc:.3f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "figures/pred_dist_split.png", dpi=120)
        plt.close()

    # summary.txt
    with open(out_dir / "summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EGFR-only 方案 A:固定 70/15/15 分层切分\n")
        f.write("=" * 60 + "\n")
        f.write(f"Case 总数: {len(cases)}  (EGFR+: {sum(egfr_label[c] for c in cases)})\n")
        f.write(f"\n── 切分(按 any_9gene_positive 分层)──\n")
        f.write(f"  train: {len(train_cases)}  (EGFR+={sum(egfr_label[c] for c in train_cases)})\n")
        f.write(f"  val  : {len(val_cases)}  (EGFR+={sum(egfr_label[c] for c in val_cases)})\n")
        f.write(f"  test : {len(test_cases)}  (EGFR+={sum(egfr_label[c] for c in test_cases)})\n")
        f.write(f"\n── 模型/超参 ──\n")
        f.write(f"  EGFROnlyModel hidden_dim={args.hidden_dim}, dropout={args.dropout}, wd={args.weight_decay}\n")
        f.write(f"  lr={args.lr}, epochs={args.epochs}, patience={args.patience}\n")
        f.write(f"  train pos_weight={res['pos_weight']:.3f}\n")
        f.write(f"  best val AUC = {res['best_val_auc']:.3f} (stopped at epoch {res['stopped_epoch']})\n")
        f.write(f"\n── 最终评估 ──\n")
        f.write(f"  val  AUC = {val_auc:.3f}  (n={len(val_cases)})\n")
        f.write(f"  test AUC = {test_auc:.3f}  (n={len(test_cases)})\n")
        f.write(f"\n── 备注 ──\n")
        f.write(f"  test 集只有 {int(y_t.sum())} 个 EGFR+ 样本,AUC 估计抖动大。\n")
        f.write(f"  真实评估请用 LOO 结果(`step3_egfr_loo.py`)。\n")
        f.write(f"  本方案 A 用途:模型选择 + 训练/val/test 流程演练。\n")
    print(f"\n[Step3-EGFR-Split] outputs -> {out_dir}/")
    print(f"[Step3-EGFR-Split] done (total {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()