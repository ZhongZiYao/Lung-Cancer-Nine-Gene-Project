"""
step3_egfr_loo.py
─────────────────
EGFR-only 留一法(LOO)训练 + 评估。
考虑到 13 个阳性样本太少,5-fold 抖动大,改用 LOO:每个 case 轮流当测试集。

对每个 test case i:
  - 训练集:其余 82 case
  - 验证集:从训练集里**按 EGFR 分层**抽 8 case(~10%)用于早停
  - 测试集:仅 1 个 case,记录模型预测概率
最后用 83 个 test 概率 + 83 个真实标签计算 OOF AUC。

输出:
  outputs_egfr_loo/loo_predictions.csv
  outputs_egfr_loo/loo_summary.csv
  outputs_egfr_loo/summary.txt
  outputs_egfr_loo/figures/roc_loo.png
  outputs_egfr_loo/figures/pred_dist_loo.png

用法:
  python step3_egfr_loo.py
  python step3_egfr_loo.py --hidden-dim 64 --epochs 60 --patience 10 --val-frac 0.1
"""
from __future__ import annotations
import argparse
import csv
import functools
import sys
import time
import warnings
from pathlib import Path

# 强制 unbuffered:逐行 flush,实时看进度
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

from egfr_only_model import EGFROnlyModel


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
        d = np.load(self.bags_dir / f"{cid}.npz", allow_pickle=True)
        return {
            "case_id": cid,
            "patch_feats": torch.from_numpy(d["patch_features"]).float(),
            "sm_indices": torch.from_numpy(d["slice_ids"]).long(),
            "label": torch.tensor(self.labels[cid], dtype=torch.float32),
        }


def collate(batch):
    return batch[0]


# ─────────────── 单 case LOO 训练 ───────────────
def train_one_loo(test_case, train_pool, val_cases, bags_dir, labels,
                  hidden_dim, epochs, patience, device, lr, weight_decay,
                  dropout, seed):
    """训练一个 LOO 折:train=train_pool 验证=val_cases 测试=test_case"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CaseDataset(train_pool, bags_dir, labels)
    val_ds = CaseDataset(val_cases, bags_dir, labels)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )

    model = EGFROnlyModel(in_dim=1024, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # pos_weight 由训练集统计
    train_labels = np.array([labels[c] for c in train_pool])
    n_pos = int(train_labels.sum())
    n_neg = len(train_pool) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1), 10.0)
    pos_weight_t = torch.tensor([pos_weight], device=device)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

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
            logit, _ = model(patch_feats, sm_idx, n_sm)
            loss = F.binary_cross_entropy_with_logits(
                logit.unsqueeze(0), label, pos_weight=pos_weight_t
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / max(n, 1)

        # 验证:val_cases 上算 AUC
        model.eval()
        preds, lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                patch_feats = batch["patch_feats"].to(device)
                sm_idx = batch["sm_indices"].to(device)
                label = batch["label"]
                n_sm = int(sm_idx.max().item()) + 1
                logit, _ = model(patch_feats, sm_idx, n_sm)
                p = torch.sigmoid(logit).cpu().numpy()
                preds.append(float(p))
                lbls.append(int(label.item()))
        y_v = np.array(lbls)
        p_v = np.array(preds)
        n_pos_v = int(y_v.sum())
        val_auc = float(np.nan) if not (0 < n_pos_v < len(y_v)) else \
            float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_v, p_v))

        elapsed = time.time() - t0
        improved = (val_auc == val_auc) and (val_auc > best_val_auc)  # NaN-safe
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or improved:
            tag = "  [best]" if improved else (f"  ({epochs_no_improve}/{patience})" if epochs_no_improve > 0 else "")
            vstr = f"{val_auc:.3f}" if val_auc == val_auc else "N/A"
            print(f"      ep{epoch:>2}: loss={train_loss:.3f} val_auc={vstr} ({elapsed:.1f}s){tag}", end="\n")

        if epochs_no_improve >= patience:
            print(f"      early stop at epoch {epoch} (best val AUC {best_val_auc:.3f})")
            break

    # 用最佳权重对 test_case 评分
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    d = np.load(bags_dir / f"{test_case}.npz", allow_pickle=True)
    patch_feats = torch.from_numpy(d["patch_features"]).float().to(device)
    sm_idx = torch.from_numpy(d["slice_ids"]).long().to(device)
    n_sm = int(sm_idx.max().item()) + 1
    with torch.no_grad():
        logit, _ = model(patch_feats, sm_idx, n_sm)
        test_pred = float(torch.sigmoid(logit).cpu().numpy())

    return {
        "test_case": test_case,
        "test_pred": test_pred,
        "test_label": int(labels[test_case]),
        "best_val_auc": float(best_val_auc) if best_val_auc == best_val_auc else float("nan"),
        "stopped_epoch": epoch,
        "n_train": len(train_pool),
        "n_val": len(val_cases),
        "pos_weight": pos_weight,
    }


# ─────────────── main ───────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags-dir",
                    default="pipeline/week_1_baseline/outputs/bags_v3")
    ap.add_argument("--manifest",
                    default="pipeline/week_1_baseline/case_manifest.csv")
    ap.add_argument("--out-dir",
                    default="pipeline/week_1_baseline/outputs_egfr_loo")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="从训练池里按 EGFR 分层抽 val 的比例(默认 0.1≈8 case)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="debug 用:只跑前 N 个 case 作为 test")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step3-EGFR-LOO] device={device}, hidden_dim={args.hidden_dim}, "
          f"val_frac={args.val_frac}, epochs={args.epochs}, patience={args.patience}")
    print(f"[Step3-EGFR-LOO] bags_dir={args.bags_dir}")

    # 读 manifest
    cases, labels = [], {}
    with open(args.manifest) as f:
        rd = csv.DictReader(f)
        for r in rd:
            cid = r["case_id"]
            cases.append(cid)
            labels[cid] = int(r["EGFR"])
    print(f"[Step3-EGFR-LOO] 共 {len(cases)} case, EGFR+={sum(labels.values())}, "
          f"-={len(cases)-sum(labels.values())}")

    bags_dir = Path(args.bags_dir)
    available = set(p.stem for p in bags_dir.glob("*.npz"))
    cases = [c for c in cases if c in available]
    print(f"[Step3-EGFR-LOO] 实际可训: {len(cases)} case, "
          f"EGFR+={sum(labels[c] for c in cases)}")

    if args.max_cases:
        cases = cases[:args.max_cases]
        print(f"[Step3-EGFR-LOO] debug: 只跑前 {len(cases)} 个 case")

    from sklearn.model_selection import StratifiedShuffleSplit

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # LOO 主循环
    n = len(cases)
    results = []
    t_start = time.time()

    for i, test_case in enumerate(cases, 1):
        # 训练池 = 剩下 n-1 个
        train_pool = [c for c in cases if c != test_case]
        # 分层抽 val
        if args.val_frac > 0 and len(train_pool) > 1:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=args.val_frac, random_state=args.seed + i
            )
            y_pool = np.array([labels[c] for c in train_pool])
            # 若池里只有单一标签,sss 会报错 → 退化
            if len(np.unique(y_pool)) >= 2:
                tr_idx, val_idx = next(sss.split(np.zeros(len(train_pool)), y_pool))
                train_cases = [train_pool[j] for j in tr_idx]
                val_cases = [train_pool[j] for j in val_idx]
            else:
                train_cases = train_pool
                val_cases = [train_pool[0]]  # 至少留 1 个
        else:
            train_cases = train_pool
            val_cases = []

        egfr_test = labels[test_case]
        n_pos_pool = sum(labels[c] for c in train_pool)
        n_pos_val = sum(labels[c] for c in val_cases)
        print(f"\n[LOO {i:>3}/{n}] test={test_case} (EGFR={egfr_test}) | "
              f"train={len(train_cases)}(+{n_pos_pool - n_pos_val}) "
              f"val={len(val_cases)}(+{n_pos_val})")
        if not val_cases:
            print(f"      (warning) val empty, train with fixed epochs={args.epochs}")

        res = train_one_loo(
            test_case=test_case,
            train_pool=train_cases,
            val_cases=val_cases,
            bags_dir=bags_dir, labels=labels,
            hidden_dim=args.hidden_dim, epochs=args.epochs, patience=args.patience,
            device=device, lr=args.lr, weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed + i,
        )
        results.append(res)
        elapsed_total = time.time() - t_start
        eta = elapsed_total / i * (n - i)
        print(f"      -> test_pred={res['test_pred']:.3f} (true={res['test_label']}) "
              f"best_val_auc={res['best_val_auc']:.3f} | "
              f"elapsed={elapsed_total/60:.1f}min eta={eta/60:.1f}min")

    # 写 predictions.csv
    with open(out_dir / "loo_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "true_EGFR", "pred_prob_EGFR", "best_val_auc", "stopped_epoch"])
        for r in results:
            w.writerow([r["test_case"], r["test_label"], round(r["test_pred"], 6),
                        round(r["best_val_auc"], 4) if r["best_val_auc"] == r["best_val_auc"] else "N/A",
                        r["stopped_epoch"]])

    # 写 loo_summary.csv(每 case 一行)
    with open(out_dir / "loo_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "true_EGFR", "pred_prob_EGFR", "best_val_auc",
                    "stopped_epoch", "pos_weight", "n_train", "n_val"])
        for r in results:
            w.writerow([r["test_case"], r["test_label"],
                        round(r["test_pred"], 6),
                        round(r["best_val_auc"], 4) if r["best_val_auc"] == r["best_val_auc"] else "N/A",
                        r["stopped_epoch"], round(r["pos_weight"], 3),
                        r["n_train"], r["n_val"]])

    # 算 OOF AUC
    from sklearn.metrics import roc_auc_score
    y = np.array([r["test_label"] for r in results])
    p = np.array([r["test_pred"] for r in results])
    overall_auc = float(roc_auc_score(y, p))

    # 分层 AUC:EGFR+ vs EGFR- 各自的预测分布
    pos_mask = y == 1
    neg_mask = y == 0
    p_pos = p[pos_mask]
    p_neg = p[neg_mask]

    # 画 ROC + 预测分布
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"EGFR-only LOO (AUC={overall_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("EGFR-only LOO ROC (n=83)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "figures/roc_loo.png", dpi=120)
    plt.close()

    # 预测分布
    plt.figure(figsize=(8, 5))
    plt.hist(p_neg, bins=20, alpha=0.6, label=f"EGFR- (n={int(neg_mask.sum())})", color="C0")
    plt.hist(p_pos, bins=20, alpha=0.6, label=f"EGFR+ (n={int(pos_mask.sum())})", color="C3")
    plt.xlabel("Predicted P(EGFR+)")
    plt.ylabel("Count")
    plt.title(f"Prediction distribution (LOO, OOF AUC={overall_auc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figures/pred_dist_loo.png", dpi=120)
    plt.close()

    # summary
    with open(out_dir / "summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EGFR-only LOO 报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"Case 数: {len(cases)}  (EGFR+: {int(pos_mask.sum())}, "
                f"EGFR-: {int(neg_mask.sum())})\n")
        f.write(f"CV: Leave-One-Out\n")
        f.write(f"Backbone: UNI (1024-dim, frozen)\n")
        f.write(f"模型: EGFROnlyModel hidden_dim={args.hidden_dim}, "
                f"dropout={args.dropout}, wd={args.weight_decay}\n")
        f.write(f"Train/val/test 划分: 留 1 case 当 test;训练集按 EGFR 分层 "
                f"抽 {args.val_frac*100:.0f}% 当 val 早停;余下训练\n")
        f.write(f"超参: lr={args.lr}, epochs={args.epochs}, patience={args.patience}\n")
        f.write(f"\n── OOF AUC ──\n  EGFR: {overall_auc:.3f}\n")
        f.write(f"\n── 预测概率统计 ──\n")
        f.write(f"  EGFR+: mean={p_pos.mean():.3f} median={np.median(p_pos):.3f} "
                f"std={p_pos.std():.3f}\n")
        f.write(f"  EGFR-: mean={p_neg.mean():.3f} median={np.median(p_neg):.3f} "
                f"std={p_neg.std():.3f}\n")
        f.write(f"\n── 早停 epoch 统计 ──\n")
        stops = [r["stopped_epoch"] for r in results]
        f.write(f"  mean={np.mean(stops):.1f} median={np.median(stops):.0f} "
                f"max={max(stops)}\n")
        f.write(f"  (用满 epochs 数量 = {sum(1 for s in stops if s == args.epochs)})\n")
    print(f"\n[Step3-EGFR-LOO] OOF AUC = {overall_auc:.3f}")
    print(f"[Step3-EGFR-LOO] predictions -> {out_dir}/loo_predictions.csv")
    print(f"[Step3-EGFR-LOO] summary -> {out_dir}/summary.txt")
    print(f"[Step3-EGFR-LOO] figures -> {out_dir}/figures/")
    print(f"\n[Step3-EGFR-LOO] done (total { (time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()