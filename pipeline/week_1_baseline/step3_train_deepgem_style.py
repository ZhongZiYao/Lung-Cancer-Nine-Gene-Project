"""
step3_train_deepgem_style.py
────────────────────────────
DeepGEM 风格训练:5-fold StratifiedKFold(按 any_positive 分层),每个 fold 训一个模型。

Loss 组成(DeepGEM + GAMIL 简化版):
  L_total = w_bag * L_bag(BCE) + w_inst * L_inst(BCE)
  L_bag:   case-level BCE(主损失,0.7)
  L_inst:  patch-level BCE 经 prototype → 聚合(辅助,0.3,前 5 epoch 衰减到 0.1)
  PartialLoss / EMA confidence: 不上(83 case 数据量不够,简化版)

类不平衡:pos_weight(逆频)
优化:AdamW + Warmup + 早停(val AUC 50 epoch 不升)
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
GENES = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF",
         "PIK3CA", "ERBB2", "NRAS"]

# 训练循环用的全局进度状态(由 main 设,trainer 读)
STATE = {}


# ─────────────── flush_print:所有 print 走这个,避免被 tee 攒住 ───────────────
# 为什么黑盒: `python -u` 只解 Python 内部缓冲; pipe + tee 还会按 4KB 攒盘
# 这里强制 flush=True 并写换行符,确保每个输出立刻落到 .log
def flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ─────────────── 数据集 ───────────────
class CaseDataset(torch.utils.data.Dataset):
    """每次返回一个 case(变长 SM 数、变长 patch 数)"""
    def __init__(self, cases: list[str], bags_dir: Path, labels: dict):
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
            "patch_feats": torch.from_numpy(d["patch_features"]).float(),  # [N, 1024]
            "sm_indices": torch.from_numpy(d["slice_ids"]).long(),          # [N]
            "label": torch.tensor([self.labels[cid][g] for g in GENES],
                                  dtype=torch.float32),                     # [9]
        }


def collate(batch):
    return batch[0]   # batch_size=1,每 case 单独处理


# ─────────────── 训练器 ───────────────
class Trainer:
    def __init__(self, model, device, num_genes=9, lr=2e-4, weight_decay=1e-4,
                 warmup_epochs=5, w_bag=0.7, w_inst_init=0.3, w_inst_final=0.1,
                 ema_m=0.9):
        self.model = model.to(device)
        self.device = device
        self.num_genes = num_genes
        self.optim = torch.optim.AdamW(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)
        self.warmup_epochs = warmup_epochs
        self.w_bag = w_bag
        self.w_inst_init = w_inst_init
        self.w_inst_final = w_inst_final
        # 用 WSI 标签给每个基因算 pos_weight
        # (在 train_one_fold 中更新)

    def get_w_inst(self, epoch: int) -> float:
        # 从 init 线性衰减到 final
        if epoch <= self.warmup_epochs:
            return 0.0  # warmup 阶段只用 bag loss
        # 10 epoch 内从 init 衰减到 final
        t = min(1.0, (epoch - self.warmup_epochs) / 10.0)
        return self.w_inst_init * (1 - t) + self.w_inst_final * t

    def train_one_epoch(self, loader, pos_weight: torch.Tensor,
                        epoch: int = 0, n_epoch: int = 0,
                        fold: int = 0, n_fold: int = 0,
                        n_cases_total: int = 0):
        self.model.train()
        total_loss = 0.0
        n = 0
        n_cases = len(loader.dataset)
        for i, batch in enumerate(loader, 1):
            patch_feats = batch["patch_feats"].to(self.device)
            sm_idx = batch["sm_indices"].to(self.device)
            label = batch["label"].to(self.device)              # [9]
            n_sm = int(sm_idx.max().item()) + 1

            bag_logits, inst_logits, aux = self.model(patch_feats, sm_idx, n_sm)
            bag_loss = F.binary_cross_entropy_with_logits(
                bag_logits, label, pos_weight=pos_weight.to(self.device)
            )
            inst_loss = F.binary_cross_entropy_with_logits(
                inst_logits, label, pos_weight=pos_weight.to(self.device)
            )

            ep = self.cur_epoch
            w_inst = self.get_w_inst(ep)
            loss = self.w_bag * bag_loss + w_inst * inst_loss

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()

            total_loss += loss.item()
            n += 1
            # case-by-case 进度(每 5 case 或最后一个 case 打印一次)
            if (i % 5 == 0) or (i == n_cases):
                # 全局进度:fold {fold/n_fold} ep {epoch/n_epoch} case {i}/{n_cases}
                case_done = (fold - 1) * n_epoch * n_cases_total + (epoch - 1) * n_cases_total + i
                case_total = n_fold * n_epoch * n_cases_total
                pct = case_done / max(case_total, 1) * 100
                flush_print(f"\r      [fold{fold}/{n_fold} ep{epoch}/{n_epoch}] "
                            f"case {i}/{n_cases}  loss_so_far={total_loss/n:.3f}  "
                            f"全局 {case_done}/{case_total} ({pct:.1f}%)",
                            end="", flush=True)
        flush_print("")  # epoch 结束换行
        return total_loss / max(n, 1)

    @torch.no_grad()
    def eval_one_epoch(self, loader, pos_weight=None):
        """返回每个 case 的 9 基因 sigmoid 概率 + 真实标签,便于算 AUC"""
        self.model.eval()
        case_ids, preds, labels = [], [], []
        for batch in loader:
            cid = batch["case_id"]
            patch_feats = batch["patch_feats"].to(self.device)
            sm_idx = batch["sm_indices"].to(self.device)
            label = batch["label"]
            n_sm = int(sm_idx.max().item()) + 1
            bag_logits, _, _ = self.model(patch_feats, sm_idx, n_sm)
            p = torch.sigmoid(bag_logits).cpu().numpy()
            case_ids.append(cid)
            preds.append(p)
            labels.append(label.numpy())
        return case_ids, np.array(preds), np.array(labels)


def compute_auc_per_gene(preds: np.ndarray, labels: np.ndarray,
                          case_ids: list[str]) -> dict:
    """对每个基因算 AUC;若该基因在所有 case 都同标签,返回 NaN。"""
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = {}
    for g_idx, g in enumerate(GENES):
        y = labels[:, g_idx]
        p = preds[:, g_idx]
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        row = {"gene": g, "n_pos": n_pos, "n_neg": n_neg,
               "pred_mean": float(p.mean()), "pred_std": float(p.std())}
        if 0 < n_pos < len(y):
            try:
                row["auc"] = float(roc_auc_score(y, p))
                row["ap"] = float(average_precision_score(y, p))
            except Exception as e:
                row["auc"] = None
                row["ap"] = None
                row["err"] = str(e)
        else:
            row["auc"] = None
            row["ap"] = None
            row["note"] = "no positive" if n_pos == 0 else "all positive"
        out[g] = row
    return out


# ─────────────── 主流程 ───────────────
def train_one_fold(train_cases, val_cases, bags_dir, labels,
                   hidden_dim, epochs, patience, device, lr,
                   weight_decay, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    from deepgem_style_model import DeepGEMStyleModel

    train_ds = CaseDataset(train_cases, bags_dir, labels)
    val_ds = CaseDataset(val_cases, bags_dir, labels)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )

    model = DeepGEMStyleModel(in_dim=1024, hidden_dim=hidden_dim, num_genes=9)
    trainer = Trainer(model, device, lr=lr, weight_decay=weight_decay)

    # 算每个基因 pos_weight = neg/pos(逆频,clip 到 10)
    train_labels = np.array([[labels[c][g] for g in GENES] for c in train_cases])
    pos_w = []
    for g_idx in range(9):
        n_pos = train_labels[:, g_idx].sum()
        n_neg = len(train_cases) - n_pos
        if n_pos == 0 or n_neg == 0:
            pw = 1.0
        else:
            pw = min(n_neg / max(n_pos, 1), 10.0)
        pos_w.append(pw)
    pos_weight = torch.tensor(pos_w, dtype=torch.float32)
    flush_print(f"  pos_weight = {dict(zip(GENES, [round(x, 2) for x in pos_w]))}")

    best_auc_mean = -1.0
    best_preds = None
    best_labels = None
    best_case_ids = None
    best_per_gene = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        trainer.cur_epoch = epoch
        t0 = time.time()
        # 全局进度:fold/ep/case
        st = globals().get("STATE", {})
        train_loss = trainer.train_one_epoch(
            train_loader, pos_weight,
            epoch=epoch, n_epoch=epochs,
            fold=st.get("fold", 0), n_fold=st.get("n_fold", 0),
            n_cases_total=st.get("n_cases_total", 0),
        )

        # 评估
        case_ids, preds, lbls = trainer.eval_one_epoch(val_loader)
        auc_per_gene = compute_auc_per_gene(preds, lbls, case_ids)
        valid_aucs = [v["auc"] for v in auc_per_gene.values()
                      if v["auc"] is not None]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
        # 退化指标:若全基因都不可算 AUC,用 0.5(随机) 作 baseline,
        # 并用 loss 最低 epoch 作 fallback
        if not valid_aucs:
            # 退化:用 1 - loss 估一个"伪 AUC",但 cap 在 [0,1]
            mean_auc_proxy = max(0.0, min(1.0, 1.0 - train_loss))
        else:
            mean_auc_proxy = mean_auc

        elapsed = time.time() - t0
        # epoch 摘要 + 9 基因分基因 AUC(只算有阳性的)
        per_gene_str = "  ".join(
            f"{g[:4]}={(auc_per_gene[g]['auc'] or 0):.2f}" if auc_per_gene[g]["auc"] is not None
            else f"{g[:4]}=N/A"
            for g in GENES
        )
        # epoch 概要行
        flush_print(f"\n  >>> ep{epoch:>2}: loss={train_loss:.3f} "
                    f"val_mean_auc={mean_auc:.3f} ({elapsed:.1f}s)")
        # 分基因 AUC 单独一行(可视化)
        flush_print(f"      per-gene AUC: {per_gene_str}", end="")

        if mean_auc_proxy > best_auc_mean:
            best_auc_mean = mean_auc_proxy
            best_preds = preds
            best_labels = lbls
            best_case_ids = case_ids
            best_per_gene = auc_per_gene
            epochs_no_improve = 0
            flush_print("  ✓ best")
        else:
            epochs_no_improve += 1
            flush_print(f"  ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            flush_print(f"    早停 at epoch {epoch}")
            break

    # 退化兜底:若 best_* 仍是 None(全 val 都算不出 AUC),用最后 epoch 的预测
    if best_preds is None:
        case_ids, preds, lbls = trainer.eval_one_epoch(val_loader)
        best_preds = preds
        best_labels = lbls
        best_case_ids = case_ids
        # 退化时也算一遍分基因 AUC
        best_per_gene = compute_auc_per_gene(preds, lbls, case_ids)

    # ============== Fold 总结 ==============
    flush_print("\n  ┌────── Fold 训练总结 ──────")
    flush_print(f"  │ best mean AUC = {best_auc_mean:.4f}")
    if best_per_gene:
        # 把 9 基因 AUC 打成一个 mini 表格
        flush_print("  │ 分基因 best AUC:")
        for g in GENES:
            v = best_per_gene.get(g, {})
            auc = v.get("auc")
            n_pos = v.get("n_pos", 0)
            n_neg = v.get("n_neg", 0)
            if auc is None:
                suffix = f"({v.get('note','N/A')})"
            else:
                suffix = f"(n_pos={n_pos}, n_neg={n_neg})"
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            flush_print(f"  │   {g:7s}  AUC={auc_str:>6s}  {suffix}")
    flush_print("  └" + "─" * 40)

    return {
        "best_auc_mean": best_auc_mean,
        "preds": best_preds,
        "labels": best_labels,
        "case_ids": best_case_ids,
        "best_per_gene": best_per_gene,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags-dir",
                    default="pipeline/week_1_baseline/outputs/bags_per_case")
    ap.add_argument("--manifest",
                    default="pipeline/week_1_baseline/case_manifest.csv")
    ap.add_argument("--out",
                    default="pipeline/week_1_baseline/outputs/predictions.csv")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    flush_print(f"[Step3] device={device}, folds={args.folds}")

    # 读 manifest
    cases, labels = [], {}
    with open(args.manifest) as f:
        rd = csv.DictReader(f)
        for r in rd:
            cases.append(r["case_id"])
            labels[r["case_id"]] = {g: int(r[g]) for g in GENES}
    flush_print(f"[Step3] 共 {len(cases)} case")

    bags_dir = Path(args.bags_dir)
    if not bags_dir.exists():
        sys.exit(f"找不到 {bags_dir},请先跑 step2")
    n_avail = len(list(bags_dir.glob("*.npz")))
    flush_print(f"[Step3] bags 目录下有 {n_avail} 个 case 的 .npz")
    # 只用已有 .npz 的 case
    available = set(p.stem for p in bags_dir.glob("*.npz"))
    cases = [c for c in cases if c in available]
    n_cases_total = len(cases)
    flush_print(f"[Step3] 实际可训: {n_cases_total} case")

    y_any = np.array([1 if any(labels[c].values()) else 0 for c in cases])
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_preds = np.zeros((len(cases), 9), dtype=np.float32)
    oof_done = np.zeros(len(cases), dtype=bool)
    fold_results = []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(cases)),
                                                          y_any), 1):
        train_cases = [cases[i] for i in tr_idx]
        test_cases = [cases[i] for i in te_idx]
        flush_print(f"\n========== Fold {fold_i}/{args.folds} ==========")
        flush_print(f"  train={len(train_cases)} test={len(test_cases)}")
        # 把 fold/epoch/case 总数传给 trainer 用来算全局进度
        STATE.clear()
        STATE.update({"fold": fold_i, "n_fold": args.folds,
                      "n_epoch": args.epochs, "n_cases_total": n_cases_total})
        res = train_one_fold(
            train_cases, test_cases, bags_dir, labels,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs, patience=args.patience,
            device=device, lr=args.lr, weight_decay=args.weight_decay,
            seed=args.seed + fold_i,
        )
        # 把预测写回 oof
        for cid, p in zip(res["case_ids"], res["preds"]):
            idx = cases.index(cid)
            oof_preds[idx] = p
            oof_done[idx] = True
        fold_results.append({"fold": fold_i, "best_auc_mean": res["best_auc_mean"]})
        flush_print(f"  >> Fold {fold_i} best mean AUC = {res['best_auc_mean']:.3f}")

    # ============== 全 5-fold 结束: 9 基因 OOF 汇总 ==============
    # 这里 oof_preds 已经覆盖了 83 case(每个 case 来自它属于的 fold 的最佳模型预测)
    oof_auc_per_gene = compute_auc_per_gene(oof_preds,
                                            np.array([[labels[c][g] for g in GENES]
                                                      for c in cases]),
                                            cases)
    flush_print("\n╔══════════════════════════════════════════════════════════════╗")
    flush_print("║                  Step3 完成 — OOF AUC 摘要                  ║")
    flush_print("╠══════════════════════════════════════════════════════════════╣")
    for g in GENES:
        v = oof_auc_per_gene.get(g, {})
        auc = v.get("auc")
        ap = v.get("ap")
        n_pos = v.get("n_pos", 0)
        n_neg = v.get("n_neg", 0)
        if auc is None:
            line = f"║   {g:7s}  AUC={'N/A':>5s}  AP={'N/A':>5s}  ({v.get('note','no signal')})"
        else:
            line = f"║   {g:7s}  AUC={auc:.3f}  AP={ap:.3f}   (n_pos={n_pos}, n_neg={n_neg})"
        flush_print(line)
    flush_print("╚══════════════════════════════════════════════════════════════╝")

    # 写 predictions.csv
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id"] +
                   [f"true_{g}" for g in GENES] +
                   [f"pred_prob_{g}" for g in GENES])
        for i, c in enumerate(cases):
            row = [c] + [labels[c][g] for g in GENES] + oof_preds[i].tolist()
            w.writerow(row)
    flush_print(f"\n[Step3] predictions → {args.out}")
    flush_print(f"[Step3] OOF 覆盖率: {oof_done.sum()}/{len(cases)}")

    # 写 fold summary
    summary = Path(args.out).with_name("fold_summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "best_auc_mean"])
        for r in fold_results:
            w.writerow([r["fold"], round(r["best_auc_mean"], 4)])
    flush_print(f"[Step3] fold summary → {summary}")
    flush_print("\n[Step3] 完成 ✓")


if __name__ == "__main__":
    main()