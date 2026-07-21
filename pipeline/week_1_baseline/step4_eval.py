"""
step4_eval.py
─────────────
读 step3 的 predictions.csv,出 9 基因 ROC-AUC 汇总 + ROC 曲线 + 报告。

输入:
  --predictions predictions.csv
  --manifest     case_manifest.csv(用于一致性校验,可选)

输出:
  --out-dir/outputs/
    metrics_per_gene.csv   每个基因的 AUC / AP / F1 + 有效 fold 数
    figures/roc_curves.png 9 子图 ROC
    figures/group_comparison.png 高频组 vs 低频组对比
    summary.txt            文字摘要(给组会看)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np

GENES = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF",
         "PIK3CA", "ERBB2", "NRAS"]
# 分组:高频组(每折基本 ≥1 阳性),低频组(某些折 0 阳性)
HIGH_FREQ = {"EGFR", "KRAS", "ALK", "TP53", "BRAF"}
LOW_FREQ = {"ROS1", "PIK3CA", "ERBB2", "NRAS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default="pipeline/week_1_baseline/outputs/predictions.csv")
    ap.add_argument("--out-dir",
                    default="pipeline/week_1_baseline/outputs")
    args = ap.parse_args()

    cases, true_y, pred_p = [], {g: [] for g in GENES}, {g: [] for g in GENES}
    with open(args.predictions) as f:
        rd = csv.DictReader(f)
        for r in rd:
            cases.append(r["case_id"])
            for g in GENES:
                true_y[g].append(int(r[f"true_{g}"]))
                pred_p[g].append(float(r[f"pred_prob_{g}"]))
    print(f"[Step4] {len(cases)} case, 9 基因")

    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, roc_curve)

    metrics_rows = []
    fig_data = []
    for g in GENES:
        y = np.array(true_y[g])
        p = np.array(pred_p[g])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        row = {"gene": g, "n_pos": n_pos, "n_neg": n_neg,
               "pred_mean": float(p.mean()), "pred_std": float(p.std()),
               "group": "high" if g in HIGH_FREQ else "low"}
        if 0 < n_pos < len(y):
            row["auc"] = float(roc_auc_score(y, p))
            row["ap"] = float(average_precision_score(y, p))
            # Youden's J 阈值
            fpr, tpr, thr = roc_curve(y, p)
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            row["best_thr"] = float(thr[best_idx])
            row["tpr_at_best"] = float(tpr[best_idx])
            row["fpr_at_best"] = float(fpr[best_idx])
            yhat = (p >= row["best_thr"]).astype(int)
            row["f1_at_best"] = float(f1_score(y, yhat, zero_division=0))
            row["acc_at_best"] = float((yhat == y).mean())
            fig_data.append((g, fpr, tpr, row["auc"], n_pos))
        else:
            row["auc"] = None
            row["ap"] = None
            row["note"] = "no positive" if n_pos == 0 else "all positive"
        metrics_rows.append(row)
        auc_str = f"{row['auc']:.3f}" if row.get("auc") else "N/A"
        print(f"  {g:<6} n_pos={n_pos:>2}  AUC={auc_str}  "
              f"thr={row.get('best_thr', 'N/A')}  "
              f"F1={row.get('f1_at_best', 'N/A')}")

    # 写 metrics_per_gene.csv
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics_per_gene.csv"
    cols = ["gene", "group", "n_pos", "n_neg", "auc", "ap", "best_thr",
            "tpr_at_best", "fpr_at_best", "f1_at_best", "acc_at_best",
            "pred_mean", "pred_std", "note"]
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in metrics_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n[Step4] 指标 → {metrics_csv}")

    # 画 9 子图 ROC
    if fig_data:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 3, figsize=(13, 11))
            for ax, (g, fpr, tpr, auc, n_pos) in zip(axes.flat, fig_data):
                ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.set_title(f"{g} (n_pos={n_pos})")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.legend(loc="lower right", fontsize=9)
                ax.grid(alpha=0.3)
            fig.suptitle("Week-1 Baseline: 9-gene ROC (5-fold OOF)", fontsize=14)
            fig.tight_layout()
            fig_path = out_dir / "figures" / "roc_curves.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=150)
            print(f"[Step4] ROC 图 → {fig_path}")
        except Exception as e:
            print(f"[Step4] ⚠️ matplotlib 画图失败: {e}")

    # 高频组 vs 低频组对比图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        high_aucs = [r["auc"] for r in metrics_rows
                     if r["gene"] in HIGH_FREQ and r.get("auc") is not None]
        low_aucs = [r["auc"] for r in metrics_rows
                    if r["gene"] in LOW_FREQ and r.get("auc") is not None]
        fig, ax = plt.subplots(figsize=(7, 5))
        labels, vals = [], []
        if high_aucs:
            labels.append(f"High-freq\n({', '.join(sorted(HIGH_FREQ))})")
            vals.append(high_aucs)
        if low_aucs:
            labels.append(f"Low-freq\n({', '.join(sorted(LOW_FREQ))})")
            vals.append(low_aucs)
        if labels:
            bp = ax.boxplot(vals, labels=labels, patch_artist=True,
                            showmeans=True)
            for patch, color in zip(bp["boxes"], ["#4CAF50", "#FF9800"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            for i, v in enumerate(vals, 1):
                ax.scatter([i] * len(v), v, alpha=0.7, s=40, color="black")
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("AUC")
            ax.set_ylim(0, 1)
            ax.set_title("High-freq vs Low-freq Gene AUC")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            gp = out_dir / "figures" / "group_comparison.png"
            gp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(gp, dpi=150)
            print(f"[Step4] 组对比图 → {gp}")
    except Exception as e:
        print(f"[Step4] ⚠️ 组对比图失败: {e}")

    # 文字摘要(给组会用)
    summary_path = out_dir / "summary.txt"
    valid_aucs = [r["auc"] for r in metrics_rows if r.get("auc") is not None]
    valid_high = [r["auc"] for r in metrics_rows
                  if r["gene"] in HIGH_FREQ and r.get("auc") is not None]
    valid_low = [r["auc"] for r in metrics_rows
                 if r["gene"] in LOW_FREQ and r.get("auc") is not None]

    lines = [
        "=" * 60,
        "Week-1 Baseline 报告 — 9-gene Lung Adenocarcinoma Mutation",
        "=" * 60,
        f"Case 数: {len(cases)}",
        f"CV: 5-fold StratifiedKFold (by any_positive)",
        f"Backbone: UNI (1024-dim, frozen)",
        f"模型: GAMIL-style Gated Attention + DeepGEM-style prototype",
        "",
        "── 全 9 基因 ──",
    ]
    for r in metrics_rows:
        auc_str = f"{r['auc']:.3f}" if r.get("auc") is not None else "N/A"
        lines.append(f"  {r['gene']:<6}  n_pos={r['n_pos']:>2}  "
                     f"AUC={auc_str}  F1@best={r.get('f1_at_best', 'N/A')}")
    lines += [
        "",
        f"── 高频组 (mean AUC) ──",
        f"  {', '.join(sorted(HIGH_FREQ))}",
        f"  mean = {np.mean(valid_high):.3f} "
        f"(min={min(valid_high):.3f}, max={max(valid_high):.3f})" if valid_high else
        "  N/A",
        f"── 低频组 (mean AUC) ──",
        f"  {', '.join(sorted(LOW_FREQ))}",
        f"  mean = {np.mean(valid_low):.3f} "
        f"(min={min(valid_low):.3f}, max={max(valid_low):.3f})" if valid_low else
        "  N/A",
        "",
        f"全部有效 AUC 数: {len(valid_aucs)}/{len(GENES)}",
        "",
        "⚠️ 注意: 83 case 训练,NRAS(2 例)等低频基因 AUC 方差大,",
        "   实际数字应配合置信区间/不同 seed 看。",
        "=" * 60,
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Step4] 摘要 → {summary_path}")
    print("\n".join(lines[:8]))
    print("\n[Step4] 完成 ✓")


if __name__ == "__main__":
    main()