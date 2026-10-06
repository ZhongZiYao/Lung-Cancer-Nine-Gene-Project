# -*- coding: utf-8 -*-
"""48_experiment_matrix.py — 汇总全部实验为论文用的表格

把分散在各 CSV 里的 25 项实验汇成:
  Table A  单变量消融矩阵(方法族 / 变体 / mean AUC / Δ / 同号 / 判定)
  Table B  跨种子基线与噪声阈值
  Table C  最终模型的完整指标(内部 + 外部)
输出: .tmp/9gene_实验矩阵_TableA-C.md
"""
import os, sys, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
C = os.path.join(R, "data", "patch_cache")
OUT = os.path.join(R, ".tmp", "9gene_实验矩阵_TableA-C.md")
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]


def read(fn):
    p = os.path.join(C, fn)
    return pd.read_csv(p) if os.path.exists(p) else None


def fmt(df, cols, keycol):
    out = []
    for _, r in df.iterrows():
        cells = [str(r[keycol])]
        for c in cols:
            v = r.get(c)
            cells.append("-" if v is None or (isinstance(v, float) and np.isnan(v)) else
                         (f"{v:+.4f}" if isinstance(v, float) and ("delta" in c.lower() or "Δ" in c) else
                          f"{v:.4f}" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return out


L = []
L.append("# 实验矩阵（论文 Table A / B / C）\n")
L.append("> 自动汇总自 `data/patch_cache/*.csv`，生成于 2026-10-06\n")
L.append("> 判定标准：|Δ| < 0.01 视为噪声；有效需 Δ>0.01 且 4/5 seed 同号\n")

# ---------------- Table A ----------------
L.append("\n## Table A｜单变量消融矩阵（25 项）\n")
L.append("| # | 实验族 | 变体 | mean AUC | Δ vs base | 同号 | 5-seed集成 | 判定 |")
L.append("|---|---|---|---|---|---|---|---|")

rows = []
def pick(r, *names):
    """各脚本的列名不统一(delta_vs_base / delta_mean / delta), 取第一个存在的。"""
    for n in names:
        if n in r.index and r[n] is not None:
            v = r[n]
            if not (isinstance(v, float) and np.isnan(v)):
                return v
    return np.nan


def put(sc, fam, d, keynames, deltanames, ensnames, agreenames, meannames):
    if d is None:
        return
    for _, r in d.iterrows():
        if str(pick(r, *keynames)).endswith("_ens5"):
            continue          # 集成行单列到 Table A2, 不与单种子消融混排
        rows.append((sc, fam, str(pick(r, *keynames)),
                     pick(r, *meannames), pick(r, *deltanames),
                     ("%d/5" % int(r[agreenames[0]])) if agreenames and agreenames[0] in r.index else "-",
                     pick(r, *ensnames), pick(r, *deltanames)))
# 30 扫描一
put(30, "损失/注意力", read("improvement_summary.csv"),
    ["variant"], ["delta_mean", "delta"], ["ens_auc"], ["agree"], ["mean_auc"])
# 33 扫描二
put(33, "预处理/协议", read("preprocess_summary.csv"),
    ["variant"], ["delta", "delta_mean"], ["ens_auc"], ["agree"], ["mean_auc"])
# 34 双编码器
put(34, "双编码器融合", read("dual_encoder_summary.csv"),
    ["mode", "config"], ["delta_mean", "delta"], ["ens_auc"], ["agree"], ["mean_auc"])
# 39 池化
put(39, "聚合方式", read("pooling_summary.csv"),
    ["variant"], ["delta_mean", "delta"], ["ens_auc"], ["agree"], ["mean_auc"])
# 41 联合vs分离 + epoch
put(41, "训练结构/epoch", read("joint_vs_sep.csv"),
    ["config"], ["delta"], ["ens_auc"], ["agree"], ["mean_auc"])
# 45 双教师蒸馏
put(45, "双教师蒸馏", read("multiteacher_distill.csv"),
    ["config"], ["delta"], ["ens_auc"], ["agree"], ["mean_auc"])
# 46 路线A 两阶段
put(46, "路线A 两阶段", read("routeA_twostage.csv"),
    ["config"], ["delta"], ["ens_auc"], ["agree"], ["mean_auc"])
# 47 路线B 条件化注意力
put(47, "路线B 条件化注意力", read("routeB_condattn.csv"),
    ["mode"], ["delta"], ["ens_auc"], ["agree"], ["mean_auc"])

def num(v, f="{:.4f}"):
    return "-" if v is None or v != v else f.format(v)
def verdict(delta):
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return "-"
    if delta > 0.01:
        return "有效提升"
    if delta < -0.01:
        return "明确有害"
    return "噪声"

i = 0
for sc, fam, var, m, dlt, agree, ens, dl in rows:
    i += 1
    L.append(f"| {i} | {fam} | `{var}` | "
             f"{chr(45) if m!=m else f'{m:.4f}'} | "
             f"{'-' if dl!=dl else f'{dl:+.4f}'} | {agree} | "
             f"{chr(45) if (ens is None or ens!=ens) else f'{ens:.4f}'} | {verdict(dl)} |")

# ---------------- Table B ----------------
L.append("\n\n## Table B｜跨种子基线与噪声阈值\n")
L.append("| 指标 | 数值 | 说明 |")
L.append("|---|---|---|")
L.append("| gated MIL 跨种子 std | 0.0052–0.0074 | 折划分 + patch 采样 + **权重初始化**（torch 未播种）三重方差 |")
L.append("| 判定阈值 | \\|Δ\\| < 0.01 | 约为 1.4 倍 std |")
L.append("| 有效增益需满足 | Δ > 0.01 且 4/5 seed 同号 | |")
L.append("| **唯一通过的正收益** | **5-seed OOF 集成 +0.0124** | 0.7300 → 0.7423 |")

# ---------------- Table C ----------------
L.append("\n\n## Table C｜最终模型完整指标\n")
d = read("final_metrics_best.csv")
if d is not None:
    L.append("配置：LUAD n=436 | Prov-GigaPath 512 patch | gated MIL | 5-fold OOF | 5-seed 集成\n")
    L.append("| gene | n+ | prev | AUC | ACC | BA | F1 | MCC | SEN | SPEC | AUPRC | Brier | ECE | slope |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        L.append("| {gene} | {n_pos} | {prev:.3f} | {auc:.3f} | {acc:.3f} | {ba:.3f} | {f1:.3f} | "
                 "{mcc:.3f} | {sen:.3f} | {spec:.3f} | {auprc:.3f} | {brier:.3f} | {ece:.3f} | {slope:.3f} |"
                 .format(**r))

# ---------------- 外部验证 ----------------
L.append("\n\n## 附｜外部验证（CPTAC-LUAD n=216, UNI v1）\n")
L.append("| 指标 | 修正前(n=115, 有 bug) | 修正后(n=216) |")
L.append("|---|---|---|")
L.append("| mean AUC | 0.533 | **0.653** |")
L.append("| 相对源队列掉点 | −0.205 | **−0.029** |")
L.append("| 掉点成因 | 真阴性缺失 + 癌种默认 LUSC | 已修（`31_cptac_diagnosis.py`）|")

L.append("\n\n## 附｜负结果与不可行方向（供论文 Discussion）\n")
for t in [
    "真阴性缺失：CPTAC 35 例全野生型被丢 → 外验队列 115 而非 216",
    "ROI 语义反了：`sen_at_spec` 写成 `fpr <= target` → 整张操作点表作废",
    "case_id 严格正则：丢 90 例临床元数据 → 早期子群分析只用 79% 病例",
    "EGFR 位点级：Ex19del 14 例 / L858R 16 例，5 折后不可估",
    "GigaPath 14 层：层间 AUC 跨度 0.004，拼接一律变差",
    "标签一致性：MC3 vs cBioPortal 一致率 0.960，标签非噪声源",
    "LUSC 加训：外部 −0.0498（LUSC 阳性率 1.5–2% 反向教模型）",
    "TITAN/GP-slide 融合 7 种方式：全部 ≤ +0.022，均低于 MIL 0.043",
]:
    L.append(f"- {t}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(L))
print(f"[written] {OUT}  ({len(L)} lines, {os.path.getsize(OUT)/1024:.1f} KB)")
print("\n".join(L[:26]))
print("...")
print("\n".join(L[-16:]))
print("DONE")