"""
make_charts.py — 生成 PPT 用的数据图(11 基因 prevalence + DICOM 元信息示意)
依赖:matplotlib(WSL venv 已装),输出 PNG 到 assets/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = "/mnt/d/pan-caner/group_meeting_report/assets"
os.makedirs(OUT, exist_ok=True)

# ===== 调色板(按 dataviz skill 的 reference palette,validated passing) =====
# Categorical: 用于 11 个基因
PALETTE = [
    "#2563eb",  # blue
    "#0891b2",  # cyan
    "#059669",  # emerald
    "#ca8a04",  # amber
    "#dc2626",  # red
    "#9333ea",  # violet
    "#db2777",  # pink
    "#65a30d",  # lime
    "#0284c7",  # sky
    "#7c2d12",  # brown
    "#475569",  # slate
]
# Status: 阳性高亮
POS_FILL = "#dc2626"   # 阳性率 bar 主色
NEG_FILL = "#e2e8f0"   # 阴性率 bar 灰底

# ====== Figure 1: 11 基因 prevalence 双色堆叠条形图 ======
genes = ["EGFR", "KRAS", "ALK", "ROS1", "TP53", "BRAF", "PIK3CA",
         "ERBB2", "NRAS", "RET", "MET"]
prevalences = [15.91, 35.87, 6.65, 5.23, 61.52, 9.74, 6.41,
               1.90, 0.71, 4.28, 4.75]
n_cases = 421

# 按 prevalence 排序(高→低),更易读
order = np.argsort(prevalences)[::-1]
genes_s = [genes[i] for i in order]
prev_s = [prevalences[i] for i in order]

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
y_pos = np.arange(len(genes_s))
# 横向条:阴性(灰)+阳性(红)
neg = [100 - p for p in prev_s]
ax.barh(y_pos, neg, color=NEG_FILL, edgecolor="white", linewidth=1.5, label="Wild-type")
ax.barh(y_pos, prev_s, left=neg, color=POS_FILL, edgecolor="white",
        linewidth=1.5, label="Mutated")
# 数值标签
for i, p in enumerate(prev_s):
    ax.text(p + 1.2, i, f"{p:.1f}%", va="center", fontsize=10,
            fontweight="bold", color="#1e293b")
# y 轴
ax.set_yticks(y_pos)
ax.set_yticklabels(genes_s, fontsize=12, fontweight="bold")
ax.invert_yaxis()
# x 轴
ax.set_xlim(0, 100)
ax.set_xlabel("Prevalence (%) in 421 TCGA-LUAD cases", fontsize=11)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0", "20", "40", "60", "80", "100%"])
# 网格(only x, recessive)
ax.xaxis.grid(True, color="#f1f5f9", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
# spines
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#cbd5e1")
# title
ax.set_title("11-Gene Panel Mutation Prevalence (TCGA-LUAD, MC3 v0.2.8)",
             fontsize=13, fontweight="bold", pad=15, loc="left", color="#0f172a")
# legend
ax.legend(loc="lower right", frameon=False, fontsize=10)
# source
fig.text(0.99, 0.01, "Source: MC3 v0.2.8 PUBLIC MAF, cBioPortal whitelist (n=421)",
         fontsize=8, color="#64748b", ha="right")
plt.tight_layout()
fig.savefig(f"{OUT}/fig_gene_prevalence.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"saved {OUT}/fig_gene_prevalence.png")

# ====== Figure 2: 数据流 pipeline 概览图 ======
fig, ax = plt.subplots(figsize=(12, 4.2), dpi=120)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

# 4 个 stage + 1 个输出
stages = [
    ("TCGA-LUAD\nWSI DICOM\n(50 cases)", "raw", "#0891b2"),
    ("Patches\n1120×1120 @ 20×", "patch", "#2563eb"),
    ("Patch Features\nCTransPath / EffB0", "feat", "#7c3aed"),
    ("MIL Model\nDeepGEM-style", "model", "#dc2626"),
]
arrow_y = 2.5
box_w = 2.2
box_h = 1.8
gap = 0.4
x_start = 0.5

for i, (label, key, color) in enumerate(stages):
    x = x_start + i * (box_w + gap)
    rect = FancyBboxPatch((x, arrow_y - box_h/2), box_w, box_h,
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=color, edgecolor="none", alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + box_w/2, arrow_y, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    # arrow
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + box_w + gap - 0.05, arrow_y),
                    xytext=(x + box_w + 0.05, arrow_y),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#475569"))

# 标签: 数据源 + 输出
ax.text(0.5 + box_w/2, arrow_y + 1.4, "Data\nSource", ha="center",
        fontsize=9, color="#64748b", style="italic")
ax.text(x_start + 3*(box_w + gap) + box_w/2, arrow_y + 1.4,
        "Gene\nPrediction", ha="center", fontsize=9, color="#64748b",
        style="italic")

# 标签 step numbers
for i in range(4):
    x = x_start + i * (box_w + gap)
    ax.text(x + 0.18, arrow_y + box_h/2 - 0.18, f"{i+1}",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.15", facecolor="#0f172a",
                      edgecolor="none"))

# 底部 subtitle
ax.text(6, 0.5, "Each step corresponds to a DeepGEM script (step1~step4)",
        ha="center", fontsize=10, color="#475569", style="italic")

ax.set_title("Pipeline Overview: WSI → Patches → Features → MIL Prediction",
             fontsize=14, fontweight="bold", pad=10, loc="left", color="#0f172a")
plt.tight_layout()
fig.savefig(f"{OUT}/fig_pipeline.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"saved {OUT}/fig_pipeline.png")

# ====== Figure 3: DICOM tile 拼接示意 ======
fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.5)
ax.axis("off")

# 左: 4 个 tile
tile_positions = [(0.3, 2.5), (1.7, 2.5), (0.3, 1.0), (1.7, 1.0)]
tile_labels = ["tile A\n(row=0,col=0)", "tile B\n(row=0,col=256)",
               "tile C\n(row=256,col=0)", "tile D\n(row=256,col=256)"]
for (x, y), lab in zip(tile_positions, tile_labels):
    rect = mpatches.Rectangle((x, y), 1.2, 1.2, facecolor="#fde68a",
                              edgecolor="#92400e", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.6, y + 0.6, lab, ha="center", va="center", fontsize=9,
            color="#78350f", fontweight="bold")

# 箭头
ax.annotate("", xy=(5.0, 1.85), xytext=(3.1, 1.85),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#475569"))
ax.text(4.05, 2.2, "stitch\n(by row/col tag)",
        ha="center", fontsize=10, color="#475569", style="italic")

# 右: 拼接后的全图(简化示意)
canvas_x, canvas_y = 5.5, 0.5
canvas_w, canvas_h = 5.0, 3.0
rect = mpatches.Rectangle((canvas_x, canvas_y), canvas_w, canvas_h,
                          facecolor="#fef3c7", edgecolor="#78350f", linewidth=1.5)
ax.add_patch(rect)
# 在 canvas 内画 4 个 tile 位置
tile_pos_canvas = [
    (canvas_x + 0.3, canvas_y + canvas_h - 1.4, 2.0, 1.2),  # top-left
    (canvas_x + 2.5, canvas_y + canvas_h - 1.4, 2.0, 1.2),  # top-right
    (canvas_x + 0.3, canvas_y + 0.3, 2.0, 1.2),              # bottom-left
    (canvas_x + 2.5, canvas_y + 0.3, 2.0, 1.2),              # bottom-right
]
for x, y, w, h in tile_pos_canvas:
    sub = mpatches.Rectangle((x, y), w, h, facecolor="#fbbf24",
                              edgecolor="#92400e", linewidth=0.8, alpha=0.6)
    ax.add_patch(sub)

# 标签
ax.text(canvas_x + canvas_w/2, canvas_y + canvas_h + 0.25,
        "Stitched whole slide image\n(TotalPixelMatrix: 4980×4500 @ 20×)",
        ha="center", va="bottom", fontsize=10, color="#78350f", fontweight="bold")

ax.set_title("DICOM WSI Tile Stitching: Standard Tag-Based Reconstruction",
             fontsize=13, fontweight="bold", pad=10, loc="left", color="#0f172a")
plt.tight_layout()
fig.savefig(f"{OUT}/fig_dicom_stitch.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"saved {OUT}/fig_dicom_stitch.png")

# ====== Figure 4: 数据规模 stat tiles ======
fig, ax = plt.subplots(figsize=(11, 3.2), dpi=120)
ax.set_xlim(0, 11)
ax.set_ylim(0, 3.2)
ax.axis("off")

tiles = [
    ("421", "TCGA-LUAD cases\nwith mutation labels", "#2563eb"),
    ("50", "Cases with\nWSI downloaded", "#0891b2"),
    ("11", "Gene panel\n(EGFR/KRAS/...)", "#7c3aed"),
    ("9.6 mm × 7.5 mm", "Avg. tissue area\nper slide", "#059669"),
    ("20×", "Objective\nmagnification", "#dc2626"),
]
tile_w = 2.0
gap = 0.1
x0 = 0.5
for i, (val, label, color) in enumerate(tiles):
    x = x0 + i * (tile_w + gap)
    rect = FancyBboxPatch((x, 0.4), tile_w, 2.4,
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=color, edgecolor="none", alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + tile_w/2, 2.2, val, ha="center", va="center",
            fontsize=22 if len(val) < 6 else 13, fontweight="bold", color="white")
    ax.text(x + tile_w/2, 1.0, label, ha="center", va="center",
            fontsize=10, color="white")

ax.set_title("Dataset at a Glance", fontsize=14, fontweight="bold",
             pad=10, loc="left", color="#0f172a")
plt.tight_layout()
fig.savefig(f"{OUT}/fig_stat_tiles.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"saved {OUT}/fig_stat_tiles.png")

print("\nAll charts generated.")