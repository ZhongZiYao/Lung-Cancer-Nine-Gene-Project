# `pipeline/06_deepgem_inference/` — 用训好的 DeepGEM EGFR 模型做小样本 sanity

> 把 `DeepGEM` 项目里**训好的多基因 model** 拿来验证你的 patches 抽取管线
> + 9-gene label CSV 是否接得通。本期只跑 **EGFR** 这一个基因。

---

## 🎯 目标

1. **管线有效性**：把现有 `patches_direct/` 喂给 DeepGEM（用训练时同样的 patch encoder = CTransPath + 同样的特征聚合 model）能跑通 forward
2. **数据集有效性**：训好的 EGFR model 至少在 case 级别给出与 `9gene_panel_LUAD.csv` 标签一致的预测方向

⚠️ **小样本局限**：你的 `patches_direct/` 当前 5 个 case, EGFR+ 只有 1 个 (`TCGA-05-4382`)。
要在 8 个 EGFR+ 上跑才有 AUC，下面三步脚本都支持多 case batch 跑。

---

## 📦 模块拆分（三步）

```
06_deepgem_inference/
├── README.md                       ← 本文件
├── step1_extract_ctranspath.py     ← patches/*.jpg  →  per-patch 768-dim .pkl
├── step2_build_internal.py         ← per-patch  →  per-case + internal.pickle
└── step3_forward_egfr.py           ← case-level  →  per-case prob CSV
```

---

## 🚀 完整运行

### 0. 准备权重（一次）

```
# 把 ctranspath.pth 放到 ./DeepGEM/checkpoints/pretrain/ctranspath.pth
# 官方源 (Xiyue-Wang/TransPath): https://github.com/Xiyue-Wang/TransPath
# 或 HuggingFace mirror: lukasfolle/ctranspath
```

### 1. 扩 dataset 至 ≥5 EGFR+

```bash
# 已挑出 7 个 EGFR+ (从 50-case 列表里):
#   TCGA-05-4402, TCGA-05-4410, TCGA-05-5423
#   TCGA-38-4627, TCGA-38-4628, TCGA-38-4629, TCGA-38-6178
# 从 IDC (公开) 拉 WSI (无需 dbGaP):
python data_download/download_extra_egfr_pos.py

# 然后切 patches (复用主仓 pipeline;绝对路径请按你机器调):
python pipeline/extract_patches_direct.py \
  --dicom ./data/TCGA-LUAD-WSI/dicom/tcga_luad \
  --out   ./data/TCGA-LUAD-WSI/patches_direct \
  --cases ./project_resources/data_public/picked_50_case_ids.json
```

### 2. 抽特征（每张 patch → 768 维）
```bash
python pipeline/06_deepgem_inference/step1_extract_ctranspath.py \
  --patches    ./data/TCGA-LUAD-WSI/patches_direct \
  --ctranspath ./DeepGEM/checkpoints/pretrain/ctranspath.pth \
  --out        ./deepgem_feat
# 估时: 4 万 patch × RTX 3060 ≈ 30 秒
```

### 3. 合并 → case-level + label
```bash
python pipeline/06_deepgem_inference/step2_build_internal.py \
  --feat     ./deepgem_feat \
  --labels   ./project_resources/data_public/9gene_panel_LUAD.csv \
  --gene     EGFR \
  --out-feat ./deepgem_combined_feat \
  --out-pickle ./deepgem_internal.pickle
```

### 4. 跑训好的 DeepGEM 模型 → per-case prob
```bash
python pipeline/06_deepgem_inference/step3_forward_egfr.py \
  --internal-pickle ./deepgem_internal.pickle \
  --feat-dir        ./deepgem_combined_feat \
  --checkpoint      ./DeepGEM/checkpoints/DeepGEM_TCGA/modelTCGA_ExcisionalBiopsy_EGFR.pickle \
  --gene            EGFR \
  --wsi-type        ExcisionalBiopsy \
  --out             ./egfr_pred.csv
```

输出 `egfr_pred.csv`:

```csv
case_id,true_label,prob_negative,prob_positive
TCGA-05-4382,1,0.045,0.955
TCGA-05-4402,1,0.082,0.918
...
TCGA-05-4244,0,0.881,0.119
```

---

## 📐 数据约定（与 DeepGEM 原版对齐）

| DeepGEM 原版 | 我们的实现 | 一致性 |
|---|---|---|
| `combined_feat/<case_id>.pkl` = `list[dict{val:np[768], feat_name:str}]` | step2 输出 | ✓ |
| `internal.pickle[wsi_type][gene]["test"] = {pid, label, feat_fp}` | step2 输出 | ✓ |
| **feature_len = 500**（deepgem_test 训好的 model 默认） | DeepGEMDataset 截断/补零到 500 | ✓ |
| WSI 类型 `ExcisionalBiopsy`（TCGA 全部切除活检都算这个 bucket） | DeepGEM `modelTCGA_ExcisionalBiopsy_EGFR.pickle` | ✓ |
| CTransPath `swin_tiny_patch4_window7_224`, 224×224 输入, ImageNet 均值方差归一化 | step1 同 | ✓ |

⚠️ **关键限制**：deepgem 的 ckpt 里 `parameter['feature_len']=500`，意味着每个 case 最多取
前 500 个 patch。所以大型 case (18 000+ patch) 会被截断，**不影响 sanity check**
（只看 case-level prob），但写论文对比时要意识到这一点。

---

## 🧬 后续可扩展

把 `step3_forward_egfr.py` 通用化 → `step3_forward_gene.py --gene KRAS`:
- DeepGEM_TCGA 已训好 6 个基因模型（代码中路径改一下）:
  - `modelTCGA_ExcisionalBiopsy_EGFR.pickle`
  - `modelTCGA_ExcisionalBiopsy_KRAS.pickle`（如有的话）
  - `..._ALK_...`, `..._ROS1_...`, `..._TP53_...`, `..._LRP1B_...`
- 我们的 9-gene CSV 里剩 KRAS/ALK/ROS1/TP53/BRAF/PIK3CA/ERBB2/NRAS/RET
  其中 DeepGEM 没训的（BRAF/PIK3CA/...）就只能等自己训

---

## ⚙️ 依赖

```
torch>=2.1  timm>=0.9  Pillow  numpy  pandas  scikit-learn
```

不需要 DeepGEM 原版的 `albumentations / efficientnet_pytorch / rich / lifelines / yacs`，
因为我们完全复用了 DeepGEMDataset + DeepGEM model 架构（inference 子集），绕开它
的 DDP / training / config machinery。

