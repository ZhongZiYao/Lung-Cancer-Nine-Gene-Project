# Lung Cancer Nine-Gene Project (LCNGP)

> Histopathology-based multi-label mutation prediction for 9 driver genes in Lung Adenocarcinoma (TCGA-LUAD).

GitHub: https://github.com/ZhongZiYao/Lung-Cancer-Nine-Gene-Project

---

## 项目目标

用 H&E 全片病理图（WSI）作为唯一输入，预测 TCGA-LUAD 患者 9 个 driver gene 的突变状态：

`EGFR  KRAS  ALK  ROS1  TP53  BRAF  PIK3CA  ERBB2  NRAS  RET`（最终清单见 `transfer_package/02_data_public/9gene_panel_LUAD.csv`）

---

## 管线架构（从 WSI 到 9-gene logits）

```
┌────────────────────────────────────────────────────────────┐
│ TCGA-LUAD DICOM（pixelmed 转换的 multi-frame JFIF）  │
│ SM_<seriesUID>/*.dcm  ├── multi-frame ~ 200MB/series │
└────────────────────────────────────────────────────────────┘
                  │
                  ▼  extract_patches_direct.py
        ┌────────────────────────────────────┐
        │ patches/<case>/<slice>/<x>_<y>.jpg │
        │ 256×256, multi-slice 全切              │
        │ Otsu bg filter + tissue filter     │
        └────────────────────────────────────┘
                  │
                  ▼  step2_extract_ctranspath.py
        ┌────────────────────────────────────┐
        │ features/<case>.npz                │
        │ 768-dim CTransPath CLS per patch   │
        │ + slice_ids / slice_uids           │
        └────────────────────────────────────┘
                  │
                  ▼  step3_make_deepgem_pkl.py
        ┌────────────────────────────────────┐
        │ deepgem_feat/<case>/<s<i>_SM>/   │
        │   p<i>_f768.pkl   (1 patch 1 .pkl)│
        └────────────────────────────────────┘
                  │
                  ▼  step5 (DeepGEM forward)  OR  ABMIL/TransMIL (TBD)
        ┌────────────────────────────────────┐
        │ case-level logits [9]               │
        │ BCE 多标签，二分类/9 个基因 独立  │
        └────────────────────────────────────┘
```

**核心组件**：

| 模块 | 用途 | 路径 |
|---|---|---|
| Patch 切割 | multi-frame DICOM → 256×256 JPG | `pipeline/extract_patches_direct.py` |
| CTransPath 抽特征 | ResNet-50 + ViT-B, 768-dim CLS | `pipeline/deepgem_test/step2_extract_ctranspath.py` |
| DeepGEM 一致性测试 | slice-level 训练 / 前向 | `pipeline/deepgem_test/step5*.py` |

---

## 仓库结构

```
Lung-Cancer-Nine-Gene-Project/
├── README.md                          ← 你在这里
├── LICENSE                            ← MIT
├── requirements.txt                   ← pip install -r requirements.txt
├── .gitignore
│
├── pipeline/                          ← **本项目核心代码**
│   ├── extract_patches_direct.py      ← 切 patch（DICOM → 256 JPG）
│   ├── step1_extract_features_uni.py  ← UNI 抽 feature（备选路线）
│   ├── step2_to_deepgem_pkl.py        ← train_plan → deepgem .pkl
│   ├── step3_build_deepgem_inputs.py ← 构造 internal.pickle
│   ├── step4_run_deepgem_sanity.py    ← 跑训好的 DeepGEM 模型
│   └── deepgem_test/                  ← 跟 DeepGEM 项目兼容的测试管线
│       ├── step1_repatch_3cases.py
│       ├── step2_extract_ctranspath.py
│       ├── step3_make_deepgem_pkl.py
│       ├── step4_build_internal_pkl.py
│       ├── step5_predict_egfr.py      ← 单 case forward
│       ├── step5_slice_ensemble.py    ← multi-slice ensemble
│       └── README.md
│
├── transfer_package/                  ← **项目"上下游"原始资料**
│   ├── 01_code/
│   │   ├── scripts/                    ← PowerShell 跑批脚本
│   │   ├── tools/                      ← 9-gene label 抽取, profile 等
│   │   └── memory/                     ← 项目状态备忘 (不走 repo)
│   ├── 02_data_public/                 ← **推送：CSV / TSV（labels 和 case id）**
│   │   ├── 9gene_panel_LUAD.csv        ← 9 gene labels (per-case)
│   │   ├── manifest_50.tsv
│   │   ├── picked_50_case_ids.json
│   │   └── tcga_luad_cases.json
│   ├── 04_papers/                      ← 参考论文代码（少量）
│   ├── 05_docs/                        ← 项目说明 docx/md
│   ├── 06_assets/                      ← 项目图片资源
│   ├── scripts/
│   └── data/                           ← **不推**：86 GB 原始 DICOM
│
├── models/                            ← 第三方 git submodule 引用
│   ├── UNI/                           ← mahmoodlab/UNI (submodule)
│   ├── DeepGEM/                       ← tencent/DeepGEM (submodule)
│   ├── CONCH/                         ← mahmoodlab/CONCH (submodule)
│   ├── TITAN/                         ← mahmoodlab/TITAN (submodule)
│   ├── TransPath/                     ← Xiyue-Wang/TransPath (submodule)
│   ├── Prediction-of-Mutated-Genes/    ← THUML/...  (submodule)
│   └── README.md                       ← 各自 README 入口
│
└── docs/                              ← 项目进展记录
    ├── status/                        ← 阶段性小结
    └── ...
```

---

## 快速开始

### 1. 克隆（含 submodule）

```bash
git clone --recurse-submodules https://github.com/ZhongZiYao/Lung-Cancer-Nine-Gene-Project.git
cd Lung-Cancer-Nine-Gene-Project
```

### 2. 装依赖

```bash
pip install -r requirements.txt
```

**环境要求**：
- Python 3.10+
- CUDA 11.0+ (GPU 推荐 8GB+ 显存)
- pydicom ≥ 3.0（读 JPEG2000 的 DICOM）
- timm ≥ 0.9, torch ≥ 2.1

### 3. 下载第三方权重（**不**进 repo）

```bash
# UNI (~1.2 GB)
wget -O models/UNI/assets/ckpts/uni/pytorch_model.bin \
  https://huggingface.co/MahmoodLab/UNI/resolve/main/pytorch_model.bin

# CTransPath (~800 MB, already in DeepGEM repo)
# 你可以用 DeepGEM 的 scripts 拉
```

### 4. 准备 data（**不**进 repo）

```bash
# 1. TCGA-LUAD WSI 从 IDC 下载 → 50 case, ~86 GB
bash transfer_package/data_download/*.sh
# 2. MC3 v0.2.8 PUBLIC MAF (~700 MB)
# 3. 9-gene panel labels CSV (已经在这个 repo 里: transfer_package/02_data_public/9gene_panel_LUAD.csv)
```

### 5. 跑 pipeline

```bash
# 步骤 1: 切 patch（256×256, multi-slice）
python pipeline/extract_patches_direct.py \
  --dicom ./transfer_package/data/TCGA-LUAD-WSI/dicom/tcga_luad \
  --out   ./patches \
  --cases ./transfer_package/02_data_public/picked_50_case_ids.json

# 步骤 2: 抽 CTransPath features
python pipeline/deepgem_test/step2_extract_ctranspath.py \
  --patches ./patches \
  --ctranspath ./models/DeepGEM/checkpoints/pretrain/ctranspath.pth

# 步骤 3-5: 喂 DeepGEM 训好的模型 / 或自己训 ABMIL
```

---

## 数据/模型 流程总览

| 来源 | 数据/模型 | 用途 | 推送？ |
|---|---|---|---|
| **TCGA-LUAD** | DICOM WSI | patch 切割源 | ❌ DICOM 太 |
| **MC3 v0.2.8 PUBLIC** | `.maf.gz` | 9-gene label | ❌ maf.gz 70M |
| **9gene_panel_LUAD.csv** | 自己 derived | case-level 9-gene 标签 | ✅ |
| **CTransPath** | TransPath 仓库 weights | patch encoder | submodule |
| **DeepGEM** | DeepGEM 仓库 + 训好的 9 个 gene model | baseline / 模型架构 | submodule |
| **UNI** | UNI 仓库 weights | patch encoder (备选) | submodule |

---

## 进度

- ✅ 0. WSI → patch 切割 (multi-frame DICOM, multi-slice, bg filter)
- ✅ 1. CTransPath 全权重加载 (`missing=0, unexpected=0`)
- ✅ 2. per-patch & per-slice 输出对齐 DeepGEM 标准格式
- ✅ 3. 训好的 DeepGEM EGFR 模型 forward sanity check（3 case × 3 slice）
- ✅ 4. slice-level ensemble（multi-slice aggregate）验证管线
- 🔄 5. **ABMIL/TransMIL 9-gene multi-label baseline**（doing... task #21）
- 📋 6. server 上 50 case 全量跑 + 5-fold CV
- 📋 7. 与 DeepGEM 训好的 EGFR/KRAS/etc. 多基因 ensemble 对比

---

## Reference

- TCGA-LUAD 数据下载：IDC Cancer Genome Atlas DICOM collection
- 9-gene panel: 中山大学 GAMIL paper, "Prediction of Mutated Genes in Lung Adenocarcinoma Based on Weak Supervision", THUML GitHub
- 病理 patch encoder: CTransPath (Xiyue-Wang/TransPath), UNI (MahmoodLab/UNI), CONCH, TITAN
- 聚合模型: DeepGEM, CLAM ABMIL, TransMIL

---

## License

MIT

---

## Status

Last update: 2025-07-17
Maintainer: ZhongZiYao
