# `D:\pan-caner\data\` 数据契约 — TCGA-LUAD 9 基因预测项目

> 本目录存两类数据:**mutation 标签** (MC3) 和 **WSI 图像** (IDC)。
> 训练时通过 **`case_id`** 字段 join,两边物理上独立,逻辑上一一对应。

## 目录总览

```
D:\pan-caner\data\
├── README.md                        ← 本文件
├── MC3/                             ← mutation 标签 (421 cases × 11 genes)
│   ├── mc3.v0.2.8.PUBLIC.maf.gz     ← 原始 MC3 pan-cancer MAF (753 MB, 3.6M 行)
│   ├── tcga_luad_cases.json         ← cBioPortal 拉的 TCGA-LUAD 候选病例 (584 cases)
│   └── 9gene_panel_LUAD.csv         ← ⭐ 训练用的 patient-level mutation 标签
│                                       case_id × {EGFR, KRAS, ALK, ROS1, TP53, BRAF, PIK3CA, ERBB2, NRAS, RET, MET}
│                                       + any_9gene_positive + n_genes_mutated
│                                       421 rows × 14 cols, 17 KB
│
└── TCGA-LUAD-WSI/                   ← WSI 图像 (50 cases, 5-8 GB)
    ├── manifest_50.tsv              ← 50 cases × 166 series 的 IDC 元数据
    ├── picked_50_case_ids.json      ← 50 case_id 列表
    ├── dicom/
    │   └── tcga_luad/               ← IDC collection
    │       └── <case_id>/           ← 50 个病例文件夹,如 TCGA-05-4245/
    │           └── <StudyInstanceUID>/     ← 每个 case 1-3 个 study
    │               ├── SM_<series>/        ← ⭐ Slide Microscopy (真切片,训练用)
    │               │   ├── <uuid>.dcm      ← tile 文件(256×256 RGB)
    │               │   └── ...
    │               ├── SEG_<series>/       ← Segmentation (肿瘤区域 mask,辅助)
    │               ├── ANN_<series>/       ← Annotation (病理医生标注,科研对比)
    │               └── CT_<series>/        ← CT scan (3D 体数据,与 WSI 无关,忽略)
    └── *.png                        ← 测试用 smoke 图(可删)
```

## Join Key: case_id

**MC3 ↔ WSI 通过 `case_id` (e.g. `TCGA-05-4245`) 1:1 对应**:

```
MC3/9gene_panel_LUAD.csv                 TCGA-LUAD-WSI/dicom/tcga_luad/
┌──────────────────┬─────┬─────┬─────┐   ┌─────────────────────────────┐
│ case_id          │EGFR │KRAS │...  │   │ TCGA-05-4245/               │
│ TCGA-05-4244     │  1  │  0  │...  │ ←→│   <StudyInstanceUID>/       │
│ TCGA-05-4245     │  0  │  1  │...  │ ←→│     SM_<series>/<tile>.dcm  │
│ ... 421 cases    │     │     │     │   │     SEG_<series>/<mask>.dcm │
└──────────────────┴─────┴─────┴─────┘   │     ANN_<series>/<ann>.dcm  │
                                          └─────────────────────────────┘
```

`case_id` 在 DICOM 文件里存储为 `PatientID` tag `(0010, 0020)`。

## WSI Series 类型详解

| Modality 前缀 | 含义 | 训练用 | 用途 |
|---|---|---|---|
| **SM** | Slide Microscopy — 真切片图像 | ✅ 主用 | 喂 DINOv3 backbone |
| SEG | Segmentation — 肿瘤/正常区域 mask | 辅助 | 选 patch 时只抽肿瘤内 |
| ANN | Annotation — 病理医生手画标注 | ❌ | 写 paper 时做"算法 vs 专家"对比 |
| CT  | Computed Tomography — 3D 影像 | ❌ | 与 WSI 无关,忽略 |

> IDC 下载时没按 modality 过滤,所以 CT/SEG/ANN 一起拉下来了。**处理脚本按前缀名筛 SM**。

## DICOM 元信息(关键 tag)

```
PatientID (0010,0020)                        case_id, e.g. "TCGA-05-4245"
Modality (0008,0060)                        "SM" / "SEG" / "ANN" / "CT"
ContainerIdentifier (0040,0512)             "TCGA-05-4245-01Z-00-DX1" (barcode 完整版)
SOPClassUID (0008,0016)                     1.2.840.10008.5.1.4.1.1.77.1.6 (VL Whole Slide Microscopy)
Manufacturer (0008,0070)                    e.g. "Carl Zeiss"
ManufacturerModelName (0008,1090)           e.g. "Mirax converted by com.pixelmed..."
Rows / Columns (0028,0010 / 0028,0011)      单 tile 尺寸,e.g. 256×256
NumberOfFrames (0028,0008)                  帧数,通常 1 (多帧 = Z-stack focus stacking)
TotalPixelMatrixRows (0048,0001)            全图总高度(像素)
TotalPixelMatrixColumns (0048,0002)         全图总宽度(像素)
RowPositionInTotalImagePixelMatrix (0048,0070)     这块 tile 的左上角 Y 坐标
ColumnPositionInTotalImagePixelMatrix (0048,0071)   这块 tile 的左上角 X 坐标
```

读法: `pydicom.dcmread(file).<TagName>`,如 `d.PatientID` / `d.TotalPixelMatrixRows`。

## DICOM tile 拼接规则(标准化,不会拼错)

**DICOM WSI 把一张大切片切成几十到几百个小 tile(每块 ~256×256 RGB),每个 tile 是独立 .dcm 文件**。
要还原成完整切片,必须按以下规则拼接:

```python
import pydicom
import numpy as np
from PIL import Image

def stitch_series(series_dir):
    """输入: SM_<series>/ 目录路径;输出: 拼接后的 numpy array (H, W, 3) RGB"""
    files = sorted(f for f in os.listdir(series_dir) if f.endswith(".dcm"))
    # 先读第 1 个文件拿到全图尺寸
    d0 = pydicom.dcmread(os.path.join(series_dir, files[0]))
    total_h = int(d0.TotalPixelMatrixRows)
    total_w = int(d0.TotalPixelMatrixColumns)
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)  # 白底
    for f in files:
        d = pydicom.dcmread(os.path.join(series_dir, f))
        a = d.pixel_array
        # 多帧取第 1 帧(z-stack focus stacking)
        if a.ndim == 4:
            a = a[0]
        # 灰度转 RGB(保险)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        row = int(d.RowPositionInTotalImagePixelMatrix)
        col = int(d.ColumnPositionInTotalImagePixelMatrix)
        h, w = a.shape[:2]
        canvas[row:row+h, col:col+w] = a
    return canvas
```

**为何不会拼错**:

1. **`(Row/Column)PositionInTotalImagePixelMatrix` 是 DICOM Part 3 VL WSI IOD 的强制要求 tag**,所有厂商(Philips/Leica/Hamamatsu/3DHistech)必须填,标准统一
2. **坐标系统一**:`(0,0)` 是图像左上角,向右 = col 增大,向下 = row 增大(图像坐标系)
3. **tile 不重叠**:同一像素不会被两块 tile 覆盖
4. **空白正常**:某些位置没采到,canvas 默认白色填充,不会有缺口
5. **可校验**:拼完后 `canvas[TotalPixelMatrixRows-1, TotalPixelMatrixColumns-1]` 应有有效像素

## 每个 case 的 SM series 选择规则

每个 case 通常有 1-3 个 SM series:

| Series 类型 | ImageType tag | 典型 TotalPixelMatrix | 用途 |
|---|---|---|---|
| 最高分辨率诊断切片 | `['DERIVED','PRIMARY','VOLUME','RESAMPLED']` | 4500×5000 级别 (40x 物镜) | ✅ **训练主用** |
| 中等分辨率 | 同上 | 1700×1750 级别 (10x 物镜) | 辅助 / fallback |
| 缩略图 | `['DERIVED','PRIMARY','THUMBNAIL','RESAMPLED']` | 768×365 | ❌ 丢弃 |

筛选规则: `ImageType` 包含 `VOLUME` 而非 `THUMBNAIL`,且 TotalPixelMatrix 面积最大者优先。

## 数据规模与覆盖

- **MC3 mutation label**: 584 候选 → 421 实际有数据 (cBioPortal ∩ MC3)
- **IDC WSI**: 522 patients 有 SM 数据
- **本次下载**: MC3 ∩ IDC 取前 50 cases
- **理论上可扩展**: 扩到 200-300 cases 总共 ~50-100 GB(单 case ~300 MB)

## 训练时 join 代码骨架

```python
import pandas as pd
import pydicom

labels = pd.read_csv("MC3/9gene_panel_LUAD.csv").set_index("case_id")
WSI_ROOT = "TCGA-LUAD-WSI/dicom/tcga_luad"

for case_id, row in labels.iterrows():
    case_dir = os.path.join(WSI_ROOT, case_id)
    if not os.path.isdir(case_dir):
        continue  # 没 WSI
    # 找主 SM series (VOLUME, TotalPixelMatrix 最大)
    sm_series = pick_primary_sm(case_dir)
    # 拼成完整切片 / 直接抽 patch
    canvas = stitch_series(sm_series)
    for patch in extract_patches(canvas, size=256, stride=256):
        # patch: (256, 256, 3) RGB, label: row[gene_names]
        yield patch, row[gene_names].values.astype(np.float32)
```

## 维护说明

- MC3 数据一次性下载即可,Synapse 永久有效
- IDC 数据可能偶尔更新(数据版本 ID: `series_init_idc_version` / `series_revised_idc_version`)
- 重新跑 `manifest_50.tsv` 时,`picked_50_case_ids.json` 会决定覆盖范围
- DICOM 文件命名带 UUID,**绝不依赖文件名**,所有信息从 DICOM tag 读

## 相关 memory

- `~/.claude/projects/D--pan-caner/memory/tcga-luad-wsi-pipeline.md` — 端到端管线总结
- `~/.claude/projects/D--pan-caner/memory/mc3-download-status.md` — MC3 数据详情
- `~/.claude/projects/D--pan-caner/memory/project-overview.md` — 项目背景