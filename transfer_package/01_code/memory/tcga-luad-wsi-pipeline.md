---
name: tcga-luad-wsi-pipeline
description: TCGA-LUAD WSI 切片下载与 9 基因 panel 对齐的端到端管线(python 工具 + 数据状态)
metadata: 
  node_type: memory
  type: project
  originSessionId: 8d114393-e01e-442f-9a95-2e5ca6f12bad
---

**目标**: TCGA-LUAD 421 cases × 11 基因 mutation flags + DICOM WSI patches → DINOv3 backbone → 9 基因预测模型

**已完成**(2026-07-12):

1. **MC3 mutation labels** — `D:/pan-caner/data/MC3/9gene_panel_LUAD.csv` (17 KB, 421 cases × 11 genes 0/1 flags + any_9gene_positive + n_genes_mutated)
   - 脚本 `D:/pan-caner/tools/extract_9gene_panel.py`,已用 cBioPortal `tcga_luad` 白名单过滤
2. **TCGA-LUAD WSI DICOM** — `D:/pan-caner/data/TCGA-LUAD-WSI/dicom/tcga_luad/`,50 cases × ~67 study folders × 4912 DICOM 文件
   - 每个 case 包含 SM (Slide Microscopy 真切片) + SEG (segmentation) + ANN (annotation) + 部分 CT
   - 用 `idc-index` 0.12.4 拉,无 PAT,公开
   - 50 cases manifest 在 `D:/pan-caner/data/TCGA-LUAD-WSI/manifest_50.tsv`

**未做**:

3. **从 DICOM 提取 patch-level features** — 需要装 `pydicom` + `openslide` 或 `wsidicom`,当前 WSL 网络掉线(PyPI 连不上,只有 IDC S3 还能通),需恢复网络后装
4. **DINOv3 backbone fine-tune** — 需要 GPU(合同 95% 准确率指标)
5. **外部验证** — 钟团队 CJFH 异源数据(合同要求)

**关键工程事实**(接力 Claude 必读):

- **GDC 新 API 不可用**: `cases?project_id=TCGA-LUAD` 现在返 0(PEAR-1166 legacy archive 关闭续效应)。WSI 下载必须走 **IDC (Imaging Data Commons)**, `pip install idc-index` 即可
- **IDC collection 名是下划线不是连字符**: `tcga_luad` 不是 `tcga-luad`
- **IDC `download_dicom_patients()` 不接 collection_id 参数**,要过滤 collection 必须用 `download_from_selection(collection_id=..., patientId=...)`
- **IDC 默认下全部 modality**(SM+SEG+ANN+CT),不会按 modality 过滤。50 cases × 1.2 GB ≈ 30 GB,实际下到 5-8 GB(部分 series 还在路上)
- **IDC SM DICOM 是 VL Whole Slide Microscopy 格式**,一个 slide = 多个 .dcm tile 文件,每个 tile 是图像的一小块(典型 859 KB - 几 MB),不是单文件包含整图。需要 `openslide-python` 或 `wsidicom` 拼回去
- **TCGA barcode 第 2 段是 site code,不是 disease**: `TCGA-05-XXXX` 是 site 05(Cedars-Sinai,不专是 LUAD)。MC3 LUAD 散落在 22 个 site 上,要用 cBioPortal `/api/studies/luad_tcga/samples?pageSize=1000` 拉 584 病例白名单(case_id 形式) → 喂 `extract_9gene_panel.py --luad-whitelist`

**WSL 网络坑**(接力必知):

- pwsh 端 `wsl -e bash -c '/home/...'` 会被 git-bash 把 `/home/...` 翻译成 `C:/Program Files/Git/home/...` — 命令前面加 `MSYS_NO_PATHCONV=1` 解决
- WSL 偶尔会断外网(PyPI 不可达),但 IDC S3 (s3://idc-open-data/) 通常还能通 — 因为 s5cmd 通过 AWS 公开 bucket,不需要走 WSL HTTP 栈
- 装包遇到 PyPI 超时换清华/阿里镜像:`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>`

**当前 task list 主线状态**(接力读 task-list):

- ✅ #1-#3, #4a, #5, #6, #7, #8, #10, #11, #12, #13(全部 completed)
- 阻塞项: PyPI 网络断链,无法装 pydicom/openslide/wsidicom 做 DICOM → patch 提取

**相关 memory**: [[mc3-download-status]] [[project-overview]] [[task-state]]