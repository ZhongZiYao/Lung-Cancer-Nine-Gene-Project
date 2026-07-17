---
name: mc3-download-status
description: "MC3 MAF 拉取最小方案的当前结论 — 直链走不通,走 synapseclient + PAT,脚本 bug 待修"
metadata: 
  node_type: memory
  type: project
  originSessionId: c5983032-f1b6-4a5e-bbd7-29bfd26b2c3e
---

**目标**: 拉 MC3 (Multi-Center Mutation Calling in Multiple Cancers) 公开 MAF,作为 TCGA-LUAD 9 基因突变的 patient-level 标签来源。

**已写好的工具脚本**:
- `D:/pan-caner/tools/download_mc3_maf.py` — 拉 MC3
- `D:/pan-caner/tools/extract_9gene_panel.py` — 从 MC3 抽 11 基因 panel (TCGA-LUAD filter,site code 05)
- 输出约定: `D:/pan-caner/data/MC3/9gene_panel_LUAD.csv`

**结论(2026-07-12 调研)**:
1. **MC3 没有 HTTP 真直链**。WebFetch 抓 `https://www.synapse.org/Synapse:syn7824274` 只拿到 HTML 页面,不是 .maf.gz
2. **GDC Legacy Archive 不可靠**: 速查卡里 UUID `1c8cfe5f-...` 试了两个变体都 404;且 PEAR-1166 票据显示 Legacy 正在被移除
3. **唯一稳的路**: Synapse 账号 + PAT(personal access token) + `synapseclient` Python 库
4. **MC3 PUBLIC 版本是公开的**,不需要 dbGaP 授权,只需要 Synapse 账号

**脚本 bug 待修**(在 `download_mc3_maf.py`):
- `download_via_https()` 抓 `https://www.synapse.org/Synapse:syn7824274` 会拿到 HTML,**应当删除或降级**
- `--no-synapse --try-public` 这条 fallback **实际无效**

**PAT token 存放约定**:
- 路径: `/mnt/d/pan-caner/.synapse_token` 或 `~/.synapse_token`
- 权限: 0600
- 内容: 一行 PAT 字符串

**MC3 MAF 字段**(给 `extract_9gene_panel.py` 用):
- 列名: `Hugo_Symbol`, `Variant_Classification`, `FILTER`, `Tumor_Sample_Barcode`
- LUAD 判定: barcode 第 2 段 = `05`(legacy,只覆盖 ~30 cases,不推荐)
- **正确做法**: 用 cBioPortal `/api/studies/luad_tcga/samples?pageSize=1000` 拉白名单(584 patients),`extract_9gene_panel.py --luad-whitelist <json>` 喂入,可覆盖 421 cases
- PASS 判定: `FILTER` 列 ∈ {`PASS`, `.`, ``}(MC3 没 FILTER 列,脚本默认全 PASS)
- 阳性变异类(写死在 `POSITIVE_CLASSES`): Missense/Nonsense/Frame_Shift_Del/Frame_Shift_Ins/In_Frame_Del/In_Frame_Ins/Splice_Site/Translation_Start_Site/Nonstop_Mutation

**GDC 现状(2026-07)**:
- 新 GDC API `cases?project_id=TCGA-LUAD` 现在返 count=0(PEAR-1166 legacy archive 关闭续效应)
- cBioPortal `/api/studies/luad_tcga/samples` 是当前最稳的 TCGA case_id 白名单源

**pwsh 跑 wsl 命令的 git-bash 坑**:
- 直接 `wsl -e bash -lc '/home/admin123/...'` 会被 git-bash 把 `/home/admin123` 翻译成 `C:/Program Files/Git/home/admin123/...`
- 解决: 命令前面加 `MSYS_NO_PATHCONV=1`
- 或用 `wsl -e bash -lc '"$(printf %s /home/admin123/.venv-pancaner/bin/python)" ...'`(printf 输出绝对路径绕过翻译)

**MC3 PUBLIC MAF 真实大小**(实测): 753 MB(解压后),3.6M 行,LUAD 部分 244k 行 / 421 unique cases(用 cBioPortal 白名单过滤)

**相关 memory**: [[environment-and-workflow]] [[task-state]] [[project-overview]]