---
name: task-state
description: 当前 MC3 最小方案的任务清单状态 — 接续上下文用
metadata: 
  node_type: memory
  type: project
  originSessionId: c5983032-f1b6-4a5e-bbd7-29bfd26b2c3e
---

**当前任务主线**: MC3 拉取 → 抽 9 基因标签 → 跟 WSI barcodes 交叉(最小方案,只到 9gene_panel_LUAD.csv 即可)

**任务清单状态**(2026-07-12 截至):

- [x] #1 确认 MC3 数据集最小方案的可行直链
  - 结论: 无 HTTP 直链,走 Synapse + synapseclient + PAT
- [x] #2a 修 `download_mc3_maf.py` 的 HTTPS fallback bug(2026-07-12 完成)
- [x] #2b 下载 MC3 MAF 验证管道(2026-07-12 完成,753 MB / 3.6M 行)
- [x] #3 抽 TCGA-LUAD 9 基因标签(2026-07-12 完成)
  - **关键修复**: `is_luad()` 旧版只认 site `05` 漏 19 个 LUAD site,从 30 cases 跳到 421 cases
  - cBioPortal `/api/studies/luad_tcga/samples` 拉 584 patients 白名单,实际匹配 421
  - 11 基因 prevalence 跑出来:EGFR 15.9% / KRAS 35.9% / ALK 6.7% / TP53 61.5% / BRAF 9.7% 等(均在文献合理范围,MC3 多 caller union 略偏高)
  - 产物: `D:/pan-caner/data/MC3/9gene_panel_LUAD.csv` (16.9 KB,422 行)
- [ ] #4 跟 TCGA-LUAD WSI barcodes 交叉
  - 待用户定方向:A 拉 WSI 交叉 / B 进 GDC 下 WSI patch / C 加临床元数据 / D 打住先汇报

**下一步该做什么**(接力 Claude 第一件事):
1. (已完成 ✅)WSL sanity check: venv 在、`synapseclient` 装了 (v4.13.0)
2. (已完成 ✅)把 `download_mc3_maf.py` 里 `download_via_https()` 删了,只留 synapseclient 路径
3. **当前**:提醒用户在 https://www.synapse.org/SignUp 注册 + 拿 PAT
4. 用户把 PAT 写到 `/mnt/d/pan-caner/.synapse_token` 后,跑 `wsl -e bash -lc "/home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/download_mc3_maf.py"`(注意 `-lc` 登录 shell 避免 Windows PATH 注入劫持)

**最小方案"完成"的定义**(用户原话: "先做最小方案试试"):
- 拉到 MC3 → 抽出 `9gene_panel_LUAD.csv`(几百 KB)
- 控制台打出 11 基因阳性病例数 + 百分比
- 不下 WSI、不训模型

**相关 memory**: [[mc3-download-status]] [[environment-and-workflow]]