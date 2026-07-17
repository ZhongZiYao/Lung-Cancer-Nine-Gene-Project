---
name: environment-and-workflow
description: "工作环境关键事实 — pwsh 当前终端 vs WSL Ubuntu venv-pancaner,影响脚本怎么跑"
metadata: 
  node_type: memory
  type: project
  originSessionId: c5983032-f1b6-4a5e-bbd7-29bfd26b2c3e
---

**当前终端是 Windows pwsh(不是 WSL Ubuntu)。** 这是关键,影响所有推荐命令。

**两套环境分工**:

| 任务 | 在哪跑 | 路径视角 |
|---|---|---|
| 编辑 Python 脚本、写文档 | pwsh(当前终端) | `D:/pan-caner/tools/...` |
| 跑需要 `synapseclient` / `pandas` 的 Python 脚本 | WSL Ubuntu | `/home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/xxx.py` |
| 看下载数据结果 | pwsh 或 WSL 都行(共享 D 盘) | 两边看到同一份文件 |

**WSL Ubuntu venv-pancaner**(上次跑通过的环境):
- Python: `/home/admin123/.venv-pancaner/bin/python`
- 包: `synapseclient`, `requests`, `pandas` 等(已装,但未在本会话验证)
- 数据落盘: `/mnt/d/pan-caner/data/` = Windows `D:\pan-caner\data\`

**从 pwsh 切 WSL 跑命令的标准模板**:
```bash
wsl -e bash -c "/home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/download_mc3_maf.py"
```

**反馈**: 用户明确说"当前终端在 pwsh,不在 WSL Ubuntu 系统",之前生成的命令如果没强调这一点,容易误用 Windows Python / 路径。**怎么应用**: 涉及 `venv-pancaner` / `/mnt/d/` / `/home/admin123/` 路径的命令,先提醒"切到 WSL 跑",或者直接用 `wsl -e bash -c '...'` 包装。

**相关 memory**: [[mc3-download-status]] [[task-state]]