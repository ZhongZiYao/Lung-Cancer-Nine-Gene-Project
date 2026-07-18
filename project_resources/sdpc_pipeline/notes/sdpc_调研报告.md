# .sdpc 格式调研报告

> 项目:肺癌九基因预测模型(中日友好医院冰冻切片)
> 整理日期:2026-07-09
> 工作区: `D:\pan-caner\`

---

## 0. 一句话结论

**`.sdpc` 是深圳生强(SQS / TEKSQRAY)数字病理扫描仪的私有 WSI 格式**,当前
数据来自 `SQS600P-222300AG` 这台 20x/40x 双倍率扫描仪。

- ✅ **头部元信息可以纯 Python 解析**(不依赖任何 SDK/dll),已实现 `tools/sdpc_inspect.py`
- ⚠️ **像素/缩略图读取需要生强官方解码库**,Windows 需装 `DecodeSdpcDll.dll`(无公开下载),Linux 需 `libDecodeSdpc.so`(已随 opensdpc pip 包附带)
- 📊 **数据规模**:500 个 .sdpc 文件,共 91.8 GB,最大 977 MB,最小 2.2 MB
- ❌ **xlsx 标签覆盖率仅 2.1%**:1339 个标签切片号中只有 28 个能找到对应 .sdpc;500 个 .sdpc 中 472 个无标签

---

## 1. 格式归属性证据

直接读取 .sdpc 头部 4KB,得到:

| 字段 | 值 | 含义 |
|---|---|---|
| 起始 16 字节 | `SQ1.0.3.1010\x00\x00\x00\x00` | 文件 magic + 版本 |
| SqExtraInfo.model | `FV-025G-X4C` | 物镜型号(Olympus 20x 平场复消色差) |
| SqExtraInfo.serial | `SQS600P-222300AG` | 扫描仪序列号,确认是**深圳生强 SQS600P** |
| SqExtraInfo.barCode | `895983003` | 切片条码,与文件名 stem 一致 |

**生强**是 2024 年前三季度国内数字病理扫描仪金额市占率第二的厂商(约 24%,仅次于江丰 43%),SQS-40P/SQS-120P/SQS-12P/SQS-600P 是其主要型号。TEKSQRAY(sqray.com)是其软件品牌。

---

## 2. 文件大小与扫描参数分布(全部 500 个样本)

| 指标 | 值 |
|---|---|
| 文件数 | 500 |
| 总大小 | 91.8 GB(85.5 GiB) |
| 最小 / 中位 / 最大 | 2.2 MB / 155 MB / 977 MB |
| 唯一扫描仪序列号 | 1 个(全是 SQS600P-222300AG) |
| 唯一物镜 | FV-025G-X4C(20x 平场) |
| 软件版本 | 1.0.3.1010(全 500 个) |
| 倍率 | 40(全 500 个;`rate` 字段,实际像素 0.4132 或 0.2068 μm/px) |
| 金字塔层级 | 2~8 层(与文件大小正相关) |
| 缩略图尺寸 | 300×300(几乎所有) |
| 瓦片大小 | 288×288(几乎所有) |
| 患者信息 | **全部空**(已脱敏) |

**ruler 字段**(每像素物理尺寸)出现两种值:

- `0.4132 μm/px` — 对应 20x 物镜,src 分辨率约 6.9k×6.9k ~ 38k×38k
- `0.2068 μm/px` — 对应 40x(数字放大),src 分辨率约 45k×73k ~ 126k×107k

**这是同一台扫描仪通过不同扫描程序产生的两种倍率样本**,不是两台机器。

详细数据见 `docs/sdpc_metadata_all.csv`(29 列 × 500 行)。

---

## 3. 可用的工具与读取方案

### 3.1 已验证可行(纯 Python,无外部依赖)

| 工具 | 状态 | 能力 |
|---|---|---|
| **`tools/sdpc_inspect.py`**(本项目自写) | ✅ 跑通 | 读 SqPicHead / SqPersonInfo / SqExtraInfo 全部元信息 |
| openslide(>=4.0)+ openslide-bin | ✅ 装好 | 可处理 .svs/.ndpi/.scn/.mrxs 等 23 种标准 WSI |
| openslide-python | ✅ 装好 | — |

### 3.2 已装但 Windows 上不可直接用

| 包 | 状态 | 问题 |
|---|---|---|
| `opensdpc` (pip 装的 0.x) | ⚠️ 半成品 | Windows 需要 `DecodeSdpcDll.dll`(pip 包没带);Linux 有 `libDecodeSdpc.so` 但 x86-64 ELF,Windows 不能直接 ctypes 加载 |

### 3.3 需要额外安装

| 工具 | 怎么拿 | 说明 |
|---|---|---|
| **TEKSQRAY 数字病理浏览器** | https://www.sqray.com/Download | 生强官方桌面软件,可看 .sdpc + 导出 SVS/TIFF;需要该厂商务或售后给到客户端 |
| **opensdpc(Windows 全套)** | github.com/WonderLandxD/opensdpc clone 源码 + 找 WINDOWS/dll | README 提到 "TEKSQRAY reading software" 是 Windows 端依赖 |
| **WSL2 (Ubuntu) + opensdpc** | `wsl --install` 启用 | pip 装的 `libDecodeSdpc.so` 在 Linux 上可直接用,接口与 OpenSlide 对齐(`OpenSdpc(path)` 返回 `read_region()` / `get_thumbnail()`) |
| **PIANO(配套 AI pipeline)** | github.com/WonderLandxD/PIANO | 批量 patch 提取 + 归一化 + 后续 AI 推理 |

### 3.4 完全无解的路径(目前公开资料来看)

- ❌ **官方 SDK / 文档**:生强官网 sqray.com 公开的下载页只能拿浏览器客户端,**没有给 Windows 解码 dll 单独下载**
- ❌ **OpenSlide 原生支持**:OpenSlide 4.1.0 原生 23 种格式里**没有 sdpc**,需要写 vendor 插件或装 opensdpc
- ❌ **TIFF-based 转换工具**:.sdpc 不是 TIFF 容器(虽然头 16 字节是 ASCII `SQ1.0.3.1010`),无法用 vips/tifffile 直接打开

### 3.5 推荐路径(按可执行度排序)

1. **0 天**:用 `tools/sdpc_inspect.py` 把 500 个 .sdpc 全部跑出 metadata,join xlsx 标签,做数据摸底(✅ 已完成)
2. **1-3 天**:联系生强 / TEKSQRAY 客服,拿到:
   - Windows 端 .dll(或完整的 TEKSQRAY 浏览器安装包)→ 装到本机 → `opensdpc` 就能 `import`
   - 或要求对方提供 SVS/TIFF 批量转换服务(常见的国产扫描仪配套)
3. **3-5 天**:若 Windows 解码 dll 拿不到,在 **WSL2 Ubuntu** 里 `pip install opensdpc` 直接用,做个文件代理 / WSL 调用
4. **5-7 天**(备选):自写 vendor plugin 给 OpenSlide(参考 OpenSlide vendor 接口 `openslide_ops`),这样 OpenSlide 生态(qupath / histopathology.jl / DeepGEM)都能用

---

## 4. 数据现状诊断

### 4.1 数据对齐结果(用 `02_测试集+验证集.csv` 的"切片号" join)

| | 数量 |
|---|---|
| xlsx 中的切片号(全部) | 1339 |
| disk 上的 .sdpc 文件 | 500 |
| **xlsx 与 .sdpc 匹配(可立即训练)** | **28 (2.1%)** |
| xlsx 有但 disk 上没文件(缺数据) | 1311 |
| disk 上有但 xlsx 没标签(可做无标签 / 自监督) | 472 |

### 4.2 这意味着什么

- **真实可用训练样本只有 28 例** —— 远低于合同要求的 9 基因 ≥95% 准确率所需的样本量(钟论文用了 1,999 例)
- 500 个 .sdpc 中 472 个无标签,可考虑:
  - 走 DeepGEM 的弱监督 / noisy label 训练范式(不需要准确标签)
  - 走 DINO / DINOv2 自监督预训练,得到 backbone 后做小样本 fine-tune
- xlsx 标签表本身有 782/1339 行"模型判定"列为空,说明原始数据中相当一部分本来就是"良性/未判定",做训练时需要按列 8(冰冻诊断三分类)和列 10(石蜡切片诊断三分类)做 label 对齐

### 4.3 xlsx 字段含义(部分还原)

| 列 | openpyxl 读出的 mojibake | 真实含义 | 还原状态 |
|---|---|---|---|
| 1 | 切片号 | 切片号 | ✅ |
| 2 | 鎬у埆 | 性别 | ⚠️ 部分(可机读但显示乱) |
| 3 | 骞撮緞 | 年龄 | ⚠️ 同上 |
| 4 | 住院号 | 住院号 | ✅ |
| 5 | 鏍囨湰鍚嶇О | 标本名称 | ⚠️ |
| 6 | 涓村簥璇婃柇 | 临床诊断 | ⚠️ |
| 7 | 鍐板喕璇婃柇 | 冰冻诊断 | ⚠️ |
| 8 | 恶性2/交界1/良性0(冰冻诊断三分类) | 冰冻三分类 | ✅ |
| 9 | 石蜡最终诊断 | 石蜡诊断 | ✅ |
| 10 | 石蜡切片诊断三分类 | 石蜡三分类 | ✅ |
| 11 | 预测概率值 | Softmoe 模型概率 | ✅ |
| 12 | 妯″瀷鍒ゅ畾 | 模型判定 | ⚠️ |
| 13 | 鍣ㄥ畼鍒嗙被 | 良恶分类 | ⚠️ |
| 14 | 模型与真实一致性 | 一致性 | ✅ |
| 15 | 可视化评估 | 可视化结果 | ✅ |
| 16 | 是否为测试集 | split 标记 | ✅ |
| 17 | 新概率 | 复查概率 | ✅ |

**根本原因**:xlsx 在 Windows GBK 系统上写时,sheet 名 + 部分单元格被 GBK 编码后又被错误地以 latin-1 解读 / 截断,导致 openpyxl 读出 `ԭʼ` 这种 mojibake 字符。**建议用 Excel/WPS 打开源 xlsx → 另存为 .xlsx (UTF-8) → 重新 dump 即可彻底解决**。

---

## 5. 落地的工作区文件

```
D:\pan-caner\
├── tools\
│   ├── sdpc_inspect.py          # .sdpc 头部纯 Python 解析器(自写)
│   └── xlsx_view.py             # xlsx mojibake 还原 + dump
├── docs\
│   ├── sdpc_调研报告.md          # 本文件
│   ├── sdpc_metadata_all.csv    # 500 个 .sdpc 全部元信息(29 列)
│   ├── sdpc_metadata_samples.csv # 3 个代表样本(最小/中位/最大)
│   └── sdpc_sample_full.json    # 一个 .sdpc 的完整结构化输出
├── data\
│   ├── 中日冰冻切片\             # 500 个 .sdpc 文件(91.8 GB)
│   ├── 石蜡结果+冰冻大模型.xlsx  # 原始 xlsx 标签表
│   ├── _xlsx_dump\              # xlsx dump 出的 JSON/CSV/MD(UTF-8)
│   │   ├── README.md
│   │   ├── 01_原始数据集.{json,csv,md}
│   │   ├── 02_测试集+验证集.{json,csv,md}
│   │   ├── 03_原始测试集LLM.{json,csv,md}
│   │   └── 04_冰冻结果统计.{json,csv,md}
│   └── _sdpc_evidence\          # .sdpc 二进制 header 证据
│       ├── header_4KB.bin
│       ├── header_4KB.hex.txt
│       ├── header_strings.txt
│       ├── header_strings_medium.txt
│       └── stats.json
```

---

## 6. 立即可做的下一步(给项目方)

1. **跑 `python tools/sdpc_inspect.py data/中日冰冻切片 --csv docs/sdpc_metadata_all.csv`** — 重新生成最新元信息表
2. **联系生强客服(https://www.sqray.com)**,告知序列号 `SQS600P-222300AG`,要:
   - Windows 端浏览器客户端(用于查看/导出 SVS)
   - 或 SVS/TIFF 批量转换服务
3. **xlsx 重新导出**:用 Excel/WPS 打开 `石蜡结果+冰冻大模型.xlsx` → 另存为 .xlsx → 重跑 `tools/xlsx_view.py`,可解决 mojibake
4. **数据补齐**:与医院确认 1311 个"xlsx 有但无 .sdpc"的切片是否在医院系统中存在,如能补齐,可大幅扩大训练集
5. **472 个无标签 .sdpc**:可走 DINO/DINOv2 自监督预训练,作为 backbone 增强

---

## 7. 参考资料

| 来源 | 链接 |
|---|---|
| opensdpc(GitHub) | https://github.com/WonderLandxD/opensdpc |
| PIANO 配套 AI pipeline | https://github.com/WonderLandxD/PIANO (推断) |
| OpenSlide 官方 | https://openslide.org |
| 生强官网 | https://www.sqray.com |
| CSDN 数字病理格式综述 | https://blog.csdn.net/qq_45404805/article/details/145717420 |
| DeepGEM(本项目参考) | https://github.com/TencentAILabHealthcare/DeepGEM |
| GAMIL(本项目参考) | https://github.com/zjsmzn/Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision |
