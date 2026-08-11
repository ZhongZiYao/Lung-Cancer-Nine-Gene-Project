#!/bin/bash
# pull_on_server.sh (v2)
#
# 服务器端拉取 TCGA-LUAD-WSI + 处理 9 基因 panel。
# 适用: Linux / WSL Ubuntu / Git Bash on Windows
# 用法: bash pull_on_server.sh
#
# 跟原版区别:
# 1. picked_50_case_ids.json 和 9gene_panel_LUAD.csv 路径用 ENV 变量, 默认 ./data/...
# 2. 加 Windows GBK 兼容 (idc-index _load_text monkey-patch)
# 3. 跳过 MC3 MAF (NIH 区域封锁), 用已 derived 的 9-gene CSV
# 4. 后台启动下载 + 写日志 + 给你 PID
#
# 前置:
#   - 服务器已装 Python 3.10+
#   - 已有 data/TCGA-LUAD-WSI/picked_50_case_ids.json (本机从 project_resources/data_public/ scp)
#   - 已有 data/MC3/9gene_panel_LUAD.csv (同上)
#   - 网络通 IDC S3 (公共免 dbGaP)

set -e

cd "$(dirname "$0")"
ROOT="$PWD"
echo "============================================"
echo "  服务器端: 拉 WSI + 处理 9 基因 panel"
echo "  工作目录: $ROOT"
echo "============================================"
echo ""

# ---- 1. 装依赖 (Linux/Mac 用 pip) ----
echo "[1/5] 安装 Python 依赖..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet \
    idc-index pydicom wsidicom numpy pillow pandas \
    2>&1 | tail -3
echo ""

# ---- 2. Windows GBK 兼容 (only matters on Windows clients) ----
echo "[2/5] 检查 Windows GBK 兼容性..."
python -c "
import idc_index_data as _iid
_orig = _iid._load_text
def _patched(path):
    if path is None: return None
    try:
        with open(str(path), 'rb') as f:
            raw = f.read()
        for enc in ('utf-8', 'gbk', 'latin-1'):
            try: return raw.decode(enc)
            except UnicodeDecodeError: continue
        return raw.decode('utf-8', errors='replace')
    except FileNotFoundError: return None
_iid._load_text = _patched
print('  patched _load_text for cross-platform')
" 2>&1 | head -3
echo ""

# ---- 3. 准备目录 ----
echo "[3/5] 准备目录..."
mkdir -p data/TCGA-LUAD-WSI/dicom
mkdir -p data/MC3
echo ""

# ---- 4. 检查必要文件 ----
echo "[4/5] 检查必要文件..."
need_files=(
    "data/TCGA-LUAD-WSI/picked_50_case_ids.json"
    "data/MC3/9gene_panel_LUAD.csv"
)
missing=0
for f in "${need_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  [MISSING] $f"
        missing=$((missing+1))
    else
        sz=$(du -h "$f" | cut -f1)
        echo "  [OK]      $f ($sz)"
    fi
done
echo ""

if [ $missing -gt 0 ]; then
    echo "============================================"
    echo "  ERROR: $missing 文件缺失,需要先 scp"
    echo "============================================"
    echo ""
    echo "在本机 PowerShell 跑:"
    echo ""
    echo "  scp D:/Lungcancer/Lung-Cancer-Nine-Gene-Project/project_resources/data_public/picked_50_case_ids.json \\"
    echo "      user@server:~/ZhongZiyao/data/TCGA-LUAD-WSI/"
    echo ""
    echo "  scp D:/Lungcancer/Lung-Cancer-Nine-Gene-Project/project_resources/data_public/9gene_panel_LUAD.csv \\"
    echo "      user@server:~/ZhongZiyao/data/MC3/"
    echo ""
    echo "  scp D:/Lungcancer/Lung-Cancer-Nine-Gene-Project/project_resources/data_public/tcga_luad_cases.json \\"
    echo "      user@server:~/ZhongZiyao/data/MC3/"
    echo ""
    exit 1
fi

# ---- 5. 后台拉 WSI ----
echo "[5/5] 后台拉 50 cases WSI..."
echo "  target: $(python -c "import json; print(len(json.load(open('data/TCGA-LUAD-WSI/picked_50_case_ids.json'))))") cases"
echo "  estimated disk size: 30-42 GB"
echo ""
nohup python -c "
from idc_index import IDCClient
import json, time

# Apply GBK patch (in case it wasn't applied above)
import idc_index_data as _iid
_orig = _iid._load_text
def _patched(path):
    if path is None: return None
    try:
        with open(str(path), 'rb') as f:
            raw = f.read()
        for enc in ('utf-8', 'gbk', 'latin-1'):
            try: return raw.decode(enc)
            except UnicodeDecodeError: continue
        return raw.decode('utf-8', errors='replace')
    except FileNotFoundError: return None
_iid._load_text = _patched

c = IDCClient()
ids = json.load(open('./data/TCGA-LUAD-WSI/picked_50_case_ids.json'))
print(f'[start] {len(ids)} cases', flush=True)
t0 = time.time()
result = c.download_from_selection(
    downloadDir='./data/TCGA-LUAD-WSI/dicom/',
    collection_id='tcga_luad',
    patientId=ids,
    show_progress_bar=True,
    quiet=False,
)
print(f'[done] {(time.time()-t0)/60:.1f} min', flush=True)
print('[verify] Downloading test...')
import os
total = sum(1 for _ in os.walk('./data/TCGA-LUAD-WSI/dicom/'))
total_files = sum(len(fs) for _, _, fs in os.walk('./data/TCGA-LUAD-WSI/dicom/'))
print(f'  {total} subdirs, {total_files} files', flush=True)
" > wsi_download.log 2>&1 &

PID=$!
echo "  Background PID: $PID"
echo "  Log file: $ROOT/wsi_download.log"
echo ""

echo "============================================"
echo "  启动成功!"
echo "============================================"
echo ""
echo "WSI 下载状态:"
echo "  tail -f $ROOT/wsi_download.log"
echo ""
echo "下载完后,切换到本机或本地:"
echo "  bash pipeline/06_deepgem_inference/run_pipeline.sh"
echo ""
echo "或者直接在本机继续 (本机已经 5 case):"
echo "  python pipeline/06_deepgem_inference/step1_extract_ctranspath.py"
echo "  python pipeline/06_deepgem_inference/step23_v2.py"
echo "============================================"
