# DeepGEM Windows 一键跑通脚本
# 用法：在 PowerShell 中执行  .\run_sample_test.ps1
# 这是给 sample test 用的轻量脚本——只验证推理 pipeline，不切 WSI

# ============= 1. 装 Python 依赖 =============
Write-Host "==> [1/4] 升级 pip" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "==> [2/4] 安装 Windows 适配版 requirements" -ForegroundColor Cyan
python -m pip install -r .\requirements_windows.txt

# ============= 2. 装 CTransPath 用的 timm =============
# DeepGEM 论文用了 CTransPath，ctranspath.pth 是改过的 Swin Transformer
# 不在 PyPI 上，需要单独准备
# 如果没有 ctranspath.pth，sample test 跑不起来
Write-Host "==> [3/4] 检查 CTransPath 预训练权重" -ForegroundColor Cyan
$ctranspath = ".\checkpoints\pretrain\ctranspath.pth"
if (-not (Test-Path $ctranspath)) {
    Write-Host "    ! ctranspath.pth 不存在——sample test 会卡在这一步" -ForegroundColor Yellow
    Write-Host "    请去 https://github.com/TencentAILabHealthcare/DeepGEM 检查数据获取方式" -ForegroundColor Yellow
}

# ============= 3. 跑 sample test =============
Write-Host "==> [4/4] 跑 sample test" -ForegroundColor Cyan
python .\main\test.py `
    --cfg .\configs\sample.yaml `
    --input_data .\sample_data\sample.csv `
    --feat_dir .\sample_data\combined_feat `
    --checkpoint .\checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_EGFR.pickle `
    --gene EGFR `
    --wsi_type ExcisionalBiopsy `
    --save_testfile True

Write-Host "==> 跑完" -ForegroundColor Green
