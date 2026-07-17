$root = 'D:\pan-caner'
$tp = "$root\transfer_package"

Write-Host "=== Step 1: Copy Python tools (src only, no __pycache__) ===" -ForegroundColor Cyan
$srcDir = "$tp\01_code\tools"
New-Item -ItemType Directory -Path $srcDir -Force | Out-Null
Copy-Item -Path "$root\tools\*.py" -Destination $srcDir -Force
Get-ChildItem $srcDir | ForEach-Object { "  copied: $($_.Name)" }

Write-Host ""
Write-Host "=== Step 2: Copy group_meeting_report scripts ===" -ForegroundColor Cyan
$grpDir = "$tp\01_code\group_meeting_report"
New-Item -ItemType Directory -Path $grpDir -Force | Out-Null
Copy-Item -Path "$root\group_meeting_report\*.py" -Destination $grpDir -Force
Copy-Item -Path "$root\group_meeting_report\*.pptx" -Destination $grpDir -Force
Copy-Item -Path "$root\group_meeting_report\*.docx" -Destination $grpDir -Force
Get-ChildItem $grpDir | ForEach-Object { "  copied: $($_.Name)" }

Write-Host ""
Write-Host "=== Step 3: Copy public MC3 labels + small CSV/JSON ===" -ForegroundColor Cyan
$pubDataDir = "$tp\02_data_public"
New-Item -ItemType Directory -Path $pubDataDir -Force | Out-Null
# These are small, shareable
$pubFiles = @(
    "$root\data\MC3\9gene_panel_LUAD.csv",
    "$root\data\MC3\tcga_luad_cases.json",
    "$root\data\TCGA-LUAD-WSI\manifest_50.tsv",
    "$root\data\TCGA-LUAD-WSI\picked_50_case_ids.json",
    "$root\data\README.md"
)
foreach ($f in $pubFiles) {
    if (Test-Path $f) {
        Copy-Item -Path $f -Destination $pubDataDir -Force
        "  copied: $($f.Substring($root.Length))"
    }
}

Write-Host ""
Write-Host "=== Step 4: Copy docs (sdpc metadata, planning, zhong paper) ===" -ForegroundColor Cyan
$docsDir = "$tp\05_docs"
New-Item -ItemType Directory -Path $docsDir -Force | Out-Null
if (Test-Path "$root\docs") {
    Copy-Item -Path "$root\docs\*" -Destination $docsDir -Recurse -Force
    "  copied docs/"
}
if (Test-Path "$root\zhong_paper_cn") {
    Copy-Item -Path "$root\zhong_paper_cn\*" -Destination "$docsDir\zhong_paper_cn" -Recurse -Force
    "  copied zhong_paper_cn/"
}
if (Test-Path "$root\肺癌九基因项目数据集与方向规划v2.md") {
    Copy-Item -Path "$root\肺癌九基因项目数据集与方向规划v2.md" -Destination $docsDir -Force
    "  copied 规划v2.md"
}
if (Test-Path "$root\TCGA_LUAD_下载速查卡.md") {
    Copy-Item -Path "$root\TCGA_LUAD_下载速查卡.md" -Destination $docsDir -Force
    "  copied 速查卡.md"
}

Write-Host ""
Write-Host "=== Step 5: Copy Zhong/GAMIL reference code ===" -ForegroundColor Cyan
$gamilSrc = "$root\Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision"
$gamilDst = "$tp\04_papers\zhong_GAMIL_code"
New-Item -ItemType Directory -Path $gamilDst -Force | Out-Null
if (Test-Path $gamilSrc) {
    # README + key modules
    Copy-Item -Path "$gamilSrc\README.md" -Destination $gamilDst -Force
    Copy-Item -Path "$gamilSrc\GAMIL\*" -Destination "$gamilDst\GAMIL" -Recurse -Force
    "  copied README + GAMIL/"
}
# Tencent DeepGEM code
$deepgemSrc = "$root\DeepGEM"
$deepgemDst = "$tp\04_papers\tencent_DeepGEM_code"
if (Test-Path $deepgemSrc) {
    # Copy only main + configs (skip checkpoints which are huge)
    New-Item -ItemType Directory -Path "$deepgemDst\main" -Force | Out-Null
    New-Item -ItemType Directory -Path "$deepgemDst\data_prepare" -Force | Out-Null
    New-Item -ItemType Directory -Path "$deepgemDst\configs" -Force | Out-Null
    Copy-Item -Path "$deepgemSrc\README.md" -Destination $deepgemDst -Force
    Copy-Item -Path "$deepgemSrc\main\*" -Destination "$deepgemDst\main" -Recurse -Force -Exclude "*.pyc"
    Copy-Item -Path "$deepgemSrc\data_prepare\*" -Destination "$deepgemDst\data_prepare" -Recurse -Force -Exclude "*.pyc"
    Copy-Item -Path "$deepgemSrc\configs\*" -Destination "$deepgemDst\configs" -Recurse -Force
    Copy-Item -Path "$deepgemSrc\requirements.txt" -Destination $deepgemDst -Force
    "  copied DeepGEM main + data_prepare + configs"
}

Write-Host ""
Write-Host "=== Step 6: Copy transfer scripts (.ps1 + scan) ===" -ForegroundColor Cyan
$scriptsDir = "$tp\01_code\scripts"
New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null
Copy-Item -Path "$root\tmp\scan.ps1" -Destination $scriptsDir -Force
"  copied scan.ps1"

Write-Host ""
Write-Host "=== Step 7: Copy group_meeting_report assets (figures + smoke patches) ===" -ForegroundColor Cyan
$assetsDir = "$tp\06_assets\group_meeting_report"
New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null
Copy-Item -Path "$root\group_meeting_report\assets\*" -Destination $assetsDir -Force
"  copied assets/"

Write-Host ""
Write-Host "=== Step 8: Copy memory bank (Claude 上下文) ===" -ForegroundColor Cyan
$memoryDir = "$tp\01_code\memory"
New-Item -ItemType Directory -Path $memoryDir -Force | Out-Null
$memSrc = 'C:\Users\admin\.claude\projects\D--pan-caner\memory'
if (Test-Path $memSrc) {
    Copy-Item -Path "$memSrc\*" -Destination $memoryDir -Force
    "  copied memory/ ($((Get-ChildItem $memSrc).Count) files)"
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Sensitive data NOT copied (you decide):" -ForegroundColor Yellow
Write-Host "  - data/MC3/mc3.v0.2.8.PUBLIC.maf.gz (753 MB) -- re-download from Synapse with your PAT"
Write-Host "  - data/TCGA-LUAD-WSI/dicom/ (~42 GB) -- re-download with idc-index Python script"
Write-Host "  - data/中日冰冻切片/*.sdpc (~87 GB) -- PRIVATE CJFH frozen slides, MUST carry manually"
Write-Host "  - data/patches/ (558 MB) -- patches extracted from .sdpc, can regenerate"
Write-Host "  - DeepGEM/checkpoints/ (heavy) -- re-download from Zenodo"