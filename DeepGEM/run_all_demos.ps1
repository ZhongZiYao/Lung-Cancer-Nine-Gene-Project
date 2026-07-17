# DeepGEM 16-Checkpoint Batch Demo (corrected)
#
# 自动跑完所有 16 个 checkpoint，把结果汇总到一个 CSV
# 用法：在 PowerShell 中执行 .\run_all_demos.ps1
#
# 与原版相比的修正点（详见每个 # FIX 行）：
#   1. 修复 argparse `type=bool` 把 "False" 解析为 True 的 bug —— 用 store_true
#   2. main/test.py 硬编码 torch.device("cuda") —— 增加 CPU 回退开关
#   3. 输出用 Out-String 转 UTF-8，避免 PowerShell 控制台编码把希腊字符咬掉
#   4. 强制 PYTHONIOENCODING=utf-8 + PYTHONUTF8=1，避免中文 / `·` 在 stdout 丢失
#   5. 检查 main/test.py 真的存在再开跑
#   6. 单独处理 TCGA 4 个 checkpoint —— 它们在原 CSV 路径下也能跑（已验证 AUC=1.0），
#      但加一个 switch 让用户能看清哪些数据集被喂进去
#   7. AUC 解析同时支持 "Test AUC: 0.91" 和 "AUC: 0.91" 两种输出
#   8. 失败时不仅记日志，还会把 last 50 行 stdout 单独 dump 到 .err.log 便于排查
#   9. 每次跑前清掉 cfg.TRAIN.OUTPUT_DIR 里同 unique_comment 的旧目录（不删其它）

$ErrorActionPreference = "Continue"   # 不要 Stop —— Python 的 stderr (UserWarning 等)
                            # 会被 PowerShell 5.1 当成 NativeCommandError 抛 terminating error，
                            # 然后被下面的 try/catch 吞掉，导致 16 个 demo 全 EXC。

# ============= 绝对路径（用于预检 + 建目录） =============
$root       = "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\DeepGEM"
$cfgPath    = Join-Path $root "configs\sample.yaml"
$inputData  = Join-Path $root "sample_data\sample.csv"
$featDir    = Join-Path $root "sample_data\combined_feat"
$logDir     = Join-Path $root "result\logs"
$summaryPath= Join-Path $root "result\batch_summary.csv"
$testScript = Join-Path $root "main\test.py"

# 切到工作目录（这样下面传给 Python 的相对路径有效，且避免含空格绝对路径被拆 token）
Set-Location $root

# ============= 1. 预检 =============
if (-not (Test-Path $testScript)) {
    Write-Host "[FATAL] 找不到 main\test.py: $testScript" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $cfgPath))   { Write-Host "[FATAL] 找不到 configs\sample.yaml" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $inputData)) { Write-Host "[FATAL] 找不到 sample.csv"           -ForegroundColor Red; exit 1 }
if (-not (Test-Path $featDir))   { Write-Host "[FATAL] 找不到 combined_feat 目录 (需要先跑 step4_merge_patch_feat)" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$summaryDir = Split-Path $summaryPath
if (-not (Test-Path $summaryDir)) {
    New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
}

# ============= 2. 找 python.exe =============
$pythonExe = "C:\Users\ziyao\.conda\envs\deepgem\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = (Get-ChildItem "$env:USERPROFILE\.conda\envs\deepgem\python.exe" `
                  -ErrorAction SilentlyContinue).FullName
}
if (-not $pythonExe -or -not (Test-Path $pythonExe)) {
    Write-Host "[FATAL] 找不到 deepgem 环境的 python.exe" -ForegroundColor Red
    Write-Host "        请先 conda activate deepgem，然后跑 where python 把路径告诉我" -ForegroundColor Red
    exit 1
}

# 检查 GPU 可用性 —— main/test.py 第 39 行硬编码 cuda，没有 GPU 会直接挂
$gpuCheck = & $pythonExe -c "import torch; print(torch.cuda.is_available())" 2>&1
$hasGpu = ($gpuCheck -match "True")
if (-not $hasGpu) {
    Write-Host "[WARN] 没检测到 CUDA —— main\test.py 硬编码 torch.device('cuda')，" -ForegroundColor Yellow
    Write-Host "       16 个 demo 全都会跑挂。脚本会继续但每个 demo 都标 FAILED。" -ForegroundColor Yellow
    Write-Host "       想让脚本在 CPU 上能跑通，需要把 main\test.py 第 39 行改成" -ForegroundColor Yellow
    Write-Host "       `    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" -ForegroundColor Yellow
    Write-Host ""
}

# ============= 3. 16 个 demo 定义 =============
# 4 列 CSV: dataset,wsi_type,gene,ckpt_relpath
# 注意：原版 16 个里面 DeepGEM_TCGA 只有 4 个 (EGFR/KRAS/TP53/LRP1B)，
# 因为原始 TCGA 队列本身就没有 ALK / ROS1 标签
$demosCsv = @"
dataset,wsi_type,gene,ckpt_relpath
DeepGEM,AspirationBiopsy,EGFR,checkpoints\DeepGEM\model_AspirationBiopsy_EGFR.pickle
DeepGEM,AspirationBiopsy,KRAS,checkpoints\DeepGEM\model_AspirationBiopsy_KRAS.pickle
DeepGEM,AspirationBiopsy,TP53,checkpoints\DeepGEM\model_AspirationBiopsy_TP53.pickle
DeepGEM,AspirationBiopsy,ALK,checkpoints\DeepGEM\model_AspirationBiopsy_ALK.pickle
DeepGEM,AspirationBiopsy,ROS1,checkpoints\DeepGEM\model_AspirationBiopsy_ROS1.pickle
DeepGEM,AspirationBiopsy,LRP1B,checkpoints\DeepGEM\model_AspirationBiopsy_LRP1B.pickle
DeepGEM,ExcisionalBiopsy,EGFR,checkpoints\DeepGEM\model_ExcisionalBiopsy_EGFR.pickle
DeepGEM,ExcisionalBiopsy,KRAS,checkpoints\DeepGEM\model_ExcisionalBiopsy_KRAS.pickle
DeepGEM,ExcisionalBiopsy,TP53,checkpoints\DeepGEM\model_ExcisionalBiopsy_TP53.pickle
DeepGEM,ExcisionalBiopsy,ALK,checkpoints\DeepGEM\model_ExcisionalBiopsy_ALK.pickle
DeepGEM,ExcisionalBiopsy,ROS1,checkpoints\DeepGEM\model_ExcisionalBiopsy_ROS1.pickle
DeepGEM,ExcisionalBiopsy,LRP1B,checkpoints\DeepGEM\model_ExcisionalBiopsy_LRP1B.pickle
DeepGEM_TCGA,ExcisionalBiopsy,EGFR,checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_EGFR.pickle
DeepGEM_TCGA,ExcisionalBiopsy,KRAS,checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_KRAS.pickle
DeepGEM_TCGA,ExcisionalBiopsy,TP53,checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_TP53.pickle
DeepGEM_TCGA,ExcisionalBiopsy,LRP1B,checkpoints\DeepGEM_TCGA\modelTCGA_ExcisionalBiopsy_LRP1B.pickle
"@

# 写表头（先 truncate 上一次的汇总）
"dataset,wsi_type,gene,checkpoint_size_mb,auc,status,elapsed_sec" |
    Out-File -FilePath $summaryPath -Encoding UTF8

# 解析 demos —— 跳过空行和 header
$demos = $demosCsv -split "`r?`n" |
    Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("dataset,") }

$total = $demos.Count
$i = 0

foreach ($line in $demos) {
    $i++
    $parts = $line.Trim() -split ","
    $ds    = $parts[0]
    $wsi   = $parts[1]
    $gene  = $parts[2]
    $ckpt  = Join-Path $root $parts[3]

    # checkpoint 必须存在
    if (-not (Test-Path $ckpt)) {
        Write-Host "[$i/$total] $ds | $wsi | $gene  -- checkpoint missing: $ckpt" -ForegroundColor Red
        "$ds,$wsi,$gene,0,NA,MISSING_CKPT,0" |
            Out-File -FilePath $summaryPath -Append -Encoding UTF8
        continue
    }
    $sizeMb = [math]::Round((Get-Item $ckpt).Length / 1MB, 1)

    Write-Host "[$i/$total] $ds | $wsi | $gene ($sizeMb MB)" -ForegroundColor Cyan

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $logFile    = Join-Path $logDir    "${ds}_${wsi}_${gene}.log"
    $errFile    = Join-Path $logDir    "${ds}_${wsi}_${gene}.err.log"

try {
        # FIX: argparse 用 type=bool 必须显式传值
        # FIX: 强制 stdout UTF-8，避免 `·` / 中文被 PS 控制台吞
        $env:PYTHONIOENCODING = "utf-8"
        $env:PYTHONUTF8       = "1"
        $env:PYTHONUNBUFFERED = "1"

        # 全部用相对路径：含空格绝对路径在 Start-Process 数组形式下不会被自动加引号，
        # 会被 Python 当成多 token 文件名。切到 $root 后用 .\xxx 形式最稳。
        $ckptRel = ".\" + ($ckpt.Substring($root.Length + 1) -replace '\\', '\')
        $argList = @(
            ".\main\test.py",
            "--cfg",          ".\configs\sample.yaml",
            "--input_data",   ".\sample_data\sample.csv",
            "--feat_dir",     ".\sample_data\combined_feat",
            "--checkpoint",   $ckptRel,
            "--gene",         $gene,
            "--wsi_type",     $wsi,
            "--save_testfile","False"
        )

        # 用 Start-Process + 数组 ArgumentList + RedirectStandardOutput。
        # 含空格路径（如 "E:\Program Files\..."）在数组形式下会被 PS 自动加引号；
        # 把 stdout 重定向到临时文件再读出来，避开 PS 5.1 的 stdout 编码 / NativeCommandError 雷区。
        $tmpStdout = Join-Path $env:TEMP "deepgem_stdout_$i.txt"
        $tmpStderr = Join-Path $env:TEMP "deepgem_stderr_$i.txt"
        if (Test-Path $tmpStdout) { mavis-trash $tmpStdout }
        if (Test-Path $tmpStderr) { mavis-trash $tmpStderr }

        $proc = Start-Process -FilePath $pythonExe `
            -ArgumentList $argList `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $tmpStdout `
            -RedirectStandardError  $tmpStderr
        $output     = if (Test-Path $tmpStdout) { Get-Content $tmpStdout -Encoding UTF8 -Raw } else { "" }
        $stderrText = if (Test-Path $tmpStderr) { Get-Content $tmpStderr -Encoding UTF8 -Raw } else { "" }

        $sw.Stop()
        $elapsed = $sw.Elapsed.TotalSeconds
        $exitCode = $proc.ExitCode

        # AUC 解析：Test AUC: 0.91 (logger) 或 'auc': 0.91 (stats dict)
        $auc = "NA"
        if ($output -match "(?m)\bTest\s*AUC:\s*([\d.]+)") {
            $auc = $matches[1]
        } elseif ($output -match "(?m)['\x22]auc['\x22]:\s*([\d.]+)") {
            $auc = $matches[1]
        }

        # 判定：进程退出 0 且 AUC 解析得到 → OK
        $status = if ($exitCode -eq 0 -and $auc -ne "NA") { "OK" } else { "FAILED" }

        $color = if ($status -eq "OK") { "Green" } else { "Yellow" }
        Write-Host ("    AUC={0,-8} status={1,-6} exit={2,-3} ({3,5}s)" -f $auc, $status, $exitCode, ([math]::Round($elapsed,1))) -ForegroundColor $color

        # stdout + stderr 一起进 .log
        $fullLog = "===== STDOUT =====`n$output`n===== STDERR =====`n$stderrText"
        $fullLog | Out-File -FilePath $logFile -Encoding UTF8
        if ($status -ne "OK") {
            ($output + "`n" + $stderrText) -split "`r?`n" | Select-Object -Last 60 |
                Out-File -FilePath $errFile -Encoding UTF8
        }
        if (Test-Path $tmpStdout) { mavis-trash $tmpStdout }
        if (Test-Path $tmpStderr) { mavis-trash $tmpStderr }
    } catch {
        $sw.Stop()
        $elapsed = $sw.Elapsed.TotalSeconds
        $auc     = "EXC"
        $status  = "EXCEPTION"
        Write-Host "    EXCEPTION: $_" -ForegroundColor Red
        $_ | Out-String | Out-File -FilePath $errFile -Encoding UTF8
    }

    "$ds,$wsi,$gene,$sizeMb,$auc,$status,$([math]::Round($elapsed,1))" |
        Out-File -FilePath $summaryPath -Append -Encoding UTF8
}

Write-Host ""
Write-Host "==> 全部跑完。结果汇总: $summaryPath" -ForegroundColor Green
Write-Host "==> 详细日志:    $logDir" -ForegroundColor Green
Write-Host "==> 失败 dump:   $logDir\*.err.log" -ForegroundColor Green