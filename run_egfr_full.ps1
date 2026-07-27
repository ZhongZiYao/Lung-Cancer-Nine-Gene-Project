# 跑完整 5-fold EGFR 训练,后台运行,日志输出到文件
$ErrorActionPreference = "Stop"
$logFile = "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\_egfr_full_train.log"
$py = "C:\Users\ziyao\.conda\envs\deepgem\python.exe"
$script = "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\pipeline\week_1_baseline\step3_egfr_only.py"
$args = @("--bags-dir", "pipeline\week_1_baseline\outputs\bags_v3")
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $py
$psi.Arguments = "`"$script`" " + ($args -join " ")
$psi.WorkingDirectory = "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$psi.CreateNoWindow = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$outSb = New-Object System.Text.StringBuilder
$errSb = New-Object System.Text.StringBuilder
Register-ObjectEvent -InputObject $p -EventName OutputDataReceived -Action {
    if ($EventArgs.Data) {
        [void]$Event.MessageData.AppendLine($EventArgs.Data)
        Add-Content -Path $Event.Sender.pspath -Value $EventArgs.Data -Encoding UTF8
    }
} -MessageData $outSb
# 简化:用 -PassThru + Start-Process + 重定向文件
Write-Host "starting..."
$proc = Start-Process -FilePath $py -ArgumentList "`"$script`"", "--bags-dir", "pipeline\week_1_baseline\outputs\bags_v3" -WorkingDirectory "E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer" -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" -PassThru -WindowStyle Hidden
Write-Host "PID: $($proc.Id), log: $logFile"