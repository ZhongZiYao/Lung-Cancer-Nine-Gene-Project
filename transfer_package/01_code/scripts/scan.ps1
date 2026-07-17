$root = 'D:\pan-caner'
Write-Host "=== File type counts (top 20) ===" -ForegroundColor Cyan
Get-ChildItem $root -Recurse -Force -ErrorAction SilentlyContinue |
  Where-Object { -not $_.PSIsContainer } |
  Group-Object Extension |
  Sort-Object Count -Descending |
  Select-Object -First 20 |
  ForEach-Object { "{0,8}  {1}" -f $_.Count, ($_.Name -replace '^$','(none)') }

Write-Host ""
Write-Host "=== Top 30 largest files (MB) ===" -ForegroundColor Cyan
Get-ChildItem $root -Recurse -Force -ErrorAction SilentlyContinue |
  Where-Object { -not $_.PSIsContainer } |
  Sort-Object Length -Descending |
  Select-Object -First 30 |
  ForEach-Object {
    $mb = [int]($_.Length/1MB)
    "{0,8} MB  {1}" -f $mb, $_.FullName.Substring($root.Length)
  }

Write-Host ""
Write-Host "=== Top 10 dirs by size (MB) ===" -ForegroundColor Cyan
Get-ChildItem $root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
  ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -Force -ErrorAction SilentlyContinue |
             Where-Object { -not $_.PSIsContainer } |
             Measure-Object -Property Length -Sum).Sum
    $mb = [int]($size/1MB)
    [PSCustomObject]@{ SizeMB = $mb; Path = $_.FullName.Substring($root.Length) }
  } | Sort-Object SizeMB -Descending | Select-Object -First 10 | Format-Table -AutoSize

Write-Host ""
Write-Host "=== tools/ scripts ===" -ForegroundColor Cyan
Get-ChildItem "$root\tools" -Recurse -File -ErrorAction SilentlyContinue |
  ForEach-Object { "{0,8}  {1}" -f $_.Length, $_.FullName.Substring($root.Length) }

Write-Host ""
Write-Host "=== group_meeting_report/ ===" -ForegroundColor Cyan
Get-ChildItem "$root\group_meeting_report" -Recurse -File -ErrorAction SilentlyContinue |
  ForEach-Object { "{0,10:N0}  {1}" -f ($_.Length/1KB), $_.FullName.Substring($root.Length) }

Write-Host ""
Write-Host "=== docs / papers ===" -ForegroundColor Cyan
foreach ($d in 'docs','zhong_paper_cn','Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision','_contract_extracted','_docx_extracted') {
    $p = "$root\$d"
    if (Test-Path $p) {
        Write-Host "-- $d --" -ForegroundColor Yellow
        Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue |
            ForEach-Object { "{0,10:N0}  {1}" -f ($_.Length/1KB), $_.FullName.Substring($root.Length) }
    }
}

Write-Host ""
Write-Host "=== data/ top-level ===" -ForegroundColor Cyan
Get-ChildItem "$root\data" -Directory -ErrorAction SilentlyContinue |
  ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
    "{0,8} MB  {1}" -f [int]($size/1MB), $_.FullName.Substring($root.Length)
  }