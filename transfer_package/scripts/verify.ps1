$tp = "D:\pan-caner\transfer_package"
Write-Host "=== Final structure ===" -ForegroundColor Cyan
Get-ChildItem $tp -Recurse -File -ErrorAction SilentlyContinue |
  Group-Object DirectoryName |
  ForEach-Object {
    $total = ($_.Group | Measure-Object -Property Length -Sum).Sum
    $rel = $_.Name.Substring($tp.Length)
    "{0,8:N1} KB  {1}" -f ($total/1KB), $rel
  } | Sort-Object

Write-Host ""
Write-Host "=== Total package size ===" -ForegroundColor Cyan
$totalAll = (Get-ChildItem $tp -Recurse -File -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
"{0:N1} MB  ({1:N2} GB)" -f ($totalAll/1MB), ($totalAll/1GB)