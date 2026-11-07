# PaperBrief 一键启动（Windows PowerShell）
# 用法：右键 → 使用 PowerShell 运行；或在终端执行  .\start.ps1
# 说明：优先使用本机已安装的 Python；若未安装 PyMuPDF，则使用工作区自带的离线依赖目录。

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = if ($env:PORT) { $env:PORT } else { "8000" }

# 1) 找 Python
$py = $null
foreach ($c in @(
        "C:\Users\ASUS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe",
        (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    )) {
    if ($c -and (Test-Path $c)) { $py = $c; break }
}
if (-not $py) { Write-Host "未找到 Python，请先安装 Python 3.10+"; exit 1 }

# 2) 找 PyMuPDF：优先项目同级 _pylibs（离线），否则用系统环境
$libs = Join-Path (Split-Path -Parent $here) "_pylibs"
if (Test-Path (Join-Path $libs "pymupdf")) {
    $env:PYTHONPATH = $libs
    Write-Host "使用离线依赖：$libs"
} else {
    Write-Host "使用系统依赖（如未安装请执行: pip install pymupdf）"
}

# 3) 官方测试集目录（可改）
if (-not $env:PAPERBRIEF_PDF_DIR) {
    $cand = Join-Path (Split-Path -Parent $here) "ref_kxyzs\pdf"
    if (Test-Path $cand) { $env:PAPERBRIEF_PDF_DIR = $cand }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PORT = $port

Write-Host ""
Write-Host "PaperBrief 启动中…" -ForegroundColor Cyan
Write-Host "  地址:    http://127.0.0.1:$port" -ForegroundColor Green
Write-Host "  演示页:  http://127.0.0.1:$port/?auto=1706.03762v7.pdf&brief=1" -ForegroundColor Green
Write-Host "  按 Ctrl+C 停止" -ForegroundColor DarkGray
Write-Host ""

Start-Process "http://127.0.0.1:$port"
& $py (Join-Path $here "backend\server.py")
