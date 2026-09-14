# bootstrap.ps1 — 다른 PC에서 첫 실행: venv 생성 + 개발 설치. 요구사항은 Python >= 3.10 하나.
#   .\bootstrap.ps1            # pip 범위 설치 (pyproject)
#   .\bootstrap.ps1 -Locked    # requirements-lock.txt의 정확한 버전으로 재현
#   .\bootstrap.ps1 -Gui       # + PySide6 GUI        -Build: + PyInstaller (tools/build_zip.ps1 용)
param(
    [switch]$Locked,
    [switch]$Gui,
    [switch]$Build
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) { Write-Error "Python 3.10+ 이 필요합니다. https://www.python.org/downloads/"; exit 1 }

if (-not (Test-Path ".venv")) {
    Write-Host "[bootstrap] venv 생성" -ForegroundColor Cyan
    & $py.Source -m venv .venv
}
$pip = Join-Path $root ".venv\Scripts\python.exe"
& $pip -m pip install --upgrade pip --quiet
if ($Locked -and (Test-Path "requirements-lock.txt")) {
    Write-Host "[bootstrap] requirements-lock.txt 로 설치" -ForegroundColor Cyan
    & $pip -m pip install -r requirements-lock.txt --quiet
}
$extras = "dev"
if ($Gui) { $extras += ",gui" }
if ($Build) { $extras += ",gui,build" }
Write-Host "[bootstrap] pip install -e .[$extras]" -ForegroundColor Cyan
& $pip -m pip install -e ".[$extras]" --quiet
& $pip -m anograft --version
Write-Host "[bootstrap] 완료. 활성화: .venv\Scripts\Activate.ps1" -ForegroundColor Green
