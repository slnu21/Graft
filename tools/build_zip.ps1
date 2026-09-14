# tools/build_zip.ps1 — Windows 배포 zip (PyInstaller onedir + recipes/ + 라이선스 문서). 파이썬 없는 PC용.
#   .\tools\build_zip.ps1              # → dist/anograft-<ver>-win64.zip  (+ dist/anograft-<ver>-win64/ 스테이징)
#   .\tools\build_zip.ps1 -SkipBuild   # PyInstaller 재실행 없이 스테이징·zip만 다시
# 전제: .venv 에 pip install -e ".[gui,build]"  (build = pyinstaller). 버전은 pyproject.toml 단일원.
param(
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$ver = (Select-String -Path pyproject.toml -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
if (-not $ver) { Write-Error "pyproject.toml 에서 version 을 찾지 못했습니다"; exit 1 }
$name = "anograft-$ver-win64"
$stage = Join-Path $root "dist\$name"
$zip = Join-Path $root "dist\$name.zip"

& $py -c "import PyInstaller, PySide6" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Error "pyinstaller·PySide6 가 없습니다: pip install -e `".[gui,build]`""; exit 1 }

if (-not $SkipBuild) {
    Write-Host "[build_zip] pyinstaller packaging/anograft.spec" -ForegroundColor Cyan
    # PyInstaller 는 INFO 로그를 stderr 에 쓴다 — PS 5.1 에서 Stop 이면 그게 종료 오류가 되므로 잠시 Continue
    $eap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $py -m PyInstaller packaging/anograft.spec --noconfirm --clean --log-level WARN 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE; $ErrorActionPreference = $eap
    if ($code -ne 0) { Write-Error "PyInstaller 실패 (exit $code)"; exit 1 }
}
if (-not (Test-Path "dist\anograft\anograft.exe")) { Write-Error "dist\anograft\anograft.exe 가 없습니다 (빌드 먼저)"; exit 1 }

Write-Host "[build_zip] 스테이징 $stage" -ForegroundColor Cyan
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
Copy-Item -Recurse "dist\anograft" $stage
Copy-Item -Recurse "recipes" (Join-Path $stage "recipes")
Copy-Item -Recurse "assets" (Join-Path $stage "assets")     # README 이미지
foreach ($f in "README.md", "CHANGELOG.md", "LICENSE", "THIRD-PARTY-NOTICES.md", "PRIVACY.md") {
    if (-not (Test-Path $f)) { Write-Error "배포 필수 파일 없음: $f"; exit 1 }
    Copy-Item $f $stage
}

# 스모크: 스테이징된 exe 가 뜨는지 (버전 문자열 = pyproject 버전)
$out = & (Join-Path $stage "anograft.exe") --version
if ($out -ne "anograft $ver") { Write-Error "exe 버전 불일치: '$out' (기대 'anograft $ver' — src/anograft/__init__.py 확인)"; exit 1 }

Write-Host "[build_zip] zip $zip" -ForegroundColor Cyan
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
$dirMb = [math]::Round((Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "[build_zip] 완료: $zip ($mb MB, 풀면 $dirMb MB)" -ForegroundColor Green
