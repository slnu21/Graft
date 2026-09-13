#!/bin/sh
# bootstrap.sh — Linux/macOS 첫 실행: venv 생성 + 개발 설치. 요구사항은 Python >= 3.10 하나.
#   ./bootstrap.sh            # pip 범위 설치 (pyproject)
#   ./bootstrap.sh --locked   # requirements-lock.txt의 정확한 버전으로 재현
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python) || { echo "Python 3.10+ 이 필요합니다."; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
PIP=".venv/bin/python -m pip"
$PIP install --upgrade pip --quiet
if [ "$1" = "--locked" ] && [ -f requirements-lock.txt ]; then
  $PIP install -r requirements-lock.txt --quiet
fi
$PIP install -e ".[dev]" --quiet
.venv/bin/python -m anograft --version
echo "[bootstrap] 완료. 활성화: source .venv/bin/activate"
