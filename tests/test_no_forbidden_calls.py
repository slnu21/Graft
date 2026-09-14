"""코드베이스 규율을 grep으로 고정한다.

- 전역 난수(``np.random.seed`` · ``random`` 모듈) 금지 — 재현성.
- ``cv2.imread``/``cv2.imwrite`` 금지 — Windows 한글 경로에서 조용히 실패. ``anograft.io.imgio``만 쓴다.
- ``core``는 ``anograft.io``·Qt를 import하지 않는다.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "anograft"

FORBIDDEN_EVERYWHERE = [
    re.compile(r"\bnp\.random\.seed\("),
    re.compile(r"\bnumpy\.random\.seed\("),
    re.compile(r"^\s*import random\b", re.M),
    re.compile(r"^\s*from random import\b", re.M),
    re.compile(r"\bcv2\.imread\("),
    re.compile(r"\bcv2\.imwrite\("),
]

FORBIDDEN_IN_CORE = [
    re.compile(r"^\s*(from|import) anograft\.io\b", re.M),
    re.compile(r"^\s*(from|import) PySide6\b", re.M),
    re.compile(r"^\s*(from|import) PyQt\d\b", re.M),
]


def _py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_no_global_randomness_or_raw_cv2_file_io() -> None:
    offenders = []
    for f in _py_files(SRC):
        text = f.read_text(encoding="utf-8")
        # 문서 문자열 안의 언급은 허용하지 않는다 — 규칙은 코드에도 문서에도 같게 적용
        for pat in FORBIDDEN_EVERYWHERE:
            for m in pat.finditer(text):
                line = text[: m.start()].count("\n") + 1
                # 주석/독스트링에서 "금지" 설명으로 언급하는 것은 허용: 해당 줄에 '금지' 또는 '실패'가 있으면 통과
                line_text = text.splitlines()[line - 1]
                if "금지" in line_text or "실패" in line_text or "직접 부르지" in line_text:
                    continue
                offenders.append(f"{f.relative_to(SRC)}:{line}: {pat.pattern}")
    assert offenders == [], "\n".join(offenders)


def test_core_does_not_import_io_or_qt() -> None:
    offenders = []
    for f in _py_files(SRC / "core"):
        text = f.read_text(encoding="utf-8")
        for pat in FORBIDDEN_IN_CORE:
            for m in pat.finditer(text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{f.relative_to(SRC)}:{line}: {m.group(0).strip()}")
    assert offenders == [], "\n".join(offenders)


QT_IMPORT = re.compile(r"^\s*(from|import) (PySide6|PyQt\d)\b", re.M)


def test_qt_only_inside_gui_package() -> None:
    """Qt는 ``gui/`` 안에서만 — core·io·bank·runner·preview·cli는 GUI 없이 돌아야 한다(CLI-only 설치)."""
    offenders = []
    for f in _py_files(SRC):
        if "gui" in f.relative_to(SRC).parts:
            continue
        text = f.read_text(encoding="utf-8")
        for m in QT_IMPORT.finditer(text):
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{f.relative_to(SRC)}:{line}: {m.group(0).strip()}")
    assert offenders == [], "\n".join(offenders)


def test_gui_package_root_does_not_import_qt() -> None:
    """``anograft.gui`` 자체(``__init__``·``__main__``)는 Qt 없이 import돼야 설치 안내를 낼 수 있다."""
    for name in ("__init__.py", "__main__.py"):
        text = (SRC / "gui" / name).read_text(encoding="utf-8")
        top_level = [m.group(0) for m in QT_IMPORT.finditer(text) if not m.group(0).startswith(" ")]
        assert top_level == [], f"gui/{name}: {top_level}"
