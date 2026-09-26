"""최근에 연 경로 — 다섯 화면의 "열기" 칸이 같은 경로를 두 번 치지 않게 한다 (U7).

**Qt 없음 · 런타임 의존성 0**(stdlib ``json``). 저장 위치는 ``~/.anograft/recent.json`` 하나다
(사용자 결정 2026-09-25) — 브라우저를 바꿔도 남고, 나중에 Qt ``QSettings`` 와 합칠 여지가 있으며,
"화면은 API 호출만 한다"는 규약에 프론트가 상태를 드는 예외를 만들지 않는다.

종류는 **화면이 아니라 칸의 뜻**으로 나눈다 — ③ 미리보기에서 저장한 레시피가 ④ 일괄 생성의 목록에
그대로 뜨는 것이 이 단위가 노리는 것이기 때문이다.

모든 함수는 **fail-soft** 다: 읽다 깨지면 빈 목록, 쓰다 막히면 조용히 넘어간다. 편의 기능 하나 때문에
화면이 못 뜨면 안 된다(홈이 읽기 전용인 PC·샌드박스도 있다).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: 칸의 뜻. ``bank`` = 결함 보관함 폴더(① 열기 · ② 저장 대상) · ``image`` = 결함 사진(② 열기) ·
#: ``recipe`` = 레시피 YAML(③ 열기·저장 · ④ 열기) · ``output`` = 출력 폴더(④ 출력 · ⑤ 열기) ·
#: ``loop`` = 루프 설정 `loop.yaml`(⑥ 열기 — 레시피와 뜻이 다르다: 현장 경로·평가셋이 든 개인 설정이라
#: 레시피 밖에 있는 파일이고, ③·④ 의 레시피 목록에 섞이면 안 된다).
KINDS: tuple[str, ...] = ("bank", "image", "recipe", "output", "loop")

#: 종류마다 기억하는 최대 개수 — 목록이 길면 고르는 데 더 오래 걸린다.
LIMIT = 10

#: 홈을 바꿔 끼우는 환경변수(테스트 · 이동식 설치). 없으면 ``~/.anograft``.
HOME_ENV = "ANOGRAFT_HOME"

FILE_NAME = "recent.json"


def home_dir() -> Path:
    """설정 폴더. ``ANOGRAFT_HOME`` 이 있으면 그것, 없으면 ``~/.anograft``."""
    override = os.environ.get(HOME_ENV, "").strip()
    return Path(override) if override else Path.home() / ".anograft"


def store_path() -> Path:
    return home_dir() / FILE_NAME


def load() -> dict[str, list[str]]:
    """저장된 목록 전부. 파일이 없거나 깨졌으면 **빈 목록**(예외를 올리지 않는다)."""
    empty: dict[str, list[str]] = {k: [] for k in KINDS}
    try:
        raw = json.loads(store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(raw, dict):
        return empty
    for kind in KINDS:
        items = raw.get(kind)
        if isinstance(items, list):
            empty[kind] = [str(v) for v in items if isinstance(v, str) and v.strip()][:LIMIT]
    return empty


def recent(kind: str) -> list[str]:
    """한 종류의 목록 — 최근 것이 앞."""
    return load().get(kind, []) if kind in KINDS else []


def remember(kind: str, value: str) -> list[str]:
    """연 경로 하나를 맨 앞에 둔다(같은 값은 위로 올라올 뿐 중복되지 않는다).

    **화면이 "기억해 줘"라고 부르는 API 는 없다** — 무언가를 연 행위가 곧 기록이라서, 각 화면의
    ``open`` 핸들러가 성공한 뒤에 이걸 부른다. 반환은 갱신된 목록(쓰기에 실패해도 그 모양).
    """
    text = (value or "").strip()
    if kind not in KINDS or not text:
        return recent(kind)
    data = load()
    items = [v for v in data.get(kind, []) if v != text]
    items.insert(0, text)
    data[kind] = items[:LIMIT]
    try:
        path = store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    except OSError:
        pass  # fail-soft — 기억하지 못할 뿐, 화면은 그대로 돈다
    return data[kind]
