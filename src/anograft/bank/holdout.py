"""평가셋 누수 방지 — `<bank>/holdout.txt` 에 적힌 원본은 **은행에 들어갈 수 없다**(설계 §6.3).

왜 코드가 막아야 하나: 자동 편입이 평가셋 이미지를 은행에 넣으면 그 뒤의 모든 라운드 비교가
**조용히** 무의미해진다(모델이 시험지를 학습한다). 사람 규율로는 자동 루프에서 반드시 깨지므로,
임포터가 지나는 **한 지점**(`BankWriter.add`)에서 거부하고 사유를 남긴다.

형식은 사람이 손으로 쓸 수 있게 단순하게:

    # 고정 평가셋 — 2026-09-24 기준
    plate_0007
    plate_0031.png      # 확장자를 적어도 된다
    images/plate_0042.png   # 경로를 적어도 된다

비교는 **stem**(확장자·폴더를 뗀 이름)으로 한다 — 같은 사진이 폴더를 옮기거나 확장자가 바뀌어도
같은 것으로 본다. 반대로 서로 다른 폴더에 같은 이름이 있으면 둘 다 막는다(안전한 쪽으로 틀린다).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

HOLDOUT_FILE = "holdout.txt"


def normalize_stem(name: str) -> str:
    """`images/plate_0007.png` · `plate_0007.png` · `plate_0007` → `plate_0007`."""
    return Path(str(name).strip().replace("\\", "/")).stem


def parse_holdout(text: str) -> set[str]:
    """목록 텍스트 → stem 집합. `#` 주석과 빈 줄은 건너뛴다."""
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        stem = normalize_stem(line)
        if stem:
            out.add(stem)
    return out


def read_holdout(bank_root: str | Path) -> set[str]:
    """은행 폴더의 `holdout.txt` — 없으면 빈 집합(fail-soft: 목록이 없다고 임포트를 막지 않는다)."""
    path = Path(bank_root) / HOLDOUT_FILE
    if not path.is_file():
        return set()
    return parse_holdout(path.read_text(encoding="utf-8-sig"))


def is_held_out(origin: str, holdout: Iterable[str]) -> bool:
    """이 원본이 평가셋인가. `origin` 이 비어 있으면(출처를 모르면) 막지 않는다."""
    if not origin:
        return False
    return normalize_stem(origin) in set(holdout)


def violations(sources, holdout: Iterable[str]) -> list[str]:
    """**이미 은행에 들어가 있는** 평가셋 소스의 id 목록 — 목록을 나중에 만들었을 때를 잡는다.

    거부(import 시점)만으로는 "어제 넣고 오늘 목록에 추가한" 경우를 못 잡는다. `bank ls`·`bank verify`
    가 이걸 보여 주고, 사람이 지우기로 결정한다(**자동으로 지우지 않는다** — 은행 삭제는 되돌릴 수 없다).
    """
    held = set(holdout)
    return [s.id for s in sources if is_held_out(getattr(s, "origin", ""), held)]
