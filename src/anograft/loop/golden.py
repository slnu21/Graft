"""이중 평가셋 — **고정**(동결·회귀 감시)과 **최근**(갱신·승급 판단) (설계 §2b.6, 작업 단위 T16).

§2 규약 2 는 "평가셋을 동결한다"인데, 운영에서 시간이 지나면 동결된 평가셋이 **지금 공정을 대표하지
못한다**(램프 교체·렌즈 청소·로트 변경). 갱신하면 기준선이 끊기고, 안 하면 현실과 멀어진다. 둘 다 둔다:

==========  ================================  ==========================
            구성                              쓰임
==========  ================================  ==========================
고정        최초 실제 결함, **영원히 동결**    회귀 감시 — 떨어지면 승급 없음
최근        최근 N개월 실제 결함, 주기 갱신    지금 공정 성능 — 승급 판단
==========  ================================  ==========================

승급 = **고정에서 회귀 없음 ∧ 최근에서 개선**(`policy.should_promote` — 인자 자리는 처음부터 있었다).

**Graft 는 최근 평가셋을 큐레이션하지 않는다.** "최근 N개월"이 몇 달인지는 현장이 정하고(설계 §8 확인
게이트), 폴더를 새로 채우는 것도 사람이다. Graft 가 하는 일은 그 폴더가 **바뀐 것을 알아채는 것**이다 —
안 알아채면 두 라운드의 점수가 서로 다른 평가셋에서 나온 값인데도 나란히 그려지고, 그게 이 단위가
막으려는 유일한 거짓말이다. 그래서 여기 있는 것은 **지문**(무엇을 재고 있었나)과 **판정**(다시 재야 하나)
둘뿐이다.

지문이 바뀌면 둘 다 일어난다:

* **추이는 끊긴다** — 0.46(옛 평가셋)과 0.44(새 평가셋)는 견줄 수 있는 값이 아니다(원장 `rolling_update`
  → 현황판 `segment`).
* **판정은 끊기지 않는다** — champion 을 **새 평가셋에서 다시 재기** 때문이다(그게 계약에 `eval` verb 가
  필요한 두 번째 이유다. 첫째는 "학습 없이 재는 것"). 기준선 재설정(T15)과 다른 점이 정확히 여기다:
  그쪽은 정답이 바뀌어 **견줄 수 없게** 된 것이고, 이쪽은 **다시 재면 견줄 수 있다**.

`ingest.py` 처럼 **순수 함수와 IO 를 한 파일에 둔다** — 판정(`rolling_plan`·`fingerprint`)은 인자만 보고,
디스크를 읽는 것은 `split_entries`/`split_fingerprint` 뿐이다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 파일 하나의 신원 — ``(목록 안의 이름, 바이트 수, 내용 해시 앞 8)``.
Entry = tuple[str, int, str]

#: 내용 해시를 몇 자까지 볼지. 커서(`ingest`)와 같은 8 — 평가셋 한 폴더 안에서 충돌은 사실상 없다.
HASH_CHARS = 8
#: 지문 길이(16진). 사람이 로그·표에서 눈으로 대조할 수 있는 길이.
FINGERPRINT_CHARS = 16


def file_entry(name: str, path: Path) -> Entry:
    """파일 하나 → 신원. 읽을 수 없으면 크기 −1 + 빈 해시(없는 파일과 못 읽는 파일을 구분하지 않는다)."""
    try:
        data = path.read_bytes()
    except OSError:
        return (name, -1, "")
    return (name, len(data), hashlib.sha1(data).hexdigest()[:HASH_CHARS])


def fingerprint(entries: Sequence[Entry]) -> str:
    """신원 목록 → 지문 한 줄. **순수**하고 순서에 무관하다(폴더 나열 순서가 지문을 바꾸면 안 된다).

    크기만 보지 않고 **내용까지** 보는 이유: 여기서 "안 바뀌었다"를 잘못 말하면 서로 다른 평가셋의 점수가
    조용히 나란히 그려진다 — 이 단위가 막으려는 바로 그 거짓말이다. 라운드마다 한 번만 부른다
    (한 바퀴가 학습을 몇 시간 돌리는 일이라, 평가셋 수십~수백 장을 한 번 읽는 값은 싸다).
    """
    joined = "\n".join(f"{name}:{size}:{digest}" for name, size, digest in sorted(entries))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]


def split_entries(split: Mapping[str, Path | None]) -> list[Entry]:
    """평가셋 한 쌍(이미지 + 라벨/마스크)의 신원 목록. **IO** — 파일을 읽는다.

    정답(라벨·마스크)도 함께 본다 — 같은 사진의 GT 를 고쳐 놓은 것도 "다른 평가셋"이다.
    """
    from anograft.io import imgio

    images = split.get("images")
    entries: list[Entry] = []
    if images is not None and Path(images).is_dir():
        for p in imgio.list_images(Path(images)):
            entries.append(file_entry(f"img/{p.name}", p))
    for key, prefix in (("labels", "gt"), ("masks", "gt")):
        second = split.get(key)
        if second is None or not Path(second).is_dir():
            continue
        for p in sorted(Path(second).iterdir()):
            if p.is_file():
                entries.append(file_entry(f"{prefix}/{p.name}", p))
    return entries


def split_fingerprint(split: Mapping[str, Path | None]) -> tuple[str, int]:
    """``(지문, 이미지 장수)``. 빈 폴더면 ``("", 0)`` — 지문이 없으면 판정도 하지 않는다."""
    entries = split_entries(split)
    images = [e for e in entries if e[0].startswith("img/")]
    if not images:
        return "", 0
    return fingerprint(entries), len(images)


@dataclass(frozen=True)
class RollingBaseline:
    """champion 이 **지금 최근 평가셋에서** 받은 점수 — `loop.state.json` 의 세 번째 포인터.

    여기 있는 이유는 champion 의 점수가 **평가셋마다 다르기** 때문이다. 고정 평가셋은 영원히 같아서
    `Champion.metric` 한 값으로 끝나지만, 최근 평가셋은 갱신되므로 "어느 평가셋에서 잰 값인가"
    (`fingerprint`)와 "누구를 잰 값인가"(`model`)를 값과 함께 들고 있어야 견줄 수 있다.
    """

    fingerprint: str
    metric: float
    round: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "metric": self.metric,
            "round": self.round,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RollingBaseline | None:
        """모르는/깨진 모양은 ``None`` — 기준선이 없으면 champion 을 다시 재면 된다(fail-soft)."""
        fp = str(d.get("fingerprint", "") or "")
        try:
            metric = float(d.get("metric"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not fp:
            return None
        return cls(
            fingerprint=fp,
            metric=metric,
            round=int(d.get("round", 0) or 0),
            model=str(d.get("model", "") or ""),
        )


@dataclass(frozen=True)
class RollingPlan:
    """이 라운드에 최근 평가셋으로 무엇을 할지 — **순수 판정**(파일·프로세스를 모른다)."""

    #: champion 을 새로(또는 처음) 재야 하는가. 기준선이 없거나 죽었으면 참.
    remeasure: bool
    #: 평가셋이 갱신되었는가 — **추이를 끊는다**(판정은 재측정이 살려 준다).
    changed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"remeasure": self.remeasure, "changed": self.changed, "reason": self.reason}


def rolling_plan(
    *, baseline: RollingBaseline | None, fingerprint: str, champion_model: str
) -> RollingPlan:
    """기준선 · 지금 평가셋 지문 · 지금 champion → 무엇을 할지.

    판정 순서에 뜻이 있다:

    1. **champion 이 없으면** 견줄 것이 없다(첫 라운드) — 재지 않는다. 그래도 옛 지문과 다르면
       갱신은 갱신이라 추이는 끊는다.
    2. **기준선이 없으면** 잰다 — 최근 평가셋을 이제 막 붙인 경우다. 끊을 추이는 없다.
    3. **지문이 달라졌으면** 잰다 + 추이를 끊는다.
    4. **champion 이 바뀌었는데 기준선이 옛 모델의 것이면** 잰다(승급 때 같이 적히므로 보통 안 생긴다 —
       원장·상태가 손으로 고쳐진 경우의 안전판).
    5. 그 밖에는 그대로 견준다.
    """
    if not champion_model:
        changed = baseline is not None and baseline.fingerprint != fingerprint
        return RollingPlan(
            remeasure=False,
            changed=changed,
            reason="champion 이 없어 최근 평가셋으로 견줄 것이 없습니다",
        )
    if baseline is None:
        return RollingPlan(
            remeasure=True, changed=False, reason="최근 평가셋 기준선이 없어 champion 을 잽니다"
        )
    if baseline.fingerprint != fingerprint:
        return RollingPlan(
            remeasure=True,
            changed=True,
            reason=(
                f"최근 평가셋이 갱신되었습니다({baseline.fingerprint} → {fingerprint}) — "
                "champion 을 새 평가셋에서 다시 재고 추이는 여기서 끊습니다"
            ),
        )
    if baseline.model != champion_model:
        return RollingPlan(
            remeasure=True,
            changed=False,
            reason="기준선이 지금 champion 의 것이 아니어서 다시 잽니다",
        )
    return RollingPlan(
        remeasure=False,
        changed=False,
        reason=f"최근 평가셋 그대로({fingerprint}) — 기준선을 그대로 씁니다",
    )


__all__ = [
    "FINGERPRINT_CHARS",
    "HASH_CHARS",
    "Entry",
    "RollingBaseline",
    "RollingPlan",
    "file_entry",
    "fingerprint",
    "rolling_plan",
    "split_entries",
    "split_fingerprint",
]
