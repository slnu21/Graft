"""유입 커서 — 지난 라운드 이후 **새로 들어온 이미지만** 스코어링한다 (설계 §2b.2, 작업 단위 T13).

지금까지 루프의 입력(`inputs.targets` · `loop.yaml` 의 `field`)은 **정적**이었다. 사람이 한 번 돌리는
배치 라운드에서는 그걸로 됐지만, 라인에서 이미지가 계속 들어오는 운영에서는 "지난번 이후 새로 들어온 것"을
알아야 한다. 그걸 아는 방법이 이 파일이다.

**커서를 mtime 단일 값으로 두지 않는다.** 늦게 도착한 파일(late arrival) · 되돌린 파일 · 시계 역행에
조용히 구멍이 난다("마지막 처리 시각보다 옛날이니 처리한 것" = 틀렸다). 정답은 **처리 이력(집합)** 이고,
그래서 파일 하나가 아니라 append-only JSONL 이다::

    <out>/processed.jsonl
    {"path":"…/f0.png","size":12345,"mtime":1790000000.5,"digest":"a1b2c3d4","round":2,"at":"…"}

한 줄이 "이 이미지를 이 라운드에서 스코어링했다"는 사실 하나다. 지우지 않고 더하기만 하므로 감사도 된다.

**신원은 두 겹**이다 — `(경로·크기·mtime)` 은 **빠른 길**이고(아는 파일을 매번 해시하면 4K 폴더에서 못 쓴다),
`(크기·내용 해시)` 가 **진짜 신원**이다. 그래서 파일을 다른 이름으로 복사해 와도 두 번 스코어링하지 않고,
같은 경로의 내용이 바뀌면 다시 스코어링한다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: 라운드 출력 루트에 놓이는 커서 파일 이름 (`loop.state.json` 과 나란히).
PROCESSED_FILE = "processed.jsonl"
#: 내용 해시 앞 8자(설계 §2b.2). 짧지만 **크기와 함께** 키로 쓰기 때문에 실무 규모에서 충돌하지 않는다.
DIGEST_CHARS = 8
#: 해시는 한 번에 1 MiB 씩 읽는다 — 4096×2732 이미지를 통째로 메모리에 올리지 않는다.
_CHUNK = 1 << 20


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize(path: str | Path) -> str:
    """경로를 커서에 적을 형태로 — 절대 posix. 대소문자까지 맞추지는 않는다(내용 해시가 진짜 신원)."""
    try:
        return Path(path).resolve().as_posix()
    except OSError:  # 네트워크 드라이브가 끊겼을 때도 커서는 돌아야 한다
        return Path(path).as_posix()


# --------------------------------------------------------------------------------------
# 1. 파일 하나의 신원
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileStamp:
    """한 파일의 신원 — 경로 · 크기 · mtime · 내용 해시 앞 8.

    `digest` 가 빈 문자열이면 **아직 읽지 않았다**는 뜻이다(빠른 길에서 걸러진 파일).
    """

    path: str
    size: int
    mtime: float
    digest: str = ""

    @property
    def stat_key(self) -> str:
        """빠른 길의 키 — 경로·크기·mtime 이 그대로면 같은 파일로 본다(git 의 stat 캐시와 같은 생각)."""
        return f"{self.path}|{self.size}|{self.mtime:.3f}"

    @property
    def content_key(self) -> str:
        """진짜 신원 — 이름이 달라도 내용이 같으면 같은 이미지다. 빈 해시는 키가 없다."""
        return f"{self.size}:{self.digest}" if self.digest else ""

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    def to_json(self, *, round_no: int = 0, at: str | None = None) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime": round(self.mtime, 3),
            "digest": self.digest,
            "round": int(round_no),
            "at": at or _now(),
        }


def content_digest(path: str | Path, *, chars: int = DIGEST_CHARS) -> str:
    """내용 해시 앞 `chars` 자. 순차 읽기 한 번뿐이라 큰 이미지에서도 싸다."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:chars]


def stat_stamp(path: str | Path) -> FileStamp:
    """해시 없이 stat 만 — 빠른 길."""
    st = Path(path).stat()
    return FileStamp(path=normalize(path), size=int(st.st_size), mtime=round(st.st_mtime, 3))


def stamp(path: str | Path) -> FileStamp:
    """내용까지 읽은 완전한 신원."""
    return replace(stat_stamp(path), digest=content_digest(path))


def stamp_all(paths: Iterable[str | Path]) -> tuple[list[FileStamp], list[str]]:
    """여러 파일을 찍는다. 읽을 수 없는 것은 **건너뛰고 사유를 남긴다**(fail-soft)."""
    out: list[FileStamp] = []
    warns: list[str] = []
    for p in paths:
        try:
            out.append(stamp(p))
        except OSError as exc:
            warns.append(f"파일을 읽을 수 없어 커서에 적지 않습니다: {p} ({exc})")
    return out, warns


# --------------------------------------------------------------------------------------
# 2. 처리 이력 (append-only)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedEntry:
    """커서 한 줄 — "이 이미지를 이 라운드에서 스코어링했다"."""

    stamp: FileStamp
    round: int = 0
    at: str = ""

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ProcessedEntry | None:
        """깨진 줄은 ``None`` — 커서 한 줄이 상하면 루프가 멈추는 게 아니라 그 줄만 버린다."""
        try:
            path = str(data["path"])
            size = int(data["size"])
            mtime = float(data.get("mtime", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            stamp=FileStamp(
                path=path, size=size, mtime=mtime, digest=str(data.get("digest", "") or "")
            ),
            round=int(data.get("round", 0) or 0),
            at=str(data.get("at", "") or ""),
        )


@dataclass
class ProcessedLog:
    """커서 전체. 두 겹의 색인(stat · 내용)을 들고 있어 판정이 O(1) 이다."""

    entries: list[ProcessedEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _stat: set[str] = field(default_factory=set, init=False, repr=False)
    _content: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self._stat = {e.stamp.stat_key for e in self.entries}
        self._content = {e.stamp.content_key for e in self.entries if e.stamp.content_key}

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def last_round(self) -> int:
        return max((e.round for e in self.entries), default=0)

    def stat_seen(self, s: FileStamp) -> bool:
        return s.stat_key in self._stat

    def content_seen(self, s: FileStamp) -> bool:
        return bool(s.content_key) and s.content_key in self._content

    def before_round(self, number: int) -> ProcessedLog:
        """``loop tick --since n`` — 라운드 n 부터의 이력을 **없는 것으로 보고** 다시 처리한다.

        어댑터를 갈거나 버그를 고친 뒤 "그때 스코어링한 것들을 다시"가 필요하다. 커서를 지우지 않고
        읽을 때 걸러내므로 이력은 그대로 남는다(감사가 끊기지 않는다).
        """
        return ProcessedLog(
            entries=[e for e in self.entries if e.round < number], warnings=list(self.warnings)
        )


def is_seen(s: FileStamp, log: ProcessedLog) -> bool:
    """이미 처리한 파일인가 — **판정 규칙이 사는 단 한 곳**(순수).

    빠른 길(stat 일치)이 먼저이고, 그게 아니면 내용으로 본다. 해시를 아직 읽지 않은 stamp 는
    내용 판정을 하지 않는다(그래서 호출부가 "빠른 길 → 해시 → 다시 판정" 순서로 쓸 수 있다).
    """
    return log.stat_seen(s) or log.content_seen(s)


def unprocessed(stamps: Sequence[FileStamp], log: ProcessedLog) -> list[FileStamp]:
    """아직 처리하지 않은 것만 (설계 §3 `unprocessed`) — 순수, 파일을 읽지 않는다."""
    return [s for s in stamps if not is_seen(s, log)]


def dedupe_stems(stamps: Sequence[FileStamp]) -> tuple[list[FileStamp], list[FileStamp]]:
    """같은 이름(stem)이 둘이면 이번 라운드에는 **하나만** 넣는다.

    예측 출력은 ``scores/<stem>.json`` 이라 이름이 겹치면 **조용히 덮어쓴다**(§3.1 함정 — MVTec 폴더별
    ``000.png`` 25장이 5장이 됐다). 어댑터는 `count: 25` 를 보고하므로 아무도 모른다.

    미룬 것은 커서에 적지 않으므로 **다음 라운드에 들어온다**(먼저 들어간 쪽이 그때는 이미 처리됨) —
    한 라운드에 하나씩, 경고와 함께 저절로 풀린다. 버리는 게 아니라 미루는 것이 요점이다.
    """
    kept: list[FileStamp] = []
    deferred: list[FileStamp] = []
    seen: set[str] = set()
    for s in stamps:
        if s.stem in seen:
            deferred.append(s)
            continue
        seen.add(s.stem)
        kept.append(s)
    return kept, deferred


def read_processed(path: str | Path) -> ProcessedLog:
    """커서를 읽는다. 없으면 빈 이력, 깨진 줄은 사유와 함께 건너뛴다(fail-soft)."""
    p = Path(path)
    if not p.is_file():
        return ProcessedLog()
    entries: list[ProcessedEntry] = []
    warns: list[str] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return ProcessedLog(warnings=[f"{p} 를 읽을 수 없습니다: {exc}"])
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except ValueError:
            warns.append(f"{p.name} {n}번째 줄을 읽지 못했습니다(건너뜀)")
            continue
        entry = ProcessedEntry.from_json(data) if isinstance(data, Mapping) else None
        if entry is None:
            warns.append(f"{p.name} {n}번째 줄에 경로·크기가 없습니다(건너뜀)")
            continue
        entries.append(entry)
    return ProcessedLog(entries=entries, warnings=warns)


def append_processed(
    path: str | Path, stamps: Sequence[FileStamp], *, round_no: int = 0, at: str | None = None
) -> int:
    """처리한 것을 커서에 **더한다**(덮어쓰지 않는다). 돌려주는 값은 적은 줄 수."""
    if not stamps:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    when = at or _now()
    lines = [json.dumps(s.to_json(round_no=round_no, at=when), ensure_ascii=False) for s in stamps]
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


# --------------------------------------------------------------------------------------
# 3. 이번 라운드가 볼 것 고르기
# --------------------------------------------------------------------------------------


@dataclass
class IngestPlan:
    """이번 라운드가 스코어링할 것 · 건너뛴 것 · 미룬 것."""

    fresh: list[FileStamp] = field(default_factory=list)
    seen: int = 0
    deferred: list[FileStamp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        return [Path(s.path) for s in self.fresh]

    @property
    def empty(self) -> bool:
        return not self.fresh

    def line(self) -> str:
        parts = [f"새 이미지 {len(self.fresh)}장", f"이미 처리 {self.seen}장"]
        if self.deferred:
            parts.append(f"이름 겹쳐 다음 라운드로 {len(self.deferred)}장")
        return " · ".join(parts)


def plan_ingest(candidates: Sequence[str | Path], log: ProcessedLog) -> IngestPlan:
    """후보에서 아직 처리하지 않은 것을 고른다 — **stat 먼저, 해시는 모르는 파일만**.

    아는 파일을 매번 해시하면 4K 이미지 수천 장 폴더에서 tick 이 몇 분씩 걸린다. 그래서 빠른 길을
    먼저 보고, 거기서 걸리지 않은 것만 읽는다(= 새로 들어왔거나 내용이 바뀐 것).
    """
    plan = IngestPlan(warnings=list(log.warnings))
    stamps: list[FileStamp] = []
    for candidate in candidates:
        try:
            quick = stat_stamp(candidate)
        except OSError as exc:
            plan.warnings.append(f"파일을 볼 수 없어 건너뜁니다: {candidate} ({exc})")
            continue
        if is_seen(quick, log):  # 빠른 길 — 해시를 읽지 않는다
            plan.seen += 1
            continue
        try:
            full = replace(quick, digest=content_digest(candidate))
        except OSError as exc:
            plan.warnings.append(f"파일을 읽을 수 없어 건너뜁니다: {candidate} ({exc})")
            continue
        if is_seen(full, log):  # 이름만 다른 같은 이미지 · 되돌린 파일
            plan.seen += 1
            continue
        stamps.append(full)
    plan.fresh, plan.deferred = dedupe_stems(stamps)
    for s in plan.deferred:
        plan.warnings.append(
            f"이름이 겹쳐 이번 라운드에서 미룹니다(다음 라운드에 들어옵니다): {s.path}"
        )
    return plan


def quick_new_count(candidates: Sequence[str | Path], log: ProcessedLog) -> int:
    """해시를 읽지 않고 **새로 들어온 듯한** 장수만 센다 — 트리거 판정용(T14).

    stat 만 보므로 이름을 바꿔 복사한 사진을 새것으로 셀 수 있다(과다 계수). "N장 모이면 돈다"는 임계에는
    충분하고, 정확한 선별은 predict 직전의 `plan_ingest` 가 한다 — **트리거가 라운드마다 폴더 전체를
    해시하면 스케줄러가 부르는 5분마다 디스크를 통째로 읽는다.**
    """
    n = 0
    for candidate in candidates:
        try:
            quick = stat_stamp(candidate)
        except OSError:
            continue
        if not log.stat_seen(quick):
            n += 1
    return n


__all__ = [
    "DIGEST_CHARS",
    "PROCESSED_FILE",
    "FileStamp",
    "IngestPlan",
    "ProcessedEntry",
    "ProcessedLog",
    "append_processed",
    "content_digest",
    "dedupe_stems",
    "is_seen",
    "normalize",
    "plan_ingest",
    "quick_new_count",
    "read_processed",
    "stamp",
    "stamp_all",
    "stat_stamp",
    "unprocessed",
]
