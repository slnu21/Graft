"""은행 스냅샷 — 라운드 재현을 위한 **소스 id + 내용 해시 목록**(설계 §6.2).

**은행을 복사하지 않는다.** 두 가지 이유:

1. `inputs.bank` **경로**는 `pipeline_hash` 에 들어간다(T9 가 확인하고 남긴 주의) — 라운드마다 은행을
   복사해 새 경로에 두면 "같은 합성인가"를 해시로 비교할 수 없게 된다.
2. 은행은 수백 MB 가 되고 라운드마다 복사하면 디스크가 라운드 수에 비례해 는다.

대신 그 시점의 **내용**을 적어 둔다. 나중에 "라운드 5 는 무슨 결함으로 만들었나"를 물으면
스냅샷과 지금 은행을 견주어 **없어진 것 · 새로 들어온 것 · 바뀐 것**을 답한다.

해시는 소스마다 세 파일(크롭·마스크·메타)을 순서대로 넣어 만든다 — 마스크를 다듬거나 클래스·태그를
고쳐도 바뀐 것으로 잡힌다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from anograft.bank.bank import BANK_FILE, IMAGE_SUFFIX, MASK_SUFFIX, META_SUFFIX, BankError

SNAPSHOT_VERSION = 1
#: 해시 앞 16자만 적는다 — 충돌 걱정 없이 사람이 눈으로 비교할 수 있는 길이.
DIGEST_CHARS = 16


@dataclass(frozen=True)
class SourceEntry:
    id: str  # "<class>/<name>"
    cls: str
    digest: str


@dataclass(frozen=True)
class Snapshot:
    """어느 시점의 은행 내용. `bank` 는 참고용 경로이고 **비교에는 쓰지 않는다**(폴더를 옮겨도 같다)."""

    name: str
    created: str
    classes: list[str]
    sources: list[SourceEntry]
    bank: str = ""
    version: int = SNAPSHOT_VERSION

    @property
    def ids(self) -> set[str]:
        return {s.id for s in self.sources}

    def digest_of(self, source_id: str) -> str | None:
        for s in self.sources:
            if s.id == source_id:
                return s.digest
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "bank": self.bank,
            "created": self.created,
            "classes": list(self.classes),
            "sources": [{"id": s.id, "cls": s.cls, "digest": s.digest} for s in self.sources],
        }


@dataclass
class SnapshotDiff:
    """스냅샷 이후 무엇이 달라졌나. 셋 다 비면 같은 은행이다."""

    missing: list[str] = field(default_factory=list)  # 스냅샷엔 있는데 지금 없다(지워짐)
    added: list[str] = field(default_factory=list)  # 지금 있는데 스냅샷엔 없다(새로 들어옴)
    changed: list[str] = field(default_factory=list)  # 같은 id 인데 내용이 다르다(마스크 다듬기 등)

    @property
    def same(self) -> bool:
        return not (self.missing or self.added or self.changed)

    def text(self) -> str:
        if self.same:
            return "스냅샷과 같습니다."
        parts = []
        if self.missing:
            parts.append(f"없어짐 {len(self.missing)}")
        if self.added:
            parts.append(f"새로 들어옴 {len(self.added)}")
        if self.changed:
            parts.append(f"내용 바뀜 {len(self.changed)}")
        return " · ".join(parts)


def source_digest(bank_root: Path, source_id: str) -> str:
    """소스 하나의 내용 해시 — 크롭·마스크·메타 세 파일을 순서대로."""
    cls, _, name = source_id.partition("/")
    base = Path(bank_root) / cls / name
    digest = hashlib.sha256()
    for suffix in (IMAGE_SUFFIX, MASK_SUFFIX, META_SUFFIX):
        path = base.with_name(name + suffix)
        if not path.is_file():
            raise BankError(f"소스 파일이 없습니다: {path}")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:DIGEST_CHARS]


def take(bank_root: str | Path) -> Snapshot:
    """지금 은행 상태를 찍는다. 소스는 **이름 정렬 순**(은행 로더와 같은 규칙 — OS 열람 순서는 못 믿는다)."""
    from anograft.bank.bank import Bank

    root = Path(bank_root)
    if not (root / BANK_FILE).is_file():
        raise BankError(f"은행이 아닙니다(bank.yaml 없음): {root}")
    bank = Bank.load(root)
    entries = [
        SourceEntry(id=s.id, cls=s.cls, digest=source_digest(root, s.id))
        for s in sorted(bank.sources(), key=lambda s: s.id)
    ]
    return Snapshot(
        name=bank.name,
        bank=root.as_posix(),
        created=date.today().isoformat(),
        classes=list(bank.classes),
        sources=entries,
    )


def write(snapshot: Snapshot, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return out


def load(path: str | Path) -> Snapshot:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != SNAPSHOT_VERSION:
        raise BankError(f"모르는 스냅샷 버전입니다: {data.get('version')}")
    return Snapshot(
        name=str(data.get("name", "")),
        bank=str(data.get("bank", "")),
        created=str(data.get("created", "")),
        classes=[str(c) for c in data.get("classes", [])],
        sources=[
            SourceEntry(id=str(s["id"]), cls=str(s.get("cls", "")), digest=str(s["digest"]))
            for s in data.get("sources", [])
        ],
        version=int(data["version"]),
    )


def diff(snapshot: Snapshot, bank_root: str | Path) -> SnapshotDiff:
    """스냅샷 ↔ 지금 은행. 경로가 달라도 내용이 같으면 `same` 이다."""
    now = take(bank_root)
    now_by_id = {s.id: s.digest for s in now.sources}
    out = SnapshotDiff()
    for entry in snapshot.sources:
        current = now_by_id.get(entry.id)
        if current is None:
            out.missing.append(entry.id)
        elif current != entry.digest:
            out.changed.append(entry.id)
    out.added = sorted(now_by_id.keys() - snapshot.ids)
    return out
