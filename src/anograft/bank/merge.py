"""은행 병합(v0.7.x) — 여러 은행(제품별·촬영별)을 하나로. ``anograft bank merge a b --out c``.

- 소스는 **다시 크롭하지 않고** 세 파일(png · mask.png · json)을 그대로 복사한다(크롭 margin·마스크·추정 점수 보존). 메타는
  클래스 이름 바꾸기(``--rename old=new``)·태그 추가(``--tags``)·은행 기본 ``um_per_px`` 를 소스 메타에 **실체화**(은행마다 기본값이
  달라도 병합 뒤 소스별 값이 남게)만 손댄다.
- 클래스는 **이름 기준 병합**(``BankWriter.ensure_classes`` — 대상에 있던 순서 유지, 새 이름은 끝에). id 순서가 바뀔 수 있는 건
  새 이름뿐이라 기존 대상 은행의 ``data.yaml`` 은 흔들리지 않는다.
- id 충돌(같은 클래스·같은 이름)은 ``-dup<n>`` + 경고(``BankWriter._unique_id`` 와 같은 규칙) — 은행 A·B 에 같은 원본 이름이 있으면
  둘 다 남는다. 같은 파일을 두 번 병합해도 덮어쓰지 않는다(중복 소스가 생긴다 — 의도: 병합은 누적).
- 대상이 이미 은행이면 이어 쓴다. 대상 = 소스 중 하나면 거부(자기 자신에 병합).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anograft.bank.bank import (
    BANK_FILE,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    META_SUFFIX,
    read_bank_meta,
)
from anograft.bank.importers.common import BankWriter

Logger = Callable[[str], None]


class MergeError(RuntimeError):
    pass


@dataclass
class MergeSummary:
    out: Path
    banks: int = 0
    copied: int = 0
    duplicates: int = 0
    renamed: int = 0
    classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_rename(specs: Sequence[str]) -> dict[str, str]:
    """``old=new`` 목록 → dict. 형식이 틀리면 ``MergeError``."""
    out: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise MergeError(f"--rename 은 old=new 형식입니다: {spec!r}")
        old, new = (s.strip() for s in spec.split("=", 1))
        if not old or not new or "/" in new or "\\" in new:
            raise MergeError(f"--rename 값이 비었거나 경로 구분자가 있습니다: {spec!r}")
        out[old] = new
    return out


def merge_banks(
    sources: Sequence[str | Path],
    out: str | Path,
    *,
    tags: Sequence[str] = (),
    rename: Mapping[str, str] | None = None,
    log: Logger | None = None,
) -> MergeSummary:
    log = log or (lambda _m: None)
    rename = dict(rename or {})
    out_p = Path(out)
    srcs = [Path(s) for s in sources]
    if not srcs:
        raise MergeError("병합할 은행이 없습니다")
    for s in srcs:
        if not (s / BANK_FILE).is_file():
            raise MergeError(f"은행이 아닙니다 ({BANK_FILE} 없음): {s}")
        if s.resolve() == out_p.resolve():
            raise MergeError(f"대상이 소스와 같습니다: {s}")
    writer = BankWriter(out_p, log=log)
    summary = MergeSummary(out=out_p)
    extra_tags = [t.strip() for t in tags if t.strip()]

    for s in srcs:
        meta = read_bank_meta(s / BANK_FILE)
        default_um = meta.get("um_per_px")
        classes = [str(c) for c in meta.get("classes") or []]
        # 폴더에 있는 클래스도(bank.yaml 에 빠졌어도)
        for d in sorted(p for p in s.iterdir() if p.is_dir()):
            if d.name not in classes and any(d.glob(f"*{META_SUFFIX}")):
                classes.append(d.name)
        for cls in classes:
            new_cls = rename.get(cls, cls)
            if new_cls != cls:
                summary.renamed += 1
            writer.ensure_classes([new_cls])
            cdir = s / cls
            if not cdir.is_dir():
                continue
            for meta_file in sorted(cdir.glob(f"*{META_SUFFIX}")):
                sid = meta_file.name[: -len(META_SUFFIX)]
                img, msk = cdir / f"{sid}{IMAGE_SUFFIX}", cdir / f"{sid}{MASK_SUFFIX}"
                if not (img.is_file() and msk.is_file()):
                    summary.warnings.append(
                        f"{s.name}/{cls}/{sid}: 이미지·마스크 파일 없음 — 건너뜀"
                    )
                    log(summary.warnings[-1])
                    continue
                try:
                    m = json.loads(meta_file.read_text(encoding="utf-8"))
                except ValueError as e:
                    summary.warnings.append(f"{s.name}/{cls}/{sid}: 메타 읽기 실패({e}) — 건너뜀")
                    log(summary.warnings[-1])
                    continue
                new_id = writer._unique_id(new_cls, sid)
                if new_id != sid:
                    summary.duplicates += 1
                dst = out_p / new_cls
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(img, dst / f"{new_id}{IMAGE_SUFFIX}")
                shutil.copyfile(msk, dst / f"{new_id}{MASK_SUFFIX}")
                m["class"] = new_cls
                if m.get("um_per_px") is None and default_um is not None:
                    m["um_per_px"] = default_um  # 은행 기본값을 소스에 실체화
                old_tags = [str(t) for t in m.get("tags") or []]
                m["tags"] = old_tags + [t for t in extra_tags if t not in old_tags]
                m["merged_from"] = f"{s.as_posix()}::{cls}/{sid}"
                (dst / f"{new_id}{META_SUFFIX}").write_text(
                    json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                summary.copied += 1
        summary.banks += 1
    writer.finish(
        {
            "importer": "merge",
            "sources": [s.as_posix() for s in srcs],
            "rename": rename,
            "tags": extra_tags,
            "n_sources": summary.copied,
        }
    )
    summary.classes = writer.classes
    summary.warnings += [w for w in writer.stats.warnings if w not in summary.warnings]
    return summary
