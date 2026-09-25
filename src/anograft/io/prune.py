"""검수 결과 반영 — ``review.csv``(index, verdict, note) 와 정리본 내보내기(``prune_dataset``). Qt 없음 — 검수 탭·CLI ``dataset prune`` 공용.

- ``review.csv`` 는 출력 루트에 두고 verdict ∈ ``accept | reject | ""``(미검수). 파일이 없으면 전부 미검수.
- ``prune_dataset(root, out, review)`` 는 **반려(reject)만 뺀** 사본을 만든다(미검수는 남긴다 — "검수 안 한 것"을 버리는 건
  검수자의 선택이어야 한다; ``drop_unreviewed=True`` 면 채택만). 정상 이미지 행은 항상 복사. manifest 는 남은 행만으로 다시 쓰고,
  ``recipe.resolved.yaml``·``data.yaml`` 은 그대로 복사, ``annotations.json``(coco)은 남은 이미지로 걸러 다시 쓴다.
  ``mvtec/`` 레이아웃 사본은 manifest ``label`` 열이 가리키는 파일만 복사한다.
- 원본은 건드리지 않는다(정리본은 새 폴더).
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from anograft.io.manifest import MANIFEST_FILE, read_manifest, write_manifest

REVIEW_FILE = "review.csv"
#: 루프의 검토 대기 폴더만 가지는 표식(`loop/queue.py` 가 쓴다) — 검수 화면이 "합성"과
#: "검토 대기"를 가르는 데 쓴다. 이름의 한 원천을 여기 둔다(`review.csv` 와 같은 폴더에 산다).
QUEUE_FILE = "queue.csv"
VERDICTS: tuple[str, ...] = ("accept", "reject", "")
COPY_AS_IS: tuple[str, ...] = ("recipe.resolved.yaml", "data.yaml")


class PruneError(ValueError):
    pass


def read_review(path: str | Path) -> dict[str, tuple[str, str]]:
    """``{index: (verdict, note)}``. 없으면 빈 dict."""
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, tuple[str, str]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict") or "").strip()
            if v not in VERDICTS:
                raise PruneError(f"{p.name}: verdict {v!r} (accept | reject | 빈칸)")
            out[str(row.get("index", "")).strip()] = (v, row.get("note") or "")
    return out


def write_review(path: str | Path, verdicts: dict[str, tuple[str, str]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "verdict", "note"])
        for idx in sorted(verdicts, key=lambda s: (len(s), s)):
            v, note = verdicts[idx]
            w.writerow([idx, v, note])
    return p


@dataclass
class PruneSummary:
    out: Path
    kept: int = 0
    dropped: int = 0
    normals: int = 0
    skipped: int = 0
    files: int = 0
    warnings: list[str] = field(default_factory=list)


def _copy(root: Path, out: Path, rel: str, summary: PruneSummary) -> None:
    if not rel:
        return
    src = root / rel
    if not src.is_file():
        summary.warnings.append(f"파일 없음: {rel}")
        return
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    summary.files += 1


def prune_dataset(
    root: str | Path,
    out: str | Path,
    review: dict[str, tuple[str, str]] | None = None,
    *,
    drop_unreviewed: bool = False,
    drop_indices: Iterable[str] = (),
) -> PruneSummary:
    """반려를 뺀 정리본. ``review`` 가 None 이면 ``<root>/review.csv``. ``drop_indices`` 는 판정과 무관하게 뺄 index
    (예: 검수 세션의 조명 뒤집힘 의심 — ``dataset prune --drop-flipped``)."""
    drop_set = {str(i) for i in drop_indices}
    root, out = Path(root), Path(out)
    if not (root / MANIFEST_FILE).is_file():
        raise PruneError(f"manifest.csv 가 없습니다: {root}")
    if out.resolve() == root.resolve():
        raise PruneError("정리본은 원본과 다른 폴더여야 합니다")
    if review is None:
        review = read_review(root / REVIEW_FILE)
    summary = PruneSummary(out=out)
    rows = read_manifest(root / MANIFEST_FILE)
    kept_rows: list[dict[str, str]] = []
    kept_images: set[str] = set()
    for r in rows:
        status = r.get("status", "")
        idx = str(r.get("index", ""))
        if status == "skipped":
            summary.skipped += 1
            continue  # 파일이 없는 행 — 정리본 manifest 에도 남기지 않는다
        if status == "ok":
            verdict = review.get(idx, ("", ""))[0]
            if verdict == "reject" or (drop_unreviewed and verdict != "accept") or idx in drop_set:
                summary.dropped += 1
                continue
            summary.kept += 1
        else:
            summary.normals += 1
        kept_rows.append(r)
        for col in ("image", "mask", "sidecar"):
            _copy(root, out, r.get(col, ""), summary)
        label = r.get("label", "")
        if label and "#" not in label and not label.endswith("/"):
            _copy(root, out, label, summary)
        # 형식 writer 사본(mvtec 레이아웃 등)은 사이드카 writer 항목이 가리킨다
        sc = r.get("sidecar", "")
        if sc and (root / sc).is_file():
            try:
                w = json.loads((root / sc).read_text(encoding="utf-8")).get("writer") or {}
            except ValueError:
                w = {}
            for key in ("image", "mask"):
                rel = w.get(key)
                if isinstance(rel, str) and rel and rel != r.get("image") and rel != r.get("mask"):
                    _copy(root, out, rel, summary)
        if r.get("image"):
            kept_images.add(Path(r["image"]).name)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / MANIFEST_FILE, kept_rows)
    _filter_queue_csv(root, out, {str(r.get("index", "")) for r in kept_rows}, summary)
    for name in COPY_AS_IS:
        if (root / name).is_file():
            _copy(root, out, name, summary)
    ann = root / "annotations.json"
    if ann.is_file():
        doc = json.loads(ann.read_text(encoding="utf-8"))
        images = [
            im for im in doc.get("images", []) if Path(im.get("file_name", "")).name in kept_images
        ]
        ids = {im["id"] for im in images}
        doc["images"] = images
        doc["annotations"] = [a for a in doc.get("annotations", []) if a.get("image_id") in ids]
        (out / "annotations.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        summary.files += 1
    if review:
        write_review(out / REVIEW_FILE, {k: v for k, v in review.items() if v[0] != "reject"})
    return summary


def _filter_queue_csv(root: Path, out: Path, kept: set[str], summary: PruneSummary) -> None:
    """검토 대기 표식(``queue.csv``)도 **남긴 행만** 옮긴다.

    안 옮기면 정리본이 자기를 다시 "합성"이라 부르고(표식이 없으니까), 그대로 복사하면 반려한 항목의
    사유가 남아 목록과 어긋난다. 열 구성은 쓴 쪽(`loop/queue.py`)이 정하므로 그대로 따른다.
    """
    src = root / QUEUE_FILE
    if not src.is_file():
        return
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [r for r in reader if str(r.get("index", "")) in kept]
    if not fields:
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / QUEUE_FILE).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary.files += 1
