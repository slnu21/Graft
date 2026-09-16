"""출력 폴더 병합 — ``anograft run`` 정본(이미지+마스크+사이드카+manifest) 여러 개를 한 학습셋으로. Qt 없음 — CLI ``dataset merge`` 용.

- 입력 루트마다 파일 이름 앞에 ``d<k>_``(``--prefix`` 로 바꿈)를 붙여 충돌을 피우고, 합성 행은 0 부터 다시 번호를 매긴다
  (manifest ``index`` = 사이드카 ``index``; 원래 자리는 사이드카 ``merged_from: {root, index}``). 정상 행·skipped 행도 그대로
  옮긴다(skipped 는 파일이 없으므로 행만, 정리본을 넣으면 애초에 없다).
- **같은 writer 형식 · 같은 클래스 이름(순서까지)** 이어야 한다 — YOLO 라벨의 class id 가 은행 순서라서. 다르면 ``MergeError``.
  근거는 ``recipe.resolved.yaml`` 의 ``output.writer.format`` 과 ``data.yaml`` 의 ``names``(없으면 사이드카 ``defects[].source.class_id``
  로는 판별 못 하므로 형식만 본다).
- 형식 파일: ``yolo`` 는 ``labels/<name>.txt`` 도 접두어 · ``coco`` 는 ``annotations.json`` 을 images/annotations id 오프셋으로 이어 붙임 ·
  ``mvtec`` 은 ``mvtec/<cat>/…/<name>.png`` 파일명에 접두어(사이드카 writer 항목의 경로도 함께 고침). ``data.yaml``·``recipe.resolved.yaml``
  은 첫 루트 것을 복사하고 ``merge.json`` 에 루트별 요약을 남긴다. ``review.csv`` 는 새 index 로 합친다.
- 원본은 건드리지 않는다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from anograft.io.manifest import MANIFEST_FILE, read_manifest, write_manifest
from anograft.io.prune import REVIEW_FILE, read_review, write_review

MERGE_FILE = "merge.json"
PATH_COLUMNS: tuple[str, ...] = ("image", "mask", "sidecar", "label")


class MergeError(ValueError):
    pass


@dataclass
class MergeSummary:
    out: Path
    roots: list[Path]
    synthetic: int = 0
    normals: int = 0
    normals_dropped: int = 0  # dedupe_normals 로 뺀 정상
    skipped: int = 0
    files: int = 0
    per_root: list[dict[str, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def prefixed(rel: str, prefix: str) -> str:
    """상대경로의 **파일 이름**에만 접두어 — ``images/000000.png`` → ``images/d0_000000.png``. 빈 문자열·``#`` 참조(coco)는 그대로."""
    if not rel or "#" in rel or rel.endswith("/"):
        return rel
    p = Path(rel)
    return (p.parent / f"{prefix}{p.name}").as_posix()


def _writer_format(root: Path) -> str | None:
    rr = root / "recipe.resolved.yaml"
    if not rr.is_file():
        return None
    try:
        doc = yaml.safe_load(rr.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    return str(((doc.get("output") or {}).get("writer") or {}).get("format") or "pairs")


def _names(root: Path) -> list[str] | None:
    dy = root / "data.yaml"
    if not dy.is_file():
        return None
    try:
        doc = yaml.safe_load(dy.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    names = doc.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names] if isinstance(names, list) else None


def check_compatible(roots: list[Path]) -> tuple[str, list[str] | None]:
    """writer 형식·클래스 이름이 전부 같은지 — 다르면 ``MergeError``. 반환 (형식, names)."""
    if len(roots) < 2:
        raise MergeError("병합할 출력 폴더를 둘 이상 주세요")
    fmts, names_list = [], []
    for r in roots:
        if not (r / MANIFEST_FILE).is_file():
            raise MergeError(f"manifest.csv 가 없습니다: {r}")
        fmts.append(_writer_format(r) or "pairs")
        names_list.append(_names(r))
    if len(set(fmts)) > 1:
        raise MergeError(
            "writer 형식이 다릅니다: "
            + " · ".join(f"{r.name}={f}" for r, f in zip(roots, fmts, strict=True))
        )
    known = [n for n in names_list if n is not None]
    if known and any(n != known[0] for n in known):
        raise MergeError(
            "클래스 이름(data.yaml names)이 다릅니다 — YOLO class id 가 어긋납니다: "
            + " · ".join(f"{r.name}={n}" for r, n in zip(roots, names_list, strict=True))
        )
    return fmts[0], known[0] if known else None


def _copy(src: Path, dst: Path, summary: MergeSummary) -> bool:
    if not src.is_file():
        summary.warnings.append(f"파일 없음: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    summary.files += 1
    return True


def _load_sidecar(src: Path, summary: MergeSummary) -> dict[str, Any]:
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        summary.warnings.append(f"사이드카 읽기 실패 {src.name}: {e}")
        return {}


def writer_files(doc: dict[str, Any], row: dict[str, str]) -> list[str]:
    """사이드카 writer 항목이 가리키는 형식 파일(정본 image/mask 와 다른 것만) — yolo ``label`` · mvtec ``image``/``mask``."""
    w = doc.get("writer")
    out: list[str] = []
    if not isinstance(w, dict):
        return out
    for key in ("label", "image", "mask"):
        rel = w.get(key)
        if (
            isinstance(rel, str)
            and rel
            and "#" not in rel
            and not rel.endswith("/")
            and rel not in (row.get("image"), row.get("mask"))
            and rel not in out
        ):
            out.append(rel)
    return out


def _write_sidecar(
    doc: dict[str, Any],
    dst: Path,
    *,
    prefix: str,
    new_index: int | None,
    root_name: str,
    old_index: str,
    summary: MergeSummary,
) -> None:
    """index·merged_from·writer 경로를 고쳐 쓴다."""
    doc = dict(doc)
    if new_index is not None:
        doc["index"] = new_index
    doc["merged_from"] = {"root": root_name, "index": old_index}
    w = doc.get("writer")
    if isinstance(w, dict):
        w = dict(w)
        for key in ("label", "image", "mask"):
            if isinstance(w.get(key), str):
                w[key] = prefixed(w[key], prefix)
        doc["writer"] = w
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    summary.files += 1


def merge_datasets(
    roots: list[str | Path],
    out: str | Path,
    *,
    prefix: str = "d{k}_",
    dedupe_normals: bool = False,
) -> MergeSummary:
    """``roots`` 를 ``out`` 으로 병합. ``prefix`` 의 ``{k}`` 는 루트 순번. ``dedupe_normals`` 면 같은 대상(manifest ``target``)의
    정상 이미지는 첫 루트 것만(같은 정상 폴더로 돌린 레시피 여러 개를 합칠 때 정상이 n배로 불지 않게)."""
    roots_p = [Path(r) for r in roots]
    out_p = Path(out)
    fmt, _ = check_compatible(roots_p)
    if any(out_p.resolve() == r.resolve() for r in roots_p):
        raise MergeError("병합 결과는 입력과 다른 폴더여야 합니다")
    summary = MergeSummary(out=out_p, roots=roots_p)
    out_p.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, str]] = []
    review_out: dict[str, tuple[str, str]] = {}
    coco: dict[str, Any] | None = None
    next_index = 0
    seen_targets: set[str] = set()
    for k, root in enumerate(roots_p):
        pre = prefix.format(k=k)
        counts = {"synthetic": 0, "normals": 0, "skipped": 0}
        skipped_normal_names: set[str] = set()  # coco images 에서도 빼기 위해
        review = read_review(root / REVIEW_FILE)
        for r in read_manifest(root / MANIFEST_FILE):
            row = dict(r)
            status = row.get("status", "")
            old_index = str(row.get("index", ""))
            new_index: int | None = None
            if status == "skipped":
                counts["skipped"] += 1
                summary.skipped += 1
                new_index = next_index
                next_index += 1
                row["index"] = str(new_index)
                rows_out.append(row)
                continue
            if status == "ok":
                counts["synthetic"] += 1
                summary.synthetic += 1
                new_index = next_index
                next_index += 1
                row["index"] = str(new_index)
                if old_index in review:
                    review_out[str(new_index)] = review[old_index]
            else:
                target = str(row.get("target", ""))
                if dedupe_normals and target:
                    if target in seen_targets:
                        counts["normals_dropped"] = counts.get("normals_dropped", 0) + 1
                        summary.normals_dropped += 1
                        skipped_normal_names.add(Path(r.get("image", "")).name)
                        continue
                    seen_targets.add(target)
                counts["normals"] += 1
                summary.normals += 1
            for col in PATH_COLUMNS:
                row[col] = prefixed(row.get(col, ""), pre)
            for col in ("image", "mask"):
                if r.get(col):
                    _copy(root / r[col], out_p / row[col], summary)
            extra: list[
                str
            ] = []  # 형식 writer 파일(yolo labels · mvtec 레이아웃 사본) — 정본과 별개
            label = r.get("label", "")
            if label and "#" not in label and not label.endswith("/"):
                extra.append(label)
            if r.get("sidecar") and (root / r["sidecar"]).is_file():
                doc = _load_sidecar(root / r["sidecar"], summary)
                extra += [rel for rel in writer_files(doc, r) if rel not in extra]
                _write_sidecar(
                    doc,
                    out_p / row["sidecar"],
                    prefix=pre,
                    new_index=new_index,
                    root_name=root.name,
                    old_index=old_index,
                    summary=summary,
                )
            for rel in extra:
                _copy(root / rel, out_p / prefixed(rel, pre), summary)
            rows_out.append(row)
        ann = root / "annotations.json"
        if fmt == "coco" and ann.is_file():
            doc = json.loads(ann.read_text(encoding="utf-8"))
            if coco is None:
                coco = {"images": [], "annotations": [], "categories": doc.get("categories", [])}
                for key in doc:
                    if key not in coco:
                        coco[key] = doc[key]
            elif [c.get("name") for c in doc.get("categories", [])] != [
                c.get("name") for c in coco["categories"]
            ]:
                raise MergeError(f"coco categories 가 다릅니다: {root.name}")
            img_off = max((im["id"] for im in coco["images"]), default=0)
            ann_off = max((a["id"] for a in coco["annotations"]), default=0)
            for im in doc.get("images", []):
                if Path(str(im.get("file_name", ""))).name in skipped_normal_names:
                    continue
                im = dict(im)
                im["id"] = int(im["id"]) + img_off
                im["file_name"] = prefixed(im.get("file_name", ""), pre)
                coco["images"].append(im)
            for a in doc.get("annotations", []):
                a = dict(a)
                a["id"] = int(a["id"]) + ann_off
                a["image_id"] = int(a["image_id"]) + img_off
                coco["annotations"].append(a)
        summary.per_root.append(counts)
    write_manifest(out_p / MANIFEST_FILE, rows_out)
    for name in ("recipe.resolved.yaml", "data.yaml"):
        if (roots_p[0] / name).is_file():
            _copy(roots_p[0] / name, out_p / name, summary)
    if coco is not None:
        (out_p / "annotations.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        summary.files += 1
    if review_out:
        write_review(out_p / REVIEW_FILE, review_out)
    (out_p / MERGE_FILE).write_text(
        json.dumps(
            {
                "roots": [r.as_posix() for r in roots_p],
                "prefix": prefix,
                "writer": fmt,
                "per_root": summary.per_root,
                "synthetic": summary.synthetic,
                "normals": summary.normals,
                "normals_dropped": summary.normals_dropped,
                "skipped": summary.skipped,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return summary
