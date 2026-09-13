"""``manifest.csv`` (설계 §8.4) — 헤더 위 주석 없음(pandas 바로 읽기). 파이프라인 해시는 ``recipe.resolved.yaml``에.

열: ``index,status,image,mask,sidecar,label,target,classes,source_ids,n_defects,area_px,blend,fallback,reason``
(``classes``·``source_ids``는 ``;`` 구분 — 이미지에 결함이 여럿일 수 있다)
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MANIFEST_FILE = "manifest.csv"
COLUMNS: tuple[str, ...] = (
    "index",
    "status",
    "image",
    "mask",
    "sidecar",
    "label",
    "target",
    "classes",
    "source_ids",
    "n_defects",
    "area_px",
    "blend",
    "fallback",
    "reason",
)


def row_from_sidecar(
    index: int | str,
    status: str,
    sidecar: Mapping[str, Any],
    *,
    image: str = "",
    mask: str = "",
    meta: str = "",
    label: str = "",
    reason: str | None = None,
) -> dict[str, Any]:
    """사이드카(§8.3)에서 manifest 한 행을 만든다. skipped면 파일 열은 비어 있다."""
    defects = [d for d in sidecar.get("defects", []) if "gt" in d]
    classes = ";".join(str(d.get("source", {}).get("class", "")) for d in defects)
    source_ids = ";".join(str(d.get("source", {}).get("source_id", "")) for d in defects)
    blends = {str(d.get("blend", {}).get("method", "")) for d in defects}
    fallback = any(bool(d.get("blend", {}).get("fallback")) for d in defects)
    return {
        "index": index,
        "status": status,
        "image": image,
        "mask": mask,
        "sidecar": meta,
        "label": label,
        "target": sidecar.get("target", {}).get("file", ""),
        "classes": classes,
        "source_ids": source_ids,
        "n_defects": len(defects),
        "area_px": sidecar.get("gtmask", {}).get("area_px_total", 0),
        "blend": ";".join(sorted(b for b in blends if b)),
        "fallback": int(fallback),
        "reason": reason or "",
    }


def write_manifest(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return p


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
