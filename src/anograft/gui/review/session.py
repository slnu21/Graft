"""검수 세션 — **Qt 없음**. 출력 루트(``manifest.csv`` + ``meta/*.json``)를 열어 합성 결과 목록·채택/반려·분포 통계를 다룬다(v0.7 검수 탭).

- ``load(root)`` 는 manifest 행을 읽고 ok 행의 사이드카를 **지연** 로드(``item(index)``). ``review.csv`` 가 있으면 verdict 복원.
- ``set_verdict(index, verdict, note)`` → ``save()`` 가 ``review.csv`` 로(자동 저장은 위젯 몫).
- 분포: 합성 인스턴스(사이드카 ``gtmask.instances[].area_px``·bbox 긴 변) vs 실제(은행 소스 마스크 면적·긴 변). 은행은
  ``recipe.resolved.yaml`` 의 ``inputs.bank`` 를 cwd 기준으로 열어 보고, 없으면 실제 분포 없이(fail-soft).
  ``histogram(values_a, values_b, bins)`` 는 두 계열을 **같은 로그 구간**으로 세는 순수 함수 — 위젯이 막대만 그린다.
- ``prune(out, drop_unreviewed)`` → ``io.prune.prune_dataset``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from anograft.bank import Bank
from anograft.bank.bank import BankError
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import (
    REVIEW_FILE,
    VERDICTS,
    PruneSummary,
    prune_dataset,
    read_review,
    write_review,
)
from anograft.io.report import ReportData, skipped_reason_counts, write_report

FILTERS: tuple[str, ...] = ("all", "unreviewed", "accept", "reject", "fallback", "skipped")
DIST_KEYS: tuple[str, ...] = ("area", "length", "contrast")  # contrast 는 선형 구간(음수 가능)
RING_PX = 8


class ReviewError(ValueError):
    pass


@dataclass
class ReviewItem:
    """manifest 한 행 + (ok 면) 사이드카 요약."""

    index: str
    status: str  # ok | normal | skipped
    image: str
    mask: str
    sidecar: str
    classes: tuple[str, ...]
    source_ids: tuple[str, ...]
    area_px: int
    blend: str
    fallback: bool
    reason: str
    target: str
    verdict: str = ""
    note: str = ""
    instances: list[dict[str, Any]] = field(default_factory=list)  # gtmask.instances
    warnings: list[str] = field(default_factory=list)
    loaded: bool = False
    contrasts: list[float] | None = None  # 인스턴스별 대비(이미지·마스크를 읽어야 해서 지연)


@dataclass(frozen=True)
class Histogram:
    edges: tuple[float, ...]  # bins+1
    a: tuple[int, ...]  # 합성
    b: tuple[int, ...]  # 실제
    log: bool

    @property
    def bins(self) -> int:
        return len(self.a)


def histogram(
    a: Sequence[float], b: Sequence[float], *, bins: int = 12, log: bool = True
) -> Histogram:
    """두 계열을 같은 구간으로. 값이 하나도 없으면 빈 히스토그램(edges 0..1). 로그 구간은 양수만 센다."""
    va = [float(v) for v in a if v > 0] if log else [float(v) for v in a]
    vb = [float(v) for v in b if v > 0] if log else [float(v) for v in b]
    allv = va + vb
    if not allv:
        return Histogram(tuple(np.linspace(0, 1, bins + 1)), (0,) * bins, (0,) * bins, log)
    lo, hi = min(allv), max(allv)
    if log:
        lo, hi = math.log10(lo), math.log10(hi)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    xa = np.log10(va) if log and va else np.array(va)
    xb = np.log10(vb) if log and vb else np.array(vb)
    ca, _ = np.histogram(xa, bins=edges) if len(xa) else (np.zeros(bins, dtype=int), None)
    cb, _ = np.histogram(xb, bins=edges) if len(xb) else (np.zeros(bins, dtype=int), None)
    real_edges = tuple(float(10**e) if log else float(e) for e in edges)
    return Histogram(real_edges, tuple(int(v) for v in ca), tuple(int(v) for v in cb), log)


def mask_contrast(gray: np.ndarray, mask: np.ndarray, *, ring_px: int = RING_PX) -> float | None:
    """마스크 안 평균 그레이 − 둘레 링(폭 ``ring_px``, 마스크 제외) 평균. 어느 쪽이든 비면 None. 라벨 탭 통계와 같은 정의."""
    m = mask > 0
    if not m.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = cv2.dilate(m.astype(np.uint8), k) > 0
    ring &= ~m
    if not ring.any():
        return None
    g = gray.astype(np.float32)
    return round(float(g[m].mean() - g[ring].mean()), 2)


def _split(s: str) -> tuple[str, ...]:
    return tuple(x for x in s.split(";") if x)


class ReviewSession:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.items: list[ReviewItem] = []
        self.review: dict[str, tuple[str, str]] = {}
        self.recipe_meta: dict[str, Any] = {}
        self.bank: Bank | None = None
        self.warnings: list[str] = []
        self.dirty = False

    # ------------------------------------------------------------------ 열기

    @property
    def loaded(self) -> bool:
        return self.root is not None

    def load(self, root: str | Path, *, load_bank: bool = True) -> list[ReviewItem]:
        r = Path(root)
        if not (r / MANIFEST_FILE).is_file():
            raise ReviewError(f"출력 폴더가 아닙니다 ({MANIFEST_FILE} 없음): {r}")
        self.root = r
        self.warnings = []
        try:
            self.review = read_review(r / REVIEW_FILE)
        except ValueError as e:
            self.review = {}
            self.warnings.append(str(e))
        self.items = []
        for row in read_manifest(r / MANIFEST_FILE):
            idx = str(row.get("index", ""))
            v, note = self.review.get(idx, ("", ""))
            self.items.append(
                ReviewItem(
                    index=idx,
                    status=row.get("status", ""),
                    image=row.get("image", ""),
                    mask=row.get("mask", ""),
                    sidecar=row.get("sidecar", ""),
                    classes=_split(row.get("classes", "")),
                    source_ids=_split(row.get("source_ids", "")),
                    area_px=int(row.get("area_px") or 0),
                    blend=row.get("blend", ""),
                    fallback=bool(int(row.get("fallback") or 0)),
                    reason=row.get("reason", ""),
                    target=row.get("target", ""),
                    verdict=v,
                    note=note,
                )
            )
        self.recipe_meta = {}
        self.bank = None
        rr = r / "recipe.resolved.yaml"
        if rr.is_file():
            try:
                data = yaml.safe_load(rr.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.recipe_meta = data
            except yaml.YAMLError as e:
                self.warnings.append(f"recipe.resolved.yaml 읽기 실패: {e}")
        if load_bank:
            self._load_bank()
        self.dirty = False
        return list(self.items)

    def _load_bank(self) -> None:
        bank_path = (self.recipe_meta.get("inputs") or {}).get("bank")
        if not bank_path:
            return
        try:
            self.bank = Bank.load(bank_path)
        except (BankError, OSError) as e:
            self.warnings.append(f"은행을 열 수 없어 실제 분포 없음 ({bank_path}): {e}")

    def item(self, index: str) -> ReviewItem:
        """사이드카를 처음 볼 때 읽는다(gtmask.instances·warnings)."""
        it = next((x for x in self.items if x.index == index), None)
        if it is None:
            raise ReviewError(f"항목이 없습니다: {index}")
        if not it.loaded and it.sidecar and self.root is not None:
            p = self.root / it.sidecar
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                it.instances = list((d.get("gtmask") or {}).get("instances") or [])
                it.warnings = [str(w) for w in d.get("warnings") or []]
            except (OSError, ValueError) as e:
                it.warnings = [f"사이드카 읽기 실패: {e}"]
            it.loaded = True
        return it

    # ------------------------------------------------------------------ 조회

    def filtered(self, which: str = "all", cls: str | None = None) -> list[ReviewItem]:
        if which not in FILTERS:
            raise ReviewError(f"필터 {which!r} (선택: {', '.join(FILTERS)})")
        out = []
        for it in self.items:
            if which == "skipped":
                ok = it.status == "skipped"
            elif it.status != "ok":
                ok = which == "all"
            elif which == "unreviewed":
                ok = it.verdict == ""
            elif which in ("accept", "reject"):
                ok = it.verdict == which
            elif which == "fallback":
                ok = it.fallback
            else:
                ok = True
            if ok and cls is not None and cls not in it.classes:
                ok = False
            if ok:
                out.append(it)
        return out

    def classes(self) -> list[str]:
        return sorted({c for it in self.items for c in it.classes})

    def counts(self) -> dict[str, int]:
        ok = [it for it in self.items if it.status == "ok"]
        return {
            "ok": len(ok),
            "normal": sum(1 for it in self.items if it.status == "normal"),
            "skipped": sum(1 for it in self.items if it.status == "skipped"),
            "accept": sum(1 for it in ok if it.verdict == "accept"),
            "reject": sum(1 for it in ok if it.verdict == "reject"),
            "unreviewed": sum(1 for it in ok if it.verdict == ""),
            "fallback": sum(1 for it in ok if it.fallback),
        }

    def summary_text(self) -> str:
        if self.root is None:
            return "출력 폴더 없음"
        c = self.counts()
        return (
            f"{self.root.name}: 합성 {c['ok']} · 정상 {c['normal']} · skipped {c['skipped']} — "
            f"채택 {c['accept']} · 반려 {c['reject']} · 미검수 {c['unreviewed']}"
            + (f" · 폴백 {c['fallback']}" if c["fallback"] else "")
        )

    # ------------------------------------------------------------------ 판정

    def set_verdict(self, index: str, verdict: str, note: str | None = None) -> ReviewItem:
        if verdict not in VERDICTS:
            raise ReviewError(f"verdict {verdict!r} (accept | reject | 빈칸)")
        it = self.item(index)
        if it.status != "ok":
            raise ReviewError("합성 이미지만 판정합니다")
        it.verdict = verdict
        if note is not None:
            it.note = note
        if verdict or it.note:
            self.review[index] = (verdict, it.note)
        else:
            self.review.pop(index, None)
        self.dirty = True
        return it

    def save(self) -> Path:
        if self.root is None:
            raise ReviewError("출력 폴더를 먼저 여세요")
        p = write_review(self.root / REVIEW_FILE, self.review)
        self.dirty = False
        return p

    # ------------------------------------------------------------------ 분포

    def synthetic_values(self, key: str = "area") -> list[float]:
        """합성 인스턴스의 면적·긴 변(bbox)·대비(마스크 안 평균 그레이 − 링 평균) — 채택/미검수만(반려는 제외)."""
        if key not in DIST_KEYS:
            raise ReviewError(f"분포 키 {key!r} (선택: {', '.join(DIST_KEYS)})")
        vals: list[float] = []
        for it in self.items:
            if it.status != "ok" or it.verdict == "reject":
                continue
            self.item(it.index)
            if key == "contrast":
                vals += self._item_contrasts(it)
                continue
            for inst in it.instances:
                if key == "area":
                    vals.append(float(inst.get("area_px") or 0))
                else:
                    bb = inst.get("bbox") or [0, 0, 0, 0]
                    vals.append(float(max(bb[2], bb[3])))
        return vals

    def _item_contrasts(self, it: ReviewItem) -> list[float]:
        """인스턴스 bbox 창 안에서 GT 마스크 vs 링 대비 — 이미지·마스크를 한 번 읽어 캐시."""
        if it.contrasts is not None:
            return it.contrasts
        it.contrasts = []
        if self.root is None or not it.image or not it.mask:
            return it.contrasts
        try:
            image, _g = imgio.read_image(self.root / it.image)
            mask = imgio.read_mask(self.root / it.mask)
        except (OSError, imgio.ImageReadError):
            return it.contrasts
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        for inst in it.instances:
            bb = inst.get("bbox") or None
            if not bb:
                continue
            x, y, bw, bh = (int(v) for v in bb)
            x0, y0 = max(0, x - RING_PX), max(0, y - RING_PX)
            x1, y1 = min(w, x + bw + RING_PX), min(h, y + bh + RING_PX)
            c = mask_contrast(gray[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            if c is not None:
                it.contrasts.append(c)
        return it.contrasts

    def real_values(self, key: str = "area") -> list[float]:
        if key not in DIST_KEYS:
            raise ReviewError(f"분포 키 {key!r} (선택: {', '.join(DIST_KEYS)})")
        if self.bank is None:
            return []
        vals: list[float] = []
        for s in self.bank.sources():
            if key == "area":
                vals.append(float(np.count_nonzero(s.mask)))
            elif key == "contrast":
                c = mask_contrast(cv2.cvtColor(s.image, cv2.COLOR_BGR2GRAY), s.mask)
                if c is not None:
                    vals.append(c)
            else:
                ys, xs = np.nonzero(s.mask)
                if len(xs) == 0:
                    continue
                vals.append(float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
        return vals

    def distribution(self, key: str = "area", *, bins: int = 12) -> Histogram:
        return histogram(
            self.synthetic_values(key), self.real_values(key), bins=bins, log=key != "contrast"
        )

    # ------------------------------------------------------------------ 리포트

    def report_data(self) -> ReportData:
        """HTML 리포트 입력(Qt 없음). 사이드카는 여기서 전부 읽는다(클래스별 인스턴스 수)."""
        if self.root is None:
            raise ReviewError("출력 폴더를 먼저 여세요")
        per_class: dict[str, int] = {}
        rejected: list[tuple[str, str, str]] = []
        for it in self.items:
            if it.status != "ok":
                continue
            self.item(it.index)
            if it.verdict == "reject":
                rejected.append((it.index, ", ".join(it.classes), it.note))
                continue
            for inst in it.instances:
                c = str(inst.get("class") or "?")
                per_class[c] = per_class.get(c, 0) + 1
        meta = self.recipe_meta
        pipe = meta.get("pipeline") or {}
        return ReportData(
            root=self.root.as_posix(),
            recipe_name=str(meta.get("name") or ""),
            seed=meta.get("seed"),
            preset=str(pipe.get("preset") or ""),
            pipeline_hash=self._pipeline_hash(),
            counts=self.counts(),
            per_class=per_class,
            rejected=rejected,
            skipped_reasons=skipped_reason_counts(
                [it.reason for it in self.items if it.status == "skipped"]
            ),
            hist_area=self.distribution("area"),
            hist_length=self.distribution("length"),
            hist_contrast=self.distribution("contrast"),
            bank_name=self.bank.name if self.bank is not None else "",
            warnings=list(self.warnings),
        )

    def _pipeline_hash(self) -> str:
        """``recipe.resolved.yaml`` 헤더 주석의 pipeline_hash(없으면 첫 사이드카)."""
        assert self.root is not None
        rr = self.root / "recipe.resolved.yaml"
        if rr.is_file():
            head = "\n".join(rr.read_text(encoding="utf-8").splitlines()[:5])
            m = re.search(r"pipeline_hash\s*[=:]\s*([0-9a-f]+)", head)
            if m:
                return m.group(1)
        for it in self.items:
            if it.status == "ok" and it.sidecar:
                try:
                    return str(
                        json.loads((self.root / it.sidecar).read_text(encoding="utf-8")).get(
                            "pipeline_hash"
                        )
                        or ""
                    )
                except (OSError, ValueError):
                    return ""
        return ""

    def write_report(self, path: str | Path | None = None) -> Path:
        if self.root is None:
            raise ReviewError("출력 폴더를 먼저 여세요")
        out = Path(path) if path else self.root / "review-report.html"
        return write_report(out, self.report_data())

    # ------------------------------------------------------------------ 정리본

    def prune(self, out: str | Path, *, drop_unreviewed: bool = False) -> PruneSummary:
        if self.root is None:
            raise ReviewError("출력 폴더를 먼저 여세요")
        if self.dirty:
            self.save()
        try:
            return prune_dataset(self.root, out, self.review, drop_unreviewed=drop_unreviewed)
        except ValueError as e:
            raise ReviewError(str(e)) from e
