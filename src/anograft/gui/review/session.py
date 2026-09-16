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
from anograft.core.appearance import (  # noqa: F401 — 검수 탭·테스트가 여기서도 import 한다
    APPEARANCE_FN,
    LIGHT_MIN_N,
    LIGHT_REAL_MIN,
    LIGHT_RING_PX,
    RING_PX,
    angle_diff,
    circular_concentration,
    circular_mean,
    is_directional,
    mask_contrast,
    mask_lighting,
    mask_sharpness,
    mask_texture,
)
from anograft.core.types import DefectSource
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

FILTERS: tuple[str, ...] = (
    "all",
    "unreviewed",
    "accept",
    "reject",
    "fallback",
    "skipped",
    "flipped",  # 조명 뒤집힘 의심(실제 클래스 방향과 > FLIP_DEG)
)
FLIP_DEG = 90.0  # 실제 클래스 평균 방향에서 이보다 벗어난 합성 인스턴스 = 조명 뒤집힘 의심
DIST_KEYS: tuple[str, ...] = ("area", "length", "contrast", "texture", "sharpness", "lighting")
IMAGE_KEYS: tuple[str, ...] = (
    "contrast",
    "texture",
    "sharpness",
    "lighting",
)  # 이미지·마스크를 읽어 계산(지연·캐시)
LINEAR_KEYS: tuple[str, ...] = (
    IMAGE_KEYS  # 외형 지표는 0·음수가 정상값 → 선형 구간(면적·길이는 로그)
)
FIXED_RANGE: dict[str, tuple[float, float]] = {"lighting": (-180.0, 180.0)}  # 각도는 구간 고정


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
    appearance: dict[str, list[float]] = field(
        default_factory=dict
    )  # key → 인스턴스별 값(contrast·texture·sharpness·lighting)
    appearance_cls: dict[str, list[str]] = field(
        default_factory=dict
    )  # 위와 나란한 인스턴스 클래스


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
    a: Sequence[float],
    b: Sequence[float],
    *,
    bins: int = 12,
    log: bool = True,
    value_range: tuple[float, float] | None = None,
) -> Histogram:
    """두 계열을 같은 구간으로. 값이 하나도 없으면 빈 히스토그램(edges 0..1). 로그 구간은 양수만 센다.
    ``value_range`` 는 선형 구간을 데이터 대신 고정할 때(각도 −180..180)."""
    va = [float(v) for v in a if v > 0] if log else [float(v) for v in a]
    vb = [float(v) for v in b if v > 0] if log else [float(v) for v in b]
    allv = va + vb
    if not allv:
        lo0, hi0 = value_range if (value_range is not None and not log) else (0.0, 1.0)
        return Histogram(tuple(np.linspace(lo0, hi0, bins + 1)), (0,) * bins, (0,) * bins, log)
    lo, hi = min(allv), max(allv)
    if value_range is not None and not log:
        lo, hi = value_range
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


def _split(s: str) -> tuple[str, ...]:
    return tuple(x for x in s.split(";") if x)


class ReviewSession:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.items: list[ReviewItem] = []
        self.review: dict[str, tuple[str, str]] = {}
        self.recipe_meta: dict[str, Any] = {}
        self.bank: Bank | None = None
        self.real_csv: Path | None = None  # 실측 CSV(있으면 그 열들은 은행 대신)
        self.real_rows: list[dict[str, Any]] = []
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
        self.real_csv = None
        self.real_rows = []
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
        flipped = self.flipped_lighting() if which == "flipped" else set()
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
            elif which == "flipped":
                ok = it.index in flipped
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
            if key in IMAGE_KEYS:
                vals += self._item_appearance(it, key)
                continue
            for inst in it.instances:
                if key == "area":
                    vals.append(float(inst.get("area_px") or 0))
                else:
                    bb = inst.get("bbox") or [0, 0, 0, 0]
                    vals.append(float(max(bb[2], bb[3])))
        return vals

    def _item_appearance(self, it: ReviewItem, key: str) -> list[float]:
        """인스턴스 bbox(±RING_PX) 창에서 외형 지표 — 이미지·마스크를 한 번 읽어 세 지표를 같이 캐시."""
        if key in it.appearance:
            return it.appearance[key]
        for k in IMAGE_KEYS:
            it.appearance[k] = []
            it.appearance_cls[k] = []
        it.contrasts = it.appearance["contrast"]
        if self.root is None or not it.image or not it.mask:
            return it.appearance[key]
        try:
            image, _g = imgio.read_image(self.root / it.image)
            mask = imgio.read_mask(self.root / it.mask)
        except (OSError, imgio.ImageReadError):
            return it.appearance[key]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        for inst in it.instances:
            bb = inst.get("bbox") or None
            if not bb:
                continue
            x, y, bw, bh = (int(v) for v in bb)
            x0, y0 = max(0, x - RING_PX), max(0, y - RING_PX)
            x1, y1 = min(w, x + bw + RING_PX), min(h, y + bh + RING_PX)
            g_win, m_win = gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            for k, fn in APPEARANCE_FN.items():
                v = fn(g_win, m_win)
                if v is not None:
                    it.appearance[k].append(v)
                    it.appearance_cls[k].append(str(inst.get("class") or "?"))
        return it.appearance[key]

    def synthetic_by_class(self, key: str) -> dict[str, list[float]]:
        """외형 지표를 클래스별로(채택/미검수만). 조명 일관성처럼 클래스마다 성격이 다른 지표용."""
        if key not in IMAGE_KEYS:
            raise ReviewError(f"클래스별 분포는 외형 지표만 (선택: {', '.join(IMAGE_KEYS)})")
        out: dict[str, list[float]] = {}
        for it in self.items:
            if it.status != "ok" or it.verdict == "reject":
                continue
            self.item(it.index)
            vals = self._item_appearance(it, key)
            for v, c in zip(vals, it.appearance_cls.get(key, ()), strict=True):
                out.setdefault(c, []).append(v)
        return out

    # ------------------------------------------------------------------ 실측 CSV(실제 분포를 은행 대신 현장 측정값으로)

    def load_real_csv(self, path: str | Path) -> int:
        """실제 분포를 은행 소스 대신 **실측 CSV** 로 — 열 = ``class``(선택) + ``area``·``length``·``contrast``·``texture``·``sharpness``·
        ``lighting`` 중 있는 것(숫자, 빈 칸은 건너뜀). 현장에서 잰 결함 크기 분포(예: 현미경 µm → px 환산)를 합성과 견줄 때.
        CSV 에 있는 키는 CSV 가, 없는 키는 여전히 은행이 실제 값을 댄다. 반환: 읽은 행 수. 형식이 틀리면 ``ReviewError``."""
        import csv

        p = Path(path)
        try:
            with open(p, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                cols = [c.strip() for c in (reader.fieldnames or [])]
                rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in reader]
        except (OSError, csv.Error) as e:
            raise ReviewError(f"실측 CSV 를 읽을 수 없습니다: {e}") from e
        keys = [c for c in cols if c in DIST_KEYS]
        if not keys:
            raise ReviewError(
                f"실측 CSV 에 분포 열이 없습니다 (가능: {', '.join(DIST_KEYS)}; 선택: class)"
            )
        parsed: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {"class": r.get("class", "")}
            for k in keys:
                v = r.get(k, "")
                if v == "":
                    continue
                try:
                    item[k] = float(v)
                except ValueError as e:
                    raise ReviewError(f"실측 CSV {k} 열에 숫자가 아닌 값: {v!r}") from e
            parsed.append(item)
        self.real_csv = p
        self.real_rows = parsed
        return len(parsed)

    def clear_real_csv(self) -> None:
        self.real_csv = None
        self.real_rows = []

    def real_csv_keys(self) -> set[str]:
        return {k for r in self.real_rows for k in r if k != "class"}

    def real_label(self) -> str:
        """실제 분포의 출처 — '실측 x.csv' 또는 '은행 name' 또는 ''."""
        if self.real_csv is not None:
            return f"실측 {self.real_csv.name}"
        return f"은행 {self.bank.name}" if self.bank is not None else ""

    def _csv_values(self, key: str) -> list[float] | None:
        """CSV 가 그 키를 갖고 있으면 값 목록(레시피 클래스 필터 적용), 아니면 None(은행으로)."""
        if key not in self.real_csv_keys():
            return None
        allowed = self.real_classes()
        return [
            float(r[key])
            for r in self.real_rows
            if key in r and (allowed is None or not r.get("class") or r["class"] in allowed)
        ]

    def real_classes(self) -> list[str] | None:
        """'실제' 분포에 쓸 은행 클래스 — 레시피가 뽑은 클래스만(`source.classes` → `class_ratio` 키 → 전부 None).
        은행에 스크래치·얼룩이 같이 있어도 pit 만 합성한 출력이면 실제 쪽도 pit 만 세어야 비교가 된다."""
        pipe = self.recipe_meta.get("pipeline") or {}
        src = pipe.get("source") or {}
        if isinstance(src.get("classes"), list) and src["classes"]:
            return [str(c) for c in src["classes"]]
        ratio = (self.recipe_meta.get("output") or {}).get("class_ratio")
        if isinstance(ratio, dict) and ratio:
            return [str(c) for c in ratio]
        return None

    def real_sources(self) -> list[DefectSource]:
        """실제 분포의 소스 목록 = 은행 소스 중 `real_classes()` 에 드는 것(None 이면 전부)."""
        if self.bank is None:
            return []
        allowed = self.real_classes()
        srcs = self.bank.sources()
        return srcs if allowed is None else [s for s in srcs if s.cls in allowed]

    def real_by_class(self, key: str) -> dict[str, list[float]]:
        if key not in IMAGE_KEYS:
            raise ReviewError(f"클래스별 분포는 외형 지표만 (선택: {', '.join(IMAGE_KEYS)})")
        out: dict[str, list[float]] = {}
        if key in self.real_csv_keys():
            allowed = self.real_classes()
            for r in self.real_rows:
                c = str(r.get("class") or "")
                if key in r and c and (allowed is None or c in allowed):
                    out.setdefault(c, []).append(float(r[key]))
            return out
        if self.bank is None:
            return out
        for s in self.real_sources():
            v = APPEARANCE_FN[key](cv2.cvtColor(s.image, cv2.COLOR_BGR2GRAY), s.mask)
            if v is not None:
                out.setdefault(s.cls, []).append(v)
        return out

    def real_values(self, key: str = "area") -> list[float]:
        if key not in DIST_KEYS:
            raise ReviewError(f"분포 키 {key!r} (선택: {', '.join(DIST_KEYS)})")
        csv_vals = self._csv_values(key)
        if csv_vals is not None:
            return csv_vals
        if self.bank is None:
            return []
        vals: list[float] = []
        for s in self.real_sources():
            if key == "area":
                vals.append(float(np.count_nonzero(s.mask)))
            elif key in IMAGE_KEYS:
                c = APPEARANCE_FN[key](cv2.cvtColor(s.image, cv2.COLOR_BGR2GRAY), s.mask)
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
            self.synthetic_values(key),
            self.real_values(key),
            bins=bins,
            log=key not in LINEAR_KEYS,
            value_range=FIXED_RANGE.get(key),
        )

    def class_options(self, key: str) -> list[str]:
        """클래스별 분포를 그릴 수 있는 클래스 — 외형 지표(``IMAGE_KEYS``)만, 합성·실제 어느 쪽이든 값이 있는 클래스(이름순)."""
        if key not in IMAGE_KEYS:
            return []
        return sorted(set(self.synthetic_by_class(key)) | set(self.real_by_class(key)))

    def distribution_by_class(self, key: str, cls: str, *, bins: int = 12) -> Histogram:
        """한 클래스만의 합성 vs 실제 히스토그램(외형 지표) — 조명 방향은 클래스마다 다르므로 전체 분포는 섞여 보인다.
        ``lighting_r_for(key, cls)`` 와 같은 값 집합."""
        if key not in IMAGE_KEYS:
            raise ReviewError(f"클래스별 분포는 외형 지표만 (선택: {', '.join(IMAGE_KEYS)})")
        return histogram(
            self.synthetic_by_class(key).get(cls, []),
            self.real_by_class(key).get(cls, []),
            bins=bins,
            log=False,
            value_range=FIXED_RANGE.get(key),
        )

    def lighting_r_for(self, cls: str) -> tuple[float | None, float | None, int, int]:
        """한 클래스의 조명 일관성 (R 합성, R 실제, n 합성, n 실제)."""
        syn = self.synthetic_by_class("lighting").get(cls, [])
        real = self.real_by_class("lighting").get(cls, [])
        return circular_concentration(syn), circular_concentration(real), len(syn), len(real)

    def lighting_histograms_by_class(self, *, bins: int = 12) -> dict[str, Histogram]:
        """리포트용 — 실제 방향이 유의한 클래스(`directional_classes`)의 조명 방향 히스토그램. 없으면 빈 dict."""
        return {
            c: self.distribution_by_class("lighting", c, bins=bins)
            for c in self.directional_classes()
        }

    def lighting_concentration(self) -> tuple[float | None, float | None]:
        """조명 일관성 R — (합성, 실제) 전체. 합성이 실제보다 뚜렷이 낮으면 회전 범위가 조명 방향을 깨고 있다(→ dent-graft)."""
        return (
            circular_concentration(self.synthetic_values("lighting")),
            circular_concentration(self.real_values("lighting")),
        )

    def real_lighting_direction(self) -> dict[str, float]:
        """조명 방향이 뚜렷한 클래스(실제 R ≥ LIGHT_REAL_MIN, n ≥ LIGHT_MIN_N)의 실제 평균 방향(°)."""
        out: dict[str, float] = {}
        for c, vals in self.real_by_class("lighting").items():
            r = circular_concentration(vals)
            if is_directional(r, len(vals)):
                m = circular_mean(vals)
                if m is not None:
                    out[c] = m
        return out

    def flipped_lighting(self, *, max_deg: float = FLIP_DEG) -> set[str]:
        """조명 뒤집힘 의심 — 인스턴스의 조명 방향이 그 클래스의 실제 평균 방향에서 ``max_deg`` 넘게 벗어난 항목(index).
        은행이 없거나 방향이 뚜렷한 클래스가 없으면 비어 있다. 반려 판정과 무관(사용자가 보고 판정)."""
        real = self.real_lighting_direction()
        if not real:
            return set()
        out: set[str] = set()
        for it in self.items:
            if it.status != "ok":
                continue
            self.item(it.index)
            vals = self._item_appearance(it, "lighting")
            for v, c in zip(vals, it.appearance_cls.get("lighting", ()), strict=True):
                if c in real and angle_diff(v, real[c]) > max_deg:
                    out.add(it.index)
                    break
        return out

    def lighting_concentration_by_class(self) -> dict[str, tuple[float | None, float | None]]:
        """클래스별 R — 방향성 없는 클래스(얼룩·스크래치)가 전체 R 을 희석하므로 판단은 클래스별로."""
        syn, real = self.synthetic_by_class("lighting"), self.real_by_class("lighting")
        return {
            c: (circular_concentration(syn.get(c, [])), circular_concentration(real.get(c, [])))
            for c in sorted(set(syn) | set(real))
        }

    def directional_classes(self) -> list[str]:
        """실제 소스의 조명 방향이 유의하게 뚜렷한 클래스(`is_directional`) — 리포트·제목의 '뒤집힘' 판정 대상."""
        return [
            c
            for c, vals in self.real_by_class("lighting").items()
            if is_directional(circular_concentration(vals), len(vals))
        ]

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
            hist_lighting=self.distribution("lighting"),
            hist_lighting_class=self.lighting_histograms_by_class(),
            lighting_r=self.lighting_concentration(),
            lighting_r_class=self.lighting_concentration_by_class(),
            directional=self.directional_classes(),
            geometry=self.geometry_text(),
            flipped=sorted(self.flipped_lighting()),
            bank_name=self.real_label(),
            warnings=list(self.warnings),
        )

    def geometry_text(self) -> str:
        """resolved 레시피의 기하 한 줄 — scale·rotate·flip + per_class(준 필드만). 조명 분포의 맥락."""
        geo = (self.recipe_meta.get("pipeline") or {}).get("geometry") or {}
        if not geo:
            return ""

        def rng(v: Any) -> str:
            return f"{v[0]:g}~{v[1]:g}" if isinstance(v, (list, tuple)) and len(v) == 2 else str(v)

        parts = [
            f"scale {rng(geo.get('scale'))}",
            f"rotate {rng(geo.get('rotate'))}°",
            f"flip {geo.get('flip')}",
        ]
        for c, o in (geo.get("per_class") or {}).items():
            bits = [
                f"{k} {rng(v) if k != 'flip' else v}" for k, v in (o or {}).items() if v is not None
            ]
            parts.append(f"per_class {c}: {' · '.join(bits) or '(변경 없음)'}")
        return " · ".join(parts)

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
