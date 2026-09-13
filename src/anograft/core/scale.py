"""µm/px 축척 정합 (설계 §3.4 · §6 geometry).

다른 카메라로 찍은 결함을 그대로 붙이면 크기가 비현실적이 된다. 소스·대상 모두 픽셀 피치(µm/px)가 있으면
``factor = source / target`` — 소스 1px가 대상에서 몇 px인지. 어느 한쪽이라도 없으면 1.0 + 사유.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleResult:
    factor: float
    applied: bool
    reason: str | None = None


def physical_scale(source_um_per_px: float | None, target_um_per_px: float | None) -> ScaleResult:
    if source_um_per_px is None and target_um_per_px is None:
        return ScaleResult(1.0, False, "소스·대상 모두 µm/px 없음")
    if source_um_per_px is None:
        return ScaleResult(1.0, False, "소스 µm/px 없음")
    if target_um_per_px is None:
        return ScaleResult(1.0, False, "대상 µm/px 없음")
    if source_um_per_px <= 0 or target_um_per_px <= 0:
        raise ValueError(
            f"µm/px must be > 0, got source={source_um_per_px}, target={target_um_per_px}"
        )
    return ScaleResult(source_um_per_px / target_um_per_px, True, None)


def physical_area_um2(area_px: int, um_per_px: float | None) -> float | None:
    """마스크 면적(px) → 실제 면적(µm²). 피치가 없으면 None."""
    if um_per_px is None:
        return None
    return float(area_px) * um_per_px * um_per_px
