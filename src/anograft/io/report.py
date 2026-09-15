"""검수 리포트(v0.7.x) — 출력 폴더 + ``review.csv`` 를 **HTML 한 장**으로(의존성 0: 인라인 SVG 히스토그램, 인라인 CSS).
학습 로그에 첨부하거나 정리본 폴더에 같이 둔다. Qt 없음 — CLI ``dataset report`` 와 검수 탭 공용.

내용: 레시피 이름·시드·파이프라인 해시·프리셋 · 합성/정상/skipped · 채택/반려/미검수 · 클래스별 인스턴스 수 · 폴백 수 ·
합성 vs 실제(은행) 면적·긴 변 히스토그램(합성 ``#00A188`` / 실제 ``#C8841C``) · 반려 목록(index · 클래스 · 메모) · skipped 사유 상위.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anograft.core.appearance import LIGHT_REAL_MIN, LIGHT_SYNTH_MAX

SYNTH = "#00A188"
REAL = "#C8841C"


@dataclass
class ReportData:
    """리포트에 필요한 전부 — 검수 세션이 채운다(Qt 없음)."""

    root: str
    recipe_name: str = ""
    seed: int | None = None
    preset: str = ""
    pipeline_hash: str = ""
    counts: Mapping[str, int] = field(
        default_factory=dict
    )  # ok normal skipped accept reject unreviewed fallback
    per_class: Mapping[str, int] = field(default_factory=dict)  # 채택+미검수 인스턴스 수
    rejected: Sequence[tuple[str, str, str]] = ()  # (index, classes, note)
    skipped_reasons: Mapping[str, int] = field(default_factory=dict)
    hist_area: Any = None  # Histogram(edges, a, b, log)
    hist_length: Any = None
    hist_contrast: Any = None  # 선형 구간(음수 가능)
    hist_lighting: Any = None  # 조명 방향 각도(−180..180)
    lighting_r: tuple[float | None, float | None] = (None, None)  # 조명 일관성 R(합성, 실제)
    lighting_r_class: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    flipped: Sequence[str] = ()  # 조명 뒤집힘 의심 index(실제 클래스 방향에서 > 90°)
    bank_name: str = ""
    warnings: Sequence[str] = ()


def lighting_broken_classes(
    per_class: Mapping[str, tuple[float | None, float | None]],
) -> list[str]:
    """실제는 한 방향(R ≥ LIGHT_REAL_MIN)인데 합성은 무작위(R < LIGHT_SYNTH_MAX)인 클래스."""
    return [
        c
        for c, (rs, rr) in per_class.items()
        if rs is not None and rr is not None and rr >= LIGHT_REAL_MIN and rs < LIGHT_SYNTH_MAX
    ]


def flipped_line(flipped: Sequence[str]) -> str:
    """조명 뒤집힘 의심 목록 한 줄(검수 탭 필터 '조명 뒤집힘 의심'과 같은 집합)."""
    if not flipped:
        return ""
    shown = ", ".join(html.escape(i) for i in list(flipped)[:12])
    more = f" 외 {len(flipped) - 12}" if len(flipped) > 12 else ""
    return (
        f'<p class="muted">조명 뒤집힘 의심 <b>{len(flipped)}</b>건 — 실제 클래스 방향에서 90° 넘게 벗어난 인스턴스가 있는 이미지: '
        f"{shown}{more} (검수 탭 필터 '조명 뒤집힘 의심')</p>"
    )


def lighting_line(
    r: tuple[float | None, float | None],
    per_class: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> str:
    """조명 일관성 R 한 줄(전체 + 클래스별) — 어느 클래스든 실제는 한 방향인데 합성이 무작위면 dent-graft 안내."""
    rs, rr = r
    if rs is None and rr is None:
        return ""

    def f(v: float | None) -> str:
        return "–" if v is None else f"{v:.2f}"

    per_class = per_class or {}
    cls_part = " · ".join(f"{html.escape(c)} {f(a)}/{f(b)}" for c, (a, b) in per_class.items())
    broken = lighting_broken_classes(per_class)
    hint = ""
    if broken:
        hint = (
            f" — <b>{html.escape(', '.join(broken))}</b>: 실제는 한 방향인데 합성은 무작위 → 회전 범위가 조명 방향을 "
            "뒤집고 있다. 프리셋 <code>dent-graft</code>(±15°, flip 끔)"
        )
    return (
        f'<p class="muted">조명 일관성 R — 합성 {f(rs)} · 실제 {f(rr)}'
        f"{(' · 클래스별(합성/실제): ' + cls_part) if cls_part else ''} "
        f"(1 = 하이라이트가 모두 같은 방향, 0 = 무작위){hint}</p>"
    )


def _fmt(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def svg_histogram(hist: Any, title: str, *, width: int = 520, height: int = 200) -> str:
    """두 계열 막대 SVG(합성 teal · 실제 amber). ``hist`` 는 ``edges/a/b`` 속성만 필요."""
    if hist is None or (sum(hist.a) + sum(hist.b)) == 0:
        return f'<p class="muted">{html.escape(title)} — 분포 없음</p>'
    n = len(hist.a)
    left, top, right, bottom = 36, 28, width - 12, height - 26
    peak = max(max(hist.a), max(hist.b), 1)
    bw = (right - left) / n
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="16" class="t">{html.escape(title)}</text>',
        f'<text x="{right - 190}" y="16" fill="{SYNTH}">■ 합성 {sum(hist.a)}</text>',
        f'<text x="{right - 100}" y="16" fill="{REAL}">■ 실제 {sum(hist.b)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#888"/>',
    ]
    for i in range(n):
        x0 = left + i * bw
        for k, (count, color) in enumerate(((hist.a[i], SYNTH), (hist.b[i], REAL))):
            if count <= 0:
                continue
            bh = (bottom - top) * count / peak
            x = x0 + 1 + k * (bw / 2 - 1)
            parts.append(
                f'<rect x="{x:.1f}" y="{bottom - bh:.1f}" width="{max(1.0, bw / 2 - 2):.1f}" height="{bh:.1f}" fill="{color}">'
                f"<title>{'합성' if k == 0 else '실제'} {count} · [{_fmt(hist.edges[i])}, {_fmt(hist.edges[i + 1])})</title></rect>"
            )
    for i in (0, n // 2, n):
        x = left + i * bw
        label = _fmt(hist.edges[i]) if getattr(hist, "log", True) else f"{hist.edges[i]:+.0f}"
        parts.append(
            f'<text x="{min(x, right - 30):.0f}" y="{height - 8}" class="ax">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_report(d: ReportData) -> str:
    c = d.counts
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(i)}</td><td>{e(cls)}</td><td>{e(note)}</td></tr>"
        for i, cls, note in d.rejected
    )
    per_class = "".join(
        f"<tr><td>{e(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(d.per_class.items(), key=lambda kv: -kv[1])
    )
    skipped = "".join(
        f"<tr><td>{e(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(d.skipped_reasons.items(), key=lambda kv: -kv[1])[:8]
    )
    warns = "".join(f"<li>{e(w)}</li>" for w in d.warnings[:10])
    total_ok = c.get("ok", 0)
    reviewed = c.get("accept", 0) + c.get("reject", 0)
    pct = f"{(reviewed / total_ok * 100):.0f}%" if total_ok else "–"
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Graft 검수 리포트 — {e(d.recipe_name or d.root)}</title>
<style>
body{{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:28px auto;max-width:980px;color:#1d232a;padding:0 16px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px;border-bottom:1px solid #e3e6ea;padding-bottom:4px}}
.muted{{color:#697683}} .grid{{display:flex;gap:10px;flex-wrap:wrap}}
.tile{{border:1px solid #e3e6ea;border-radius:8px;padding:10px 14px;min-width:120px}} .tile b{{font-size:20px;display:block}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #eef0f2;padding:5px 8px;text-align:left;vertical-align:top}}
th{{color:#697683;font-weight:600}} svg .t{{font-size:12px;fill:#1d232a}} svg .ax{{font-size:10px;fill:#697683}} svg text{{font-size:11px}}
code{{background:#f4f6f8;padding:1px 4px;border-radius:4px}}
</style></head><body>
<h1>Graft 검수 리포트 <span class="muted">Review report</span></h1>
<p class="muted">{e(d.root)} · 레시피 <code>{e(d.recipe_name)}</code> · seed {d.seed if d.seed is not None else "–"} · 프리셋 <code>{e(d.preset)}</code> · pipeline_hash <code>{e(d.pipeline_hash)}</code>{(" · 은행 " + e(d.bank_name)) if d.bank_name else ""}</p>
<div class="grid">
<div class="tile"><b>{total_ok}</b>합성 synthetic</div>
<div class="tile"><b>{c.get("normal", 0)}</b>정상 normal</div>
<div class="tile"><b>{c.get("skipped", 0)}</b>skipped</div>
<div class="tile"><b style="color:{SYNTH}">{c.get("accept", 0)}</b>채택 accepted</div>
<div class="tile"><b style="color:#c0392b">{c.get("reject", 0)}</b>반려 rejected</div>
<div class="tile"><b>{c.get("unreviewed", 0)}</b>미검수 unreviewed</div>
<div class="tile"><b>{pct}</b>검수율 reviewed</div>
<div class="tile"><b>{c.get("fallback", 0)}</b>폴백 fallback</div>
</div>
<h2>분포 Distribution <span class="muted">— 합성(반려 제외) vs 실제(은행 소스), 로그 구간</span></h2>
<div class="grid">{svg_histogram(d.hist_area, "면적 area (px)")}{svg_histogram(d.hist_length, "긴 변 length (px)")}{svg_histogram(d.hist_contrast, "대비 contrast (gray, 마스크 − 링)")}{svg_histogram(d.hist_lighting, "조명 방향 lighting (°, 0 = →, 90 = ↓)")}</div>
{lighting_line(d.lighting_r, d.lighting_r_class)}
{flipped_line(d.flipped)}
<h2>클래스별 인스턴스 <span class="muted">(채택 + 미검수)</span></h2>
<table><tr><th>클래스</th><th>인스턴스</th></tr>{per_class or '<tr><td colspan="2" class="muted">없음</td></tr>'}</table>
<h2>반려 Rejected ({len(d.rejected)})</h2>
<table><tr><th>index</th><th>클래스</th><th>메모</th></tr>{rows or '<tr><td colspan="3" class="muted">없음</td></tr>'}</table>
<h2>skipped 사유</h2>
<table><tr><th>사유</th><th>수</th></tr>{skipped or '<tr><td colspan="2" class="muted">없음</td></tr>'}</table>
{("<h2>경고</h2><ul>" + warns + "</ul>") if warns else ""}
<p class="muted">anograft · 정리본은 <code>anograft dataset prune {e(d.root)} --out &lt;dir&gt;</code></p>
</body></html>
"""


def write_report(path: str | Path, d: ReportData) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(d), encoding="utf-8")
    return p


def skipped_reason_counts(reasons: Sequence[str]) -> dict[str, int]:
    """사유 문자열을 접두(첫 ' — ' 앞)로 묶어 센다."""
    return dict(Counter((r.split(" — ", 1)[0] or "(사유 없음)") for r in reasons))
