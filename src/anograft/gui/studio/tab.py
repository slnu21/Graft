"""``StudioTab`` — 스튜디오 탭 조립·배선. 상태는 ``StudioSession``, 계산은 ``PreviewWorker``, 이 클래스는 둘을 잇는다.

흐름::

    열기(은행·대상) → PrepareJob → prepared → 레일 채움 + 썸네일 잡 → request_previews()
    프리셋/시드/method/대상/변형 수/축소 변경 → 세션 갱신(generation+1) → request_previews()
    request_previews(): 지난 세대 잡 폐기 → 변형 카드 자리표 → k = 0..N-1 (선택된 k 먼저) PreviewJob
    finished_preview(res): res.job.generation == session.generation 일 때만 반영. k == 선택이면 캔버스·파이프라인 패널 갱신
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anograft import runner
from anograft.core import recipe as R
from anograft.core.appearance import flipped_instances
from anograft.core.channels import promote_to_bgr
from anograft.gui.studio.canvas import CompareCanvas
from anograft.gui.studio.jobs import (
    KIND_PREPARE,
    KIND_PREVIEW,
    KIND_THUMB,
    JobError,
    PreviewJob,
    PreviewResult,
    ThumbResult,
)
from anograft.gui.studio.panels import InputsPanel, PipelinePanel, StripBar, TargetRail
from anograft.gui.studio.session import SessionError, StudioSession
from anograft.gui.studio.variants import VariantStrip
from anograft.gui.studio.worker import PreviewWorker


def field_error_detail(message: str) -> str:
    """``SessionError`` 본문에서 카드에 넣을 한 줄 — 헤더("레시피 검증 실패:")·loc 경로·"Value error," 접두를 뗀다."""
    lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
    if not lines:
        return message
    detail = lines[-1] if len(lines) > 1 else lines[0]
    if detail.startswith("pipeline.") and ": " in detail:
        detail = detail.split(": ", 1)[1]
    return detail.replace("Value error, ", "", 1)


class StudioTab(QWidget):
    status = Signal(str)
    context = Signal(str)
    send_to_batch_requested = Signal()
    recipe_opened = Signal(
        str
    )  # 레시피 파일을 열거나 저장했다(posix) — 메인 창이 최근 레시피로 기억

    def __init__(
        self, session: StudioSession, worker: PreviewWorker, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.worker = worker
        self._results: dict[int, PreviewResult] = {}
        self._fit: runner.FitDiagnostic | None = (
            None  # 마지막 미리보기의 배치 가능성 진단(테스트·상태용)
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.strip = StripBar()
        root.addWidget(self.strip)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # 왼쪽 레일: 입력 + 대상 목록
        left = QWidget()
        left.setObjectName("Rail")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        self.inputs = InputsPanel()
        self.rail = TargetRail()
        ll.addWidget(self.inputs)
        ll.addWidget(self.rail, 1)
        left.setMinimumWidth(220)
        split.addWidget(left)
        # 가운데: 캔버스 + 오버레이 바
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        self.canvas = CompareCanvas()
        ml.addWidget(self.canvas, 1)
        bar = QWidget()
        bar.setObjectName("Strip")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 5, 14, 5)
        self.cb_gt = QCheckBox("정답 영역")
        self.cb_gt.setToolTip("GT mask — 학습에 쓰일 정답 영역을 겹쳐 보기")
        self.cb_gt.setChecked(True)
        self.cb_roi = QCheckBox("붙일 수 있는 영역")
        self.cb_roi.setToolTip("ROI — 결함을 놓아도 되는 영역을 겹쳐 보기")
        self.cb_lab = QCheckBox("인스턴스 라벨")
        self.cb_lab.setChecked(True)
        for cb in (self.cb_gt, self.cb_roi, self.cb_lab):
            bl.addWidget(cb)
        self.zoom_info = QLabel("")
        self.zoom_info.setObjectName("Muted")
        self.zoom_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.zoom_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addStretch(1)
        bl.addWidget(self.zoom_info)
        ml.addWidget(bar)
        split.addWidget(mid)
        # 오른쪽: 파이프라인
        self.pipe = PipelinePanel()
        self.pipe.setMinimumWidth(
            352
        )  # 카드(라벨 108 + 두 스핀박스 · per_class 두 줄)가 잘리지 않는 최소 폭
        split.addWidget(self.pipe)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([240, 900, 330])
        root.addWidget(split, 1)

        self.variants = VariantStrip()
        root.addWidget(self.variants)

        self._wire()
        self.sync_widgets()

    # ------------------------------------------------------------------ 배선

    def _wire(self) -> None:
        s = self.strip
        s.preset_changed.connect(lambda name: self._edit(lambda: self.session.set_preset(name)))
        s.gallery_clicked.connect(self.open_preset_gallery)
        s.seed_changed.connect(lambda v: self._edit(lambda: self.session.set_seed(v)))
        s.variants_changed.connect(self._on_variants)
        s.long_side_changed.connect(self._on_long_side)
        s.open_clicked.connect(self.open_recipe_dialog)
        s.save_clicked.connect(self.save_recipe_dialog)
        s.batch_clicked.connect(self.send_to_batch)
        self.inputs.open_requested.connect(self.open_inputs)
        self.inputs.um_per_px_changed.connect(self._on_um_per_px)
        self.rail.selected.connect(self._on_target)
        self.pipe.method_changed.connect(
            lambda st, m: self._edit(lambda: self.session.set_method(st, m))
        )
        self.pipe.field_changed.connect(self._on_field)
        self.pipe.fix_requested.connect(self._on_fix)
        self.variants.selected.connect(self._on_variant)
        self.cb_gt.toggled.connect(lambda v: self.canvas.set_overlays(gt=v))
        self.cb_roi.toggled.connect(lambda v: self.canvas.set_overlays(roi=v))
        self.cb_lab.toggled.connect(lambda v: self.canvas.set_overlays(labels=v))
        self.canvas.zoom_changed.connect(lambda _z: self._update_zoom_info())
        w = self.worker
        w.prepared.connect(self._on_prepared)
        w.finished_preview.connect(self._on_preview)
        w.finished_thumb.connect(self._on_thumb)
        w.failed.connect(self._on_failed)

    # ------------------------------------------------------------------ 세션 → 위젯

    def sync_widgets(self) -> None:
        ses = self.session
        self.strip.sync(ses.recipe, ses.n_variants, ses.long_side)
        self.pipe.sync(ses.recipe)
        self._sync_fix()  # 카드 편집(reprepare)으로 경고가 생기거나 사라졌을 수 있다
        summary = ""
        if ses.prepared is not None:
            b = ses.prepared.bank
            if ses.recipe.bankless:
                head = f"보관함 없음 — {ses.recipe.pipeline.source.method} (클래스 {ses.recipe.pipeline.source.cls})"
            else:
                rows = ", ".join(
                    f"{r.cls} {r.count}"
                    + (
                        f"(추정 {r.estimated}"
                        + (f" · 신뢰도 낮음 {r.low_conf}" if r.low_conf else "")
                        + ")"
                        if r.estimated
                        else ""
                    )
                    for r in b.summary()
                )
                head = f"보관함 {b.name}: 조각 {len(b)}개 — {rows}"
            summary = f"{head}\n바탕 이미지 {len(ses.prepared.targets)}장 · 설정 지문 {ses.prepared.pipeline_hash[:12]}"
        self.inputs.sync(ses.recipe, summary)
        self.strip.note.setText(self._note())
        self.context.emit(self._context())

    def _note(self) -> str:
        ses = self.session
        parts = []
        if ses.long_side:
            parts.append(
                f"미리보기는 긴 변 {ses.long_side}px 축소본 · 실제 결과는 일괄 생성(원본 해상도)에서"
            )
        if ses.warnings:
            parts.append(f"경고 {len(ses.warnings)}건: {ses.warnings[0]}")
        return " · ".join(parts)

    def _context(self) -> str:
        ses = self.session
        t = ses.target
        return (
            f"바탕 {t.name if t else '–'} · 보관함 {ses.prepared.bank.name if ses.prepared else '–'} · "
            f"레시피 {ses.recipe.name} ({ses.recipe.pipeline.preset}, seed {ses.recipe.seed})"
        )

    def directional_classes(self) -> list[str]:
        """은행에서 조명 의존으로 판정된 클래스(lightR ≥ 0.5, n ≥ 3) 중 이 레시피가 뽑는 것."""
        prep = self.session.prepared
        if prep is None or prep.recipe.bankless:
            return []
        selected = set(prep.recipe.effective_classes(prep.bank))
        return [r.cls for r in prep.bank.summary() if r.cls in selected and r.directional]

    def _sync_fix(self) -> None:
        """기하 카드의 '고치기' 버튼 — prepare 경고에 `geometry:`(조명) 가 있을 때만."""
        has = any(w.startswith("geometry:") for w in self.session.warnings)
        self.pipe.cards["geometry"].set_fix("▶ 빛 방향 클래스만 ±15°로 좁히기" if has else None)

    def _on_fix(self, stage: str) -> None:
        """기하 카드 '고치기' — 조명 의존 클래스에 `DENT_OVERRIDE` 를 per_class 로(다른 클래스는 그대로)."""
        if stage != "geometry":
            return
        classes = self.directional_classes()
        if not classes:
            self.status.emit("조명 의존 클래스가 없습니다")
            return
        per = {
            c: o.model_dump() for c, o in self.session.recipe.pipeline.geometry.per_class.items()
        }
        prep = self.session.prepared
        dirs = {r.cls: r.light_dir for r in prep.bank.summary()} if prep is not None else {}
        for c in classes:
            per[c] = {**per.get(c, {}), **R.dent_override_for(dirs.get(c))}
        self._edit(lambda: self.session.set_stage_field("geometry", "per_class", per))
        self.status.emit(
            "geometry.per_class ← "
            + ", ".join(f"{c}(±15°, flip {per[c]['flip']})" for c in classes)
        )

    def _fit_warnings(self, res: PreviewResult, roi: np.ndarray | None) -> list[str]:
        """이 대상의 허용 영역 최대 폭(미리보기 ROI ÷ 축소 배율 = 원본 px) vs 클래스별 패치 폭 — 빠듯/불가면 배치 카드 ⚠."""
        prep = self.session.prepared
        if prep is None or roi is None:
            return []
        from anograft.core.stages.placement import roi_max_width

        margin = int(getattr(self.session.recipe.pipeline.placement, "margin_px", 0))
        width = roi_max_width(roi, margin) / max(res.scale, 1e-6)
        short_side = float(min(roi.shape[:2])) / max(res.scale, 1e-6)
        fit = runner.fit_from_widths(
            prep, [(res.target.path.name, round(width, 1))], target_short_side=short_side
        )
        self._fit = fit
        if fit is None:
            return []
        return [w for w in (fit.source_warning(), fit.warning()) if w]

    def _update_zoom_info(self) -> None:
        w, h = self.canvas.image_size()
        res = self._results.get(self.session.variant_index)
        scale = f" · {res.scale:.0%}" if res is not None and res.scale < 1 else ""
        self.zoom_info.setText(f"{w} × {h}{scale} · 줌 {self.canvas.zoom * 100:.0f}%")

    # ------------------------------------------------------------------ 편집(전부 세션을 거친다)

    def _edit(self, fn) -> None:
        try:
            fn()
        except SessionError as e:
            self.status.emit(str(e).splitlines()[0])
            QMessageBox.warning(self, "레시피 오류", str(e))
            self.sync_widgets()
            return
        self.pipe.clear_errors()
        self.sync_widgets()
        self.request_previews()

    def _on_field(self, stage: str, field: str, value: object) -> None:
        """카드 폼 값 변경 — 세션 재검증. 실패하면 대화상자 대신 **그 카드에 빨간 줄** + 위젯은 세션 값으로 되돌린다."""
        try:
            self.session.set_stage_field(stage, field, value)
        except SessionError as e:
            msg = f"{field}: {field_error_detail(str(e))}"
            self.pipe.set_error(stage, field, msg)
            self.status.emit(msg)
            self.pipe.sync(self.session.recipe)  # 위젯 되돌리기 (오류 줄은 유지)
            return
        self.pipe.clear_errors()
        self.sync_widgets()
        self.request_previews()

    def _on_variants(self, n: int) -> None:
        self.session.set_n_variants(n)
        self.request_previews()

    def _on_long_side(self, px: int) -> None:
        self.session.set_long_side(px)
        self.sync_widgets()
        self.request_previews()

    def _on_target(self, index: int) -> None:
        self.session.select_target(index)
        self.context.emit(self._context())
        self.request_previews()

    def _on_variant(self, k: int) -> None:
        self.session.select_variant(k)
        res = self._results.get(k)
        if res is not None:
            self._show(res)
        else:
            self.status.emit(f"v{k + 1} 계산 중…")

    # ------------------------------------------------------------------ 열기·저장

    def _on_um_per_px(self, value: object) -> None:
        """대상 픽셀 피치 — 레시피 inputs.um_per_px(재검증 → 재준비 → 미리보기). 은행 피치가 없으면 경고가 그대로 알린다."""
        try:
            self.session.set_field(("inputs", "um_per_px"), value)
        except SessionError as e:
            QMessageBox.warning(self, "µm/px", str(e))
            self.sync_widgets()
            return
        self.sync_widgets()
        self.request_previews()

    def open_inputs(self, bank: str, targets: str) -> None:
        try:
            # bank 빈칸 = 은행 없음(self-cut·perlin 프리셋). bank 소스인데 비어 있으면 레시피 검증이 막는다(SessionError)
            self.session.set_paths(bank, targets or None)
        except SessionError as e:
            QMessageBox.warning(self, "입력 오류", str(e))
            return
        self.session.prepared = None
        self.worker.set_prepared(None)
        self.canvas.clear("보관함·바탕 이미지 읽는 중…")
        self.status.emit("보관함·바탕 이미지 읽는 중…")
        self.worker.submit(
            PreviewJob(
                KIND_PREPARE, self.session.generation, recipe=self.session.recipe, priority=0
            )
        )

    def open_recipe(self, path: str | Path) -> None:
        try:
            self.session.load(path)
        except SessionError as e:
            QMessageBox.warning(self, "레시피 열기", str(e))
            return
        self.sync_widgets()
        if self.session.needs_prepare():
            self.open_inputs(
                self.session.recipe.inputs.bank_key(),
                self.session.recipe.inputs.targets.as_posix(),
            )
        else:
            self.request_previews()
        msg = f"레시피 열림: {Path(path).as_posix()}"
        if self.session.path_notes:
            msg += (
                f" · 경로 {len(self.session.path_notes)}개를 레시피 파일 기준으로 해석: "
                + "; ".join(self.session.path_notes)
            )
        self.status.emit(msg)
        self.recipe_opened.emit(Path(path).as_posix())

    def gallery_thumbs(self) -> dict[str, Any] | None:
        """갤러리 썸네일 — 준비된 세션 + 고른 바탕 이미지가 있을 때만(없으면 None → 문안만)."""
        from anograft.gui.studio.gallery import render_preset_thumbs

        ses = self.session
        if ses.prepared is None or ses.target is None:
            return None
        try:
            return render_preset_thumbs(ses.prepared, ses.recipe, ses.target)
        except Exception as e:  # fail-soft
            self.status.emit(f"프리셋 썸네일 실패: {e}")
            return None

    def open_preset_gallery(self) -> str | None:
        """프리셋 갤러리 → 고르면 콤보를 바꿔(= set_preset) 카드·미리보기가 따라온다. 반환 = 고른 이름(취소면 None)."""
        from anograft.gui.studio.preset_gallery import PresetGalleryDialog

        dlg = PresetGalleryDialog(self.session.recipe.pipeline.preset, self.gallery_thumbs(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.chosen:
            return None
        self.apply_preset(dlg.chosen)
        return dlg.chosen

    def apply_preset(self, name: str) -> None:
        i = self.strip.preset.findData(name)
        if i >= 0 and i != self.strip.preset.currentIndex():
            self.strip.preset.setCurrentIndex(i)  # → preset_changed → set_preset

    def open_recipe_dialog(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "레시피 열기", "recipes", "레시피 (*.yaml *.yml)")
        if f:
            self.open_recipe(f)

    def save_recipe_dialog(self) -> None:
        start = (
            self.session.recipe_path.as_posix()
            if self.session.recipe_path
            else "recipes/studio.yaml"
        )
        f, _ = QFileDialog.getSaveFileName(self, "레시피 저장", start, "레시피 (*.yaml)")
        if f:
            p = self.session.save(f)
            self.status.emit(f"레시피 저장: {p.as_posix()} → {self.session.run_command()}")
            self.recipe_opened.emit(p.as_posix())

    def send_to_batch(self) -> None:
        """현재 레시피를 배치 탭으로(저장하지 않아도 된다 — 배치 탭이 Recipe 객체를 받는다). 메인 창이 탭을 전환한다.
        CLI 한 줄도 클립보드에 남긴다."""
        cmd = self.session.run_command()
        QGuiApplication.clipboard().setText(cmd)
        self.send_to_batch_requested.emit()
        self.status.emit(f"일괄 생성 탭으로 보냈습니다 · 클립보드에 CLI 명령: {cmd}")

    # ------------------------------------------------------------------ 미리보기 요청·수신

    def request_previews(self) -> None:
        ses = self.session
        if ses.prepared is None or ses.target is None:
            return
        gen = ses.generation
        self.worker.invalidate(gen)
        self.worker.set_prepared(ses.prepared)  # reprepare 로 파이프라인이 바뀌었을 수 있다
        self._results.clear()
        self.variants.reset(ses.n_variants, ses.variant_index)
        self.pipe.set_trace(
            None, self.session.warnings
        )  # prepare 경고(접두 있는 것)는 카드에 남긴다
        order = [ses.variant_index] + [k for k in range(ses.n_variants) if k != ses.variant_index]
        for k in order:
            self.worker.submit(
                PreviewJob(
                    KIND_PREVIEW,
                    gen,
                    target=ses.target,
                    index=k,
                    long_side=ses.long_side,
                    priority=0 if k == ses.variant_index else 1,
                )
            )
        self.status.emit(f"미리보기 계산 중… ({ses.n_variants}장)")

    def _on_prepared(self, prep: runner.Prepared) -> None:
        if not self.session.accept_prepared(prep):
            return
        self.worker.set_prepared(self.session.prepared)
        self.rail.set_targets(self.session.targets, self.session.target_index)
        for p in self.session.targets:
            self.worker.submit(
                PreviewJob(KIND_THUMB, self.session.generation, target=p, priority=2)
            )
        self.sync_widgets()
        n_warn = len(self.session.warnings)
        self.pipe.set_classes([] if prep.recipe.bankless else list(prep.bank.classes))
        self.pipe.sync(self.session.recipe)  # 표가 생긴 뒤 값 채우기 · 정보 줄 숨김
        self._sync_fix()
        self.status.emit(
            f"준비 완료 — 결함 조각 {len(prep.bank)}개 · 바탕 이미지 {len(prep.targets)}장"
            + (f" · 경고 {n_warn}건" if n_warn else "")
        )
        self.request_previews()

    def _on_thumb(self, res: ThumbResult) -> None:
        if res.job.target is not None:
            self.rail.set_thumb(res.job.target, res.image, res.shape)

    def _on_preview(self, res: PreviewResult) -> None:
        if res.job.generation != self.session.generation:
            return
        k = res.job.index
        self._results[k] = res
        r = res.result
        if r.status == "ok":
            caption = ", ".join(sorted({i.cls for i in r.instances})) or "ok"
            low = self._low_confidence_sources(r)
            flipped = self._flipped_instances(r)
            tooltip = "결함 조각: " + ", ".join(self._source_ids(r))
            if low:
                caption += " ⚠"
                tooltip += (
                    "\n⚠ 마스크 신뢰도가 낮은 조각: "
                    + ", ".join(low)
                    + " — 보관함 탭에서 다듬으세요"
                )
            if flipped:
                caption += " ↯"
                tooltip += (
                    "\n↯ 빛 방향이 뒤집힌 듯함: "
                    + ", ".join(f"{r.instances[i].cls}#{i + 1}" for i in flipped)
                    + " — 하이라이트가 실제 조각과 반대쪽입니다(크기·회전 카드의 고치기 또는 프리셋 dent-graft)"
                )
            self.variants.set_result(k, promote_to_bgr(r.image), caption, tooltip)
        else:
            self.variants.set_failed(k, r.reason or "skipped")
        if k == self.session.variant_index:
            self._show(res)

    @staticmethod
    def _source_ids(r) -> list[str]:
        return [
            str(d.get("source", {}).get("source_id", ""))
            for d in r.sidecar.get("defects", [])
            if "gt" in d
        ]

    def _flipped_instances(self, r) -> list[int]:
        """이 변형의 인스턴스 중 조명이 실제 클래스 방향과 반대(> 90°)인 것 — 검수 탭 '조명 뒤집힘 의심'을 미리보기에서.
        은행에 방향이 유의한 클래스가 없으면 빈 목록(1024 축소본이라 각도는 근사)."""
        prep = self.session.prepared
        if prep is None or prep.recipe.bankless or r.status != "ok" or not r.instances:
            return []
        real = prep.bank.real_lighting_direction()
        if not real:
            return []
        gray = cv2.cvtColor(promote_to_bgr(r.image), cv2.COLOR_BGR2GRAY)
        return flipped_instances(gray, [(i.cls, i.mask) for i in r.instances], real)

    def _low_confidence_sources(self, r) -> list[str]:
        """이 변형에 쓰인 소스 중 confidence < 0.5(KNOWN-ISSUES #3) — 미리보기가 그럴듯해도 마스크가 헐거울 수 있다."""
        prep = self.session.prepared
        if prep is None or prep.bank is None:
            return []
        low = {s.id for s in prep.bank.low_confidence()}
        return [sid for sid in self._source_ids(r) if sid in low]

    def _on_failed(self, err: JobError) -> None:
        if err.job.kind == KIND_PREPARE:
            self.canvas.clear("보관함·바탕 이미지를 열 수 없습니다")
            QMessageBox.warning(self, "준비 실패", err.message)
            self.status.emit(err.message.splitlines()[0])
        elif err.job.kind == KIND_PREVIEW and err.job.generation == self.session.generation:
            self.variants.set_failed(err.job.index, err.message.splitlines()[0])
            self.status.emit(f"v{err.job.index + 1} 실패: {err.message.splitlines()[0]}")

    def _show(self, res: PreviewResult) -> None:
        r = res.result
        roi = res.steps[0].ctx.roi if res.steps and res.steps[0].stage == "roi" else None
        synthetic: np.ndarray | None = promote_to_bgr(r.image) if r.status == "ok" else None
        self.canvas.set_images(
            res.target.image, synthetic, r.gt_mask if r.status == "ok" else None, r.instances, roi
        )
        self.pipe.set_trace(  # prepare 경고(geometry: 조명 …) + 결과의 fail-soft 경고 + 배치 가능성 진단을 카드에
            res.steps, [*self.session.warnings, *r.warnings, *self._fit_warnings(res, roi)]
        )
        self._update_zoom_info()
        defects = [d for d in r.sidecar.get("defects", []) if "gt" in d]
        what = " · ".join(
            f"{d['source']['class']} {d['source']['source_id']} ({d['blend']['method']})"
            for d in defects
        )
        if r.status == "ok":
            self.status.emit(
                f"v{res.job.index + 1}: {what} · {res.elapsed_s * 1000:.0f} ms"
                + (f" · 축소 {res.scale:.0%}" if res.scale < 1 else "")
            )
        else:
            extra = len(r.warnings) - 1
            self.status.emit(
                f"v{res.job.index + 1}: skipped — {r.reason}"
                + (f" (경고 {extra}건 더 — 파이프라인 카드 참조)" if extra > 0 else "")
            )
