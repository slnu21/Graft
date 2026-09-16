"""``ReviewTab`` — 검수 탭(v0.7): 출력 폴더(``manifest.csv``)를 열어 합성 결과를 썸네일 그리드로 보고 **채택(A)/반려(R)/보류(U)** 를
매긴다(``review.csv`` 자동 저장). 오른쪽 상세(이미지 + GT 윤곽·bbox, 메타·경고·메모) · 아래 **분포 히스토그램**(합성 인스턴스 vs 은행
실제 소스 — 면적/긴 변, 로그 구간; 합성 ``#00A188`` / 실제 ``#C8841C``) · **정리본 내보내기**(반려 제외 사본, ``io.prune``).

썸네일은 정본 ``images/`` 를 읽어 GT 마스크 윤곽을 얹은 뒤 긴 변 ``THUMB`` 로 줄인다(index 로 캐시). 4K 출력 100장도 첫 열기 몇 초.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anograft.core.channels import promote_to_bgr
from anograft.gui.qt_image import to_qpixmap
from anograft.gui.review.session import FILTERS, Histogram, ReviewError, ReviewItem, ReviewSession
from anograft.gui.studio.panels import flat_icon, h4
from anograft.gui.theme import COLORS, REAL_COLOR, SYNTH_COLOR  # 검수 계열색 고정(색각 검증)
from anograft.io import imgio
from anograft.io.report import lighting_broken_classes
from anograft.preview import GT_EDGE

THUMB = 176
FILTER_LABELS: dict[str, str] = {
    "all": "전체 all",
    "unreviewed": "미검수 unreviewed",
    "accept": "채택 accepted",
    "reject": "반려 rejected",
    "fallback": "폴백 fallback",
    "skipped": "skipped",
    "flipped": "조명 뒤집힘 의심 flipped lighting",
}
VERDICT_MARK: dict[str, str] = {"accept": "✓ ", "reject": "✗ ", "": ""}


def overlay_image(image: np.ndarray, mask: np.ndarray | None, *, long_side: int) -> np.ndarray:
    """이미지 + GT 윤곽(초록) → 긴 변 ``long_side`` 축소(INTER_AREA)."""
    img = promote_to_bgr(image).copy()
    if mask is not None and mask.shape[:2] == img.shape[:2]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = max(1, round(max(img.shape[:2]) / 400))
        cv2.drawContours(img, contours, -1, GT_EDGE, thick)
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if s < 1:
        img = cv2.resize(
            img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
        )
    return img


class HistogramWidget(QWidget):
    """두 계열 막대(합성 teal · 실제 amber)를 같은 구간에 나란히. ``set_histogram`` 만 부르면 된다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.hist: Histogram | None = None
        self.title = ""
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_histogram(self, hist: Histogram | None, title: str = "") -> None:
        self.hist, self.title = hist, title
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt 규약)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(COLORS["stage"]))
        p.setPen(QColor(COLORS["tx3"]))
        p.drawText(8, 16, self.title)
        h = self.hist
        if h is None or (sum(h.a) + sum(h.b)) == 0:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "분포 없음 — 출력 폴더를 여세요")
            p.end()
            return
        left, top, right, bottom = 8.0, 40.0, self.width() - 8.0, self.height() - 22.0
        n = h.bins
        peak = max(max(h.a), max(h.b), 1)
        bw = (right - left) / n
        for i in range(n):
            x0 = left + i * bw
            for k, (count, color) in enumerate(((h.a[i], SYNTH_COLOR), (h.b[i], REAL_COLOR))):
                if count <= 0:
                    continue
                bh = (bottom - top) * count / peak
                rect = QRectF(x0 + 1 + k * (bw / 2 - 1), bottom - bh, bw / 2 - 2, bh)
                p.fillRect(rect, QColor(color))
        p.setPen(QColor(COLORS["tx3"]))
        p.drawLine(int(left), int(bottom), int(right), int(bottom))
        for i in (0, n // 2, n):
            v = h.edges[i]
            label = (f"{v:.0f}" if v >= 10 else f"{v:.1f}") if h.log else f"{v:+.0f}"
            x = left + i * bw
            p.drawText(int(min(x, right - 40)), self.height() - 6, label)
        p.setPen(QColor(SYNTH_COLOR))
        p.drawText(8, 32, f"■ 합성 {sum(h.a)}")
        p.setPen(QColor(REAL_COLOR))
        p.drawText(8 + 90, 32, f"■ 실제(은행) {sum(h.b)}")
        p.end()


class ReviewTab(QWidget):
    status = Signal(str)
    review_saved = Signal(str)  # review.csv 경로(posix)

    def __init__(self, session: ReviewSession | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session or ReviewSession()
        self._thumbs: dict[str, QPixmap] = {}
        self._rows: list[ReviewItem] = []
        self.autosave = True
        self.open_report_in_browser = True  # 테스트는 False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._strip())
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._left())
        split.addWidget(self._center())
        split.addWidget(self._right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([230, 900, 420])
        root.addWidget(split, 1)
        self._wire()
        self.refresh()

    # ------------------------------------------------------------------ 조립

    def _strip(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Strip")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 7, 14, 7)
        h.setSpacing(10)
        lab = QLabel("출력 Output")
        lab.setObjectName("Muted")
        h.addWidget(lab)
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("검수할 출력 폴더 (manifest.csv 가 있는 곳)")
        self.root_edit.setMinimumWidth(260)
        self.btn_pick = QPushButton("폴더")
        self.btn_pick.setFixedWidth(44)
        self.btn_open = QPushButton("열기 Open")
        self.btn_prune = QPushButton("정리본 내보내기 Export")
        self.btn_report = QPushButton("리포트 Report")
        h.addWidget(self.root_edit, 1)
        h.addWidget(self.btn_pick)
        h.addWidget(self.btn_open)
        h.addWidget(self.btn_prune)
        h.addWidget(self.btn_report)
        self.summary = QLabel("")
        self.summary.setObjectName("Muted")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        h.addWidget(self.summary, 1)
        return bar

    def _left(self) -> QWidget:
        left = QWidget()
        left.setObjectName("Inputs")
        v = QVBoxLayout(left)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        v.addWidget(h4("필터 Filter"))
        self.f_which = QComboBox()
        for k in FILTERS:
            self.f_which.addItem(FILTER_LABELS[k], k)
        self.f_cls = QComboBox()
        v.addWidget(self.f_which)
        v.addLayout(self._kv("클래스", self.f_cls))
        v.addSpacing(8)
        v.addWidget(h4("분포 Distribution"))
        self.dist_key = QComboBox()
        self.dist_key.addItem("면적 area (px)", "area")
        self.dist_key.addItem("긴 변 length (px)", "length")
        self.dist_key.addItem("대비 contrast (gray)", "contrast")
        self.dist_key.addItem("질감 texture (∇ 평균)", "texture")
        self.dist_key.addItem("선명도 sharpness (∇² 분산)", "sharpness")
        self.dist_key.addItem("조명 방향 lighting (°)", "lighting")
        v.addWidget(self.dist_key)
        self.dist_class = (
            QComboBox()
        )  # 외형 지표만 클래스별(조명 방향은 클래스마다 달라 전체는 섞인다)
        self.dist_class.setToolTip(
            "클래스별 분포 — 외형 지표(대비·질감·선명도·조명 방향)만 · Per-class distribution"
        )
        v.addWidget(self.dist_class)
        self.hist = HistogramWidget()
        v.addWidget(self.hist, 1)
        self.count = QLabel("")
        self.count.setObjectName("Hint")
        self.count.setWordWrap(True)
        v.addWidget(self.count)
        left.setMinimumWidth(220)
        return left

    def _center(self) -> QWidget:
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setSpacing(6)
        self.grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(False)
        ml.addWidget(self.grid, 1)
        bar = QWidget()
        bar.setObjectName("Strip")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 5, 14, 5)
        bl.setSpacing(10)
        self.btn_accept = QPushButton("채택 Accept (A)")
        self.btn_accept.setObjectName("Primary")
        self.btn_reject = QPushButton("반려 Reject (R)")
        self.btn_clear = QPushButton("보류 Unreview (U)")
        bl.addWidget(self.btn_accept)
        bl.addWidget(self.btn_reject)
        bl.addWidget(self.btn_clear)
        self.cb_next = QCheckBox("판정 후 다음으로")
        self.cb_next.setChecked(True)
        bl.addWidget(self.cb_next)
        self.btn_reject_shown = QPushButton("표시된 것 전부 반려")
        self.btn_reject_shown.setToolTip(
            "현재 필터로 보이는 합성 결과를 모두 반려 — 예: 필터 '조명 뒤집힘 의심' 뒤에 한 번에"
        )
        bl.addWidget(self.btn_reject_shown)
        self.btn_next_unreviewed = QPushButton("다음 미검수 (N)")
        self.btn_next_unreviewed.setToolTip(
            "현재 위치 다음의 미검수 합성 결과로 이동(끝이면 처음부터)"
        )
        bl.addWidget(self.btn_next_unreviewed)
        bl.addStretch(1)
        ml.addWidget(bar)
        return mid

    def _right(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Pipe")
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        v.addWidget(h4("결과 Result"))
        self.detail = QLabel("")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setMinimumHeight(260)
        self.detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addWidget(self.detail)
        self.meta = QLabel("합성 결과를 고르면 상세가 여기에")
        self.meta.setObjectName("Hint")
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.meta)
        self.note = QLineEdit()
        self.note.setPlaceholderText("메모 (review.csv note)")
        v.addLayout(self._kv("메모", self.note))
        v.addStretch(1)
        self.result = QLabel("")
        self.result.setObjectName("Muted")
        self.result.setWordWrap(True)
        v.addWidget(self.result)
        side.setMinimumWidth(360)
        return side

    @staticmethod
    def _muted(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("Muted")
        return lab

    def _kv(self, key: str, widget: QWidget) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(6)
        lab = self._muted(key)
        lab.setFixedWidth(46)
        h.addWidget(lab)
        h.addWidget(widget, 1)
        return h

    def _wire(self) -> None:
        self.btn_pick.clicked.connect(self._pick_root)
        self.btn_open.clicked.connect(lambda: self.open_root(self.root_edit.text()))
        self.root_edit.returnPressed.connect(lambda: self.open_root(self.root_edit.text()))
        self.btn_prune.clicked.connect(self.export_pruned_dialog)
        self.btn_report.clicked.connect(self.write_report)
        self.f_which.currentIndexChanged.connect(lambda _i: self.refresh_grid())
        self.f_cls.currentIndexChanged.connect(lambda _i: self.refresh_grid())
        self.dist_key.currentIndexChanged.connect(lambda _i: self.refresh_hist())
        self.dist_class.currentIndexChanged.connect(lambda _i: self.refresh_hist())
        self.grid.itemSelectionChanged.connect(self._on_select)
        self.btn_accept.clicked.connect(lambda: self.verdict("accept"))
        self.btn_reject.clicked.connect(lambda: self.verdict("reject"))
        self.btn_reject_shown.clicked.connect(lambda: self.verdict_shown("reject"))
        self.btn_clear.clicked.connect(lambda: self.verdict(""))
        self.note.editingFinished.connect(self._on_note)
        QShortcut(QKeySequence("A"), self.grid, activated=lambda: self.verdict("accept"))
        QShortcut(QKeySequence("R"), self.grid, activated=lambda: self.verdict("reject"))
        QShortcut(QKeySequence("U"), self.grid, activated=lambda: self.verdict(""))
        QShortcut(QKeySequence("N"), self.grid, activated=self.next_unreviewed)
        self.btn_next_unreviewed.clicked.connect(self.next_unreviewed)

    # ------------------------------------------------------------------ 열기

    def _pick_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "검수할 출력 폴더", self.root_edit.text() or ".")
        if d:
            self.open_root(d)

    def open_root(self, root: str | Path) -> bool:
        text = str(root).strip()
        if not text:
            self._on_error("출력 폴더를 지정하세요")
            return False
        try:
            self.session.load(text)
        except ReviewError as e:
            self._on_error(str(e))
            return False
        self.root_edit.setText(Path(text).as_posix())
        self._thumbs.clear()
        self._sync_filters()
        self.refresh()
        msg = f"검수 열림: {self.session.summary_text()}"
        if self.session.warnings:
            msg += f" · {self.session.warnings[0]}"
        self.status.emit(msg)
        return True

    def _sync_filters(self) -> None:
        cur = self.f_cls.currentText()
        self.f_cls.blockSignals(True)
        self.f_cls.clear()
        self.f_cls.addItem("(전체)")
        self.f_cls.addItems(self.session.classes())
        idx = self.f_cls.findText(cur)
        self.f_cls.setCurrentIndex(idx if idx >= 0 else 0)
        self.f_cls.blockSignals(False)

    # ------------------------------------------------------------------ 그리드

    def _thumb(self, it: ReviewItem) -> QPixmap:
        pm = self._thumbs.get(it.index)
        if pm is None:
            assert self.session.root is not None
            try:
                image, _g = imgio.read_image(self.session.root / it.image)
                mask = imgio.read_mask(self.session.root / it.mask) if it.mask else None
                pm = to_qpixmap(overlay_image(image, mask, long_side=THUMB))
            except (OSError, imgio.ImageReadError):
                pm = QPixmap(THUMB, THUMB)
                pm.fill(QColor(COLORS["stage"]))
            self._thumbs[it.index] = pm
        return pm

    def refresh_grid(self) -> None:
        which = str(self.f_which.currentData() or "all")
        cls = self.f_cls.currentText()
        cls_f = None if cls in ("(전체)", "") else cls
        self.grid.blockSignals(True)
        self.grid.clear()
        self._rows = self.session.filtered(which, cls_f) if self.session.loaded else []
        for it in self._rows:
            if it.status == "skipped":
                pm = QPixmap(THUMB, THUMB)
                pm.fill(QColor(COLORS["stage"]))
                item = QListWidgetItem(flat_icon(pm), f"skip {it.index}")
                item.setToolTip(f"{it.index} skipped: {it.reason}")
            else:
                mark = VERDICT_MARK.get(it.verdict, "")
                label = (
                    f"{mark}{it.index}" if it.status == "ok" else f"정상 {Path(it.image).stem[2:]}"
                )
                item = QListWidgetItem(flat_icon(self._thumb(it)), label)
                tip = f"{it.index} · {', '.join(it.classes) or '정상'} · {it.area_px:,} px"
                if it.fallback:
                    tip += " · 폴백"
                if it.note:
                    tip += f" · {it.note}"
                item.setToolTip(tip)
            item.setData(Qt.ItemDataRole.UserRole, it.index)
            self.grid.addItem(item)
        self.grid.blockSignals(False)
        c = self.session.counts() if self.session.loaded else {}
        self.count.setText(
            f"표시 {len(self._rows)} · 채택 {c.get('accept', 0)} · 반려 {c.get('reject', 0)} · 미검수 {c.get('unreviewed', 0)}"
            if c
            else ""
        )
        self._on_select()

    def refresh_hist(self) -> None:
        if not self.session.loaded:
            self.hist.set_histogram(None)
            return
        key = str(self.dist_key.currentData() or "area")
        cls = self._sync_class_combo(key)
        h = self.session.distribution_by_class(key, cls) if cls else self.session.distribution(key)
        title = {
            "area": "면적 px (로그 구간)",
            "length": "긴 변 px (로그 구간)",
            "contrast": "대비 gray (마스크 − 링, 선형)",
            "texture": "질감 — 마스크 안 그래디언트 평균 (선형)",
            "sharpness": "선명도 — 마스크 안 라플라시안 분산 (선형)",
            "lighting": "조명 방향 ° (0 = →, 90 = ↓; 링에서 밝은 쪽)",
        }.get(key, key)
        if cls:
            title += f" · 클래스 {cls}"
        if key == "lighting":
            fmt = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
            if cls:
                rs, rr, ns, nr = self.session.lighting_r_for(cls)
                title += f" · R 합성 {fmt(rs)} (n {ns}) / 실제 {fmt(rr)} (n {nr})"
                if cls in self.session.directional_classes():
                    title += " · 실제 방향 유의"
            else:
                rs, rr = self.session.lighting_concentration()
                title += f" · 일관성 R 합성 {fmt(rs)} / 실제 {fmt(rr)}"
            per_class = self.session.lighting_concentration_by_class()
            broken = lighting_broken_classes(per_class, self.session.directional_classes())
            if broken and (not cls or cls in broken):
                title += f" · ⚠ {', '.join(broken)} 회전이 조명을 뒤집음 → dent-graft"
        if self.session.bank is None:
            title += " — 은행 없음"
        elif self.session.real_classes():
            title += f" · 실제 = {', '.join(self.session.real_classes() or [])}"
        self.hist.set_histogram(h, title)

    def _sync_class_combo(self, key: str) -> str:
        """분포 키에 맞춰 클래스 콤보를 채운다(외형 지표만). 반환: 선택된 클래스("" = 전부)."""
        options = self.session.class_options(key)
        prev = str(self.dist_class.currentData() or "")
        self.dist_class.blockSignals(True)
        self.dist_class.clear()
        self.dist_class.addItem("전체 all classes", "")
        for c in options:
            self.dist_class.addItem(c, c)
        i = self.dist_class.findData(prev) if prev in options else 0
        self.dist_class.setCurrentIndex(max(i, 0))
        self.dist_class.setEnabled(bool(options))
        self.dist_class.blockSignals(False)
        return str(self.dist_class.currentData() or "")

    def selected_indices(self) -> list[str]:
        return [str(it.data(Qt.ItemDataRole.UserRole)) for it in self.grid.selectedItems()]

    def select_index(self, index: str) -> None:
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            it.setSelected(str(it.data(Qt.ItemDataRole.UserRole)) == index)
            if str(it.data(Qt.ItemDataRole.UserRole)) == index:
                self.grid.setCurrentItem(it)

    def current_index(self) -> str | None:
        ids = self.selected_indices()
        return ids[0] if ids else None

    # ------------------------------------------------------------------ 상세

    def _on_select(self) -> None:
        ids = self.selected_indices()
        ok = bool(ids) and all(
            next((x for x in self._rows if x.index == i), None) is not None
            and next(x for x in self._rows if x.index == i).status == "ok"
            for i in ids
        )
        for b in (self.btn_accept, self.btn_reject, self.btn_clear):
            b.setEnabled(ok)
        self.note.setEnabled(ok and len(ids) == 1)
        if not ids:
            self.detail.clear()
            self.meta.setText(
                "합성 결과를 고르면 상세가 여기에"
                if self.session.loaded
                else "출력 폴더를 열어 주세요"
            )
            self.note.setText("")
            return
        if len(ids) > 1:
            self.detail.clear()
            self.meta.setText(f"{len(ids)}개 선택 — A/R/U 로 한꺼번에 판정")
            self.note.setText("")
            return
        it = self.session.item(ids[0])
        if it.status == "ok" or it.status == "normal":
            assert self.session.root is not None
            try:
                image, _g = imgio.read_image(self.session.root / it.image)
                mask = imgio.read_mask(self.session.root / it.mask) if it.mask else None
                self.detail.setPixmap(to_qpixmap(overlay_image(image, mask, long_side=380)))
            except (OSError, imgio.ImageReadError) as e:
                self.detail.clear()
                self.meta.setText(f"이미지 읽기 실패: {e}")
                return
        else:
            self.detail.clear()
        lines = [
            f"<b>{it.index}</b> · {it.status}" + (f" · <b>{it.verdict}</b>" if it.verdict else "")
        ]
        if it.status == "ok":
            lines.append(
                f"클래스 {', '.join(it.classes)} · 결함 {len(it.classes)} · GT {it.area_px:,} px · blend {it.blend}"
                + (" · <b>폴백</b>" if it.fallback else "")
            )
            lines.append("소스 " + ", ".join(it.source_ids))
            for inst in it.instances:
                bb = inst.get("bbox") or [0, 0, 0, 0]
                lines.append(
                    f"· {inst.get('class')} {inst.get('area_px', 0):,} px bbox {bb[0]},{bb[1]} {bb[2]}×{bb[3]}"
                )
        elif it.status == "skipped":
            lines.append(f"skipped: {it.reason}")
        lines.append(f"대상 {it.target}")
        for w in it.warnings[:3]:
            lines.append(f"⚠ {w}")
        self.meta.setText("<br>".join(lines))
        self.note.setText(it.note)

    # ------------------------------------------------------------------ 판정

    def verdict(self, verdict: str) -> int:
        ids = [i for i in self.selected_indices()]
        if not ids:
            return 0
        n = 0
        for idx in ids:
            try:
                self.session.set_verdict(idx, verdict)
                n += 1
            except ReviewError as e:
                self._on_error(str(e))
        if n:
            self._after_change(ids[-1] if len(ids) == 1 else None)
        return n

    def verdict_shown(self, verdict: str) -> int:
        """현재 필터로 표시된 합성 결과 전부에 같은 판정 — 뒤집힘 의심·폴백 같은 필터 뒤의 일괄 처리."""
        ids = [it.index for it in self._rows if it.status == "ok"]
        n = 0
        for idx in ids:
            try:
                self.session.set_verdict(idx, verdict)
                n += 1
            except ReviewError as e:
                self._on_error(str(e))
        if n:
            self._after_change(None)
            self.status.emit(f"표시된 {n}건 {'반려' if verdict == 'reject' else verdict}")
        return n

    def next_unreviewed(self) -> str | None:
        """현재 행 다음의 미검수 합성 결과를 선택(끝이면 처음부터 다시). 없으면 None."""
        n = self.grid.count()
        if n == 0:
            return None
        start = self.grid.currentRow() + 1 if self.grid.currentRow() >= 0 else 0
        for step in range(n):
            it = self.grid.item((start + step) % n)
            idx = str(it.data(Qt.ItemDataRole.UserRole))
            row = next((x for x in self._rows if x.index == idx), None)
            if row is not None and row.status == "ok" and row.verdict == "":
                self.select_index(idx)
                return idx
        self.status.emit("미검수가 없습니다 — 전부 판정됨")
        return None

    def _on_note(self) -> None:
        idx = self.current_index()
        if idx is None:
            return
        it = self.session.item(idx)
        if it.status != "ok" or it.note == self.note.text():
            return
        self.session.set_verdict(idx, it.verdict, self.note.text())
        self._after_change(None)

    def _after_change(self, advance_from: str | None) -> None:
        if self.autosave:
            try:
                p = self.session.save()
                self.review_saved.emit(p.as_posix())
            except ReviewError as e:
                self._on_error(str(e))
        # 그리드 라벨(✓/✗) 갱신 — 선택은 유지, "판정 후 다음으로" 면 다음 항목으로
        current = self.current_index()
        row = self.grid.currentRow()
        self.refresh_grid()
        self.refresh_hist()
        if advance_from is not None and self.cb_next.isChecked():
            nxt = row + 1 if row + 1 < self.grid.count() else row
            if 0 <= nxt < self.grid.count():
                self.select_index(str(self.grid.item(nxt).data(Qt.ItemDataRole.UserRole)))
                return
        if current is not None:
            self.select_index(current)
        self.summary.setText(self.session.summary_text())
        self.status.emit(self.session.summary_text())

    # ------------------------------------------------------------------ 정리본

    def export_pruned_dialog(self) -> None:
        if not self.session.loaded:
            self._on_error("출력 폴더를 먼저 여세요")
            return
        start = (self.session.root.parent / f"{self.session.root.name}-pruned").as_posix()  # type: ignore[union-attr]
        d = QFileDialog.getExistingDirectory(self, "정리본 폴더 (반려 제외 사본)", start)
        if d:
            self.export_pruned(d)

    def export_pruned(self, out: str | Path, *, drop_unreviewed: bool = False) -> bool:
        try:
            s = self.session.prune(out, drop_unreviewed=drop_unreviewed)
        except ReviewError as e:
            self._on_error(str(e))
            return False
        msg = (
            f"정리본: {s.out.as_posix()} — 합성 {s.kept} 유지 · {s.dropped} 제외 · 정상 {s.normals} · 파일 {s.files}"
            + (f" · 경고 {len(s.warnings)}" if s.warnings else "")
        )
        self.result.setText(msg)
        self.status.emit(msg)
        return True

    def write_report(self) -> Path | None:
        """``<root>/review-report.html`` 을 쓰고 기본 브라우저로 연다."""
        if not self.session.loaded:
            self._on_error("출력 폴더를 먼저 여세요")
            return None
        try:
            p = self.session.write_report()
        except ReviewError as e:
            self._on_error(str(e))
            return None
        msg = f"리포트: {p.as_posix()}"
        self.result.setText(msg)
        self.status.emit(msg)
        if self.open_report_in_browser:
            QDesktopServices.openUrl(QUrl.fromLocalFile(p.as_posix()))
        return p

    # ------------------------------------------------------------------ 표시

    def _on_error(self, msg: str) -> None:
        self.result.setText(msg)
        self.status.emit(msg)

    def refresh(self) -> None:
        self.summary.setText(self.session.summary_text() if self.session.loaded else "")
        self.btn_prune.setEnabled(self.session.loaded)
        self.btn_report.setEnabled(self.session.loaded)
        self.refresh_grid()
        self.refresh_hist()

    def ask_discard(
        self,
    ) -> bool:  # 창 닫기 등에서 부르라고 남겨 둔다 — 자동 저장이라 보통 dirty 아님
        if not self.session.dirty:
            return True
        r = QMessageBox.question(
            self,
            "검수 저장",
            "저장하지 않은 판정이 있습니다. 저장할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if r == QMessageBox.StandardButton.Yes:
            self.session.save()
        return True
