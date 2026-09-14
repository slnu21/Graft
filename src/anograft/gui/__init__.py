"""GUI (PySide6, v0.2+). ``core``·``runner``를 **호출만** 한다 — 합성 로직은 여기에 없다.

PySide6는 선택 의존성(``pip install -e ".[gui]"``). 이 패키지의 최상위 모듈은 Qt를 import하지 않는다 — 없는 환경에서도
``anograft.gui.qt_available()``로 이유를 물을 수 있고, ``python -m anograft.gui``는 설치 안내 한 줄로 끝난다(fail-soft).

구성::

    gui/
      __main__.py       # main(): Qt 확인 → MainWindow
      theme.py          # 목업(docs/mockup.html)의 팔레트·QSS
      qt_image.py       # numpy(BGR) ↔ QImage/QPixmap
      app.py            # MainWindow: 상단 바 · 5탭(은행·라벨·스튜디오·배치·검수) · 상태바
      studio/
        session.py      # StudioSession — GUI 상태의 단일 원천(Recipe + Prepared 캐시). Qt 없음
        jobs.py         # PreviewJob·preview_target(1024 축소)·run_preview·LatestOnlyQueue. Qt 없음
        worker.py       # PreviewWorker(QThread): 큐 → run_preview → 시그널
        canvas.py       # CompareCanvas: A/B 와이프 · GT/ROI 오버레이 · 줌/팬
        panels.py       # StripBar · InputsPanel · TargetRail · PipelinePanel
        variants.py     # VariantStrip: 시드 변형 카드
        tab.py          # StudioTab: 조립·배선
      label/            # v0.5 라벨 탭
        session.py      # LabelSession — 이미지+마스크 편집 상태·되돌리기·통계·은행 저장(ImportRecord). Qt 없음
        canvas.py       # LabelCanvas(CompareCanvas): 마스크 오버레이(버퍼 공유 QImage)·브러시/지우개/폴리곤/자동 선택 박스
        tab.py          # LabelTab: 도구열·이미지 목록·결함 정보·통계·은행에 저장, bank_saved → 스튜디오 재준비
"""

from __future__ import annotations

import importlib.util

INSTALL_HINT = (
    'GUI를 쓰려면 PySide6가 필요합니다: pip install -e ".[gui]"  (또는 .\\bootstrap.ps1 -Gui)'
)


def qt_available() -> tuple[bool, str | None]:
    """PySide6 import 가능 여부 ``(ok, reason)``. import 자체는 하지 않는다(가볍게)."""
    if importlib.util.find_spec("PySide6") is None:
        return False, INSTALL_HINT
    return True, None
