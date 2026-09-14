"""MVTec AD 어댑터 (설계 §3.4) — 파일 1개 = 데이터셋 1개. CC BY-NC-SA 4.0: 재배포 금지, 로컬 사본 읽기만.

레이아웃(``<root>`` = ``<mvtec_anomaly_detection>/<category>``)::

    <root>/
      train/good/<idx>.png
      test/good/<idx>.png                # 정상 (제외)
      test/<defect>/<idx>.png            # 결함 이미지 — <defect> 가 클래스
      ground_truth/<defect>/<idx>_mask.png

- ``defects()``: ``test/<defect>/*.png``(``good`` 제외) ↔ ``ground_truth/<defect>/<idx>_mask.png``. 마스크 없으면 건너뛰고
  경고(호출자가 `warn`을 받는다). id = ``<category>-<defect>-<idx>``(같은 idx가 결함 폴더마다 있으므로).
- ``normals()``: ``train/good/*.png`` 이름 정렬 — 합성 대상 풀(``recipe init --targets``).
- 흑백 카테고리(grid·screw·zipper)는 1ch PNG — 임포터가 그대로 보존한다(은행 크롭도 1ch).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from anograft.bank.importers.pairs import PairRecord
from anograft.datasets.base import DatasetError, DatasetInfo, register
from anograft.io import imgio

CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
GRAY_CATEGORIES: frozenset[str] = frozenset({"grid", "screw", "zipper"})

INFO = DatasetInfo(
    name="mvtec-ad",
    title="MVTec Anomaly Detection Dataset (Bergmann et al., CVPR 2019)",
    license="CC BY-NC-SA 4.0 — 비상업·동일조건. 재배포 금지, 로컬 사본만",
    url="https://www.mvtec.com/company/research/datasets/mvtec-ad",
    layout_help=(
        "<root>/<category>/\n"
        "  train/good/<idx>.png              # 정상 → normals (합성 대상)\n"
        "  test/<defect>/<idx>.png           # 결함 (test/good 은 제외)\n"
        "  ground_truth/<defect>/<idx>_mask.png\n"
        "import-dataset 의 <root> 는 <category> 폴더까지 (예: mvtec_ad/metal_nut)"
    ),
    categories=CATEGORIES,
    note="grid·screw·zipper 는 흑백 1ch. 픽셀 피치 정보 없음 → um_per_px null(축척 자동 정합 no-op)",
)


class MvtecAdAdapter:
    info = INFO

    def category(self, root: Path) -> str:
        return Path(root).name

    def check_layout(self, root: Path) -> None:
        r = Path(root)
        missing = [d for d in ("test", "ground_truth") if not (r / d).is_dir()]
        if missing:
            raise DatasetError(
                f"{r}: {', '.join(missing)} 폴더가 없습니다 — <root>는 카테고리 폴더여야 합니다.\n"
                + INFO.layout_help
            )

    def defects(
        self, root: Path, *, warn: Callable[[str], None] | None = None
    ) -> Iterable[PairRecord]:
        r = Path(root)
        self.check_layout(r)
        cat = self.category(r)
        warn = warn or (lambda _m: None)
        for ddir in sorted(p for p in (r / "test").iterdir() if p.is_dir()):
            defect = ddir.name
            if defect == "good":
                continue
            for img in imgio.list_images(ddir):
                mask = r / "ground_truth" / defect / f"{img.stem}_mask.png"
                if not mask.is_file():
                    warn(f"{cat}/test/{defect}/{img.name}: ground_truth 마스크 없음 — 건너뜀")
                    continue
                yield PairRecord(
                    image=img,
                    mask=mask,
                    cls=defect,
                    origin=f"{cat}/test/{defect}/{img.name}",
                    id_hint=f"{cat}-{defect}-{img.stem}",
                    tags=(INFO.name, cat),
                )

    def normals(self, root: Path) -> list[Path]:
        d = Path(root) / "train" / "good"
        if not d.is_dir():
            raise DatasetError(f"{d}: train/good 이 없습니다.\n" + INFO.layout_help)
        return imgio.list_images(d)


ADAPTER = register(MvtecAdAdapter())
