"""VisA 어댑터 (설계 §3.4, v0.4) — 파일 1개 = 데이터셋 1개. CC BY-NC-SA 4.0: 재배포 금지, 로컬 사본 읽기만.

레이아웃(``<root>`` = ``<VisA>/<category>``, 공식 tar 를 푼 그대로)::

    <root>/
      Data/Images/Normal/<idx>.JPG        # 정상 → normals (합성 대상)
      Data/Images/Anomaly/<idx>.JPG       # 결함 이미지
      Data/Masks/Anomaly/<idx>.png        # GT 마스크 (같은 stem)
      image_anno.csv                      # image,label,mask — label ∈ {normal, anomaly} (결함 유형 세분 없음)

- VisA 공개본은 결함 **유형**을 폴더·CSV 어디에도 나누지 않는다(``label`` 은 normal/anomaly 뿐) → 은행 클래스는
  ``anomaly`` 하나. 카테고리는 id·tags 에 남는다(``<category>-anomaly-<idx>``, tags ``("visa", <category>)``).
- ``image_anno.csv`` 가 있으면 그 목록을 정본으로(경로는 ``<category>/Data/...`` 형태 — 파일명만 취해 ``<root>`` 기준으로
  다시 잇는다), 없으면 ``Data/Images/Anomaly`` 폴더를 훑는다. 마스크가 없는 결함 이미지는 건너뛰고 경고.
- 마스크 PNG 값이 0/255 가 아닌 사본(0/1 라벨맵)이 보고된 적이 있어 ``mask_threshold=0``(0 이 아니면 결함)으로 읽는다.
- 이미지는 JPG(대개 컬러) — 임포터가 그대로 보존한다. 픽셀 피치 정보 없음.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from pathlib import Path

from anograft.bank.importers.pairs import PairRecord
from anograft.datasets.base import DatasetError, DatasetInfo, register
from anograft.io import imgio

CATEGORIES: tuple[str, ...] = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
ANOMALY_CLASS = "anomaly"

INFO = DatasetInfo(
    name="visa",
    title="VisA — Visual Anomaly dataset (Zou et al., SPot-the-Difference, ECCV 2022)",
    license="CC BY-NC-SA 4.0 — 비상업·동일조건. 재배포 금지, 로컬 사본만",
    url="https://github.com/amazon-science/spot-diff",
    layout_help=(
        "<root>/<category>/\n"
        "  Data/Images/Normal/<idx>.JPG      # 정상 → normals (합성 대상)\n"
        "  Data/Images/Anomaly/<idx>.JPG     # 결함\n"
        "  Data/Masks/Anomaly/<idx>.png      # GT 마스크 (같은 stem)\n"
        "  image_anno.csv                    # image,label,mask (있으면 정본)\n"
        "import-dataset 의 <root> 는 <category> 폴더까지 (예: VisA/candle)"
    ),
    categories=CATEGORIES,
    note=(
        "결함 유형 세분이 없어 은행 클래스는 'anomaly' 하나(카테고리는 tags). "
        "마스크는 0 이 아니면 결함으로 읽는다(0/1 라벨맵 사본 대비). 픽셀 피치 정보 없음 → um_per_px null"
    ),
)


class VisaAdapter:
    info = INFO

    def category(self, root: Path) -> str:
        return Path(root).name

    def check_layout(self, root: Path) -> None:
        r = Path(root)
        missing = [d for d in ("Data/Images/Anomaly", "Data/Masks/Anomaly") if not (r / d).is_dir()]
        if missing:
            raise DatasetError(
                f"{r}: {', '.join(missing)} 폴더가 없습니다 — <root>는 카테고리 폴더여야 합니다.\n"
                + INFO.layout_help
            )

    def _anomaly_images(self, r: Path, warn: Callable[[str], None]) -> list[Path]:
        """``image_anno.csv`` 의 anomaly 행(파일명만 취해 <root> 기준으로) — 없거나 못 읽으면 폴더 목록."""
        anno = r / "image_anno.csv"
        folder = imgio.list_images(r / "Data" / "Images" / "Anomaly")
        if not anno.is_file():
            return folder
        try:
            with anno.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error) as e:
            warn(f"{anno.name}: 읽기 실패({e}) — Data/Images/Anomaly 폴더 목록을 씁니다")
            return folder
        by_name = {p.name: p for p in folder}
        out: list[Path] = []
        for row in rows:
            if (row.get("label") or "").strip().lower() != ANOMALY_CLASS:
                continue
            name = Path((row.get("image") or "").strip()).name
            p = by_name.get(name)
            if p is None:
                warn(f"{anno.name}: {name} 이 Data/Images/Anomaly 에 없음 — 건너뜀")
                continue
            out.append(p)
        return out or folder

    def defects(
        self, root: Path, *, warn: Callable[[str], None] | None = None
    ) -> Iterable[PairRecord]:
        r = Path(root)
        self.check_layout(r)
        cat = self.category(r)
        warn = warn or (lambda _m: None)
        mask_dir = r / "Data" / "Masks" / "Anomaly"
        for img in self._anomaly_images(r, warn):
            mask = mask_dir / f"{img.stem}.png"
            if not mask.is_file():
                warn(
                    f"{cat}/Data/Images/Anomaly/{img.name}: Data/Masks/Anomaly 마스크 없음 — 건너뜀"
                )
                continue
            yield PairRecord(
                image=img,
                mask=mask,
                cls=ANOMALY_CLASS,
                origin=f"{cat}/Data/Images/Anomaly/{img.name}",
                id_hint=f"{cat}-{ANOMALY_CLASS}-{img.stem}",
                tags=(INFO.name, cat),
                mask_threshold=0,
            )

    def normals(self, root: Path) -> list[Path]:
        d = Path(root) / "Data" / "Images" / "Normal"
        if not d.is_dir():
            raise DatasetError(f"{d}: Data/Images/Normal 이 없습니다.\n" + INFO.layout_help)
        return imgio.list_images(d)


ADAPTER = register(VisaAdapter())
