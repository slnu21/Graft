"""DTD 어댑터 (v0.7) — Describable Textures Dataset(Cimpoi et al., CVPR 2014). 결함 데이터셋이 **아니라** 텍스처셋이라
``import-dataset`` 은 거부하고, ``perlin-texture`` 소스의 ``texture_dir`` 로 쓸 **텍스처 목록**(``dataset textures dtd``)을 만든다.

레이아웃(``<root>`` = 공식 tar 를 푼 ``dtd/``)::

    <root>/
      images/<category>/<category>_NNNN.jpg    # 47 카테고리 · 5640장
      labels/labels_joint_anno.txt              # (선택) 이미지별 속성 — 쓰지 않는다
      imdb/imdb.mat                             # (선택)

- 라이선스: 이미지는 Flickr 등에서 모은 것으로 **연구 목적** 사용 조건(각 이미지의 원 라이선스 별도). 앱은 내려받지도 재배포하지도
  않는다 — 로컬 사본을 읽어 목록만 만든다. 은행에 넣지 않는다(결함이 아니다).
- ``textures(root, categories, limit, seed)`` → 이름 정렬한 경로 목록(카테고리 필터, ``limit`` 이면 ``seed`` 로 결정적 추출).
  ``dataset textures dtd <root> --out textures.txt`` 가 목록 파일을 쓰고, 레시피 ``source.texture_dir: textures.txt`` 가 그것을
  읽는다(``runner.load_textures`` 가 ``.txt`` 목록도 받는다 — 경로는 목록 파일 기준).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from anograft.bank.importers.pairs import PairRecord
from anograft.datasets.base import DatasetError, DatasetInfo, register
from anograft.io import imgio

CATEGORIES: tuple[str, ...] = (
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
    "crosshatched", "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy",
    "grid", "grooved", "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled",
    "matted", "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted", "porous",
    "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
    "studded", "swirly", "veined", "waffled", "woven", "wrinkled", "zigzagged",
)  # fmt: skip
DEFECT_LIKE: tuple[str, ...] = (
    "blotchy",
    "bumpy",
    "cracked",
    "crystalline",
    "fibrous",
    "flecked",
    "pitted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "sprinkled",
    "stained",
    "veined",
    "wrinkled",
)  # fmt: skip — 결함 텍스처로 쓸 만한 것(DRAEM 이 쓴 DTD 전체 중 얼룩·균열·구멍 계열)

INFO = DatasetInfo(
    name="dtd",
    title="DTD — Describable Textures Dataset (Cimpoi et al., CVPR 2014)",
    license="연구 목적 사용(이미지별 원 라이선스 별도, Flickr 등) — 재배포 금지, 로컬 사본만. 은행에 넣지 않는다",
    url="https://www.robots.ox.ac.uk/~vgg/data/dtd/",
    layout_help=(
        "<root>/                                  # 공식 dtd-r1.0.1.tar.gz 를 푼 dtd/\n"
        "  images/<category>/<category>_NNNN.jpg  # 47 카테고리 · 5640장\n"
        "결함 데이터셋이 아니므로 import-dataset 은 없다. 대신:\n"
        "  anograft dataset textures dtd <root> --out textures.txt [--categories cracked,stained] [--limit 300]\n"
        "  → 레시피 source: {method: perlin-texture, texture: dir, texture_dir: textures.txt}"
    ),
    categories=CATEGORIES,
    note=(
        f"결함처럼 보이는 카테고리(--categories 기본): {', '.join(DEFECT_LIKE)}. "
        "DRAEM 은 DTD 전체를 이상 텍스처 소스로 썼다"
    ),
    importable=False,
)


class DtdAdapter:
    info = INFO

    def category(self, root: Path) -> str:
        return "textures"

    def check_layout(self, root: Path) -> None:
        r = Path(root)
        if not (r / "images").is_dir():
            raise DatasetError(
                f"{r}: images/ 폴더가 없습니다 — <root>는 dtd/ (images/<category>/…) 여야 합니다.\n"
                + INFO.layout_help
            )

    def defects(
        self, root: Path, *, warn: Callable[[str], None] | None = None
    ) -> Iterable[PairRecord]:
        raise DatasetError(
            "DTD 는 결함 데이터셋이 아닙니다(마스크 없음) — 은행에 넣지 않습니다. "
            "텍스처 목록: anograft dataset textures dtd <root> --out textures.txt"
        )

    def normals(self, root: Path) -> list[Path]:
        return []

    def available_categories(self, root: Path) -> list[str]:
        self.check_layout(root)
        return sorted(p.name for p in (Path(root) / "images").iterdir() if p.is_dir())

    def textures(
        self,
        root: Path,
        *,
        categories: Sequence[str] | None = None,
        limit: int | None = None,
        seed: int = 0,
        warn: Callable[[str], None] | None = None,
    ) -> list[Path]:
        """텍스처 이미지 경로(이름 정렬). ``categories`` None = ``DEFECT_LIKE`` 중 있는 것, ``["*"]`` = 전부.
        ``limit`` 이면 ``seed`` 로 결정적 부분 추출(카테고리 섞어서)."""
        warn = warn or (lambda _m: None)
        have = self.available_categories(root)
        if categories is None:
            wanted = [c for c in DEFECT_LIKE if c in have]
        elif list(categories) == ["*"]:
            wanted = have
        else:
            wanted = []
            for c in categories:
                if c in have:
                    wanted.append(c)
                else:
                    warn(f"카테고리 없음: {c} (있는 것: {', '.join(have[:8])}…)")
        paths: list[Path] = []
        for c in wanted:
            paths += imgio.list_images(Path(root) / "images" / c)
        paths.sort()
        if limit is not None and 0 < limit < len(paths):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(paths), size=limit, replace=False))
            paths = [paths[int(i)] for i in idx]
        return paths


def write_texture_list(list_file: Path, paths: Sequence[Path]) -> Path:
    """목록 파일(한 줄 = 경로, **목록 파일 기준 상대** — 가능하면). ``runner.load_textures`` 가 읽는다."""
    lf = Path(list_file)
    lf.parent.mkdir(parents=True, exist_ok=True)
    base = lf.resolve().parent
    lines = []
    for p in paths:
        rp = Path(p).resolve()
        try:
            lines.append(Path(os.path.relpath(rp, base)).as_posix())  # '..' 도 허용
        except ValueError:  # 다른 드라이브
            lines.append(rp.as_posix())
    lf.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lf


register(DtdAdapter())
