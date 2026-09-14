"""numpy(BGR/gray/BGRA uint8) ↔ ``QImage``/``QPixmap``.

- ``QImage``는 버퍼를 **참조**하므로 numpy 배열이 먼저 사라지면 깨진다 → 항상 ``.copy()``로 소유권을 QImage에 준다.
- 행 stride를 명시한다(``ascontiguousarray`` 뒤 ``strides[0]``) — 잘라낸 배열(view)에서 줄이 밀리는 함정.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def to_qimage(image: np.ndarray) -> QImage:
    """HxW(gray) · HxWx3(BGR) · HxWx4(BGRA) uint8 → QImage(소유권 있는 복사본)."""
    if image.dtype != np.uint8:
        raise TypeError(f"uint8만 지원: {image.dtype}")
    arr = np.ascontiguousarray(image)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        fmt = QImage.Format.Format_Grayscale8
    elif arr.shape[2] == 3:
        fmt = QImage.Format.Format_BGR888
    elif arr.shape[2] == 4:
        fmt = QImage.Format.Format_ARGB32  # 메모리 순서 BGRA (리틀엔디언)
    else:
        raise ValueError(f"지원하지 않는 채널 구성: {arr.shape}")
    qimg = QImage(arr.data, w, h, int(arr.strides[0]), fmt)
    return qimg.copy()


def to_qpixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(to_qimage(image))


def from_qimage(qimg: QImage) -> np.ndarray:
    """QImage → HxWx3 BGR uint8 (테스트 왕복용)."""
    img = qimg.convertToFormat(QImage.Format.Format_BGR888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    buf = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes()).reshape(h, img.bytesPerLine())
    return np.ascontiguousarray(buf[:, : w * 3].reshape(h, w, 3))
