"""6단계 열화 — ``none`` · ``camera`` (설계 §6). **이미지 전체**에 적용한다(카메라 재현이므로).

- 무작위 소비 순서(고정): ``σ_noise = uniform`` → ``σ_blur = uniform`` → (jpeg면) ``q = integers`` → (σ_noise>0면) ``normal`` 필드.
- 적용 순서는 카메라 물리 순서 — **블러(광학) → 노이즈(센서) → JPEG(압축)**. 노이즈 뒤에 블러를 걸면 노이즈가 뭉개져 설정한 σ가 무의미해진다.
- 흑백 대상: 노이즈는 HxW 한 장을 세 채널에 같이 더하고 JPEG는 채널 0만 1ch로 인코딩해 "세 채널 동일" 승격 불변식을 지킨다.
- σ_blur < 0.1이면 블러 생략. 출력 dtype uint8, 클립.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.channels import promote_to_bgr
from anograft.core.recipe import CameraDegradeConfig, NoneDegradeConfig
from anograft.core.registry import register
from anograft.core.types import Context

MIN_BLUR_SIGMA = 0.1


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < MIN_BLUR_SIGMA:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def add_noise(image: np.ndarray, sigma: float, rng: np.random.Generator, gray: bool) -> np.ndarray:
    if sigma <= 0:
        return image
    h, w = image.shape[:2]
    if gray:
        noise = rng.normal(0.0, sigma, (h, w)).astype(np.float32)[:, :, None]
    else:
        noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(np.rint(image.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def jpeg_roundtrip(image: np.ndarray, quality: int, gray: bool) -> np.ndarray:
    src = image[:, :, 0] if gray else image
    ok, buf = cv2.imencode(".jpg", src, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return promote_to_bgr(out) if gray else out


@register
class NoneDegrade:
    stage: ClassVar[str] = "degrade"
    methods: ClassVar[tuple[str, ...]] = ("none",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: NoneDegradeConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        return ctx.with_log("degrade", {"method": "none"})


@register
class CameraDegrade:
    stage: ClassVar[str] = "degrade"
    methods: ClassVar[tuple[str, ...]] = ("camera",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: CameraDegradeConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        cfg, rng = self.cfg, ctx.rng
        sigma_noise = float(rng.uniform(cfg.noise_sigma[0], cfg.noise_sigma[1]))
        sigma_blur = float(rng.uniform(cfg.blur_sigma[0], cfg.blur_sigma[1]))
        quality: int | None = None
        if cfg.jpeg_quality is not None:
            quality = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))

        out = gaussian_blur(ctx.composite, sigma_blur)
        out = add_noise(out, sigma_noise, rng, ctx.target.gray)
        if quality is not None:
            out = jpeg_roundtrip(out, quality, ctx.target.gray)
        log = {
            "method": "camera",
            "noise_sigma": sigma_noise,
            "blur_sigma": sigma_blur,
            "blur_applied": sigma_blur >= MIN_BLUR_SIGMA,
            "jpeg_quality": quality,
        }
        return replace(ctx, composite=out).with_log("degrade", log)
