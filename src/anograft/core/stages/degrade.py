"""6단계 열화 — ``none`` · ``camera`` (설계 §6). **이미지 전체**에 적용한다(카메라 재현이므로).

- 무작위 소비 순서(고정): ``σ_noise = uniform`` → ``σ_blur = uniform`` → (jpeg면) ``q = integers`` →
  (motion_blur_px 면) ``len = uniform`` → ``angle = uniform`` → (vignette 면) ``s = uniform`` → (gamma 면) ``g = uniform`` →
  (σ_noise>0면) ``normal`` 필드. **v0.4 옵션은 null 이면 소비 0** — v0.1 레시피의 스트림·골든이 그대로다.
- 적용 순서는 카메라 물리 순서 — **모션 블러(흔들림) → 가우시안 블러(광학) → 비네팅(렌즈 감광) → 노이즈(센서) → 감마(ISP 톤 커브)
  → JPEG(압축)**. 노이즈 뒤에 블러를 걸면 노이즈가 뭉개져 설정한 σ가 무의미해진다.
- 흑백 대상: 노이즈는 HxW 한 장을 세 채널에 같이 더하고 JPEG는 채널 0만 1ch로 인코딩해 "세 채널 동일" 승격 불변식을 지킨다.
  모션 블러·비네팅·감마는 채널마다 같은 연산이라 불변식을 깨지 않는다.
- σ_blur < 0.1 이면 가우시안 블러 생략, 모션 길이 < 1px 이면 모션 블러 생략. 출력 dtype uint8, 클립.
"""

from __future__ import annotations

import math
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
MIN_MOTION_PX = 1.0


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < MIN_BLUR_SIGMA:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """길이 ``length_px`` 의 선형 모션 블러 커널(합 1). 각도는 이미지 좌표(+x 에서 +y 쪽, y 아래) — 0 = 가로, 90 = 세로."""
    k = max(3, math.ceil(length_px) | 1)  # 홀수, 최소 3
    ker = np.zeros((k, k), dtype=np.float32)
    c = (k - 1) / 2.0
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    half = length_px / 2.0
    p0 = (round(c - dx * half), round(c - dy * half))
    p1 = (round(c + dx * half), round(c + dy * half))
    cv2.line(ker, p0, p1, 1.0, 1, lineType=cv2.LINE_8)
    total = float(ker.sum())
    if total <= 0:
        ker[int(c), int(c)] = 1.0
        total = 1.0
    return ker / total


def motion_blur(image: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
    if length_px < MIN_MOTION_PX:
        return image
    return cv2.filter2D(
        image, -1, motion_kernel(length_px, angle_deg), borderType=cv2.BORDER_REFLECT_101
    )


def vignette(image: np.ndarray, strength: float) -> np.ndarray:
    """중심 1 → 모서리 ``1 - strength`` 로 떨어지는 2차 감광(``r²`` 프로파일, r 은 모서리까지 거리로 정규화)."""
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((xs - cx) ** 2 + (ys - cy) ** 2) / float(cx * cx + cy * cy or 1.0)
    gain = 1.0 - strength * r2
    out = image.astype(np.float32) * (gain[:, :, None] if image.ndim == 3 else gain)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def gamma_curve(image: np.ndarray, gamma: float) -> np.ndarray:
    """LUT ``255·(x/255)^gamma``. 1 이면 항등(바이트 동일)."""
    if gamma == 1.0:
        return image
    lut = np.clip(np.rint(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** gamma), 0, 255)
    return cv2.LUT(image, lut.astype(np.uint8))


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
        motion_px: float | None = None
        motion_angle: float | None = None
        if cfg.motion_blur_px is not None:
            motion_px = float(rng.uniform(cfg.motion_blur_px[0], cfg.motion_blur_px[1]))
            motion_angle = float(rng.uniform(cfg.motion_angle[0], cfg.motion_angle[1]))
        vig: float | None = None
        if cfg.vignette is not None:
            vig = float(rng.uniform(cfg.vignette[0], cfg.vignette[1]))
        gamma: float | None = None
        if cfg.gamma is not None:
            gamma = float(rng.uniform(cfg.gamma[0], cfg.gamma[1]))

        out = ctx.composite
        if motion_px is not None and motion_angle is not None:
            out = motion_blur(out, motion_px, motion_angle)
        out = gaussian_blur(out, sigma_blur)
        if vig is not None:
            out = vignette(out, vig)
        out = add_noise(out, sigma_noise, rng, ctx.target.gray)
        if gamma is not None:
            out = gamma_curve(out, gamma)
        if quality is not None:
            out = jpeg_roundtrip(out, quality, ctx.target.gray)
        log = {
            "method": "camera",
            "noise_sigma": sigma_noise,
            "blur_sigma": sigma_blur,
            "blur_applied": sigma_blur >= MIN_BLUR_SIGMA,
            "jpeg_quality": quality,
            "motion_blur_px": motion_px,
            "motion_angle_deg": motion_angle,
            "motion_applied": motion_px is not None and motion_px >= MIN_MOTION_PX,
            "vignette": vig,
            "gamma": gamma,
        }
        return replace(ctx, composite=out).with_log("degrade", log)
