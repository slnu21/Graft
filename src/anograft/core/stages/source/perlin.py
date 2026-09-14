"""1단계 소스 — ``perlin-texture``(DRAEM, Zavrtanik et al. 2021). 펄린 노이즈 임계 마스크 + 텍스처. 은행 불필요.

- 창: 한 변 ``size_ratio × min(H, W)``(``uniform``)의 정사각형. 마스크 = ``core.perlin.perlin_mask``(축별 해상도 2^k,
  k ∈ ``scale_range``, ``threshold``, 회전). 면적 < ``min_area_px``면 재생성(``max_tries``), 끝내 안 되면 skip.
- 텍스처: ``self`` = 대상 자신의 다른 창(``integers``×2로 자리) · ``dir`` = ``deps["textures"]``(runner가 ``texture_dir``를
  읽어 주입; 없거나 비면 self로 폴백 + 경고) 중 ``integers`` 1장 — 창보다 크면 무작위 크롭(``integers``×2), 작으면 리사이즈.
  흑백 대상이면 텍스처도 흑백(3채널 동일)으로.
- 증강(``augment``, DRAEM 흉내): 9종 중 ``rng.choice(3, replace=False)`` — gamma · brightness(mul+add) · invert · equalize ·
  autocontrast · solarize · posterize · rot90 · hue_shift. 각 항목의 파라미터는 그 자리에서 rng 소비.
- 결과는 마스크 bbox ± ``pad_px``(4)로 잘라 패치를 조밀하게 한다(placement가 창 전체가 아니라 bbox만 ROI에 넣으면 되게).
- rng 소비 순서(고정): 창 크기 ``uniform`` → [마스크: ``integers``×2 → ``random`` → ``uniform``] × 시도 → 텍스처 자리
  (self: ``integers``×2 / dir: ``integers`` + 크롭 ``integers``×2) → 증강 ``choice`` + 항목별.
- 로그 ``source = {method, source_id, class, class_id, mask_origin: "perlin", window_px, perlin{res_y,res_x,rotate,area_px},
  texture: self|dir, texture_index, augment: [...], tries}``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.perlin import perlin_mask
from anograft.core.recipe import PerlinSourceConfig
from anograft.core.registry import register
from anograft.core.types import Context, DefectSource

PAD_PX = 4


# --- 증강 (각각 BGR uint8 → BGR uint8, 자기 파라미터를 rng에서 뽑는다) --------------------


def _aug_gamma(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    g = float(rng.uniform(0.5, 2.0))
    lut = (np.clip((np.arange(256) / 255.0) ** g, 0, 1) * 255.0).astype(np.uint8)
    return lut[x], f"gamma {g:.2f}"


def _aug_brightness(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    mul = float(rng.uniform(0.8, 1.2))
    add = float(rng.uniform(-30.0, 30.0))
    y = np.clip(np.rint(x.astype(np.float32) * mul + add), 0, 255).astype(np.uint8)
    return y, f"brightness x{mul:.2f}{add:+.0f}"


def _aug_invert(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    return (255 - x).astype(np.uint8), "invert"


def _aug_equalize(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    y = np.stack([cv2.equalizeHist(x[..., c]) for c in range(3)], axis=-1)
    return y, "equalize"


def _aug_autocontrast(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    y = x.astype(np.float32)
    for c in range(3):
        lo, hi = float(y[..., c].min()), float(y[..., c].max())
        if hi > lo:
            y[..., c] = (y[..., c] - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(y), 0, 255).astype(np.uint8), "autocontrast"


def _aug_solarize(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    t = int(rng.integers(32, 129))
    return np.where(x >= t, 255 - x, x).astype(np.uint8), f"solarize {t}"


def _aug_posterize(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    bits = int(rng.integers(2, 6))
    keep = np.uint8(0xFF << (8 - bits) & 0xFF)
    return (x & keep).astype(np.uint8), f"posterize {bits}"


def _aug_rot90(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    k = int(rng.integers(1, 4))
    return np.ascontiguousarray(np.rot90(x, k)), f"rot90 x{k}"


def _aug_hue(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    shift = float(rng.uniform(-45.0, 45.0))  # OpenCV H 0..180
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] + shift, 180.0)
    return cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR), f"hue {shift:+.0f}"


AUGMENTS = (
    _aug_gamma,
    _aug_brightness,
    _aug_invert,
    _aug_equalize,
    _aug_autocontrast,
    _aug_solarize,
    _aug_posterize,
    _aug_rot90,
    _aug_hue,
)
N_AUG_PICK = 3


def augment_texture(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """DRAEM식: 9종 중 3종을 비복원 추출해 순서대로 적용. 정사각 창이므로 rot90도 크기를 유지한다."""
    picks = rng.choice(len(AUGMENTS), N_AUG_PICK, replace=False)
    names: list[str] = []
    for i in picks:
        x, name = AUGMENTS[int(i)](x, rng)
        names.append(name)
    return x, names


def _to_gray3(x: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _skip(ctx: Context, reason: str) -> Context:
    return (
        replace(ctx, source=None)
        .warn(f"source: {reason}")
        .with_log("source", {"method": "perlin-texture", "skipped": True, "reason": reason})
    )


@register
class PerlinSource:
    stage: ClassVar[str] = "source"
    methods: ClassVar[tuple[str, ...]] = ("perlin-texture",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: PerlinSourceConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg
        ids: Mapping[str, int] | None = deps.get("class_ids")
        self.class_id: int = int(ids.get(cfg.cls, 0)) if ids else 0
        textures: Sequence[np.ndarray] | None = deps.get("textures")
        self.textures: list[np.ndarray] = [t for t in (textures or []) if t is not None]

    # --- 텍스처 ------------------------------------------------------------

    def _texture(self, ctx: Context, side: int) -> tuple[np.ndarray, str, int | None, str | None]:
        """창 크기의 텍스처 (image, mode, index, warning)."""
        rng = ctx.rng
        img = ctx.target.image
        h_img, w_img = img.shape[:2]
        warning: str | None = None
        if self.cfg.texture == "dir":
            if self.textures:
                idx = int(rng.integers(len(self.textures)))
                tex = self.textures[idx]
                th, tw = tex.shape[:2]
                if th >= side and tw >= side:
                    ty = int(rng.integers(0, th - side + 1))
                    tx = int(rng.integers(0, tw - side + 1))
                    tex = tex[ty : ty + side, tx : tx + side]
                else:
                    tex = cv2.resize(tex, (side, side), interpolation=cv2.INTER_LINEAR)
                tex = np.ascontiguousarray(tex)
                if ctx.target.gray:
                    tex = _to_gray3(tex)
                return tex, "dir", idx, None
            warning = "texture_dir 에 쓸 수 있는 이미지가 없어 대상 자신(self)으로 폴백"
        ty = int(rng.integers(0, max(1, h_img - side + 1)))
        tx = int(rng.integers(0, max(1, w_img - side + 1)))
        tex = np.ascontiguousarray(img[ty : ty + side, tx : tx + side])
        if tex.shape[0] != side or tex.shape[1] != side:  # 대상이 창보다 작을 때
            tex = cv2.resize(tex, (side, side), interpolation=cv2.INTER_LINEAR)
        return tex, "self", None, warning

    # --- apply --------------------------------------------------------------

    def apply(self, ctx: Context) -> Context:
        img = ctx.target.image
        h_img, w_img = img.shape[:2]
        cfg = self.cfg
        rng = ctx.rng
        base = min(h_img, w_img)
        side = round(float(rng.uniform(cfg.size_ratio[0], cfg.size_ratio[1])) * base)
        side = max(8, min(side, base))
        mask = None
        plog: dict[str, Any] = {}
        tries = 0
        while tries < cfg.max_tries:
            tries += 1
            m, plog = perlin_mask(
                rng,
                side,
                side,
                scale_range=cfg.scale_range,
                threshold=cfg.threshold,
                rotate=cfg.rotate,
            )
            if int(plog["area_px"]) >= cfg.min_area_px:
                mask = m
                break
        if mask is None:
            return _skip(
                ctx,
                f"perlin 마스크 면적이 {cfg.max_tries}번 모두 {cfg.min_area_px}px 미만 (창 {side}px, threshold {cfg.threshold})",
            )
        tex, mode, tidx, warning = self._texture(ctx, side)
        aug_names: list[str] = []
        if cfg.augment:
            tex, aug_names = augment_texture(tex, rng)
            if ctx.target.gray:
                tex = _to_gray3(tex)  # 증강(hue 등)이 채널을 갈라도 흑백 불변식 유지

        # 마스크 bbox ± PAD 로 조밀하게
        x, y, w, h = cv2.boundingRect(mask)
        y0, x0 = max(0, y - PAD_PX), max(0, x - PAD_PX)
        y1, x1 = min(side, y + h + PAD_PX), min(side, x + w + PAD_PX)
        patch = np.ascontiguousarray(tex[y0:y1, x0:x1])
        pmask = np.ascontiguousarray(mask[y0:y1, x0:x1])

        k = len(ctx.defect_logs)
        src = DefectSource(
            id=f"perlin/{k:02d}",
            cls=cfg.cls,
            image=patch,
            mask=pmask,
            um_per_px=ctx.target.um_per_px,
            origin=ctx.target.path.as_posix() if mode == "self" else f"texture_dir[{tidx}]",
            mask_origin="perlin",
        )
        log: dict[str, Any] = {
            "method": "perlin-texture",
            "source_id": src.id,
            "class": cfg.cls,
            "class_id": self.class_id,
            "mask_origin": "perlin",
            "window_px": side,
            "perlin": plog,
            "texture": mode,
            "texture_index": tidx,
            "augment": aug_names,
            "tries": tries,
        }
        out = replace(ctx, source=src).with_log("source", log)
        if warning:
            out = out.warn(f"source: {warning}")
        return out
