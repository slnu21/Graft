"""레지스트리 + 파이프라인 골격. 실제 스테이지 없이 더미 스테이지로 루프 구조·결정성·trace를 고정한다."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Literal

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.pipeline import Pipeline
from anograft.core.types import Context, DefectSource, Instance, Placement, TargetImage

# ---------------------------------------------------------------------------
# 레지스트리
# ---------------------------------------------------------------------------


def test_schema_methods_match_design_matrix_v01() -> None:
    assert registry.schema_methods("source") == ["bank"]
    assert registry.schema_methods("geometry") == ["affine"]
    assert registry.schema_methods("roi") == ["otsu", "none", "mask_dir"]
    assert registry.schema_methods("placement") == ["sampled"]
    assert registry.schema_methods("blend") == ["paste", "alpha", "poisson", "multiband"]
    assert registry.schema_methods("harmonize") == ["none", "stats", "reinhard", "histmatch"]
    assert registry.schema_methods("degrade") == ["none", "camera"]
    assert registry.schema_methods("gtmask") == ["source", "diff", "union"]


def test_list_methods_covers_every_stage() -> None:
    infos = registry.list_methods()
    stages = {i.stage for i in infos}
    assert stages == set(registry.STAGE_ORDER)
    only_blend = registry.list_methods("blend")
    assert {i.method for i in only_blend} == {"paste", "alpha", "poisson", "multiband"}


def test_availability_reports_missing_module() -> None:
    ok, reason = registry.availability(("numpy",))
    assert ok and reason is None
    ok, reason = registry.availability(("numpy", "no_such_module_xyz_123"))
    assert not ok and "no_such_module_xyz_123" in (reason or "")


class _TestCfg(BaseModel):
    method: Literal["__test__"] = "__test__"


def test_register_build_and_unavailable() -> None:
    class Dummy:
        stage: ClassVar[str] = "blend"
        methods: ClassVar[tuple[str, ...]] = ("__test__",)
        requires: ClassVar[tuple[str, ...]] = ()

        def __init__(self, cfg: BaseModel, deps: dict) -> None:
            self.cfg = cfg
            self.deps = deps

        def apply(self, ctx: Context) -> Context:
            return ctx

    registry.register(Dummy)
    try:
        st = registry.build("blend", _TestCfg(), {"k": 1})
        assert isinstance(st, Dummy) and st.deps == {"k": 1}
        with pytest.raises(ValueError):  # 같은 키에 다른 클래스
            registry.register(type("Other", (Dummy,), {}))
    finally:
        registry.unregister("blend", "__test__")

    with pytest.raises(registry.StageNotImplementedError):
        registry.build("blend", _TestCfg())

    class NeedsTorch(Dummy):
        requires = ("no_such_module_xyz_123",)

    registry.register(NeedsTorch)
    try:
        with pytest.raises(registry.StageUnavailableError, match="no_such_module_xyz_123"):
            registry.build("blend", _TestCfg())
    finally:
        registry.unregister("blend", "__test__")


def test_register_requires_class_attrs() -> None:
    with pytest.raises(TypeError):
        registry.register(type("Bad", (), {}))
    with pytest.raises(ValueError):
        registry.register(
            type("BadStage", (), {"stage": "nope", "methods": ("x",), "requires": ()})
        )


# ---------------------------------------------------------------------------
# 파이프라인 골격 — 더미 스테이지
# ---------------------------------------------------------------------------


class _Dummy:
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: BaseModel, deps: dict) -> None:
        self.cfg = cfg
        self.deps = deps


class RoiAll(_Dummy):
    stage = "roi"
    methods = ("none",)

    def apply(self, ctx: Context) -> Context:
        h, w = ctx.target.image.shape[:2]
        return replace(ctx, roi=np.ones((h, w), dtype=bool))


class SourceFixed(_Dummy):
    stage = "source"
    methods = ("bank",)

    def apply(self, ctx: Context) -> Context:
        cls = "scratch" if ctx.rng.random() < 0.5 else "dent"
        img = np.full((6, 6, 3), 200, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:5, 1:5] = 255
        src = DefectSource(id=f"{cls}/000", cls=cls, image=img, mask=mask)
        return replace(ctx, source=src).with_log("source", {"source_id": src.id, "class": cls})


class GeomIdentity(_Dummy):
    stage = "geometry"
    methods = ("affine",)

    def apply(self, ctx: Context) -> Context:
        assert ctx.source is not None
        return replace(ctx, patch=ctx.source.image, patch_mask=ctx.source.mask).with_log(
            "geometry", {"scale": 1.0}
        )


class PlaceCenterOrFail(_Dummy):
    stage = "placement"
    methods = ("sampled",)

    def apply(self, ctx: Context) -> Context:
        if self.deps.get("fail_placement"):
            return (
                replace(ctx, placement=None)
                .warn("배치 실패(더미)")
                .with_log("placement", {"failed": True})
            )
        assert ctx.patch_mask is not None
        h, w = ctx.target.image.shape[:2]
        ph, pw = ctx.patch_mask.shape
        x = int(ctx.rng.integers(0, w - pw + 1))
        y = int(ctx.rng.integers(0, h - ph + 1))
        placed = np.zeros((h, w), dtype=np.uint8)
        placed[y : y + ph, x : x + pw] = ctx.patch_mask
        pl = Placement(
            center=(x + pw // 2, y + ph // 2), bbox=(x, y, pw, ph), offset=(x, y), tries=1
        )
        return replace(ctx, placement=pl, placed_mask=placed).with_log(
            "placement", {"bbox": list(pl.bbox)}
        )


class BlendPaste(_Dummy):
    stage = "blend"
    methods = ("paste",)

    def apply(self, ctx: Context) -> Context:
        assert ctx.patch is not None and ctx.placement is not None and ctx.patch_mask is not None
        x, y, w, h = ctx.placement.bbox
        out = ctx.composite.copy()
        region = out[y : y + h, x : x + w]
        m = ctx.patch_mask > 0
        region[m] = ctx.patch[m]
        return replace(ctx, composite=out).with_log("blend", {"method": "paste"})


class HarmonizeNone(_Dummy):
    stage = "harmonize"
    methods = ("none",)

    def apply(self, ctx: Context) -> Context:
        return ctx.with_log("harmonize", {"method": "none"})


class DegradeNone(_Dummy):
    stage = "degrade"
    methods = ("none",)

    def apply(self, ctx: Context) -> Context:
        return ctx.with_log("degrade", {"method": "none"})


class GtFromPlaced(_Dummy):
    stage = "gtmask"
    methods = ("source",)

    def apply(self, ctx: Context) -> Context:
        h, w = ctx.target.image.shape[:2]
        union = np.zeros((h, w), dtype=np.uint8)
        instances = []
        ids = self.deps.get("class_ids", {})
        for p in ctx.placed:
            x, y, bw, bh = cv2.boundingRect(p.mask)
            instances.append(
                Instance(p.cls, ids.get(p.cls, 0), p.mask, (x, y, bw, bh), int((p.mask > 0).sum()))
            )
            union = np.maximum(union, p.mask)
        return replace(ctx, gt_mask=union, instances=tuple(instances)).with_log(
            "gtmask", {"policy": "source"}
        )


def _pipeline(defects=(1, 3), **deps) -> Pipeline:
    d = {
        "version": 1,
        "name": "t",
        "seed": 42,
        "inputs": {"bank": "b", "targets": "t"},
        "output": {"root": "o", "count": 1, "defects_per_image": list(defects)},
        "pipeline": {"preset": "hard-paste"},
    }
    rec = R.Recipe.from_dict(d)
    deps = {"class_ids": {"scratch": 0, "dent": 1}, **deps}
    pipe = rec.pipeline
    stages = {
        "roi": RoiAll(pipe.placement.roi, deps),
        "source": SourceFixed(pipe.source, deps),
        "geometry": GeomIdentity(pipe.geometry, deps),
        "placement": PlaceCenterOrFail(pipe.placement, deps),
        "blend": BlendPaste(pipe.blend, deps),
        "harmonize": HarmonizeNone(pipe.harmonize, deps),
        "degrade": DegradeNone(pipe.degrade, deps),
        "gtmask": GtFromPlaced(pipe.gtmask, deps),
    }
    return Pipeline(rec, stages)


def _target(gray: bool = False) -> TargetImage:
    img = np.full((32, 48, 3), 30, dtype=np.uint8)
    return TargetImage(path=Path("t.png"), image=img, gray=gray)


def test_pipeline_requires_all_stages() -> None:
    p = _pipeline()
    with pytest.raises(ValueError, match="빠졌습니다"):
        Pipeline(p.recipe, {k: v for k, v in p.stages.items() if k != "gtmask"})


def test_loop_count_and_instances() -> None:
    p = _pipeline((2, 2))
    r = p.run_one(_target(), 0)
    assert r.status == "ok"
    assert len(r.instances) == 2
    assert r.sidecar["defects_requested"] == 2 and len(r.sidecar["defects"]) == 2
    assert set(r.sidecar["defects"][0]) == {"source", "geometry", "placement", "blend", "harmonize"}
    assert r.gt_mask.shape == (32, 48) and r.gt_mask.max() == 255
    # 붙인 자리는 밝고(200) 나머지는 어둡다(30)
    assert r.image[r.gt_mask > 0].min() == 200 and r.image[r.gt_mask == 0].max() == 30


def test_deterministic_for_same_index_and_varies_by_index() -> None:
    p = _pipeline((1, 3))
    a = p.run_one(_target(), 5)
    b = p.run_one(_target(), 5)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.gt_mask, b.gt_mask)
    assert a.sidecar == b.sidecar
    c = p.run_one(_target(), 6)
    assert not (np.array_equal(a.image, c.image) and a.sidecar["defects"] == c.sidecar["defects"])


def test_trace_captures_every_stage_output() -> None:
    p = _pipeline((2, 2))
    r, steps = p.run_one_traced(_target(), 0)
    names = [s.stage for s in steps]
    assert names[0] == "roi" and names[-2:] == ["degrade", "gtmask"]
    assert names.count("blend") == 2 and steps[1].defect_index == 0
    assert len(steps) == 1 + 2 * 5 + 2
    assert r.status == "ok"


def test_placement_failure_skips_defect_without_exception() -> None:
    p = _pipeline((1, 1), fail_placement=True)
    r = p.run_one(_target(), 0)
    assert r.status == "skipped" and r.reason and "배치 실패" in r.reason
    assert r.instances == () and r.gt_mask.max() == 0
    assert np.array_equal(r.image, _target().image)
    # 실패한 결함의 로그도 남는다 (source·geometry·placement까지)
    assert set(r.sidecar["defects"][0]) == {"source", "geometry", "placement"}


def test_gray_target_is_demoted_on_output() -> None:
    p = _pipeline((1, 1))
    r = p.run_one(_target(gray=True), 0)
    assert r.image.ndim == 2 and r.image.shape == (32, 48)
    assert r.sidecar["target"]["shape"] == [32, 48, 1]


def test_from_recipe_reports_not_implemented_clearly() -> None:
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1},
            "pipeline": {"preset": "poisson-graft"},
        }
    )
    registered = registry.registered()
    if all(("source", m) not in registered for m in registry.schema_methods("source")):
        with pytest.raises(registry.StageNotImplementedError, match=r"source\.bank"):
            Pipeline.from_recipe(rec)
