from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from anograft.core import recipe as R


def _base(**pipeline) -> dict:
    return {
        "version": 1,
        "name": "t",
        "seed": 1,
        "inputs": {"bank": "bank/x", "targets": "normals"},
        "output": {"root": "out", "count": 3},
        "pipeline": pipeline,
    }


class FakeBank:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    @property
    def classes(self) -> list[str]:
        return list(self._counts)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)


# --- 프리셋 ---------------------------------------------------------------


def test_all_presets_load_and_validate() -> None:
    names = R.preset_names()
    assert {"poisson-graft", "hard-paste", "alpha-paste", "multiband-graft"} <= set(names)
    for n in names:
        rec = R.Recipe.from_dict(_base(preset=n))
        assert rec.pipeline.preset == n


def test_preset_fills_defaults_and_user_overrides_win() -> None:
    rec = R.Recipe.from_dict(_base(preset="poisson-graft", blend={"poisson_mode": "normal"}))
    assert rec.pipeline.blend.method == "poisson"
    assert rec.pipeline.blend.poisson_mode == "normal"
    assert rec.pipeline.blend.feather_px == 3  # 프리셋 값 유지
    assert rec.pipeline.harmonize.method == "stats"


def test_changing_method_replaces_stage_block_wholesale() -> None:
    """프리셋 poisson 위에 사용자가 blend.method=alpha — poisson_mode가 섞여 들어오면 extra=forbid에 걸린다."""
    rec = R.Recipe.from_dict(
        _base(preset="poisson-graft", blend={"method": "alpha", "feather_px": 5})
    )
    assert rec.pipeline.blend.method == "alpha"
    assert rec.pipeline.blend.feather_px == 5


def test_nested_method_switch_in_roi() -> None:
    rec = R.Recipe.from_dict(_base(preset="poisson-graft", placement={"roi": {"method": "none"}}))
    assert rec.pipeline.placement.roi.method == "none"
    assert rec.pipeline.placement.margin_px == 8  # placement의 다른 키는 프리셋 유지


def test_unknown_preset() -> None:
    with pytest.raises(KeyError):
        R.Recipe.from_dict(_base(preset="nope"))


# --- 검증 규칙 -----------------------------------------------------------


def test_extra_key_is_rejected() -> None:
    with pytest.raises(ValidationError) as ei:
        R.Recipe.from_dict(_base(blend={"method": "alpha", "levels": 4}))
    assert "levels" in str(ei.value)


def test_typo_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        R.Recipe.from_dict(_base(geometry={"method": "affine", "rotat": [0, 10]}))


def test_range_lo_gt_hi_rejected() -> None:
    with pytest.raises(ValidationError):
        R.Recipe.from_dict(_base(geometry={"method": "affine", "scale": [1.5, 0.5]}))


def test_defects_per_image_min_one() -> None:
    d = _base()
    d["output"]["defects_per_image"] = [0, 2]
    with pytest.raises(ValidationError):
        R.Recipe.from_dict(d)


def test_class_ratio_normalized() -> None:
    d = _base()
    d["output"]["class_ratio"] = {"a": 2, "b": 2}
    rec = R.Recipe.from_dict(d)
    assert rec.output.class_ratio == {"a": 0.5, "b": 0.5}


def test_class_ratio_negative_or_empty_rejected() -> None:
    for bad in ({"a": -1, "b": 2}, {}, {"a": 0}):
        d = _base()
        d["output"]["class_ratio"] = bad
        with pytest.raises(ValidationError):
            R.Recipe.from_dict(d)


def test_class_ratio_must_be_subset_of_source_classes() -> None:
    d = _base(source={"method": "bank", "classes": ["a"]})
    d["output"]["class_ratio"] = {"a": 1, "b": 1}
    with pytest.raises(ValidationError):
        R.Recipe.from_dict(d)


def test_strength_unit_interval() -> None:
    with pytest.raises(ValidationError):
        R.Recipe.from_dict(_base(harmonize={"method": "stats", "strength": 1.5}))


# --- 직렬화 ---------------------------------------------------------------


def test_yaml_roundtrip_is_stable() -> None:
    rec = R.Recipe.from_dict(_base(preset="multiband-graft"))
    y1 = rec.to_yaml()
    y2 = R.Recipe.from_yaml(y1).to_yaml()
    assert y1 == y2
    assert "preset: multiband-graft" in y1


def test_hash_yaml_ignores_root_and_count_only() -> None:
    """데브로그 07 결정: ``--out``·``--count``만 다른 두 레시피는 같은 해시 입력, 그 외는 다르다."""
    base = _base(preset="poisson-graft")
    rec = R.Recipe.from_dict(base)
    moved = R.Recipe.from_dict(R.apply_overrides(dict(base), out="elsewhere/out", count=999))
    assert rec.to_yaml() != moved.to_yaml()
    assert rec.hash_yaml() == moved.hash_yaml()
    assert "root:" not in rec.hash_yaml() and "count:" not in rec.hash_yaml()
    # 뽑기에 영향 있는 키는 그대로 해시에 들어간다
    ratio = R.Recipe.from_dict({**base, "output": {**base["output"], "defects_per_image": [2, 2]}})
    assert ratio.hash_yaml() != rec.hash_yaml()
    seeded = R.Recipe.from_dict(R.apply_overrides(dict(base), seed=rec.seed + 1))
    assert seeded.hash_yaml() != rec.hash_yaml()
    # resolved 덤프(to_yaml)는 그대로 — root/count가 남아 있어야 재실행 가능
    assert "root:" in rec.to_yaml() and "count:" in rec.to_yaml()


def test_paths_serialize_posix() -> None:
    rec = R.Recipe.from_dict(_base())
    d = rec.to_dict()
    assert d["inputs"]["bank"] == "bank/x"
    assert "\\" not in yaml.safe_dump(d)


def test_load_with_overrides(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(_base(preset="hard-paste")), encoding="utf-8")
    rec = R.Recipe.load(p, seed=99, count=7, out=tmp_path / "o")
    assert rec.seed == 99 and rec.output.count == 7
    assert rec.output.root == tmp_path / "o"


def test_init_recipe_dict_is_fully_expanded() -> None:
    d = R.init_recipe_dict("alpha-paste")
    assert d["pipeline"]["blend"] == {"method": "alpha", "feather_px": 2}
    assert d["pipeline"]["harmonize"]["method"] == "reinhard"
    assert d["output"]["writer"]["format"] == "yolo"


# --- 은행 대조 -----------------------------------------------------------


def test_effective_classes_and_probabilities() -> None:
    bank = FakeBank({"a": 20, "b": 20, "c": 20})
    rec = R.Recipe.from_dict(_base())
    assert rec.effective_classes(bank) == ["a", "b", "c"]
    assert rec.class_probabilities(bank) == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})

    d = _base()
    d["output"]["class_ratio"] = {"a": 3, "b": 1}
    rec = R.Recipe.from_dict(d)
    assert rec.effective_classes(bank) == ["a", "b"]
    assert rec.class_probabilities(bank) == pytest.approx({"a": 0.75, "b": 0.25})


def test_validate_against_missing_class_raises() -> None:
    d = _base()
    d["output"]["class_ratio"] = {"zzz": 1}
    with pytest.raises(ValueError, match="은행에 없는"):
        R.Recipe.from_dict(d).validate_against(FakeBank({"a": 5}))


def test_validate_against_empty_class_raises() -> None:
    with pytest.raises(ValueError, match="0개"):
        R.Recipe.from_dict(_base()).validate_against(FakeBank({"a": 5, "b": 0}))


def test_validate_against_warns_on_few_sources() -> None:
    warnings = R.Recipe.from_dict(_base()).validate_against(FakeBank({"a": 3, "b": 50}))
    assert len(warnings) == 1 and "'a'" in warnings[0]


def test_merge_method_aware_basic() -> None:
    base = {"blend": {"method": "poisson", "poisson_mode": "mixed"}, "x": 1}
    over = {"blend": {"method": "alpha"}, "y": 2}
    assert R.merge_method_aware(base, over) == {"blend": {"method": "alpha"}, "x": 1, "y": 2}
    same = R.merge_method_aware(base, {"blend": {"poisson_mode": "normal"}})
    assert same["blend"] == {"method": "poisson", "poisson_mode": "normal"}
