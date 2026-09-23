"""은행 없이 동작하는 소스(self-cut · perlin-texture) — 레시피 검증 · runner(빈 은행·클래스·텍스처 로드) · CLI e2e ·
GUI 세션 · alpha opacity. 스테이지 단위 테스트는 ``test_source_selfcut``·``test_source_perlin``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from anograft import runner
from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.core.stages.blend.alpha import AlphaBlend
from anograft.io import imgio
from anograft.io.manifest import read_manifest
from anograft.studio.session import SessionError, StudioSession, default_recipe
from tests.fixtures import (
    blob_image,
    context,
    disk_image,
    disk_target,
    fake_yolo_dataset,
    line_defect,
    memory_bank,
)


def _base(preset: str, **over: object) -> dict:
    d: dict = {
        "version": 1,
        "name": "t",
        "seed": 1,
        "inputs": {"bank": None, "targets": "normals"},
        "output": {"root": "o", "count": 2},
        "pipeline": {"preset": preset},
    }
    d.update(over)
    return d


# --- 레시피 -----------------------------------------------------------------------


def test_bank_required_only_for_bank_source() -> None:
    with pytest.raises(ValidationError, match=r"inputs\.bank"):
        R.Recipe.from_dict(_base("poisson-graft"))
    rec = R.Recipe.from_dict(_base("self-cut"))
    assert rec.bankless and rec.inputs.bank is None and rec.inputs.bank_key() == ""
    assert rec.to_dict()["inputs"]["bank"] is None
    assert R.Recipe.from_yaml(rec.to_yaml()).to_yaml() == rec.to_yaml()  # null 왕복
    # 은행이 있어도 self-cut 은 되고, 경고만
    with_bank = R.Recipe.from_dict(
        {**_base("perlin-texture"), "inputs": {"bank": "b", "targets": "n"}}
    )
    empty = memory_bank([])
    assert with_bank.validate_against(empty) and "무시" in with_bank.validate_against(empty)[0]
    assert with_bank.effective_classes(empty) == ["anomaly"]
    assert with_bank.class_probabilities(empty) == {"anomaly": 1.0}


def test_class_ratio_must_match_cls_for_bankless() -> None:
    ok = R.Recipe.from_dict(
        {**_base("self-cut"), "output": {"root": "o", "count": 1, "class_ratio": {"cutpaste": 1}}}
    )
    assert ok.effective_classes(memory_bank([])) == ["cutpaste"]
    with pytest.raises(ValidationError, match=r"source\.cls"):
        R.Recipe.from_dict(
            {
                **_base("self-cut"),
                "output": {"root": "o", "count": 1, "class_ratio": {"scratch": 1}},
            }
        )


def test_presets_expand_and_init_recipe_sets_bank_null() -> None:
    for preset in ("self-cut", "perlin-texture"):
        d = R.init_recipe_dict(preset)
        assert d["inputs"]["bank"] is None and d["pipeline"]["preset"] == preset
        rec = R.Recipe.from_dict(d)
        assert rec.pipeline.source.method in R.BANKLESS_SOURCES
    assert R.init_recipe_dict("poisson-graft")["inputs"]["bank"] == "bank/mine"
    assert set(R.preset_names()) >= {"self-cut", "perlin-texture"}
    pt = R.Recipe.from_dict(_base("perlin-texture"))
    assert pt.pipeline.blend.method == "alpha" and pt.pipeline.blend.opacity == (0.4, 1.0)
    assert pt.pipeline.geometry.rotate == (0.0, 0.0)
    sc = R.Recipe.from_dict(_base("self-cut"))
    assert sc.pipeline.blend.method == "paste" and sc.pipeline.gtmask.policy == "source"


def test_with_method_to_bank_from_bankless_recipe_fails_clearly() -> None:
    rec = R.Recipe.from_dict(_base("self-cut"))
    with pytest.raises(ValidationError, match=r"inputs\.bank"):
        rec.with_method("source", "bank")
    # 반대 방향(bank 레시피 → self-cut)은 된다: 은행이 남아 있어도 무시
    b = R.Recipe.from_dict({**_base("hard-paste"), "inputs": {"bank": "b", "targets": "n"}})
    assert b.with_method("source", "perlin-texture").bankless


# --- alpha opacity -------------------------------------------------------------------


def test_alpha_opacity_scales_blend_and_consumes_rng_only_when_set() -> None:
    from dataclasses import replace

    from anograft.core.types import Placement

    t = disk_target(64)
    patch = np.full((16, 16, 3), 255, dtype=np.uint8)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    place = Placement(center=(32, 32), bbox=(24, 24, 16, 16), offset=(24, 24), tries=1)
    base = replace(context(t, seed=0, patch=patch, patch_mask=mask), placement=place)
    full = AlphaBlend(R.AlphaBlendConfig(feather_px=0), {}).apply(base)
    half = AlphaBlend(R.AlphaBlendConfig(feather_px=0, opacity=(0.5, 0.5)), {}).apply(base)
    assert "opacity" not in full.log["blend"] and half.log["blend"]["opacity"] == 0.5
    inside = (slice(28, 36), slice(28, 36))
    assert (full.composite[inside] == 255).all()
    expected = np.rint(t.image[inside].astype(np.float32) * 0.5 + 255 * 0.5)
    assert np.array_equal(half.composite[inside], expected.astype(np.uint8))
    # opacity 없으면 rng 를 소비하지 않는다 → 같은 rng 로 다음 뽑기가 같다
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    c1 = replace(context(t, seed=0, patch=patch, patch_mask=mask), placement=place, rng=r1)
    AlphaBlend(R.AlphaBlendConfig(feather_px=0), {}).apply(c1)
    assert r1.random() == r2.random()
    with pytest.raises(ValidationError):
        R.AlphaBlendConfig(opacity=(0.5, 1.5))


# --- runner ---------------------------------------------------------------------------


def _normals(tmp_path: Path, n: int = 3, size: int = 96) -> Path:
    folder = tmp_path / "normals"
    folder.mkdir()
    for i in range(n):
        imgio.write_image(folder / f"n{i}.png", disk_image(size))
    return folder


def test_prepare_without_bank_uses_empty_bank_and_cls(tmp_path: Path) -> None:
    normals = _normals(tmp_path)
    rec = R.Recipe.from_dict(
        {**_base("self-cut"), "inputs": {"bank": None, "targets": normals.as_posix()}}
    )
    prep = runner.prepare(rec)
    assert len(prep.bank) == 0 and prep.bank.name == "(없음)"
    assert runner.prepared_classes(rec, prep.bank) == ["cutpaste"]
    assert prep.deps["class_ids"] == {"cutpaste": 0} and prep.deps["class_probs"] == {
        "cutpaste": 1.0
    }
    assert "textures" not in prep.deps
    res = runner.run_index(prep, 0)
    assert res.status == "ok" and res.instances and res.instances[0].class_id == 0
    assert res.sidecar["defects"][0]["source"]["method"] == "self-cut"
    rows = runner.dry_run_table(prep)
    assert any("(없음 — self-cut" in v for _k, v in rows)


def test_prepare_loads_textures_fail_soft(tmp_path: Path) -> None:
    normals = _normals(tmp_path)
    tex_dir = tmp_path / "tex"
    tex_dir.mkdir()
    imgio.write_image(tex_dir / "a.png", blob_image(64))
    (tex_dir / "broken.png").write_bytes(b"not a png")
    rec = R.Recipe.from_dict(
        {
            **_base("perlin-texture"),
            "inputs": {"bank": None, "targets": normals.as_posix()},
            "pipeline": {
                "preset": "perlin-texture",
                "source": {
                    "method": "perlin-texture",
                    "texture": "dir",
                    "texture_dir": tex_dir.as_posix(),
                },
            },
        }
    )
    prep = runner.prepare(rec)
    assert len(prep.deps["textures"]) == 1 and any("텍스처 읽기 실패" in w for w in prep.warnings)
    res = runner.run_index(prep, 0)
    assert res.status == "ok" and res.sidecar["defects"][0]["source"]["texture"] == "dir"
    # 빈 폴더 → 경고 + self 폴백으로 계속 돈다
    empty = tmp_path / "empty"
    empty.mkdir()
    rec2 = rec.model_copy(
        update={
            "pipeline": rec.pipeline.model_copy(
                update={"source": rec.pipeline.source.model_copy(update={"texture_dir": empty})}
            )
        }
    )
    prep2 = runner.prepare(R.Recipe.from_dict(rec2.to_dict()))
    assert prep2.deps["textures"] == [] and any("이미지가 없습니다" in w for w in prep2.warnings)
    assert runner.run_index(prep2, 0).sidecar["defects"][0]["source"]["texture"] == "self"


# --- CLI e2e ---------------------------------------------------------------------------


def test_cli_run_bankless_recipe_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    normals = _normals(tmp_path, n=2, size=128)
    for preset, cls in (("self-cut", "cutpaste"), ("perlin-texture", "anomaly")):
        recipe = tmp_path / f"{preset}.yaml"
        root = tmp_path / f"out-{preset}"
        assert (
            main(
                [
                    "recipe",
                    "init",
                    "--preset",
                    preset,
                    "--targets",
                    normals.as_posix(),
                    "--out",
                    root.as_posix(),
                    "--count",
                    "3",
                    "--write",
                    str(recipe),
                ]
            )
            == EXIT_OK
        )
        assert yaml.safe_load(recipe.read_text(encoding="utf-8"))["inputs"]["bank"] is None
        assert main(["recipe", "check", str(recipe)]) == EXIT_OK
        assert "은행 불필요" in capsys.readouterr().out
        assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "완료: ok 3" in out
        data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        assert data["names"] == [cls] and data["nc"] == 1
        rows = read_manifest(root / "manifest.csv")
        ok = [r for r in rows if r["status"] == "ok"]
        assert len(ok) == 3 and all(r["classes"].split("|")[0] == cls for r in ok)
        for r in ok:
            lines = (root / r["label"]).read_text(encoding="utf-8").splitlines()
            assert lines and all(ln.split()[0] == "0" for ln in lines)
    # compare-methods source: bank 는 사유 타일, 나머지 둘은 ok
    grid = tmp_path / "cmp.png"
    code = main(
        [
            "preview",
            str(tmp_path / "self-cut.yaml"),
            "--index",
            "0",
            "--compare-methods",
            "source",
            "--out",
            str(grid),
            "--long-side",
            "128",
        ]
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "inputs.bank" in out and out.count("ok") >= 2 and grid.is_file()


# --- GUI 세션 (Qt 없음) --------------------------------------------------------------------


def test_studio_session_switches_between_bank_and_bankless(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
    from anograft.bank.importers import yolo as Y

    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
    )
    ses = StudioSession(
        default_recipe(bank=(tmp_path / "bank").as_posix(), targets=normals.as_posix())
    )
    ses.set_stage_field("placement", "margin_px", 4)
    ses.prepare_now()
    assert not ses.recipe.bankless
    # 프리셋을 self-cut 으로 — 은행 경로는 남아 있어도 무시(경고), prepare 키는 그대로
    ses.set_preset("self-cut")
    assert ses.recipe.bankless and not ses.needs_prepare()
    ses.prepare_now()
    assert ses.prepared is not None and ses.prepared.deps["class_ids"] == {"cutpaste": 0}
    # 은행 칸을 비우면 inputs.bank: null
    ses.set_paths("", None)
    assert ses.recipe.inputs.bank is None and ses.prepare_key()[0] == "" and ses.needs_prepare()
    prep = ses.prepare_now()
    assert len(prep.bank) == 0 and ses.summary()["bank"] == "(없음)"
    # 은행 없이 bank 프리셋으로 돌아가면 검증이 막는다(SessionError)
    with pytest.raises(SessionError, match=r"inputs\.bank"):
        ses.set_preset("poisson-graft")
    assert ses.recipe.bankless  # 실패한 변경은 적용되지 않는다


# --- 골든 이후에도 v0.1 프리셋 스트림은 그대로 (opacity/None 경로) --------------------------------


def test_bank_presets_unaffected_by_new_union_members() -> None:
    from anograft.core.pipeline import Pipeline
    from tests.fixtures import pipeline_deps

    rec = R.Recipe.from_dict(
        {
            **_base("alpha-paste"),
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [2, 2]},
        }
    )
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    p = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    r = p.run_one(disk_target(64), 0)
    assert r.status == "ok" and "opacity" not in r.sidecar["defects"][0]["blend"]
