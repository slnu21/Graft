"""미리보기 API (U5a) — 웹이 **Qt 와 같은 합성**을 보여 주는지, 쓰기가 규칙을 지키는지.

여기가 지키는 것: 합성은 `studio.jobs.run_preview` 가 낸 **그 배열** 그대로다(웹이 다시 합성하지 않는다) ·
정답 영역 오버레이는 **투명 배경 + 반투명 빨강**(U4 의 교훈) · 진단 문장은 `studio.diagnose` 가 낸다 ·
프리셋·시드 편집은 실패하면 이전 레시피를 유지한다 · 쓰기는 전부 `write=True` 로 등록된다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.bank.importers import yolo as Y
from anograft.core.channels import promote_to_bgr
from anograft.studio.jobs import KIND_PREVIEW, PreviewJob, run_preview
from anograft.studio.session import StudioSession, default_recipe
from anograft.web import studio_api
from anograft.web.api import WRITE_ROUTES, Request, handle
from tests.fixtures import fake_yolo_dataset

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _clean():
    studio_api.reset()
    yield
    studio_api.reset()


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
    """샘플 은행 + 정상 목록 + 이 입력으로 실제로 도는 레시피 한 장."""
    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
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
    ses.set_stage_field("roi", "erode_px", 2)
    ses.set_field(("pipeline", "source", "min_sources_warn"), 1)
    return ses.save(tmp_path / "studio.yaml")


def _open(recipe: Path) -> dict:
    res = handle(Request("/api/studio/open", "POST", json={"recipe": str(recipe)}))
    assert res.status == 200, res.payload
    return res.payload


def _preview(**body) -> dict:
    res = handle(Request("/api/studio/preview", "POST", json=body))
    assert res.status == 200, res.payload
    return res.payload


def _image(kind: str, **query) -> bytes:
    res = handle(Request("/api/studio/image", "GET", query={"kind": kind, **query}))
    assert res.status == 200, res.payload
    assert res.content_type == "image/png"
    assert res.body is not None and res.body.startswith(PNG_MAGIC)
    return res.body


def _decode(body: bytes, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(body, np.uint8), flags)


# ---------------------------------------------------------------- 상태


def test_studio_needs_open_first():
    res = handle(Request("/api/studio/state"))
    assert res.status == 200 and res.payload["open"] is False
    assert res.payload["presets"], "프리셋 목록은 열기 전에도 보여 준다(고르기 화면용)"
    res = handle(Request("/api/studio/image", "GET", query={"kind": "synth"}))
    assert res.status == 500 and "먼저 레시피" in res.payload["error"]


def test_open_reports_targets_bank_and_korean_stage_labels(recipe_file: Path):
    st = _open(recipe_file)
    assert st["open"] and st["bankN"] > 0 and st["targets"]
    assert st["recipe"]["preset"] == "poisson-graft"
    assert st["recipePath"].endswith("studio.yaml")
    labels = {r["stage"]: r["label"] for r in st["stages"]}
    # 한국어 라벨의 한 원천은 `core/help.py` — 서버가 실어 준다(프론트에 사전 사본 금지)
    assert labels["source"] == "결함 고르기" and "roi" in labels
    assert st["classes"] and st["runCommand"].startswith("anograft run ")


# ---------------------------------------------------------------- 합성


def test_preview_is_exactly_what_run_preview_makes(recipe_file: Path):
    """웹이 합성을 다시 구현하지 않았다는 증거 — 같은 세션·같은 잡이면 **배열이 같아야** 한다."""
    _open(recipe_file)
    meta = _preview(targetIndex=0)
    assert meta["status"] == "ok" and meta["defects"], meta
    assert meta["elapsedMs"] > 0 and 0 < meta["scale"] <= 1.0

    session = studio_api._SESSION
    assert session is not None
    direct = run_preview(
        session.prepared,
        PreviewJob(
            KIND_PREVIEW,
            session.generation,
            target=session.target,
            index=meta["variant"],
            long_side=session.long_side,
        ),
    )
    got = _decode(_image("synth"), cv2.IMREAD_COLOR)
    assert np.array_equal(got, promote_to_bgr(direct.result.image))
    assert [d["sourceId"] for d in meta["defects"]] == [
        str(x["source"]["source_id"]) for x in direct.result.sidecar["defects"] if "gt" in x
    ]


def test_preview_meta_carries_size_and_diagnostics(recipe_file: Path):
    _open(recipe_file)
    meta = _preview()
    base = _decode(_image("base"), cv2.IMREAD_COLOR)
    assert base.shape[:2] == (meta["height"], meta["width"])
    assert meta["longSide"] == 1024
    assert isinstance(meta["warnings"], list) and isinstance(meta["lowConfidence"], list)
    assert all(set(f) == {"n", "cls"} for f in meta["flipped"])
    assert meta["instances"] and len(meta["instances"][0]["bbox"]) == 4


def test_gt_overlay_is_transparent_outside_mask(recipe_file: Path):
    """CSS 블렌드로 0/255 마스크를 물들이면 사진 전체가 물든다(U4) — 알파는 서버가 만든다."""
    _open(recipe_file)
    meta = _preview()
    assert meta["hasGt"]
    over = _decode(_image("gt"))
    assert over.shape[2] == 4
    alpha = over[:, :, 3]
    assert alpha.max() > 0 and alpha.min() == 0
    inside = over[alpha > 0]
    assert (inside[:, 2] > inside[:, 0]).all(), "안쪽은 빨강(BGR 의 R 이 가장 큼)"
    assert (over[alpha == 0] == 0).all(), "마스크 밖은 완전 투명"


def test_roi_overlay_follows_roi_stage(recipe_file: Path):
    _open(recipe_file)
    meta = _preview()
    assert meta["hasRoi"], "이 레시피는 otsu ROI 를 쓴다"
    over = _decode(_image("roi"))
    assert over.shape[2] == 4 and over[:, :, 3].max() > 0


def test_thumbnail_per_target(recipe_file: Path):
    st = _open(recipe_file)
    for t in st["targets"]:
        img = _decode(_image("thumb", index=str(t["index"])), cv2.IMREAD_COLOR)
        assert max(img.shape[:2]) <= studio_api.THUMB_LONG_SIDE
    res = handle(Request("/api/studio/image", "GET", query={"kind": "thumb", "index": "99"}))
    assert res.status == 404


def test_unknown_image_kind_is_400(recipe_file: Path):
    _open(recipe_file)
    _preview()
    res = handle(Request("/api/studio/image", "GET", query={"kind": "hologram"}))
    assert res.status == 400 and "모르는 그림" in res.payload["error"]


def test_target_switch_changes_the_picture(recipe_file: Path):
    st = _open(recipe_file)
    if len(st["targets"]) < 2:
        pytest.skip("바탕 이미지가 한 장뿐")
    first = _image("synth") if _preview(targetIndex=0)["status"] == "ok" else b""
    second = _image("synth") if _preview(targetIndex=1)["status"] == "ok" else b""
    assert first and second and first != second


# ---------------------------------------------------------------- 편집


def test_preset_switch_replaces_pipeline_and_drops_stale_picture(recipe_file: Path):
    _open(recipe_file)
    _preview()
    res = handle(Request("/api/studio/preset", "POST", json={"name": "hard-paste"}))
    assert res.status == 200, res.payload
    methods = {r["stage"]: r["method"] for r in res.payload["stages"]}
    assert res.payload["recipe"]["preset"] == "hard-paste" and methods["blend"] == "paste"
    # 파라미터가 바뀌었으니 지난 그림은 버린다 — 옛 그림을 새 설정의 결과로 보여 주면 거짓말이 된다
    assert handle(Request("/api/studio/image", "GET", query={"kind": "synth"})).status == 404
    assert _preview()["status"] in ("ok", "skipped")


def test_unknown_preset_is_rejected(recipe_file: Path):
    _open(recipe_file)
    res = handle(Request("/api/studio/preset", "POST", json={"name": "no-such-preset"}))
    assert res.status == 400 and "프리셋이 없습니다" in res.payload["error"]


def test_seed_edit_and_rejection_keeps_recipe(recipe_file: Path):
    st = _open(recipe_file)
    before = st["recipe"]["seed"]
    ok = handle(Request("/api/studio/seed", "POST", json={"seed": before + 1}))
    assert ok.status == 200 and ok.payload["recipe"]["seed"] == before + 1
    bad = handle(Request("/api/studio/seed", "POST", json={"seed": -5}))
    assert bad.status == 400
    assert handle(Request("/api/studio/state")).payload["recipe"]["seed"] == before + 1


def test_different_seed_gives_a_different_picture(recipe_file: Path):
    _open(recipe_file)
    _preview()
    a = _image("synth")
    handle(Request("/api/studio/seed", "POST", json={"seed": 99}))
    _preview()
    assert _image("synth") != a


def test_preset_image_uses_the_gallery(recipe_file: Path):
    _open(recipe_file)
    res = handle(Request("/api/studio/preset-image", "GET", query={"name": "hard-paste"}))
    assert res.status == 200 and res.body is not None and res.body.startswith(PNG_MAGIC)
    missing = handle(Request("/api/studio/preset-image", "GET", query={"name": "nope"}))
    assert missing.status == 404


def test_save_writes_recipe_yaml(recipe_file: Path, tmp_path: Path):
    _open(recipe_file)
    out = tmp_path / "saved" / "studio.yaml"
    res = handle(Request("/api/studio/save", "POST", json={"path": str(out)}))
    assert res.status == 200, res.payload
    assert out.exists() and "preset" in out.read_text(encoding="utf-8")
    assert res.payload["runCommand"] == f"anograft run {out.as_posix()}"
    res = handle(Request("/api/studio/save", "POST", json={"path": ""}))
    assert res.status == 400


def test_open_needs_a_path():
    res = handle(Request("/api/studio/open", "POST", json={}))
    assert res.status == 400 and "필요합니다" in res.payload["error"]


def test_open_reports_recipe_errors(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: -1\n", encoding="utf-8", newline="\n")
    res = handle(Request("/api/studio/open", "POST", json={"recipe": str(bad)}))
    assert res.status == 400 and res.payload["error"]


# ---------------------------------------------------------------- 쓰기 등록


def test_all_mutating_routes_are_write_registered():
    """쓰기가 GET 으로 열리면 CSRF 방어가 통째로 무의미해진다 — 등록을 테스트로 고정한다."""
    handle(Request("/api/health"))  # 라우트 등록을 강제
    for path in (
        "/api/studio/open",
        "/api/studio/preview",
        "/api/studio/preset",
        "/api/studio/seed",
        "/api/studio/save",
    ):
        assert path in WRITE_ROUTES, path
        assert handle(Request(path, "GET")).status == 405
    for path in ("/api/studio/state", "/api/studio/presets", "/api/studio/image"):
        assert path not in WRITE_ROUTES
        assert handle(Request(path, "POST")).status == 405
