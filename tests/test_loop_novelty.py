"""처음 보는 형상 · 미분류 · 기준선 재설정 (설계 §2b.5·§6.7, 작업 단위 T15).

여기가 못 박는 것:

1. **처음 보는 형상은 분류기가 아니라 우선순위**다 — 보관함 조각들과의 최근접 거리를 로버스트 단위로 재고,
   비교할 조각이 적으면(`MIN_REFS`) 판정하지 않는다(작은 보관함에서는 무엇이든 멀어 보인다).
2. **미분류(`__unsorted__`)는 언제나 클래스 목록 맨 끝**이다 — 그래야 합성·출력에서 빼도 다른 클래스의
   class id 가 움직이지 않는다(= 기존 모델과 호환).
3. **틀린 이름보다 이름 없음이 낫다** — 처음 보는 형상은 채택해도 미분류로 들어가고, 이름은 사람이
   `bank promote` 로 준다(마스크·크롭은 건드리지 않는다).
4. **클래스 신설은 라운드를 멈춘다** — 평가셋에 그 클래스 정답이 없으면 잘 잡을수록 점수가 떨어지므로
   Δ 비교가 끊긴다. 사람이 `loop baseline-reset` 을 찍으면 그 다음 라운드는 점수를 견주지 않는다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from anograft.bank import Bank
from anograft.bank.bank import UNSORTED
from anograft.bank.importers.common import BankWriter
from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, main
from anograft.core.classes import named_classes, order_classes
from anograft.core.novelty import (
    DEFAULT_THRESHOLD,
    MIN_REFS,
    Feature,
    features_of,
    is_novel,
    novelty_score,
    scales,
    shape_of,
)
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop import ledger
from anograft.loop.config import load_loop_config
from anograft.loop.queue import (
    Instance,
    Prediction,
    PredictionSet,
    QueueItem,
    accept_to_bank,
    build_queue,
    novelty_scores,
    order_by_novelty,
)
from anograft.loop.round import (
    LoopError,
    baseline_reset,
    ledger_path,
    new_classes,
    run_round,
)
from tests.fixtures import blob_image, blob_mask, loop_workspace

SIZE = 64
BLOB = (32, 32, 8)
#: 비교 기준으로 쓸 얼룩 반지름들 — `MIN_REFS` 보다 많아야 판정이 켜진다
RADII = (5, 6, 7, 8, 9, 10, 11, 12, 6, 7, 8, 9)


# --------------------------------------------------------------------- 그림 재료


def _blob_pair(size: int = SIZE, radius: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return blob_image(size, [(32, 32, radius)]), blob_mask(size, [(32, 32, radius)])


def _line_pair(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """가는 대각선 — **bbox 로는 정사각형**이라 최소외접사각이 필요한 바로 그 모양."""
    image = np.full((size, size, 3), 40, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    cv2.line(image, (8, 8), (size - 8, size - 8), (225, 225, 225), 1)
    cv2.line(mask, (8, 8), (size - 8, size - 8), 255, 1)
    return image, mask


# --------------------------------------------------------------------- 순수: 외형 벡터


def test_feature_needs_a_mask() -> None:
    image, _ = _blob_pair()
    assert Feature.of(image, np.zeros((SIZE, SIZE), np.uint8)) is None


def test_shape_of_sees_a_diagonal_line_as_elongated() -> None:
    """bbox 로 재면 대각선 스크래치가 정사각형이 된다 — 가늘다는 사실이 사라진다."""
    _, mask = _line_pair()
    elongation, _fill = shape_of((mask > 0).astype(np.uint8), int(np.count_nonzero(mask)))
    assert elongation > 8  # bbox 로 재면 1.0 이 나온다

    _, blob = _blob_pair()
    e2, f2 = shape_of((blob > 0).astype(np.uint8), int(np.count_nonzero(blob)))
    assert e2 < 1.5 and f2 > 0.6
    assert elongation > e2 * 5


def test_novelty_needs_enough_references() -> None:
    """조각 셋짜리 보관함에서는 무엇이든 멀어 보인다 — 판정하지 않는다(소음이 되므로)."""
    refs = features_of([_blob_pair(radius=r) for r in (6, 7, 8)])
    feat = Feature.of(*_line_pair())
    assert len(refs) < MIN_REFS
    assert novelty_score(feat, refs) == 0.0
    # 기준이 충분해지면 같은 조각이 새 형상으로 잡힌다
    more = features_of([_blob_pair(radius=r) for r in RADII])
    assert len(more) >= MIN_REFS
    assert novelty_score(feat, more) > DEFAULT_THRESHOLD


def test_similar_pieces_are_not_novel() -> None:
    refs = features_of([_blob_pair(radius=r) for r in RADII])
    same = Feature.of(*_blob_pair(radius=8))
    score = novelty_score(same, refs)
    assert score < DEFAULT_THRESHOLD
    assert not is_novel(score)


def test_scales_are_robust_and_floored() -> None:
    """한 유형뿐인 보관함은 MAD 가 0 이다 — 바닥값이 없으면 어떤 차이도 무한히 멀어진다."""
    refs = features_of([_blob_pair(radius=8) for _ in range(MIN_REFS + 2)])
    unit = scales(refs)
    assert all(s > 0 for s in unit)
    assert novelty_score(Feature.of(*_blob_pair(radius=8)), refs, scale=unit) < 0.5


def test_threshold_zero_turns_the_signal_off() -> None:
    assert not is_novel(0.99, 0.0)


# --------------------------------------------------------------------- 미분류: 클래스 목록 불변식


def test_unsorted_is_always_last_and_ids_do_not_move() -> None:
    assert order_classes(["spot", UNSORTED, "crack"]) == ["spot", "crack", UNSORTED]
    assert named_classes(["spot", UNSORTED, "crack"]) == ["spot", "crack"]

    from anograft.core.types import DefectSource

    srcs = [
        DefectSource(
            id=f"{c}/a", cls=c, image=np.zeros((8, 8, 3), np.uint8), mask=np.zeros((8, 8), np.uint8)
        )
        for c in ("spot", UNSORTED, "crack")
    ]
    bank = Bank.from_sources(srcs, classes=["spot", UNSORTED, "crack"])
    assert bank.classes == ["spot", "crack", UNSORTED]
    assert bank.usable_classes == ["spot", "crack"]
    assert bank.class_ids["crack"] == 1  # 미분류가 끼어도 실제 클래스 id 는 그대로
    assert len(bank.unsorted()) == 1


def _bank_with_unsorted(root: Path) -> Path:
    """이름 있는 조각 여럿(MIN_REFS 이상) + 미분류 1개."""
    bank = root / "bank"
    pairs = root / "pairs"
    (pairs / "images").mkdir(parents=True)
    (pairs / "masks").mkdir(parents=True)
    for i, radius in enumerate(RADII):
        image, mask = _blob_pair(radius=radius)
        imgio.write_image(pairs / "images" / f"s{i}.png", image)
        imgio.write_image(pairs / "masks" / f"s{i}.png", mask)
    assert (
        main(
            [
                "bank",
                "import-pairs",
                "--images",
                str(pairs / "images"),
                "--masks",
                str(pairs / "masks"),
                "--class",
                "spot",
                "--out",
                str(bank),
            ]
        )
        == EXIT_OK
    )
    line = root / "line"
    (line / "images").mkdir(parents=True)
    (line / "masks").mkdir(parents=True)
    image, mask = _line_pair()
    imgio.write_image(line / "images" / "u0.png", image)
    imgio.write_image(line / "masks" / "u0.png", mask)
    assert (
        main(
            [
                "bank",
                "import-pairs",
                "--images",
                str(line / "images"),
                "--masks",
                str(line / "masks"),
                "--class",
                UNSORTED,
                "--out",
                str(bank),
            ]
        )
        == EXIT_OK
    )
    return bank


def test_novelty_refs_exclude_unsorted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """미분류를 기준에 넣으면 같은 새 유형의 두 번째 장이 "이미 아는 것"이 되어 조용히 흘러간다."""
    bank = Bank.load(_bank_with_unsorted(tmp_path))
    capsys.readouterr()
    assert len(bank.novelty_refs()) == len(RADII)  # 미분류 1개는 기준이 아니다
    assert novelty_score(Feature.of(*_line_pair()), bank.novelty_refs()) > DEFAULT_THRESHOLD


def test_unsorted_is_left_out_of_synthesis_and_writer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """이름 없는 조각으로 합성하면 출력 라벨에 뜻 없는 클래스가 생긴다 — 아예 뽑지 않는다."""
    from anograft import runner
    from anograft.core.recipe import Recipe

    bank_root = _bank_with_unsorted(tmp_path)
    targets = tmp_path / "targets"
    targets.mkdir()
    for i in range(2):
        imgio.write_image(targets / f"t{i}.png", blob_image(96, []))
    recipe = tmp_path / "r.yaml"
    assert (
        main(
            [
                "recipe",
                "init",
                "--preset",
                "hard-paste",
                "--bank",
                str(bank_root),
                "--targets",
                str(targets),
                "--out",
                str(tmp_path / "out"),
                "--count",
                "2",
                "--write",
                str(recipe),
            ]
        )
        == EXIT_OK
    )
    doc = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    doc["pipeline"]["placement"]["roi"] = {"method": "none"}
    doc["pipeline"]["source"]["min_sources_warn"] = 1
    doc["output"]["writer"] = {"format": "yolo"}
    recipe.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    prep = runner.prepare(Recipe.load(recipe))
    assert list(prep.class_probs) == ["spot"]  # 미분류는 뽑히지 않는다
    assert any("미분류" in w and "빠집니다" in w for w in prep.warnings)
    summary = runner.run(prep, workers=0)
    assert summary.writer.n_ok > 0
    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == ["spot"] and data["nc"] == 1
    capsys.readouterr()


def test_bank_promote_gives_a_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bank_root = _bank_with_unsorted(tmp_path)
    capsys.readouterr()
    assert main(["bank", "ls", str(bank_root)]) == EXIT_OK
    assert "미분류 1개" in capsys.readouterr().out

    # 무엇을 옮길지 안 정하면 거절한다(실수로 전부 옮기지 않게)
    assert main(["bank", "promote", str(bank_root), "--to", "scratch"]) == EXIT_RECIPE_ERROR
    capsys.readouterr()

    assert (
        main(["bank", "promote", str(bank_root), "--to", "scratch", "--all", "--json"]) == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["to"] == "scratch" and len(payload["moved"]) == 1

    bank = Bank.load(bank_root)
    # 빈 클래스는 목록에 남는다(id 순서 = class id 규약) — 대신 조각이 없고 맨 끝이다
    assert bank.classes == ["spot", "scratch", UNSORTED]
    assert bank.usable_classes == ["spot", "scratch"]
    assert not bank.unsorted()
    moved = bank.by_class("scratch")[0]
    assert "promoted-from:__unsorted__" in moved.tags
    assert moved.mask_origin == "png"  # 이름만 준 것이다 — 마스크는 그대로


# --------------------------------------------------------------------- 큐 · 채택


def _pred_set(
    root: Path, stems: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[PredictionSet, dict[str, Path]]:
    """현장 이미지 + 예측(마스크 포함) — `read_predictions` 가 낸 것과 같은 모양을 손으로 만든다."""
    field = root / "field"
    masks = root / "pred" / "masks"
    field.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    items: dict[str, Prediction] = {}
    images: dict[str, Path] = {}
    for stem, (image, mask) in stems.items():
        imgio.write_image(field / f"{stem}.png", image)
        imgio.write_image(masks / f"{stem}.png", mask)
        images[stem] = field / f"{stem}.png"
        items[stem] = Prediction(
            stem=stem,
            image_score=0.5,
            instances=(Instance(bbox=(8.0, 8.0, 40.0, 40.0), score=0.5, cls="spot"),),
            mask=masks / f"{stem}.png",
        )
    return PredictionSet(root=root / "pred", items=items), images


def test_queue_puts_the_unfamiliar_shape_first(tmp_path: Path) -> None:
    refs = features_of([_blob_pair(radius=r) for r in RADII])
    preds, images = _pred_set(tmp_path, {"known": _blob_pair(), "strange": _line_pair()})
    items = [
        QueueItem(stem="known", score=0.5, reason="경계", detail="경계"),
        QueueItem(stem="strange", score=0.5, reason="무작위", detail="무작위"),
    ]
    scores = novelty_scores(items, preds, images, refs)
    assert scores["strange"] > scores["known"]

    ordered = order_by_novelty(items, scores, threshold=DEFAULT_THRESHOLD)
    assert [it.stem for it in ordered] == ["strange", "known"]
    assert ordered[0].reason == "무작위"  # 사유는 그대로 — 점수만 붙고 순서가 바뀐다

    out = tmp_path / "queue"
    summary = build_queue(
        ordered, preds, images, out, trainer="noop", novelty_threshold=DEFAULT_THRESHOLD
    )
    assert summary.novel == ["strange"]
    rows = read_manifest(out / MANIFEST_FILE)
    assert Path(rows[0]["image"]).stem == "strange"
    sidecar = json.loads((out / rows[0]["sidecar"]).read_text(encoding="utf-8"))
    assert "처음 보는 형상" in sidecar["warnings"][0]  # 첫 줄 = 왜 맨 앞에 왔는가
    assert sidecar["queue"]["novelty"] > DEFAULT_THRESHOLD
    with (out / "queue.csv").open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert float(csv_rows[0]["novelty"]) > DEFAULT_THRESHOLD


def test_accepting_an_unfamiliar_shape_parks_it_in_unsorted(tmp_path: Path) -> None:
    """틀린 이름은 되돌릴 수 없고(그 클래스가 오염된다) 미분류는 한 줄로 되돌린다."""
    preds, images = _pred_set(tmp_path, {"known": _blob_pair(), "strange": _line_pair()})
    items = [
        QueueItem(stem="known", score=0.5, reason="경계", detail="경계", novelty=0.1),
        QueueItem(stem="strange", score=0.5, reason="무작위", detail="무작위", novelty=0.9),
    ]
    out = tmp_path / "queue"
    build_queue(items, preds, images, out, trainer="noop", novelty_threshold=DEFAULT_THRESHOLD)
    rows = read_manifest(out / MANIFEST_FILE)
    write_review(out / REVIEW_FILE, {r["index"]: ("accept", "") for r in rows})

    bank_root = tmp_path / "bank"
    summary = accept_to_bank(
        out, bank_root, round_no=3, novelty_threshold=DEFAULT_THRESHOLD, keep_whole=True
    )
    assert summary.accepted == 2 and summary.imported == 2
    assert summary.unsorted == ["strange"]
    bank = Bank.load(bank_root)
    assert bank.classes[-1] == UNSORTED  # 언제나 끝
    assert [s.cls for s in bank.sources()].count(UNSORTED) == 1
    parked = bank.unsorted()[0]
    assert "novel" in parked.tags and "round-3" in parked.tags
    assert parked.mask_origin.startswith("pred:")
    assert any("미분류" in w for w in summary.warnings)

    # 임계를 0 으로 주면 이 판정을 아예 하지 않는다(끔)
    out2 = tmp_path / "queue2"
    build_queue(items, preds, images, out2, trainer="noop")
    rows2 = read_manifest(out2 / MANIFEST_FILE)
    write_review(out2 / REVIEW_FILE, {r["index"]: ("accept", "") for r in rows2})
    plain = accept_to_bank(out2, tmp_path / "bank2", novelty_threshold=0.0, keep_whole=True)
    assert plain.unsorted == [] and UNSORTED not in Bank.load(tmp_path / "bank2").classes


# --------------------------------------------------------------------- 클래스 신설 · 기준선 재설정


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    ws = loop_workspace(tmp_path)
    capsys.readouterr()
    return ws


def _add_class(bank_root: Path, cls: str) -> None:
    writer = BankWriter(bank_root)
    from anograft.bank.importers.common import ImportOptions, ImportRecord

    image, mask = _blob_pair(radius=9)
    writer.add(
        ImportRecord(
            image=image,
            gray=False,
            mask=mask,
            cls=cls,
            origin=f"hand/{cls}.png",
            id_hint=cls,
        ),
        ImportOptions(keep_whole=True),
    )
    writer.finish(entry={"importer": "test"})


def test_a_new_class_stops_the_round_until_a_person_decides(loop_ws: dict[str, Path]) -> None:
    """평가셋에 그 클래스 정답이 없으면 **잘 잡을수록 점수가 떨어진다** — 자동으로 넘기면 루프가 거짓말한다."""
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)  # 1라운드: 이 라운드가 쓴 클래스 목록이 원장에 남는다
    assert new_classes(loop) == []

    _add_class(loop_ws["bank"], "blowhole")
    assert new_classes(loop) == ["blowhole"]
    with pytest.raises(LoopError, match="blowhole"):
        run_round(loop)
    assert "round-002" not in [p.name for p in Path(loop_ws["out"]).iterdir()]

    # 사람이 평가셋을 손보고 기준선을 재설정하면 다시 돈다
    result = baseline_reset(loop, note="blowhole 추가 — 평가셋 재구성")
    assert result["after_round"] == 1 and "blowhole" in result["classes"]
    assert new_classes(loop) == []
    events = ledger.read(ledger_path(loop)).of(ledger.EVENT_BASELINE_RESET)
    assert len(events) == 1 and events[0].get("note").startswith("blowhole")

    second = run_round(loop)
    assert second.record.number == 2  # 멈추지 않고 진행한다


def test_after_a_reset_the_next_round_does_not_compare_scores(
    loop_ws: dict[str, Path], tmp_path: Path
) -> None:
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)  # champion = 라운드 1

    # 현장 이미지가 없는 설정 = 합성만 도는 라운드(사람 대기 없이 끝까지 간다)
    doc = yaml.safe_load(Path(loop_ws["loop"]).read_text(encoding="utf-8"))
    doc.pop("field", None)
    quiet = tmp_path / "loop-nofield.yaml"
    quiet.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    quiet_loop = load_loop_config(quiet)

    baseline_reset(quiet_loop, note="평가셋 재구성")
    done = run_round(quiet_loop)
    judge = done.record.data["judge"]
    assert judge["baseline_reset"] is True and judge["promote"] is True
    assert "견주지 않습니다" in judge["reason"]
    assert done.state.champion is not None and done.state.champion.round == 2


def test_cli_baseline_reset_reports_new_classes(
    loop_ws: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    _add_class(loop_ws["bank"], "dent")
    capsys.readouterr()

    assert (
        main(
            ["loop", "baseline-reset", "--config", str(loop_ws["loop"]), "--note", "dent", "--json"]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["newClasses"] == ["dent"] and payload["note"] == "dent"
    assert new_classes(load_loop_config(loop_ws["loop"])) == []

    # 라운드는 막힌 자리에서 그대로 이어 간다
    assert main(["loop", "run", "--config", str(loop_ws["loop"]), "--json"]) == EXIT_OK
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["round"] == 2


def test_status_and_round_data_carry_the_unsorted_count(loop_ws: dict[str, Path]) -> None:
    """라운드 기록이 "미분류로 몇 개 들어갔나"를 들고 있어야 사람 수정률·신규 유형을 나중에 센다."""
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    second = run_round(loop)
    assert second.waiting_for_human
    rows = [
        r
        for r in read_manifest(second.round_dir / "queue" / MANIFEST_FILE)
        if r.get("status") == "ok"
    ]
    write_review(
        second.round_dir / "queue" / REVIEW_FILE, {r["index"]: ("accept", "") for r in rows}
    )
    done = run_round(loop)
    assert "unsorted" in done.record.data["accept"]
    assert "novel" in done.record.data["queue"]
