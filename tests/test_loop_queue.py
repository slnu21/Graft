"""검토 대기 큐 (설계 `v1.x-training-loop.md` T5) — 예측 읽기 · 선정(불일치 우선) · 큐 폴더 · 은행 되돌리기.

핵심 계약 둘을 테스트가 못 박는다:

1. **큐 폴더는 검수 화면이 그대로 연다** — `ReviewSession.load` 가 열고 판정이 `review.csv` 로 저장된다.
   (새 포맷을 만들면 루프가 안 닫힌다 — §1.4 와 같은 규율.)
2. **채택분은 임포터 공통 처리를 탄다** — `mask_origin: pred:*`(추정으로 센다) · 평가셋은 `BankWriter.add` 가 거부.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from anograft.bank import Bank
from anograft.bank.bank import is_estimated
from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, main
from anograft.core.seeds import split_rng
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, read_review, write_review
from anograft.loop.policy import ReviewMix
from anograft.loop.queue import (
    REASON_DISAGREE,
    Instance,
    Prediction,
    PredictionSet,
    QueueError,
    accept_to_bank,
    build_queue,
    disagreement_counts,
    index_images,
    parse_scores,
    read_predictions,
    select_queue,
)
from anograft.review import ReviewSession
from tests.fixtures import blob_image, blob_mask

SIZE = 64
BLOB = (32, 32, 8)


def _write_field(root: Path, stems: list[str]) -> Path:
    """현장 이미지 폴더."""
    images = root / "field"
    images.mkdir(parents=True, exist_ok=True)
    for s in stems:
        imgio.write_image(images / f"{s}.png", blob_image(SIZE, [BLOB]))
    return images


def _write_pred(
    root: Path,
    scores: dict[str, float],
    *,
    name: str = "pred",
    boxes: dict[str, list[tuple[float, float, float, float]]] | None = None,
    masks: bool = True,
    cls: str = "scratch",
) -> Path:
    """``scores/<stem>.json`` (+ ``masks/<stem>.png``) — 어댑터 계약의 predict 출력 형식."""
    pred = root / name
    (pred / "scores").mkdir(parents=True, exist_ok=True)
    if masks:
        (pred / "masks").mkdir(parents=True, exist_ok=True)
    for stem, score in scores.items():
        bb = (boxes or {}).get(stem, [(24.0, 24.0, 16.0, 16.0)])
        (pred / "scores" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_score": score,
                    "instances": [{"bbox": list(b), "score": score, "class": cls} for b in bb],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if masks:
            imgio.write_image(pred / "masks" / f"{stem}.png", blob_mask(SIZE, [BLOB]))
    return pred


# --------------------------------------------------------------------- 예측 읽기


def test_parse_scores_is_fail_soft_about_broken_instances() -> None:
    """예측 한 줄이 깨져도 나머지는 살린다 — 수백 장 중 하나 때문에 루프가 멈추면 안 된다."""
    pred, warns = parse_scores(
        "a",
        {
            "image_score": 0.7,
            "instances": [
                {"bbox": [1, 2, 3, 4], "score": 0.7, "class": "scratch"},
                {"bbox": [1, 2, 3], "score": 0.5},  # 길이가 4가 아니다
                "not-an-object",
                {"bbox": [1, 2, "x", 4]},
            ],
        },
    )
    assert pred.image_score == pytest.approx(0.7)
    assert [i.cls for i in pred.instances] == ["scratch"]
    assert len(warns) == 3


def test_parse_scores_without_image_score_uses_best_instance() -> None:
    pred, _ = parse_scores(
        "a",
        {"instances": [{"bbox": [0, 0, 2, 2], "score": 0.2}, {"bbox": [0, 0, 2, 2], "score": 0.9}]},
    )
    assert pred.image_score == pytest.approx(0.9)


def test_read_predictions_attaches_masks_and_survives_broken_json(tmp_path: Path) -> None:
    pred = _write_pred(tmp_path, {"a": 0.9, "b": 0.4})
    (pred / "scores" / "broken.json").write_text("{not json", encoding="utf-8")
    (pred / "masks" / "b.png").unlink()  # 마스크가 없는 예측도 있다(detect 전용 모델)

    got = read_predictions(pred)
    assert set(got.items) == {"a", "b"}
    assert got.items["a"].mask is not None and got.items["b"].mask is None
    assert any("broken.json" in w for w in got.warnings)
    assert got.scored() == [("a", pytest.approx(0.9)), ("b", pytest.approx(0.4))]


def test_read_predictions_requires_scores_dir(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(QueueError, match="scores"):
        read_predictions(tmp_path / "empty")


# --------------------------------------------------------------------- 선정(순수)


def _pset(scores: dict[str, float], boxes: dict[str, list[tuple]] | None = None) -> PredictionSet:
    items = {
        stem: Prediction(
            stem=stem,
            image_score=v,
            instances=tuple(
                Instance(bbox=b, score=v, cls="scratch")
                for b in (boxes or {}).get(stem, [(10.0, 10.0, 10.0, 10.0)])
            ),
        )
        for stem, v in scores.items()
    }
    return PredictionSet(root=Path("pred"), items=items)


def test_disagreement_counts_only_counts_unmatched_detections() -> None:
    a = _pset({"same": 0.5, "extra": 0.5, "alone": 0.5})
    b = _pset(
        {"same": 0.5, "extra": 0.5, "other": 0.5},
        boxes={
            "same": [(10.0, 10.0, 10.0, 10.0)],  # 같은 위치 → 일치
            "extra": [(10.0, 10.0, 10.0, 10.0), (40.0, 40.0, 10.0, 10.0)],  # 한쪽에만 하나 더
        },
    )
    counts = disagreement_counts(a.items, b.items)
    assert counts == {"extra": 1}  # 일치한 same 도, 한쪽에만 있는 stem 도 들어가지 않는다


def test_select_queue_puts_disagreement_first_but_caps_it() -> None:
    """불일치가 큐를 통째로 먹으면 무작위 몫(드리프트 감지)이 사라진다 — 경계 몫이 상한."""
    scores = {f"s{i:02d}": 0.5 for i in range(12)}
    a = _pset(scores)
    b = _pset(scores, boxes={s: [(40.0, 40.0, 10.0, 10.0)] for s in scores})  # 전부 어긋남

    picked = select_queue(a, b, threshold=0.5, n=10, rng=split_rng(7))
    reasons = [q.reason for q in picked]
    assert len(picked) == 10
    assert reasons[:6] == [REASON_DISAGREE] * 6  # 경계 몫 60% 가 상한
    assert REASON_DISAGREE not in reasons[6:]
    assert {"확신", "무작위"} & set(reasons[6:])
    # 둘 다 검출을 하나씩 내지만 위치가 달라 서로 한쪽에만 있는 것이 둔개다
    assert all(q.disagreement == 2 for q in picked if q.reason == REASON_DISAGREE)


def test_select_queue_without_second_model_is_boundary_first() -> None:
    a = _pset({"hot": 0.98, "edge": 0.51, "cold": 0.02, "mid": 0.60, "low": 0.10})
    picked = select_queue(a, threshold=0.5, n=3, mix=ReviewMix(1, 0, 0), rng=split_rng(1))
    assert [q.stem for q in picked] == ["edge", "mid", "low"]
    assert all(q.reason == "경계" for q in picked)
    assert "임계" in picked[0].detail


def test_select_queue_is_reproducible_for_the_same_seed() -> None:
    """무작위 몫도 ``rng`` 를 지나기 때문에 같은 시드면 같은 큐다(코어 규약 2 — 전역 난수 금지)."""
    a = _pset({f"s{i:02d}": i / 20 for i in range(20)})
    only_random = ReviewMix(0, 0, 1)
    one = select_queue(a, threshold=0.5, n=8, mix=only_random, rng=split_rng(3))
    two = select_queue(a, threshold=0.5, n=8, mix=only_random, rng=split_rng(3))
    other = select_queue(a, threshold=0.5, n=8, mix=only_random, rng=split_rng(4))
    assert [q.stem for q in one] == [q.stem for q in two]
    assert [q.stem for q in one] != [q.stem for q in other]
    assert all(q.reason == "무작위" for q in one)


def test_index_images_keeps_the_first_of_a_stem_collision() -> None:
    mapping, warns = index_images([Path("a/x.png"), Path("b/x.jpg"), Path("a/y.png")])
    assert mapping == {"x": Path("a/x.png"), "y": Path("a/y.png")}
    assert len(warns) == 1


# --------------------------------------------------------------------- 큐 폴더


def test_build_queue_writes_a_folder_the_review_screen_opens(tmp_path: Path) -> None:
    """**핵심 계약** — 새 포맷을 만들지 않는다. 검수 세션이 그대로 열고 판정이 review.csv 로 나간다."""
    stems = ["a", "b", "c"]
    images = _write_field(tmp_path, stems)
    preds = read_predictions(_write_pred(tmp_path, {s: 0.5 for s in stems}))
    mapping, _ = index_images(sorted(images.glob("*.png")))

    items = select_queue(preds, threshold=0.5, n=3, rng=split_rng(0))
    out = tmp_path / "queue"
    summary = build_queue(items, preds, mapping, out, trainer="noop", round_no=2)

    assert len(summary.written) == 3 and not summary.warnings
    assert (out / MANIFEST_FILE).is_file() and (out / REVIEW_FILE).is_file()
    assert (out / "queue.csv").is_file()
    rows = read_manifest(out / MANIFEST_FILE)
    assert [r["status"] for r in rows] == ["ok"] * 3
    assert [r["index"] for r in rows] == ["0", "1", "2"]
    assert all((out / r["image"]).is_file() and (out / r["mask"]).is_file() for r in rows)
    assert all(r["classes"] == "scratch" and int(r["area_px"]) > 0 for r in rows)
    assert read_review(out / REVIEW_FILE) == {"0": ("", ""), "1": ("", ""), "2": ("", "")}

    side = json.loads((out / rows[0]["sidecar"]).read_text(encoding="utf-8"))
    assert side["queue"]["trainer"] == "noop" and side["queue"]["round"] == 2
    assert side["gtmask"]["instances"][0]["class"] == "scratch"
    # 경로는 posix — 합성 사이드카·manifest 와 같은 규약(Windows 역슬래시 금지)
    assert "\\" not in side["target"]["file"] and side["target"]["file"].endswith("/a.png")
    assert side["target"]["shape"][:2] == [SIZE, SIZE]

    # 검수 화면(Qt 탭 · 웹 ⑤)이 여는 그 세션
    session = ReviewSession()
    session.load(out)
    counts = session.counts()
    assert counts["ok"] == 3 and counts["unreviewed"] == 3
    assert "검토 대기 사유" in session.item("0").warnings[0]
    session.set_verdict("0", "accept")
    session.save()
    assert read_review(out / REVIEW_FILE)["0"][0] == "accept"
    # 항목이 합성이 아니므로 요약 문구는 "검토 대기" 라고 말해야 한다(화면이 거짓말을 하지 않게)
    assert session.is_queue and "검토 대기 3" in session.summary_text()


def test_build_queue_drops_items_without_an_image_and_keeps_going(tmp_path: Path) -> None:
    images = _write_field(tmp_path, ["a"])
    preds = read_predictions(_write_pred(tmp_path, {"a": 0.5, "gone": 0.5}))
    mapping, _ = index_images(sorted(images.glob("*.png")))

    items = select_queue(preds, threshold=0.5, n=2, rng=split_rng(0))
    summary = build_queue(items, preds, mapping, tmp_path / "q")
    assert summary.missing_image == ["gone"]
    assert [i.stem for i in summary.written] == ["a"]
    assert any("원본 이미지를 찾지 못했습니다" in w for w in summary.warnings)


def test_build_queue_refuses_a_mask_of_the_wrong_size(tmp_path: Path) -> None:
    """이상맵을 모델 입력 해상도 그대로 낸 어댑터(T2 함정) — 마스크 없이 넣고 사유를 남긴다."""
    images = _write_field(tmp_path, ["a"])
    pred = _write_pred(tmp_path, {"a": 0.5})
    imgio.write_image(pred / "masks" / "a.png", np.zeros((16, 16), np.uint8))
    preds = read_predictions(pred)
    mapping, _ = index_images(sorted(images.glob("*.png")))

    summary = build_queue(
        select_queue(preds, threshold=0.5, n=1, rng=split_rng(0)), preds, mapping, tmp_path / "q"
    )
    assert summary.without_mask == ["a"]
    assert any("크기" in w for w in summary.warnings)
    rows = read_manifest(summary.out / MANIFEST_FILE)
    assert rows[0]["mask"] == "" and int(rows[0]["area_px"]) > 0  # 면적은 박스에서
    side = json.loads((summary.out / rows[0]["sidecar"]).read_text(encoding="utf-8"))
    assert any("결함 표시 화면" in w for w in side["warnings"])


def test_build_queue_refuses_a_non_empty_folder(tmp_path: Path) -> None:
    out = tmp_path / "q"
    out.mkdir()
    (out / "x.txt").write_text("hi", encoding="utf-8")
    images = _write_field(tmp_path, ["a"])
    preds = read_predictions(_write_pred(tmp_path, {"a": 0.5}))
    mapping, _ = index_images(sorted(images.glob("*.png")))
    with pytest.raises(QueueError, match="빈 폴더"):
        build_queue(select_queue(preds, n=1, rng=split_rng(0)), preds, mapping, out)


# --------------------------------------------------------------------- 은행으로


def _queue_with_verdicts(tmp_path: Path, verdicts: dict[str, str]) -> Path:
    stems = sorted(verdicts)
    images = _write_field(tmp_path, stems)
    preds = read_predictions(_write_pred(tmp_path, {s: 0.5 for s in stems}))
    mapping, _ = index_images(sorted(images.glob("*.png")))
    out = tmp_path / "queue"
    build_queue(
        select_queue(preds, threshold=0.5, n=len(stems), rng=split_rng(0)),
        preds,
        mapping,
        out,
        trainer="noop",
        round_no=3,
    )
    rows = read_manifest(out / MANIFEST_FILE)
    by_stem = {Path(r["image"]).stem: r["index"] for r in rows}
    write_review(out / REVIEW_FILE, {by_stem[s]: (v, "") for s, v in verdicts.items()})
    return out


def test_pruned_copy_of_a_queue_is_still_a_queue(tmp_path: Path) -> None:
    """정리본이 자기를 다시 "합성"이라 부르면 안 된다 — 표식은 남긴 행만 담아 따라간다."""
    from anograft.io.prune import QUEUE_FILE, prune_dataset

    queue = _queue_with_verdicts(tmp_path, {"a": "accept", "b": "reject", "c": ""})
    out = tmp_path / "pruned"
    summary = prune_dataset(queue, out)
    assert summary.kept == 2 and summary.dropped == 1

    session = ReviewSession()
    session.load(out)
    assert session.is_queue and "검토 대기 2" in session.summary_text()
    kept = {r["index"] for r in read_manifest(out / MANIFEST_FILE)}
    rows = list(csv.DictReader((out / QUEUE_FILE).read_text(encoding="utf-8").splitlines()))
    assert {r["index"] for r in rows} == kept


def test_accept_to_bank_imports_only_accepted_and_marks_them_estimated(tmp_path: Path) -> None:
    queue = _queue_with_verdicts(tmp_path, {"a": "accept", "b": "reject", "c": ""})
    summary = accept_to_bank(queue, tmp_path / "bank")

    assert summary.accepted == 1 and summary.imported == 1
    bank = Bank.load(tmp_path / "bank")
    (src,) = bank.sources()
    assert src.cls == "scratch"
    assert src.mask_origin == "pred:noop"
    # 모델 마스크는 추정이다 — `bank ls` 의 est 가 그렇게 세어야 사실이 된다
    assert is_estimated(src.mask_origin)
    assert bank.summary()[0].estimated == 1 and bank.summary()[0].exact == 0
    assert set(src.tags) == {"round-3", "origin:field"}
    meta = json.loads((tmp_path / "bank" / f"{src.id}.json").read_text(encoding="utf-8"))
    assert meta["origin"].endswith("a.png")  # 큐 사본이 아니라 원본 경로


def test_accept_to_bank_reports_items_without_a_mask(tmp_path: Path) -> None:
    images = _write_field(tmp_path, ["a"])
    preds = read_predictions(_write_pred(tmp_path, {"a": 0.5}, masks=False))
    mapping, _ = index_images(sorted(images.glob("*.png")))
    out = tmp_path / "queue"
    build_queue(select_queue(preds, n=1, rng=split_rng(0)), preds, mapping, out, trainer="noop")
    write_review(out / REVIEW_FILE, {"0": ("accept", "")})

    summary = accept_to_bank(out, tmp_path / "bank")
    assert summary.accepted == 1 and summary.imported == 0
    assert summary.no_mask == ["a"]
    assert any("결함 표시 화면" in w for w in summary.warnings)


def test_accept_to_bank_respects_holdout(tmp_path: Path) -> None:
    """평가셋은 코드가 막는다(T4) — 자동 루프에서 사람 규율은 반드시 깨진다."""
    queue = _queue_with_verdicts(tmp_path, {"a": "accept", "b": "accept"})
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "holdout.txt").write_text("b\n", encoding="utf-8")

    summary = accept_to_bank(queue, bank, round_no=5)
    assert summary.imported == 1
    assert [Path(h).stem for h in summary.held_out] == ["b"]
    (src,) = Bank.load(bank).sources()
    assert src.tags[0] == "round-5"  # --round 가 큐 사이드카보다 우선


def test_accept_to_bank_can_fix_the_class_name(tmp_path: Path) -> None:
    queue = _queue_with_verdicts(tmp_path, {"a": "accept"})
    accept_to_bank(queue, tmp_path / "bank", cls="dent", tags=["lot-7"])
    (src,) = Bank.load(tmp_path / "bank").sources()
    assert src.cls == "dent" and "lot-7" in src.tags


# --------------------------------------------------------------------- CLI


def test_cli_queue_then_accept_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stems = ["a", "b", "c", "d"]
    images = _write_field(tmp_path, stems)
    pred_a = _write_pred(tmp_path, {s: 0.5 for s in stems})
    pred_b = _write_pred(
        tmp_path,
        {s: 0.5 for s in stems},
        name="pred-b",
        boxes={"a": [(1.0, 1.0, 6.0, 6.0)]},  # a 만 어긋난다
    )
    out = tmp_path / "queue"

    code = main(
        [
            "loop",
            "queue",
            "--pred",
            str(pred_a),
            "--pred-b",
            str(pred_b),
            "--images",
            str(images),
            "--out",
            str(out),
            "-n",
            "4",
            "--trainer",
            "noop",
            "--round",
            "1",
            "--json",
        ]
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["count"] == 4 and payload["reasons"][REASON_DISAGREE] == 1

    session = ReviewSession()
    session.load(out)
    for it in session.filtered("all"):
        session.set_verdict(it.index, "accept")
    session.save()

    code = main(["loop", "accept", str(out), "--bank", str(tmp_path / "bank"), "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["accepted"] == 4 and payload["imported"] == 4
    assert payload["perClass"] == {"scratch": 4}
    assert all(s.mask_origin == "pred:noop" for s in Bank.load(tmp_path / "bank").sources())


def test_cli_queue_reports_a_bad_predictions_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images = _write_field(tmp_path, ["a"])
    code = main(
        [
            "loop",
            "queue",
            "--pred",
            str(tmp_path / "nope"),
            "--images",
            str(images),
            "--out",
            str(tmp_path / "q"),
        ]
    )
    assert code == EXIT_RECIPE_ERROR
    assert "scores" in capsys.readouterr().err


def test_cli_queue_rejects_a_bad_mix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    images = _write_field(tmp_path, ["a"])
    pred = _write_pred(tmp_path, {"a": 0.5})
    code = main(
        [
            "loop",
            "queue",
            "--pred",
            str(pred),
            "--images",
            str(images),
            "--out",
            str(tmp_path / "q"),
            "--mix",
            "0.6,0.4",
        ]
    )
    assert code == EXIT_RECIPE_ERROR
    assert "--mix" in capsys.readouterr().err
