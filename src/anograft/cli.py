"""anograft CLI (argparse — 의존성 추가 없음). 설계 §9.

메시지는 한국어 1줄(CLI는 내부 도구; GUI 문자열만 ko/en). 종료 코드: 0 정상 · 1 레시피/은행 오류 · 2 전부 skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from anograft import __version__, runner
from anograft.bank import Bank
from anograft.bank.bank import UNSORTED, BankError, is_estimated
from anograft.bank.importers import dataset as dataset_importer
from anograft.bank.importers import pairs as pairs_importer
from anograft.bank.importers import yolo as yolo_importer
from anograft.bank.importers.common import DEFAULT_MARGIN, DEFAULT_MIN_AREA
from anograft.bank.mask_from_box import LOW_CONFIDENCE
from anograft.bank.mask_from_box import METHODS as MASK_METHODS
from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.appearance import LIGHT_REAL_MIN, gray_of, mask_lighting
from anograft.core.help import method_help
from anograft.core.novelty import DEFAULT_THRESHOLD as NOVELTY_THRESHOLD
from anograft.datasets import DatasetError, adapter_names, get_adapter, info_lines
from anograft.io import imgio
from anograft.io.targets import load_target
from anograft.loop.queue import DEFAULT_IOU, DEFAULT_N, DEFAULT_THRESHOLD
from anograft.preview import (
    crop,
    render_compare,
    render_grid,
    render_preview,
    source_tile,
    union_bbox,
)
from anograft.samples import yolo as sample_yolo
from anograft.web import server as web_server

EXIT_OK = 0
EXIT_RECIPE_ERROR = 1
EXIT_ALL_SKIPPED = 2


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _stderr_line(line: str) -> None:
    """어댑터 로그 한 줄 — 계약상 stderr 는 로그 채널이라 그대로 흘린다(실시간)."""
    print(line, file=sys.stderr, flush=True)


def _load_recipe(args: argparse.Namespace, **overrides: object) -> R.Recipe | None:
    """레시피 로드 + CLI 오버라이드. 실패하면 stderr에 사유를 쓰고 None."""
    try:
        rec, notes = R.Recipe.load_with_notes(args.recipe, **overrides)
        for n in notes:
            _err(f"경로: {n}")
        return rec
    except ValidationError as e:
        _err(R.format_validation_error(e))
    except (KeyError, ValueError, OSError) as e:
        _err(f"레시피 로드 실패: {e}")
    return None


# ---------------------------------------------------------------------------
# 서브커맨드 구현
# ---------------------------------------------------------------------------


def cmd_methods(args: argparse.Namespace) -> int:
    infos = registry.list_methods(args.stage)
    if not infos:
        print(
            f"알 수 없는 스테이지: {args.stage} (선택: {', '.join(registry.STAGE_ORDER)})",
            file=sys.stderr,
        )
        return EXIT_RECIPE_ERROR
    width = max(len(i.method) for i in infos)
    current = None
    for i in infos:
        if i.stage != current:
            current = i.stage
            print(f"[{current}]")
        if i.usable:
            state = "ok"
        elif not i.implemented:
            state = "미구현"
        else:
            state = f"불가 — {i.reason}"
        req = f"  (requires: {', '.join(i.requires)})" if i.requires else ""
        h = method_help(i.stage, i.method)
        summary = f"  {h.label} — {h.summary}" if h else ""
        print(f"  {i.method.ljust(width)}  {state}{req}{summary}")
    print()
    print(f"프리셋: {', '.join(R.preset_names())}  (설명: anograft explain preset:<이름>)")
    return EXIT_OK


def cmd_explain(args: argparse.Namespace) -> int:
    """파라미터·method·프리셋 도움말(v0.9) — GUI 카드 툴팁과 같은 원천(core/help.py)."""
    from anograft.core import explain as E

    if args.markdown or args.out:
        md = E.markdown()
        if args.out:
            Path(args.out).write_text(md, encoding="utf-8", newline="\n")  # LF — 리다이렉트는 CRLF
            print(f"{args.out}: {len(md.splitlines())}줄")
        else:
            print(md, end="")
        return EXIT_OK
    if not args.query:
        from anograft.core.help import STAGE_HELP

        print("스테이지:")
        for k, h in STAGE_HELP.items():
            print(f"  {k:<10} {h.label} — {h.desc}")
        print("그 외: inputs · output · preset:<이름> · <stage>:<method> · <stage>.<field>")
        print("전체 표: anograft explain --markdown (= PARAMS.md)")
        return EXIT_OK
    try:
        for i, q in enumerate(args.query):
            if i:
                print()
            print(E.explain_text(q))
    except KeyError as e:
        _err(str(e.args[0]) if e.args else str(e))
        return EXIT_RECIPE_ERROR
    return EXIT_OK


def auto_dent_classes(bank_path: str | Path) -> dict[str, dict]:
    """``recipe init --auto-dent`` — 은행이 있으면 조명 의존(유의) 클래스 → 오버라이드(±15° + 방향에 안전한 flip).
    은행이 없거나 깨졌으면 빈 dict."""
    p = Path(bank_path)
    if not (p / "bank.yaml").is_file():
        return {}
    try:
        bank = Bank.load(p)
    except BankError:
        return {}
    return {r.cls: R.dent_override_for(r.light_dir) for r in bank.summary() if r.directional}


def cmd_recipe_init(args: argparse.Namespace) -> int:
    dent: dict[str, dict] = {c: dict(R.DENT_OVERRIDE) for c in (args.dent_class or [])}
    if args.auto_dent:
        found = auto_dent_classes(args.bank)
        if found:
            desc = ", ".join(f"{c}(flip {o['flip']})" for c, o in found.items())
            _err(f"참고: 은행 {args.bank} 의 조명 의존 클래스 → geometry.per_class ±15°: {desc}")
        else:
            _err(
                f"참고: 은행 {args.bank} 에서 조명 의존 클래스를 찾지 못했습니다(은행 없음 또는 lightR < {LIGHT_REAL_MIN}·유의 아님)"
            )
        for c, o in found.items():
            dent.setdefault(c, o)
    try:
        data = R.init_recipe_dict(
            args.preset,
            name=args.name,
            bank=args.bank,
            targets=args.targets,
            out=args.out,
            seed=args.seed,
            count=args.count,
            roi=args.roi,
            um_per_px=args.um_per_px,
            dent_classes=dent,
            classes=args.classes,
        )
    except KeyError as e:
        print(str(e.args[0]), file=sys.stderr)
        return EXIT_RECIPE_ERROR
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None)
    header = (
        f"# anograft {__version__} — recipe init --preset {args.preset}"
        + (f" --roi {args.roi}" if args.roi else "")
        + (f" --classes {' '.join(args.classes)}" if args.classes else "")
        + (f" --dent-class {' '.join(dent)}" if dent else "")
        + "\n"
        "# 모든 손잡이가 펼쳐져 있습니다. 스테이지의 method를 바꾸면 그 스테이지 키는 해당 method의 것만 남기세요.\n"
    )
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(
            parents=True, exist_ok=True
        )  # `recipes/new.yaml` 처럼 없는 폴더도 (KNOWN-ISSUES #10)
        out.write_text(header + text, encoding="utf-8")
        print(f"레시피를 썼습니다: {args.write}")
    else:
        sys.stdout.write(header + text)
    return EXIT_OK


def _unusable_stages(rec: R.Recipe) -> list[str]:
    """레시피가 고른 method 중 미구현·불가인 것."""
    problems: list[str] = []
    checks = [(key, getattr(rec.pipeline, key)) for key in R.STAGE_KEYS]
    checks.append(("roi", rec.pipeline.placement.roi))
    for key, cfg in checks:
        method = registry.config_method(cfg)
        info = next((i for i in registry.list_methods(key) if i.method == method), None)
        if info is None:
            problems.append(f"{key}.{method}: 스키마에 없음")
        elif not info.usable:
            problems.append(f"{key}.{method}: {info.reason}")
    return problems


def cmd_recipe_check(args: argparse.Namespace) -> int:
    try:
        rec, notes = R.Recipe.load_with_notes(args.recipe)
        for n in notes:
            _err(f"경로: {n}")
    except ValidationError as e:
        print(R.format_validation_error(e), file=sys.stderr)
        return EXIT_RECIPE_ERROR
    except (KeyError, ValueError, OSError) as e:
        print(f"레시피 로드 실패: {e}", file=sys.stderr)
        return EXIT_RECIPE_ERROR

    problems = _unusable_stages(rec)
    print(
        f"레시피 OK: {rec.name} (seed {rec.seed}, count {rec.output.count}, preset {rec.pipeline.preset})"
    )
    if problems:
        _err("실행 불가한 스테이지:")
        for p in problems:
            _err(f"  {p}")
    # 은행이 있으면 대조까지 (없으면 생략 — check는 레시피 파일만으로도 쓸 수 있어야 한다)
    bank_path = None if rec.inputs.bank is None else Path(rec.inputs.bank)
    if rec.bankless:
        print(
            f"은행 불필요: source.method {rec.pipeline.source.method} — 클래스 [{rec.pipeline.source.cls}]"
        )
    elif bank_path is not None and (bank_path / "bank.yaml").is_file():
        try:
            bank = Bank.load(bank_path)
            for w in runner.prepare_warnings(
                rec, bank
            ):  # run/GUI 와 같은 경고(대조·축척·저신뢰·조명)
                _err(f"경고: {w}")
            print(f"은행 대조 OK: {bank.name} — 클래스 {bank.classes}, 소스 {len(bank)}")
        except (BankError, ValueError, runner.PrepareError) as e:
            _err(f"은행 대조 실패: {e}")
            problems.append(str(e))
    else:
        print(
            f"은행 대조 생략: {bank_path.as_posix() if bank_path else '(없음)'} 에 bank.yaml 없음"
        )
    if args.resolved:
        sys.stdout.write(rec.to_yaml())
    return EXIT_RECIPE_ERROR if problems else EXIT_OK


# ---------------------------------------------------------------------------
# run · preview
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    rec = _load_recipe(args, seed=args.seed, count=args.count, out=args.out)
    if rec is None:
        return EXIT_RECIPE_ERROR
    try:
        prep = runner.prepare(
            rec,
            roi_cache_size=(
                args.roi_cache if args.roi_cache is not None else runner.DEFAULT_ROI_CACHE
            ),
        )
    except runner.PrepareError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    if args.dry_run:
        fit = runner.fit_diagnostic(prep)
        for k, v in runner.dry_run_table(prep, fit=fit):
            print(f"{k:>14}: {v}")
        for w in prep.warnings:
            _err(f"경고: {w}")
        if fit is not None:
            for fw in (fit.source_warning(), fit.warning()):
                if fw:
                    _err(f"경고: {fw}")
        print("(dry-run — 파일을 쓰지 않았습니다)")
        return EXIT_OK

    def progress(done: int, total: int, r: object) -> None:
        sys.stderr.write(f"\r[{done}/{total}] ")
        sys.stderr.flush()

    summary = runner.run(
        prep, workers=args.workers, progress=progress, warn=lambda w: _err(f"경고: {w}")
    )
    sys.stderr.write("\n")
    for line in runner.summary_lines(summary):
        print(line)
    for w in summary.writer.warnings:
        _err(f"경고: {w}")
    if getattr(args, "report", False) and not summary.all_skipped:
        _write_run_report(summary.writer.root)
    return EXIT_ALL_SKIPPED if summary.all_skipped else EXIT_OK


def _write_run_report(root: Path) -> None:
    """``run --report`` — 출력 폴더에 검수 리포트 HTML 을 바로(= ``dataset report <root>``). 실패해도 run 결과는 유효하니 경고만."""
    from anograft.review import ReviewError, ReviewSession

    s = ReviewSession()
    try:
        s.load(root, load_bank=True)
        out = s.write_report()
    except ReviewError as e:
        _err(f"경고: 리포트 실패 — {e}")
        return
    rs, rr = s.lighting_concentration()
    light = f" · 조명 R 합성 {rs:.2f}/실제 {rr:.2f}" if rs is not None and rr is not None else ""
    print(f"  report: {out.name} (합성 vs 실제 분포 6종{light})")


def cmd_preview(args: argparse.Namespace) -> int:
    rec = _load_recipe(args, seed=args.seed)
    if rec is None:
        return EXIT_RECIPE_ERROR
    try:
        prep = runner.prepare(rec)
    except runner.PrepareError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    for w in prep.warnings:
        _err(f"경고: {w}")
    _rng, path = runner.pick_target(prep, args.index)
    out = Path(args.out)
    if args.compare_methods:
        stage = args.compare_methods
        try:
            entries = runner.compare_methods(prep, stage, args.index)
        except runner.PrepareError as e:
            _err(str(e))
            return EXIT_RECIPE_ERROR
        tiles = []
        results = [r for _m, r, _reason in entries]
        shape = next((r.image.shape[:2] for r in results if r is not None), None)
        box = None if args.full or shape is None else union_bbox(results, shape)
        for method, r, reason in entries:
            if r is None or r.status != "ok":
                tiles.append((method, None, reason or (r.reason if r else "")))
            else:
                tiles.append((method, crop(r.image, box) if box else r.image, None))
        imgio.write_image(out, render_compare(tiles, long_side=args.long_side))
        where = f"크롭 {list(box)}" if box else "전체"
        print(
            f"비교: {out.as_posix()}  (stage {stage}, index {args.index}, seed {rec.seed}, "
            f"대상 {path.as_posix()}, {where})"
        )
        for method, r, reason in entries:
            if r is None:
                print(f"  {method:<10} — {reason}")
            elif r.status != "ok":
                print(f"  {method:<10} skipped — {r.reason}")
            else:
                d = runner.sidecar_defects(r)
                fb = sum(1 for x in d if x["blend"].get("fallback"))
                extra = f" (fallback {fb})" if fb else ""
                print(f"  {method:<10} ok · 결함 {len(d)}개{extra}")
        return EXIT_OK
    result = runner.run_index(prep, args.index)
    target = load_target(path, rec.inputs.um_per_px)
    canvas = render_preview(target.image, result, long_side=args.long_side)
    imgio.write_image(out, canvas)
    print(
        f"미리보기: {out.as_posix()}  (index {args.index}, seed {rec.seed}, 대상 {path.as_posix()})"
    )
    if result.status != "ok":
        print(f"  skipped — {result.reason}")
        return EXIT_ALL_SKIPPED
    for d in runner.sidecar_defects(result):
        s, pl, bl, gt = d["source"], d["placement"], d["blend"], d["gt"]
        fb = " (fallback)" if bl.get("fallback") else ""
        print(
            f"  {s['class']}#{s['class_id']} {s['source_id']} [{s['mask_origin']}] → "
            f"center {pl.get('center')} bbox {gt['bbox']} area {gt['area_px']}px · {bl['method']}{fb}"
        )
    for w in result.warnings:
        _err(f"  경고: {w}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# bank
# ---------------------------------------------------------------------------


def _split_csv(s: str | None) -> tuple[str, ...]:
    return tuple(x.strip() for x in (s or "").split(",") if x.strip())


def cmd_bank_import_yolo(args: argparse.Namespace) -> int:
    try:
        res = yolo_importer.import_yolo(
            args.images,
            args.labels,
            args.names,
            args.out,
            mask_from=args.mask_from,
            box_margin=args.box_margin,
            min_box=args.min_box,
            margin=args.margin,
            min_area=args.min_area,
            um_per_px=args.um_per_px,
            tags=_split_csv(args.tags),
            list_normals=args.list_normals,
            log=(lambda m: _err(f"  {m}")) if args.verbose else None,
        )
    except (FileNotFoundError, ValueError, BankError) as e:
        _err(f"임포트 실패: {e}")
        return EXIT_RECIPE_ERROR
    st = res.stats
    print(
        f"임포트 완료 → {res.bank_root.as_posix()}: 이미지 {res.n_images}장 · 소스 {len(st.added)}개 · "
        f"정상(라벨 없음) {len(res.normals)}장 · 버림(min-area) {st.dropped_small} · 중복 id {st.duplicates}"
    )
    print(f"  classes(id 순): {res.classes}")
    if st.held_out:
        # 조용히 건너뛰면 규칙이 있으나 마나다 — 몇 건을 왜 뺐는지 반드시 말한다(설계 §6.3)
        _err(
            f"평가셋(holdout.txt)이라 {len(st.held_out)}건을 넣지 않았습니다: "
            + ", ".join(st.held_out[:5])
            + (" …" if len(st.held_out) > 5 else "")
        )
    per = st.per_class()
    if per:
        print("  클래스별: " + ", ".join(f"{c} {per.get(c, 0)}" for c in res.classes))
    if res.mask_methods:
        print(
            "  박스→마스크: "
            + ", ".join(f"{m} {n}" for m, n in sorted(res.mask_methods.items()))
            + f" · 저신뢰(<{LOW_CONFIDENCE}) {res.low_confidence}"
            + (" — bank preview 로 확인, 라벨 탭에서 다듬기" if res.low_confidence else "")
        )
    if args.list_normals:
        print(
            f"  정상 목록: {Path(args.list_normals).as_posix()} ({len(res.normals)}줄) — inputs.targets 에 지정"
        )
    if res.warnings and not args.verbose:
        _err(f"경고 {len(res.warnings)}건 (--verbose 로 전부 보기). 첫 줄: {res.warnings[0]}")
    return EXIT_OK


def _print_pairs_result(res: pairs_importer.PairsImportResult, verbose: bool, what: str) -> None:
    st = res.stats
    print(
        f"임포트 완료 → {res.bank_root.as_posix()}: {what} {res.n_pairs}쌍 · 소스 {len(st.added)}개 · "
        f"마스크 없음 {res.n_missing_mask} · 버림(min-area) {st.dropped_small} · 중복 id {st.duplicates}"
    )
    print(f"  classes(id 순): {res.classes}")
    if st.held_out:
        # 조용히 건너뛰면 규칙이 있으나 마나다 — 몇 건을 왜 뺐는지 반드시 말한다(설계 §6.3)
        _err(
            f"평가셋(holdout.txt)이라 {len(st.held_out)}건을 넣지 않았습니다: "
            + ", ".join(st.held_out[:5])
            + (" …" if len(st.held_out) > 5 else "")
        )
    per = st.per_class()
    if per:
        print("  클래스별: " + ", ".join(f"{c} {per.get(c, 0)}" for c in res.classes))
    if res.warnings and not verbose:
        _err(f"경고 {len(res.warnings)}건 (--verbose 로 전부 보기). 첫 줄: {res.warnings[0]}")


def cmd_bank_import_pairs(args: argparse.Namespace) -> int:
    try:
        res = pairs_importer.import_pairs(
            args.images,
            args.masks,
            args.out,
            cls=args.cls,
            class_from_dir=args.class_from_dir,
            mask_suffix=args.mask_suffix,
            csv_path=args.csv,
            margin=args.margin,
            min_area=args.min_area,
            keep_whole=args.keep_whole,
            um_per_px=args.um_per_px,
            tags=_split_csv(args.tags),
            log=(lambda m: _err(f"  {m}")) if args.verbose else None,
        )
    except (FileNotFoundError, ValueError, BankError, imgio.ImageReadError) as e:
        _err(f"임포트 실패: {e}")
        return EXIT_RECIPE_ERROR
    _print_pairs_result(res, args.verbose, "이미지")
    return EXIT_OK


def cmd_bank_import_dataset(args: argparse.Namespace) -> int:
    try:
        res = dataset_importer.import_dataset(
            args.name,
            args.root,
            args.out,
            margin=args.margin,
            min_area=args.min_area,
            keep_whole=args.keep_whole,
            tags=_split_csv(args.tags),
            log=(lambda m: _err(f"  {m}")) if args.verbose else None,
        )
    except (DatasetError, FileNotFoundError, ValueError, BankError) as e:
        _err(f"임포트 실패: {e}")
        return EXIT_RECIPE_ERROR
    _print_pairs_result(res.pairs, args.verbose, f"{res.dataset}/{res.category}")
    if res.normals:
        d = res.normals[0].parent
        print(
            f"  정상 이미지 {len(res.normals)}장: {d.as_posix()} — recipe init --targets {d.as_posix()}"
        )
    return EXIT_OK


def cmd_bank_preview(args: argparse.Namespace) -> int:
    try:
        bank = Bank.load(args.bank)
    except BankError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    sources = bank.by_class(args.cls) if args.cls else bank.sources()
    if args.cls and not sources:
        _err(f"클래스 '{args.cls}' 소스 없음 (은행: {bank.classes})")
        return EXIT_RECIPE_ERROR
    sources = sources[: args.limit] if args.limit else sources
    tiles = [
        source_tile(
            s.image,
            s.mask,
            s.id,
            s.mask_origin,
            tile=args.tile,
            confidence=s.confidence,
            lighting_deg=mask_lighting(
                gray_of(s.image), s.mask
            ),  # 클래스 안에서 한 방향이면 조명 의존(KI #5)
        )
        for s in sources
    ]
    imgio.write_image(Path(args.out), render_grid(tiles, cols=args.cols))
    est = sum(1 for s in sources if is_estimated(s.mask_origin))
    low = sum(1 for s in sources if s.confidence is not None and s.confidence < LOW_CONFIDENCE)
    print(
        f"은행 미리보기: {Path(args.out).as_posix()} — 소스 {len(tiles)}개 (추정 마스크 {est}, 저신뢰 {low} = 빨간 테두리) · "
        f"{args.cols}열 · 타일 {args.tile}px"
    )
    return EXIT_OK


def cmd_dataset_prune(args: argparse.Namespace) -> int:
    """검수(``review.csv``)에서 반려된 합성 이미지를 뺀 정리본 사본."""
    from anograft.io.prune import PruneError, prune_dataset, read_review

    drop: set[str] = set()
    if (
        args.drop_flipped
    ):  # 은행이 있어야 실제 방향을 안다(검수 탭 필터 '조명 뒤집힘 의심'과 같은 집합)
        from anograft.review import ReviewError, ReviewSession

        sess = ReviewSession()
        try:
            sess.load(args.root, load_bank=True)
        except ReviewError as e:
            _err(f"정리 실패: {e}")
            return EXIT_RECIPE_ERROR
        if sess.bank is None:
            _err("경고: 은행을 열 수 없어 --drop-flipped 를 건너뜁니다(레시피 inputs.bank)")
        else:
            drop = sess.flipped_lighting()
            _err(f"조명 뒤집힘 의심 {len(drop)}건 제외" if drop else "조명 뒤집힘 의심 없음")
    try:
        review = read_review(args.review) if args.review else None
        s = prune_dataset(
            args.root, args.out, review, drop_unreviewed=args.drop_unreviewed, drop_indices=drop
        )
    except (PruneError, OSError) as e:
        _err(f"정리 실패: {e}")
        return EXIT_RECIPE_ERROR
    print(
        f"정리본 → {s.out.as_posix()}: 합성 {s.kept} 유지 · {s.dropped} 제외 · 정상 {s.normals} · "
        f"skipped 행 {s.skipped} 제거 · 파일 {s.files}"
    )
    for w in s.warnings:
        _err(f"경고: {w}")
    return EXIT_OK


def cmd_dataset_merge(args: argparse.Namespace) -> int:
    """출력 폴더 여러 개 → 하나(파일 접두어 · index 재부여 · coco 이어 붙임). 같은 writer 형식·클래스 이름이어야 한다."""
    from anograft.io.merge_datasets import MergeError, merge_datasets

    try:
        s = merge_datasets(
            args.roots, args.out, prefix=args.prefix, dedupe_normals=args.dedupe_normals
        )
    except (MergeError, OSError) as e:
        _err(f"병합 실패: {e}")
        return EXIT_RECIPE_ERROR
    per = " · ".join(
        f"{r.name}: 합성 {c['synthetic']} 정상 {c['normals']}"
        + (f" skipped {c['skipped']}" if c["skipped"] else "")
        for r, c in zip(s.roots, s.per_root, strict=True)
    )
    dropped = f"(중복 {s.normals_dropped} 제외)" if s.normals_dropped else ""
    print(
        f"병합 → {s.out.as_posix()}: 합성 {s.synthetic} · 정상 {s.normals}{dropped} · skipped 행 {s.skipped} · 파일 {s.files} ({per})"
    )
    for w in s.warnings:
        _err(f"경고: {w}")
    return EXIT_OK


def doctor_info() -> dict:
    """환경 진단 — 다른 PC 에서 문제를 보고받을 때 첫 줄에 붙일 것(의존성 버전·Qt·스레드·프리셋·frozen 여부)."""
    import os
    import platform

    import cv2
    import numpy as np
    import pydantic

    info: dict = {
        "anograft": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "cwd": Path.cwd().as_posix(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": int(cv2.getNumThreads()),
        "pydantic": pydantic.VERSION,
        "cpu_count": os.cpu_count(),
        "presets": R.preset_names(),
        "methods_unusable": [
            f"{i.stage}.{i.method}: {i.reason}" for i in registry.list_methods(None) if not i.usable
        ],
        "stdout_encoding": getattr(sys.stdout, "encoding", None),
    }
    try:
        from anograft.gui import qt_available

        ok, reason = qt_available()
        info["gui"] = "ok" if ok else f"불가 — {reason}"
        if ok:  # Qt 는 gui/ 안에서만 import(레이어 규약) — 버전은 메타데이터로
            from importlib.metadata import version

            info["pyside6"] = version("PySide6")
    except Exception as e:  # GUI 진단이 CLI 를 죽이면 안 된다
        info["gui"] = f"진단 실패 — {e}"
    return info


def cmd_doctor(args: argparse.Namespace) -> int:
    info = doctor_info()
    if args.json:
        import json

        print(json.dumps(info, ensure_ascii=False, indent=1))
        return EXIT_OK
    for k, v in info.items():
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) if v else "(없음)"
        print(f"{k:>18}: {v}")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    """웹 UI — `127.0.0.1` 고정. 번들이 없어도 죽지 않고 안내를 띄운다(fail-soft)."""
    return web_server.serve(
        args.port,
        api_only=args.api_only,
        open_browser=not args.no_browser,
        verbose=args.verbose,
        dev_port=args.dev_port,
    )


def _trainer_table(args: argparse.Namespace) -> tuple[Path, dict] | None:
    """``trainers.yaml`` 을 찾아 읽는다. 없으면 안내만 하고 ``None``."""
    from anograft.loop.registry import find_trainers_file, load_trainers

    path = find_trainers_file(Path(args.file) if args.file else None)
    if path is None:
        print("trainers.yaml 이 없습니다 — 학습기를 등록하면 루프가 그것을 부릅니다.")
        print("  찾은 곳: ./trainers.yaml · ~/.anograft/trainers.yaml")
        print("  예시는 trainers.example.yaml 을 복사해서 쓰세요.")
        return None
    return path, load_trainers(path)


def cmd_trainer_list(args: argparse.Namespace) -> int:
    """등록된 학습기와 **가용 여부·사유** — `methods` 가 스테이지에 하는 일을 학습기에 한다."""
    from anograft.loop.contract import TrainerError
    from anograft.loop.registry import probe_all

    try:
        found = _trainer_table(args)
    except TrainerError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return EXIT_RECIPE_ERROR
    if found is None:
        return EXIT_OK
    path, trainers = found

    print(f"{path}")
    if not trainers:
        print("  (등록된 학습기가 없습니다)")
        return EXIT_OK

    statuses = probe_all(trainers, path.parent)
    width = max(len(s.name) for s in statuses)
    for s in statuses:
        mark = "ok  " if s.available else "불가"
        print(f"  {mark} {s.name:<{width}}  {s.summary}")
    usable = sum(1 for s in statuses if s.available)
    print(f"\n  쓸 수 있는 학습기 {usable}/{len(statuses)}")
    return EXIT_OK


def cmd_trainer_info(args: argparse.Namespace) -> int:
    """학습기 하나의 선언을 그대로 보여 준다(계약 점검용)."""
    from anograft.loop.contract import TrainerError
    from anograft.loop.registry import probe

    try:
        found = _trainer_table(args)
    except TrainerError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return EXIT_RECIPE_ERROR
    if found is None:
        return EXIT_RECIPE_ERROR
    path, trainers = found

    spec = trainers.get(args.name)
    if spec is None:
        known = ", ".join(sorted(trainers)) or "(없음)"
        print(f"오류: {args.name!r} 는 {path} 에 없습니다. 등록된 것: {known}", file=sys.stderr)
        return EXIT_RECIPE_ERROR

    status = probe(args.name, spec, path.parent)
    print(f"{args.name}  ({' '.join(spec.command)})")
    if not status.available:
        print(f"  쓸 수 없음 — {status.reason}")
        return EXIT_OK
    info = status.info
    assert info is not None
    print(f"  버전           {info.version or '(없음)'}")
    print(f"  데이터 형식    {info.dataset_format}  (Graft 가 이 writer 로 내보냅니다)")
    print(
        f"  학습 입력      {'정상 이미지만 — 합성 결함은 평가에만' if info.normal_only else '라벨 포함'}"
    )
    print(f"  할 수 있는 것  {', '.join(info.capabilities)}")
    print(
        f"  결정적         {'예' if info.deterministic else '아니오 — 루프가 시드 여러 개로 평균 냅니다'}"
    )
    if spec.spec:
        print(f"  기본 spec      {spec.spec}")
    return EXIT_OK


def _trainer_spec(args: argparse.Namespace) -> tuple[Path, Any] | None:
    """등록부에서 ``args.name`` 스펙을 꺼낸다. 없으면 사유를 찍고 ``None``."""
    found = _trainer_table(args)
    if found is None:
        return None
    path, trainers = found
    spec = trainers.get(args.name)
    if spec is None:
        known = ", ".join(sorted(trainers)) or "(없음)"
        print(f"오류: {args.name!r} 는 {path} 에 없습니다. 등록된 것: {known}", file=sys.stderr)
        return None
    return path, spec


def _spec_override(path: str | None) -> dict[str, Any]:
    """``--spec`` JSON 파일(불투명). 읽기·형식 오류는 그대로 올린다 — 조용히 무시하면 학습이 엉뚱해진다."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--spec JSON 의 최상위는 오브젝트여야 합니다")
    return raw


def cmd_trainer_fit(args: argparse.Namespace) -> int:
    """등록된 학습기를 계약대로 부른다 — ``fit``. 벤치·루프가 공용으로 쓰는 진입점.

    어댑터 로그(stderr)는 **줄 단위로 그대로** 흘린다(학습은 길다). 결과는 사람이 읽는 줄 +
    ``--json`` 이면 기계가 읽는 한 줄.
    """
    from anograft.loop.contract import DEFAULT_TIMEOUT_S, TrainerError, fit, merge_spec
    from anograft.loop.registry import resolve_cwd

    try:
        found = _trainer_spec(args)
        if found is None:
            return EXIT_RECIPE_ERROR
        path, spec = found
        merged = merge_spec(spec.spec, _spec_override(args.spec))
        result = fit(
            spec.command,
            dataset=Path(args.dataset),
            out=Path(args.out),
            seed=args.seed,
            spec=merged,
            cwd=resolve_cwd(spec, path.parent),
            timeout=spec.timeout or DEFAULT_TIMEOUT_S,
            on_log=_stderr_line,
        )
    except (TrainerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return EXIT_RECIPE_ERROR

    if args.json:
        print(
            json.dumps({"model": result.model, "metrics": dict(result.metrics)}, ensure_ascii=False)
        )
    else:
        print(f"모델: {result.model}")
        for k, v in result.metrics.items():
            print(f"  {k:<16} {v}")
    return EXIT_OK


def _warn_duplicate_stems(listing: str) -> None:
    """예측 출력은 ``scores/<stem>.json`` 이라 **같은 이름이 둘이면 덮어쓴다**(계약 §1.4).

    MVTec 처럼 폴더마다 ``000.png`` 가 있는 데이터를 한 목록에 담으면 25장이 5장으로 조용히 줄어든다 —
    어댑터는 ``count: 25`` 를 돌려주므로 파일을 세어 보기 전에는 드러나지 않는다. 그래서 부르기 전에 경고한다.
    """
    try:
        paths = imgio.read_path_list(listing)
    except (OSError, ValueError):
        return  # 폴더를 준 경우 등 — 여기서 판단하지 않는다(어댑터 몫)
    seen: dict[str, Path] = {}
    dups: list[tuple[Path, Path]] = []
    for p in paths:
        if p.stem in seen:
            dups.append((seen[p.stem], p))
        else:
            seen[p.stem] = p
    if dups:
        a, b = dups[0]
        _err(
            f"경고: 목록에 같은 이름의 이미지가 {len(dups)}쌍 있습니다(예: {a} + {b}) — "
            f"예측은 이름(stem)으로 저장되므로 {len(paths)}장이 {len(seen)}장으로 덮어써집니다. "
            "이름을 구분해 복사한 뒤 목록을 만드세요."
        )


def cmd_trainer_predict(args: argparse.Namespace) -> int:
    """등록된 학습기를 계약대로 부른다 — ``predict``. 출력은 ``bank import-*`` 가 그대로 받는 형식."""
    from anograft.loop.contract import DEFAULT_TIMEOUT_S, TrainerError, merge_spec, predict
    from anograft.loop.registry import resolve_cwd

    _warn_duplicate_stems(args.images)
    try:
        found = _trainer_spec(args)
        if found is None:
            return EXIT_RECIPE_ERROR
        path, spec = found
        result = predict(
            spec.command,
            model=args.model,
            images=Path(args.images),
            out=Path(args.out),
            # 학습 때와 같은 spec 을 준다 — 추론 해상도가 달라지면 조용히 나빠진다
            spec=merge_spec(spec.spec, _spec_override(args.spec)),
            cwd=resolve_cwd(spec, path.parent),
            timeout=spec.timeout or DEFAULT_TIMEOUT_S,
            on_log=_stderr_line,
        )
    except (TrainerError, OSError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return EXIT_RECIPE_ERROR

    if args.json:
        print(
            json.dumps(
                {"predictions": str(result.predictions), "count": result.count}, ensure_ascii=False
            )
        )
    else:
        print(f"예측 {result.count}장 → {result.predictions}")
        print(
            "  masks/ 는 bank import-pairs · scores/ 의 instances 는 bank import-yolo 로 들어갑니다"
        )
    return EXIT_OK


def _review_mix(spec: str | None):
    """``--mix 0.6,0.2,0.2`` → ``ReviewMix``. 비율의 합은 자유(정규화한다)."""
    from anograft.loop.policy import ReviewMix

    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--mix 는 '경계,확신,무작위' 세 값입니다: {spec!r}")
    try:
        b, c, r = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--mix 값이 숫자가 아닙니다: {spec!r}") from exc
    return ReviewMix(boundary=b, confident=c, random=r)


def cmd_loop_queue(args: argparse.Namespace) -> int:
    """예측 → **검토 대기 폴더**(검수 화면이 그대로 여는 형식). 루프 T5.

    불일치(예측 폴더를 둘 주면) → 경계 → 확신 → 무작위 순으로 담고, 원본 이미지·예측 마스크 사본과
    사이드카를 함께 남긴다. 판정은 검수 화면에서, 되돌리기는 ``loop accept`` 에서.
    """
    from anograft.core.seeds import split_rng
    from anograft.io.targets import TargetsError, list_targets
    from anograft.loop.queue import (
        QueueError,
        build_queue,
        index_images,
        read_predictions,
        select_queue,
    )

    try:
        mix = _review_mix(args.mix)
        preds = read_predictions(args.pred)
        other = read_predictions(args.pred_b) if args.pred_b else None
        images, dup_warnings = index_images(list_targets(args.images))
    except (QueueError, TargetsError, ValueError, OSError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR

    for w in preds.warnings + (other.warnings if other else []) + dup_warnings:
        _err(f"  경고: {w}")

    items = select_queue(
        preds,
        other,
        threshold=args.threshold,
        n=args.n,
        mix=mix,
        iou_thresh=args.iou,
        rng=split_rng(args.seed),
    )
    if not items:
        _err("검토 대기로 고를 예측이 없습니다.")
        return EXIT_RECIPE_ERROR

    # 처음 보는 형상(T15) — 보관함을 줬을 때만. 고른 것만 재고 맨 앞으로 올린다
    novelty_threshold = NOVELTY_THRESHOLD if args.novelty is None else float(args.novelty)
    if args.bank and novelty_threshold > 0:
        from anograft.loop.queue import novelty_scores, order_by_novelty

        try:
            refs = Bank.load(args.bank).novelty_refs()
        except (BankError, OSError, ValueError) as exc:
            _err(f"  경고: 보관함을 읽지 못해 처음 보는 형상을 재지 않습니다 — {exc}")
            refs = []
        if refs:
            items = order_by_novelty(
                items,
                novelty_scores(items, preds, images, refs),
                threshold=novelty_threshold,
            )
    elif not args.bank:
        novelty_threshold = 0.0  # 비교 기준이 없으면 판정하지 않는다
    try:
        summary = build_queue(
            items,
            preds,
            images,
            args.out,
            threshold=args.threshold,
            trainer=args.trainer or "",
            round_no=args.round,
            pred_b=Path(args.pred_b) if args.pred_b else None,
            novelty_threshold=novelty_threshold,
        )
    except (QueueError, OSError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR

    for w in summary.warnings:
        _err(f"  경고: {w}")
    reasons = summary.reasons()
    if args.json:
        print(
            json.dumps(
                {
                    "out": str(summary.out),
                    "count": len(summary.written),
                    "reasons": reasons,
                    "withoutMask": summary.without_mask,
                    "missingImage": summary.missing_image,
                    "novel": summary.novel,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK

    out = Path(summary.out)
    print(f"검토 대기 {len(summary.written)}장 → {out.as_posix()}")
    if reasons:
        print("  사유: " + " · ".join(f"{k} {v}" for k, v in reasons.items()))
    if summary.novel:
        print(
            f"  처음 보는 형상 {len(summary.novel)}장을 맨 앞에 두었습니다 — 채택하면 미분류로 들어갑니다"
        )
    if summary.without_mask:
        print(
            f"  마스크 없음 {len(summary.without_mask)}장 — 채택해도 결함 표시 화면에서 그려야 은행에 들어갑니다"
        )
    print(f"  검수 화면에서 열기: anograft serve → 검수 · 폴더 {out.as_posix()}")
    print(f"  판정 뒤: anograft loop accept {out.as_posix()} --bank <결함 보관함>")
    return EXIT_OK


def cmd_loop_accept(args: argparse.Namespace) -> int:
    """검토가 끝난 큐의 **채택분만** 결함 보관함으로 — 임포터 공통 처리를 그대로 탄다(루프 T5)."""
    from anograft.loop.queue import QueueError, accept_to_bank

    try:
        summary = accept_to_bank(
            args.queue,
            args.bank,
            cls=args.cls,
            tags=_split_csv(args.tags),
            round_no=args.round,
            keep_whole=args.keep_whole,
            min_area=args.min_area,
            margin=args.margin,
            um_per_px=args.um_per_px,
            novelty_threshold=(NOVELTY_THRESHOLD if args.novelty is None else float(args.novelty)),
            log=(lambda m: _err(f"  {m}")) if args.verbose else None,
        )
    except (QueueError, BankError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR

    if args.json:
        print(
            json.dumps(
                {
                    "bank": str(summary.bank),
                    "accepted": summary.accepted,
                    "imported": summary.imported,
                    "perClass": summary.per_class,
                    "noMask": summary.no_mask,
                    "unsorted": summary.unsorted,
                    "heldOut": summary.held_out,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK

    print(
        f"채택 {summary.accepted}장 → 결함 조각 {summary.imported}개: {Path(summary.bank).as_posix()}"
    )
    if summary.per_class:
        print("  클래스별: " + " · ".join(f"{k} {v}" for k, v in sorted(summary.per_class.items())))
    for w in summary.warnings:
        _err(f"  경고: {w}")
    if summary.held_out:
        _err(f"  평가셋이라 뺀 것 {len(summary.held_out)}장 (holdout.txt)")
    if summary.unsorted:
        print(
            f"  처음 보는 형상 {len(summary.unsorted)}개는 **미분류**로 넣었습니다 — "
            "`anograft bank promote <보관함> --to <클래스>` 로 이름을 주면 합성·출력에 쓰입니다"
        )
    if summary.imported:
        print(
            "  마스크는 모델 초안입니다(mask_origin pred:*) — 결함 표시 화면에서 다듬으면 manual:* 이 됩니다"
        )
    return EXIT_OK


def _loop_config(args: argparse.Namespace):
    """``loop.yaml`` 을 찾아 읽는다. 없으면 안내만 하고 ``None``(fail-soft)."""
    from anograft.loop.config import LoopConfigError, find_loop_file, load_loop_config

    path = find_loop_file(Path(args.config) if args.config else None)
    if path is None:
        _err("loop.yaml 이 없습니다 — 라운드 설정을 만들어야 루프가 돕니다.")
        _err("  찾은 곳: ./loop.yaml · ~/.anograft/loop.yaml")
        _err("  예시는 loop.example.yaml 을 복사해서 쓰세요.")
        return None
    try:
        return load_loop_config(path)
    except LoopConfigError as exc:
        _err(f"오류: {exc}")
        return None


def _round_payload(result: Any) -> dict:
    """`loop run` · `loop tick` 이 같은 JSON 을 낸다 — 도구가 둘을 갈라 읽지 않게."""
    return {
        "round": result.record.number,
        "done": result.record.done,
        "waitingForHuman": result.waiting_for_human,
        "message": result.message,
        "data": result.record.data,
        "champion": result.state.champion.to_dict() if result.state.champion else None,
    }


def _print_round(result: Any) -> None:
    print(result.message)
    if result.waiting_for_human:
        print(
            f"  검수 화면: anograft serve → 검수 · 폴더 {(result.round_dir / 'queue').as_posix()}"
        )


def cmd_loop_run(args: argparse.Namespace) -> int:
    """라운드를 **다음 단계부터** 진행한다(멱등) — 사람이 판정할 차례면 멈추고 알려 준다. 루프 T10."""
    from anograft.loop.lock import LockBusyError
    from anograft.loop.round import LoopError, run_round

    loop = _loop_config(args)
    if loop is None:
        return EXIT_RECIPE_ERROR
    try:
        result = run_round(
            loop,
            on_log=None if args.json else _stderr_line,
            accept_partial=args.accept_partial,
            workers=args.workers,
            since=args.since,
        )
    except LockBusyError as exc:
        # 사람이 직접 부른 경우엔 오류다("내가 시킨 일이 안 됐다"). tick 에서는 정상 상태다.
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR

    if args.json:
        print(json.dumps(_round_payload(result), ensure_ascii=False))
        return EXIT_OK
    _print_round(result)
    return EXIT_OK


def cmd_loop_tick(args: argparse.Namespace) -> int:
    """**스케줄러가 부르는 진입점**(T13) — 멱등. Graft 는 스케줄러가 되지 않는다(설계 §2b.1).

    `loop run` 과 같은 일을 하지만 **스케줄러의 말로 답한다**: 다른 실행이 돌고 있으면 오류가 아니라
    "다음에 오라"(종료 코드 0)이고, 새로 들어온 이미지만 스코어링한다(유입 커서).
    """
    from anograft.loop.lock import LockBusyError
    from anograft.loop.round import LoopError, breaker_verdict, check_trigger, note_tick, run_round

    loop = _loop_config(args)
    if loop is None:
        return EXIT_RECIPE_ERROR

    # 자동 정지가 걸려 있으면 트리거를 보지 않는다(T12) — **오류가 아니라 상태**다(busy 와 같은 규율:
    # 5분마다 실패 알림이 오는 운영은 아무도 안 본다). 그리고 `--force` 로도 넘기지 않는다 —
    # 다시 돌리는 길은 `loop breaker-reset --note …` 하나여야 무엇을 바꿨는지가 원장에 남는다.
    try:
        breaker = breaker_verdict(loop)
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR
    if breaker.tripped:
        reason = f"자동 정지: {breaker.reason}"
        note_tick(loop, ran=False, reason=reason, log=None if args.json else _stderr_line)
        if args.json:
            print(
                json.dumps(
                    {
                        "ran": False,
                        "busy": False,
                        "breaker": True,
                        "kinds": list(breaker.kinds),
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"돌지 않았습니다 — {reason}")
            _err('  풀려면: anograft loop breaker-reset --note "무엇을 바꿨는지"')
        return EXIT_OK

    # "지금 돌 때인가" — 안 도는 이유는 **항상** 남긴다(tick.json + 사유가 바뀌면 원장, T14)
    try:
        decision, facts = check_trigger(loop)
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR
    if not decision.start and not args.force:
        note_tick(loop, ran=False, reason=decision.reason, log=None if args.json else _stderr_line)
        if args.json:
            print(
                json.dumps(
                    {
                        "ran": False,
                        "busy": False,
                        "reason": decision.reason,
                        "newLabels": facts.new_labels,
                        "newImages": facts.new_images,
                        "hoursSince": facts.hours_since,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"돌지 않았습니다 — {decision.reason}")
        return EXIT_OK
    if args.force and not decision.start:
        _err(f"--force — 트리거는 '{decision.reason}' 였지만 그대로 돕니다")

    try:
        result = run_round(
            loop,
            on_log=None if args.json else _stderr_line,
            accept_partial=args.accept_partial,
            workers=args.workers,
            since=args.since,
        )
    except LockBusyError as exc:
        note_tick(loop, ran=False, reason=str(exc))
        if args.json:
            print(json.dumps({"ran": False, "busy": True, "reason": str(exc)}, ensure_ascii=False))
        else:
            print(f"돌지 않았습니다 — {exc}")
        return EXIT_OK  # 스케줄러에게 "지금은 아니다"는 정상이다
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR

    note_tick(loop, ran=True, reason=decision.reason, round_no=result.record.number)
    if args.json:
        print(
            json.dumps({"ran": True, "busy": False, **_round_payload(result)}, ensure_ascii=False)
        )
        return EXIT_OK
    _print_round(result)
    return EXIT_OK


def cmd_loop_baseline_reset(args: argparse.Namespace) -> int:
    """**기준선 재설정**(T15) — 사람 결정 이벤트. 자동으로 부르는 곳은 없다."""
    from anograft.loop.round import LoopError, baseline_reset, new_classes

    loop = _loop_config(args)
    if loop is None:
        return EXIT_RECIPE_ERROR
    try:
        added = new_classes(loop)
        result = baseline_reset(loop, note=args.note, log=None if args.json else _stderr_line)
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR
    if args.json:
        print(json.dumps({**result, "newClasses": added}, ensure_ascii=False))
        return EXIT_OK
    print(
        f"기준선을 재설정했습니다 — 라운드 {result['after_round']} 뒤로 점수를 견주지 않습니다"
        f" (클래스 {len(result['classes'])}개)"
    )
    if added:
        print("  새 클래스: " + ", ".join(added))
    print(
        "  다음 라운드는 점수 비교 없이 현재 모델을 새로 세웁니다 — 평가셋을 이미 손봤는지 확인하세요"
    )
    return EXIT_OK


def cmd_loop_breaker_reset(args: argparse.Namespace) -> int:
    """**자동 정지 해제**(T12) — 사람 결정 이벤트. 자동으로 부르는 곳은 없다."""
    from anograft.loop.round import LoopError, breaker_reset

    loop = _loop_config(args)
    if loop is None:
        return EXIT_RECIPE_ERROR
    try:
        result = breaker_reset(loop, note=args.note, log=None if args.json else _stderr_line)
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return EXIT_OK
    if result["tripped"]:
        print(f"자동 정지를 해제했습니다 — 걸려 있던 사유: {result['reason']}")
    else:
        print("자동 정지는 걸려 있지 않았습니다 — 지점만 남깁니다")
    print(
        f"  라운드 {result['after_round']} 까지는 다음 판정에서 빠집니다"
        " (원장 rounds.jsonl 에 남습니다)"
    )
    if not args.note:
        _err("  --note 로 무엇을 바꿨는지 적어 두세요 — 다음 사람이 읽을 유일한 단서입니다")
    return EXIT_OK


def cmd_loop_status(args: argparse.Namespace) -> int:
    """지금 어디인가 — champion · 진행 중인 라운드 · 최근 이력. 루프 T10."""
    from anograft.loop.round import LoopError, status

    loop = _loop_config(args)
    if loop is None:
        return EXIT_RECIPE_ERROR
    try:
        st = status(loop)
    except (LoopError, OSError, ValueError) as exc:
        _err(f"오류: {exc}")
        return EXIT_RECIPE_ERROR
    if args.json:
        print(
            json.dumps(
                {
                    "out": str(st.out),
                    "round": st.state.round,
                    "next": st.next,
                    "champion": st.state.champion.to_dict() if st.state.champion else None,
                    "judged": st.judged,
                    "total": st.total,
                    "history": st.history,
                    "processed": st.processed,
                    "lock": st.lock.to_json() if st.lock else None,
                    "trigger": (
                        {"start": st.trigger.start, "reason": st.trigger.reason}
                        if st.trigger
                        else None
                    ),
                    "breaker": (
                        {
                            "tripped": st.breaker.tripped,
                            "reason": st.breaker.reason,
                            "kinds": list(st.breaker.kinds),
                        }
                        if st.breaker
                        else None
                    ),
                    "tick": st.tick.to_json() if st.tick else None,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_OK
    for line in st.lines():
        print(line)
    return EXIT_OK


def cmd_dataset_textures(args: argparse.Namespace) -> int:
    """텍스처셋(DTD)에서 perlin-texture 용 목록 파일을 만든다 — 은행에 넣지 않는다."""
    from anograft.datasets.dtd import DtdAdapter, write_texture_list

    try:
        adapter = get_adapter(args.name)
    except DatasetError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    if not isinstance(adapter, DtdAdapter):
        _err(f"{args.name} 은(는) 텍스처셋이 아닙니다 — dataset textures 는 dtd 만 지원")
        return EXIT_RECIPE_ERROR
    cats = _split_csv(args.categories) if args.categories else None
    warnings: list[str] = []
    try:
        paths = adapter.textures(
            Path(args.root), categories=cats, limit=args.limit, seed=args.seed, warn=warnings.append
        )
    except DatasetError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    for w in warnings:
        _err(f"경고: {w}")
    if not paths:
        _err("텍스처가 0장입니다 — 카테고리를 확인하세요 (dataset info dtd)")
        return EXIT_RECIPE_ERROR
    out = write_texture_list(Path(args.out), paths)
    cats_used = sorted({p.parent.name for p in paths})
    print(
        f"텍스처 목록 → {out.as_posix()}: {len(paths)}장 · 카테고리 {len(cats_used)} ({', '.join(cats_used[:6])}"
        f"{'…' if len(cats_used) > 6 else ''})"
    )
    print(
        "  레시피: pipeline.source: {method: perlin-texture, texture: dir, texture_dir: "
        f"{out.as_posix()}}}  (목록 안 경로는 목록 파일 기준)"
    )
    return EXIT_OK


def cmd_dataset_report(args: argparse.Namespace) -> int:
    """검수 리포트 HTML(의존성 0) — 검수 탭 '리포트' 와 같은 내용."""
    from anograft.io.report import contrast_hint_text
    from anograft.review import ReviewError, ReviewSession

    s = ReviewSession()
    try:
        s.load(args.root, load_bank=not args.no_bank)
        if args.real_csv:
            n = s.load_real_csv(args.real_csv)
            _err(
                f"실측 CSV {args.real_csv}: {n}행 ({', '.join(sorted(s.real_csv_keys()))}) — 그 열은 은행 대신"
            )
        out = s.write_report(args.out)
    except ReviewError as e:
        _err(f"리포트 실패: {e}")
        return EXIT_RECIPE_ERROR
    c = s.counts()
    print(
        f"리포트 → {out.as_posix()}: 합성 {c['ok']} · 채택 {c['accept']} · 반려 {c['reject']} · 미검수 {c['unreviewed']}"
        + (
            f" · 실제({s.real_label()}) {len(s.real_values())}"
            if (s.bank is not None or s.real_csv is not None)
            else " · 은행 없음(실제 분포 없음)"
        )
    )
    for h in (
        s.contrast_hints()
    ):  # 리포트의 대비 힌트와 같은 목록(옅어진 클래스 → relative-paste 로 갈라 보라)
        _err(f"대비 힌트: {contrast_hint_text(h)}")
    for w in s.warnings:
        _err(f"경고: {w}")
    return EXIT_OK


def cmd_dataset_info(args: argparse.Namespace) -> int:
    names = adapter_names()
    if not args.name:
        print("지원 데이터셋: " + ", ".join(names))
        print("자세히: anograft dataset info <name>")
        return EXIT_OK
    try:
        adapter = get_adapter(args.name)
    except DatasetError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    for line in info_lines(adapter.info):
        print(line)
    return EXIT_OK


def cmd_bank_merge(args: argparse.Namespace) -> int:
    from anograft.bank.merge import MergeError, merge_banks, parse_rename

    try:
        s = merge_banks(
            args.banks,
            args.out,
            tags=_split_csv(args.tags),
            rename=parse_rename(args.rename or []),
            log=(lambda m: _err(f"  {m}")) if args.verbose else None,
            dedupe=args.dedupe,
        )
    except (MergeError, OSError, ValueError) as e:
        _err(f"병합 실패: {e}")
        return EXIT_RECIPE_ERROR
    print(
        f"병합 완료 → {s.out.as_posix()}: 은행 {s.banks}개 · 소스 {s.copied}개 복사 · id 충돌 {s.duplicates} · "
        f"클래스 이름 변경 {s.renamed}"
        + (f" · 내용 중복 건너뜀 {s.skipped_same}" if s.skipped_same else "")
    )
    print(f"  classes(id 순): {s.classes}")
    if s.warnings and not args.verbose:
        _err(f"경고 {len(s.warnings)}건 (--verbose 로 전부 보기). 첫 줄: {s.warnings[0]}")
    return EXIT_OK


def bank_ls_json(bank: Bank, path: str) -> dict:
    """``bank ls --json`` — 스크립트용(요약 + 클래스별 행 + 저신뢰 id)."""
    return {
        "name": bank.name,
        "path": Path(path).as_posix(),
        "n_sources": len(bank),
        "um_per_px": bank.um_per_px,
        "no_pitch": bank.no_pitch_count(),
        "imports": len(bank.imports),
        "classes": [
            {
                "id": r.class_id,
                "class": r.cls,
                "n": r.count,
                "area_median": r.area_median,
                "exact": r.exact,
                "estimated": r.estimated,
                "low_confidence": r.low_conf,
                "no_pitch": r.no_pitch,
                "light_r": r.light_r,
                "light_n": r.light_n,
                "directional": r.directional,
                "light_dir": r.light_dir,
                "origins": dict(r.origins),
                "tags": dict(r.tags),
            }
            for r in bank.summary()
        ],
        "low_confidence_ids": [s.id for s in bank.low_confidence()],
        "tags": bank.tag_counts(),
        "warnings": list(bank.warnings),
    }


def cmd_bank_snapshot(args: argparse.Namespace) -> int:
    """은행 내용을 id+해시 목록으로. **복사가 아니다** — 경로가 해시에 들어가므로 복사하면 비교가 깨진다."""
    from anograft.bank import snapshot as snap

    try:
        shot = snap.take(args.bank)
    except BankError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    if args.out:
        path = snap.write(shot, args.out)
        print(f"스냅샷 {path.as_posix()} — 소스 {len(shot.sources)} · 클래스 {len(shot.classes)}")
    else:
        print(json.dumps(shot.to_dict(), ensure_ascii=False, indent=1))
    return EXIT_OK


def cmd_bank_verify(args: argparse.Namespace) -> int:
    """스냅샷 대조 + 평가셋 누수 검사. 둘 중 하나라도 걸리면 종료 코드 1."""
    from anograft.bank import holdout as ho
    from anograft.bank import snapshot as snap

    try:
        bank = Bank.load(args.bank)
    except BankError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR

    bad = False
    held = ho.read_holdout(args.bank)
    if held:
        leaked = ho.violations(bank.sources(), held)
        print(f"평가셋 목록 {len(held)}개 — 은행 안 위반 {len(leaked)}")
        for sid in leaked[:20]:
            _err(
                f"  평가셋이 은행에 있습니다: {sid} (사람이 지울지 결정하세요 — 자동으로 지우지 않습니다)"
            )
        bad = bad or bool(leaked)
    else:
        print("평가셋 목록(holdout.txt)이 없습니다 — 누수 검사를 건너뜁니다.")

    if args.snapshot:
        try:
            shot = snap.load(args.snapshot)
        except (BankError, OSError, ValueError, KeyError) as e:
            _err(f"스냅샷을 읽지 못했습니다: {e}")
            return EXIT_RECIPE_ERROR
        d = snap.diff(shot, args.bank)
        print(f"스냅샷 {shot.created} ({shot.name}) 대비 — {d.text()}")
        for label, ids in (
            ("없어짐", d.missing),
            ("새로 들어옴", d.added),
            ("내용 바뀜", d.changed),
        ):
            for sid in ids[:20]:
                print(f"  {label}: {sid}")
        bad = bad or not d.same
    return EXIT_RECIPE_ERROR if bad else EXIT_OK


def cmd_bank_promote(args: argparse.Namespace) -> int:
    """**미분류에 이름을 준다**(T15) — 조각을 미분류에서 실제 클래스로 옮긴다.

    자동으로 하지 않는 일이다: 클래스 이름은 공학·계약 문제이고(설계 §4 자동화 금지 목록) 새 클래스는
    평가 기준선을 끊는다(§2b.5(3)) — 그래서 옮기고 나면 루프가 한 번 **멈추고** 사람에게 묻는다.
    """
    from anograft.bank.bank import UNSORTED, BankError
    from anograft.bank.importers.common import BankWriter

    try:
        bank = Bank.load(args.bank)
    except BankError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    pool = bank.unsorted() if args.from_cls == UNSORTED else bank.by_class(args.from_cls)
    if not pool:
        _err(f"{args.from_cls} 에 옮길 조각이 없습니다")
        return EXIT_RECIPE_ERROR
    wanted = [s.strip() for s in (args.ids or "").split(",") if s.strip()]
    if wanted:
        names = {s.id.split("/", 1)[1] for s in pool}
        missing = [w for w in wanted if w not in names]
        if missing:
            _err(f"{args.from_cls} 에 없는 id: {', '.join(missing)}")
            return EXIT_RECIPE_ERROR
        chosen = wanted
    elif args.all:
        chosen = [s.id.split("/", 1)[1] for s in pool]
    else:
        _err(
            f"{args.from_cls} 에 {len(pool)}개가 있습니다 — --ids 또는 --all 로 무엇을 옮길지 정하세요"
        )
        return EXIT_RECIPE_ERROR

    tags = tuple(x.strip() for x in (args.tags or "").split(",") if x.strip())
    writer = BankWriter(args.bank, log=None if args.json else print)
    moved: list[str] = []
    for source_id in chosen:
        new_id = writer.move(
            args.from_cls, source_id, args.to, tags=(f"promoted-from:{args.from_cls}", *tags)
        )
        if new_id:
            moved.append(f"{args.to}/{new_id}")
    writer.finish(entry={"importer": "bank-promote", "from": args.from_cls, "to": args.to})

    if args.json:
        print(json.dumps({"moved": moved, "to": args.to}, ensure_ascii=False))
        return EXIT_OK
    print(f"{len(moved)}개를 {args.to} 로 옮겼습니다 (마스크·크롭은 그대로 — 이름만 준 것입니다)")
    if moved:
        print(
            "  루프를 쓰고 있으면 다음 라운드가 **새 클래스**를 보고 멈춥니다 — 평가셋을 손본 뒤 "
            "`anograft loop baseline-reset` 으로 기준선을 재설정하세요"
        )
    return EXIT_OK


def cmd_bank_ls(args: argparse.Namespace) -> int:
    try:
        bank = Bank.load(args.bank)
    except BankError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    if getattr(args, "json", False):
        import json

        print(json.dumps(bank_ls_json(bank, args.bank), ensure_ascii=False, indent=1))
        return EXIT_OK
    no_pitch = bank.no_pitch_count()
    print(
        f"은행 {bank.name} ({Path(args.bank).as_posix()}) — 소스 {len(bank)} · "
        f"um_per_px {bank.um_per_px} (미지정 {no_pitch}/{len(bank)}) · 임포트 {len(bank.imports)}회"
    )
    # 평가셋 누수는 가장 자주 보는 명령에서 눈에 띄어야 한다(설계 §6.3)
    from anograft.bank import holdout as _ho

    _held = _ho.read_holdout(args.bank)
    if _held:
        _leaked = _ho.violations(bank.sources(), _held)
        line = f"  평가셋 목록(holdout.txt) {len(_held)}개"
        if _leaked:
            _err(
                f"{line} — **은행 안에 {len(_leaked)}개가 들어 있습니다**: {', '.join(_leaked[:5])}"
            )
            _err(
                "    → bank verify 로 전체를 보고, 지울지는 사람이 정합니다(자동으로 지우지 않습니다)"
            )
        else:
            print(f"{line} · 위반 없음")
    rows = bank.summary()
    if not rows:
        print("  (클래스 없음)")
    width = max((len(r.cls) for r in rows), default=5)
    print(
        f"  {'id':>3}  {'class'.ljust(width)}  {'n':>5}  {'area_med':>9}  {'exact':>5}  {'est':>5}  "
        f"{'lowconf':>7}  {'no_um':>5}  {'lightR':>7}  origins · tags"
    )
    for r in rows:
        origins = ", ".join(f"{k}:{v}" for k, v in sorted(r.origins.items()))
        tags = ", ".join(f"{k}:{v}" for k, v in sorted(r.tags.items()))
        light = "–" if r.light_r is None else f"{r.light_r:.2f}" + ("*" if r.directional else " ")
        print(
            f"  {r.class_id:>3}  {r.cls.ljust(width)}  {r.count:>5}  {r.area_median:>9.0f}  "
            f"{r.exact:>5}  {r.estimated:>5}  {r.low_conf:>7}  {r.no_pitch:>5}  {light:>7}  {origins}"
            + (f" · {tags}" if tags else "")
        )
    directional = [r for r in rows if r.directional]
    if directional:
        _err(
            f"참고: 조명 의존 클래스(lightR* = R ≥ {LIGHT_REAL_MIN} 이고 n·R² ≥ 2.9 유의: "
            f"{', '.join(f'{r.cls} {r.light_r:.2f}' for r in directional)}) — 회전 ±180/flip 프리셋은 하이라이트를 "
            f"뒤집습니다. 프리셋 dent-graft 권장(run 이 경고합니다)"
        )
    low = bank.low_confidence()
    if low:
        est = sum(r.estimated for r in rows)
        _err(
            f"참고: 저신뢰(confidence < {LOW_CONFIDENCE}) 추정 마스크 {len(low)}/{est}개 — "
            f"bank preview 로 확인하고 라벨 탭 YOLO 초안으로 다듬거나 --mask-from otsu 로 다시 임포트. "
            f"예: {', '.join(s.id + ' ' + ('/'.join(s.flags) or '-') for s in low[:3])}"
        )
    unsorted = bank.unsorted()
    if unsorted:
        print(
            f"  미분류 {len(unsorted)}개 — 처음 보는 형상이라 이름이 없습니다. 합성·출력에서 빠집니다. "
            f"`anograft bank promote {Path(args.bank).as_posix()} --to <클래스>` 로 이름을 주세요"
        )
    if no_pitch:
        _err(
            f"참고: um_per_px 없는 소스 {no_pitch}개 — 대상 inputs.um_per_px 를 줘도 이 소스들은 축척 정합 없이(factor 1.0) "
            f"이식됩니다. 임포트 시 --um-per-px 또는 bank.yaml 의 um_per_px 로 지정"
        )
    for w in bank.warnings:
        _err(f"경고: {w}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# 파서
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anograft",
        description="Graft — 라벨링한 결함을 정상 이미지에 이식해 학습용 이상 데이터셋을 만든다",
    )
    parser.add_argument("--version", action="version", version=f"anograft {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("methods", help="스테이지별 선택 가능한 알고리즘(method)과 가용 여부")
    p.add_argument("--stage", choices=registry.STAGE_ORDER, default=None)
    p.set_defaults(func=cmd_methods)

    p = sub.add_parser(
        "explain",
        help="파라미터·method·프리셋 도움말 — 예: explain geometry.scale · explain blend:poisson · explain preset:dent-graft",
    )
    p.add_argument(
        "query",
        nargs="*",
        help="<stage> · <stage>.<field> · placement.roi.<field> · inputs[.<field>] · output[.<field>] · <stage>:<method> · preset:<이름>",
    )
    p.add_argument(
        "--markdown", action="store_true", help="전체 표를 Markdown 으로(PARAMS.md 재생성)"
    )
    p.add_argument(
        "--out", default=None, help="Markdown 을 이 파일에 LF 로 저장(예: --out PARAMS.md)"
    )
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("recipe", help="레시피 만들기·검증")
    rsub = p.add_subparsers(dest="recipe_command", metavar="<action>")

    pi = rsub.add_parser("init", help="프리셋을 펼친 레시피 YAML을 stdout(또는 --write 파일)에")
    pi.add_argument("--preset", default="poisson-graft")
    pi.add_argument("--name", default=None)
    pi.add_argument("--bank", default="./bank/mine")
    pi.add_argument("--targets", default="./normals")
    pi.add_argument("--out", default="./out/run-01")
    pi.add_argument("--seed", type=int, default=20260913)
    pi.add_argument("--count", type=int, default=100)
    pi.add_argument(
        "--roi",
        default=None,
        choices=list(R.ROI_METHODS),
        help="프리셋의 ROI method 만 교체 (예: --preset dent-graft --roi annulus — 원형 부품의 찍힘)",
    )
    pi.add_argument(
        "--um-per-px", type=float, default=None, help="대상 µm/px (inputs.um_per_px — 축척 정합)"
    )
    pi.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="source.classes — 이 프리셋을 은행의 일부 클래스에만(결함 성격별 프리셋으로 따로 돌려 dataset merge --dedupe-normals)",
    )
    pi.add_argument(
        "--dent-class",
        action="append",
        metavar="CLASS",
        help="조명 의존 클래스 — geometry.per_class 에 ±15°·flip 끔(반복 가능). 스크래치·찍힘이 섞인 은행용",
    )
    pi.add_argument(
        "--auto-dent",
        action="store_true",
        help="--bank 은행의 lightR ≥ 0.5 클래스를 --dent-class 로 자동 추가(은행이 있어야 함)",
    )
    pi.add_argument("--write", default=None, help="파일로 쓰기 (기본은 stdout)")
    pi.set_defaults(func=cmd_recipe_init)

    pc = rsub.add_parser("check", help="레시피 검증 + 스테이지 실행 가능 여부")
    pc.add_argument("recipe")
    pc.add_argument(
        "--resolved", action="store_true", help="프리셋·기본값이 채워진 resolved YAML 출력"
    )
    pc.set_defaults(func=cmd_recipe_check)

    p = sub.add_parser("run", help="레시피로 데이터셋 생성 (정본 images/masks/meta + manifest)")
    p.add_argument("recipe")
    p.add_argument("--out", default=None, help="output.root 오버라이드")
    p.add_argument("--count", type=int, default=None, help="output.count 오버라이드")
    p.add_argument("--seed", type=int, default=None, help="seed 오버라이드")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="프로세스 수. 0 = 인프로세스. 결과는 N과 무관하게 동일",
    )
    p.add_argument("--dry-run", action="store_true", help="배분·경고만 계산, 파일 안 씀")
    p.add_argument(
        "--roi-cache",
        type=int,
        default=None,
        help="대상당 ROI 캐시 항목 수(LRU, 기본 16 · 0 = 끔). 결과와 무관 — 4K 대상 수백 장에서 메모리를 아낄 때",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="끝나면 출력 폴더에 검수 리포트 HTML(= dataset report <out>) — 합성 vs 실제 분포·조명 일관성",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("preview", help="인덱스 하나를 합성해 원본|합성|GT 3패널 PNG로")
    p.add_argument("recipe")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="preview.png")
    p.add_argument("--long-side", type=int, default=1024, help="패널 긴 변 (기본 1024)")
    p.add_argument(
        "--compare-methods",
        metavar="STAGE",
        choices=registry.STAGE_ORDER,
        default=None,
        help="같은 시드로 이 스테이지의 method 전부를 한 장에 (예: blend, harmonize)",
    )
    p.add_argument(
        "--full", action="store_true", help="--compare-methods 에서 결함 주변 크롭 대신 이미지 전체"
    )
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("bank", help="결함 은행 만들기·보기")
    bsub = p.add_subparsers(dest="bank_command", metavar="<action>")

    bi = bsub.add_parser(
        "import-yolo", help="보유 YOLO 라벨(박스·폴리곤) → 은행. 박스는 마스크 추정"
    )
    bi.add_argument("--images", required=True)
    bi.add_argument("--labels", required=True)
    bi.add_argument("--names", required=True, help="data.yaml | classes.txt | a,b,c")
    bi.add_argument("--out", required=True, help="은행 폴더 (있으면 누적)")
    bi.add_argument("--mask-from", choices=MASK_METHODS, default="grabcut")
    bi.add_argument("--box-margin", type=int, default=6, help="추정 시 박스 바깥 배경 표본 폭(px)")
    bi.add_argument(
        "--min-box", type=int, default=8, help="짧은 변이 이보다 작은 박스는 바로 ellipse"
    )
    bi.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="크롭 여유(px), 최소 6")
    bi.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    bi.add_argument("--um-per-px", type=float, default=None)
    bi.add_argument("--tags", default=None, help="a,b")
    bi.add_argument(
        "--list-normals", default=None, help="라벨이 빈 이미지 경로 목록 파일 → inputs.targets"
    )
    bi.add_argument("--verbose", action="store_true", help="이미지별 경고 전부 출력")
    bi.set_defaults(func=cmd_bank_import_yolo)

    bp = bsub.add_parser("import-pairs", help="이미지 + 마스크 PNG 쌍 → 은행 (성분마다 소스 하나)")
    bp.add_argument("--images", default=None, help="이미지 폴더 (재귀)")
    bp.add_argument("--masks", default=None, help="마스크 폴더 (이미지와 같은 상대경로)")
    bp.add_argument("--out", required=True, help="은행 폴더 (있으면 누적)")
    grp = bp.add_mutually_exclusive_group()
    grp.add_argument("--class", dest="cls", default=None, help="클래스 하나 고정")
    grp.add_argument(
        "--class-from-dir", action="store_true", help="images/<class>/*.png 폴더명을 클래스로"
    )
    bp.add_argument("--csv", default=None, help="image,mask,class 열 CSV — 폴더 탐색 대신 이 목록")
    bp.add_argument(
        "--mask-suffix", default=pairs_importer.DEFAULT_MASK_SUFFIX, help="예: _mask → x_mask.png"
    )
    bp.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="크롭 여유(px), 최소 6")
    bp.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    bp.add_argument("--keep-whole", action="store_true", help="성분 분리 없이 마스크 통째로 하나")
    bp.add_argument("--um-per-px", type=float, default=None)
    bp.add_argument("--tags", default=None, help="a,b")
    bp.add_argument("--verbose", action="store_true")
    bp.set_defaults(func=cmd_bank_import_pairs)

    bd = bsub.add_parser(
        "import-dataset",
        help="표준 데이터셋(로컬 사본) → 은행. 어댑터는 (image, mask, class) 변환기",
    )
    bd.add_argument("name", help="dataset info 로 목록 (예: mvtec-ad)")
    bd.add_argument("root", help="카테고리 폴더 (예: mvtec_ad/metal_nut)")
    bd.add_argument("--out", required=True)
    bd.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    bd.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA)
    bd.add_argument("--keep-whole", action="store_true")
    bd.add_argument("--tags", default=None, help="a,b (기본 태그 <dataset>,<category>에 추가)")
    bd.add_argument("--verbose", action="store_true")
    bd.set_defaults(func=cmd_bank_import_dataset)

    bm = bsub.add_parser(
        "merge", help="여러 은행을 하나로(세 파일 복사, 클래스 이름 병합, id 충돌 -dup, 태그 추가)"
    )
    bm.add_argument("banks", nargs="+", help="소스 은행 폴더들")
    bm.add_argument("--out", required=True, help="대상 은행(있으면 이어 씀, 소스와 달라야 함)")
    bm.add_argument("--tags", default=None, help="모든 소스에 더할 태그 a,b")
    bm.add_argument(
        "--dedupe",
        action="store_true",
        help="같은 클래스에서 이미지·마스크 내용이 같은 소스는 한 번만(같은 원본을 두 은행에 임포트한 경우)",
    )
    bm.add_argument(
        "--rename",
        action="append",
        default=None,
        metavar="OLD=NEW",
        help="클래스 이름 바꾸기(여러 번)",
    )
    bm.add_argument("--verbose", action="store_true")
    bm.set_defaults(func=cmd_bank_merge)
    bp = bsub.add_parser(
        "promote",
        help="미분류 조각에 이름 주기 — 다른 클래스로 옮긴다",
        description="처음 보는 형상이라 미분류(__unsorted__)로 들어간 조각에 이름을 줍니다. 마스크·크롭은 "
        "그대로 두고 클래스만 바꿉니다(이름을 준 것이지 다듬은 것이 아닙니다). 새 클래스를 만들면 "
        "다음 라운드가 멈추고 기준선 재설정을 묻습니다 — 평가셋에 그 클래스 정답이 없으면 점수가 끊깁니다.",
    )
    bp.add_argument("bank")
    bp.add_argument("--to", required=True, help="옮길 클래스 이름(없으면 만든다)")
    bp.add_argument(
        "--from",
        dest="from_cls",
        default=UNSORTED,
        help=f"옮겨 올 클래스 (기본 {UNSORTED} = 미분류)",
    )
    bp.add_argument("--ids", help="옮길 조각 id 들(쉼표 구분) — 생략하면 --all 이 필요")
    bp.add_argument("--all", action="store_true", help="그 클래스의 조각 전부")
    bp.add_argument("--tags", help="추가 태그(쉼표 구분)")
    bp.add_argument("--json", action="store_true")
    bp.set_defaults(func=cmd_bank_promote)

    bl = bsub.add_parser("ls", help="클래스 | 소스 수 | 면적 중앙값 | 마스크 출처(정확/추정)")
    bl.add_argument("--json", action="store_true", help="JSON 으로(스크립트용)")
    bl.add_argument("bank")
    bl.set_defaults(func=cmd_bank_ls)

    bv = bsub.add_parser(
        "preview", help="소스 크롭 + 마스크 윤곽 그리드 — 추정 마스크를 눈으로 확인"
    )
    bv.add_argument("bank")
    bv.add_argument("--class", dest="cls", default=None)
    bv.add_argument("--out", default="bank-preview.png")
    bv.add_argument("--cols", type=int, default=8)
    bv.add_argument("--tile", type=int, default=128, help="타일 한 변(px)")
    bv.add_argument("--limit", type=int, default=0, help="최대 소스 수 (0 = 전부)")
    bv.set_defaults(func=cmd_bank_preview)

    bs = bsub.add_parser(
        "snapshot",
        help="지금 은행 내용을 id+해시 목록으로 적는다 (라운드 재현 — 은행을 복사하지 않는다)",
    )
    bs.add_argument("bank")
    bs.add_argument("--out", help="쓸 파일 (기본: stdout)")
    bs.set_defaults(func=cmd_bank_snapshot)

    bvf = bsub.add_parser(
        "verify", help="스냅샷과 견준다 — 없어짐·새로 들어옴·내용 바뀜 + 평가셋 누수 검사"
    )
    bvf.add_argument("bank")
    bvf.add_argument("--snapshot", help="비교할 스냅샷 파일 (없으면 평가셋 검사만)")
    bvf.set_defaults(func=cmd_bank_verify)

    p = sub.add_parser("dataset", help="표준 데이터셋 안내 (내려받지 않는다)")
    dsub = p.add_subparsers(dest="dataset_command", metavar="<action>")
    di = dsub.add_parser("info", help="이름·라이선스·URL·기대 폴더 구조·카테고리")
    di.add_argument("name", nargs="?", default=None)
    di.set_defaults(func=cmd_dataset_info)
    dt = dsub.add_parser(
        "textures", help="텍스처셋(dtd) → perlin-texture 용 목록 파일 (은행에 넣지 않는다)"
    )
    dt.add_argument("name", help="dtd")
    dt.add_argument("root", help="dtd/ 폴더 (images/<category>/…)")
    dt.add_argument("--out", required=True, help="목록 파일 (.txt) — 레시피 texture_dir 에 지정")
    dt.add_argument(
        "--categories", default=None, help="a,b,c (기본: 결함처럼 보이는 카테고리 · '*' = 전부)"
    )
    dt.add_argument("--limit", type=int, default=None, help="최대 장수 (seed 로 결정적 추출)")
    dt.add_argument("--seed", type=int, default=0)
    dt.set_defaults(func=cmd_dataset_textures)
    dr = dsub.add_parser(
        "report", help="검수 리포트 HTML 한 장(채택/반려·클래스별·합성 vs 실제 분포·반려 목록)"
    )
    dr.add_argument("root", help="anograft run 출력 폴더 (manifest.csv, review.csv)")
    dr.add_argument("--out", default=None, help="HTML 경로 (기본 <root>/review-report.html)")
    dr.add_argument("--no-bank", action="store_true", help="은행을 열지 않는다(실제 분포 생략)")
    dr.add_argument(
        "--real-csv",
        default=None,
        help="실제 분포를 은행 대신 실측 CSV 로(열: class? + area/length/contrast/texture/sharpness/lighting 중 있는 것)",
    )
    dr.set_defaults(func=cmd_dataset_report)
    dp = dsub.add_parser(
        "prune",
        help="검수 review.csv 에서 반려된 합성 이미지를 뺀 정리본 사본 (v0.7 검수 탭과 같은 동작)",
    )
    dp.add_argument("root", help="anograft run 출력 폴더 (manifest.csv)")
    dp.add_argument("--out", required=True, help="정리본 폴더 (원본과 달라야 함)")
    dp.add_argument("--review", default=None, help="review.csv 경로 (기본 <root>/review.csv)")
    dp.add_argument(
        "--drop-flipped",
        action="store_true",
        help="조명 뒤집힘 의심(실제 클래스 방향에서 > 90°)도 제외 — 검수 탭 필터와 같은 집합(은행 필요)",
    )
    dp.add_argument(
        "--drop-unreviewed", action="store_true", help="미검수도 제외하고 채택(accept)만 남긴다"
    )
    dp.set_defaults(func=cmd_dataset_prune)
    dm = dsub.add_parser(
        "merge",
        help="run 출력 폴더 여러 개를 한 학습셋으로(파일 접두어 d<k>_ · index 재부여 · coco 이어 붙임). 같은 writer·클래스여야 함",
    )
    dm.add_argument("roots", nargs="+", help="anograft run 출력 폴더 둘 이상 (정리본 권장)")
    dm.add_argument("--out", required=True, help="병합 폴더 (입력과 달라야 함)")
    dm.add_argument(
        "--prefix", default="d{k}_", help="파일 이름 접두어, {k} = 루트 순번 (기본 d{k}_)"
    )
    dm.add_argument(
        "--dedupe-normals",
        action="store_true",
        help="같은 대상(manifest target)의 정상 이미지는 첫 루트 것만 — 같은 정상 폴더로 돌린 출력 여러 개를 합칠 때",
    )
    dm.set_defaults(func=cmd_dataset_merge)

    p = sub.add_parser(
        "doctor", help="환경 진단(버전·Qt·스레드·프리셋·불가 method) — 문제 보고 첫 줄에"
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="웹 UI를 로컬에서 연다 (127.0.0.1 고정 — 오프라인·로컬)")
    p.add_argument("--port", type=int, default=web_server.DEFAULT_PORT, help="기본 8000")
    p.add_argument(
        "--api-only",
        action="store_true",
        help="JSON API만 — 화면은 web/ 에서 npm run dev (개발용)",
    )
    p.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    p.add_argument(
        "--dev-port", type=int, default=web_server.DEFAULT_DEV_PORT, help="안내에 쓸 개발 서버 포트"
    )
    p.add_argument("--verbose", action="store_true", help="요청 로그를 찍는다")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser(
        "trainer",
        help="학습기 어댑터 보기 (등록은 레시피 밖 trainers.yaml — 학습 루프 v1.x)",
    )
    tsub = p.add_subparsers(dest="trainer_cmd", required=True)
    tl = tsub.add_parser("list", help="등록된 학습기와 가용 여부·사유")
    tl.add_argument(
        "--file", help="trainers.yaml 경로 (기본: ./trainers.yaml → ~/.anograft/trainers.yaml)"
    )
    tl.set_defaults(func=cmd_trainer_list)
    ti = tsub.add_parser("info", help="학습기 하나의 선언(데이터 형식·학습 입력·결정성)")
    ti.add_argument("name")
    ti.add_argument("--file", help="trainers.yaml 경로")
    ti.set_defaults(func=cmd_trainer_info)

    tf = tsub.add_parser(
        "fit", help="등록된 학습기로 학습 — 데이터셋(writer 출력) → 모델 참조·지표"
    )
    tf.add_argument("name")
    tf.add_argument("--dataset", required=True, help="학습 데이터셋 루트(어댑터가 선언한 형식)")
    tf.add_argument("--out", required=True, help="모델이 저장될 폴더")
    tf.add_argument("--seed", type=int, default=0)
    tf.add_argument("--spec", help="추가 spec JSON 파일(불투명 — 등록부 spec 위에 얹는다)")
    tf.add_argument("--file", help="trainers.yaml 경로")
    tf.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로(도구가 읽는다)")
    tf.set_defaults(func=cmd_trainer_fit)

    tp = tsub.add_parser(
        "predict", help="등록된 학습기로 예측 — masks/·scores/ (bank import-* 입력)"
    )
    tp.add_argument("name")
    tp.add_argument("--model", required=True, help="fit 이 돌려준 모델 참조(불투명)")
    tp.add_argument("--images", required=True, help="이미지 경로 목록 .txt")
    tp.add_argument("--out", required=True, help="예측이 저장될 폴더")
    tp.add_argument("--spec", help="추가 spec JSON 파일(학습 때와 같은 값을 주는 게 기본)")
    tp.add_argument("--file", help="trainers.yaml 경로")
    tp.add_argument("--json", action="store_true")
    tp.set_defaults(func=cmd_trainer_predict)

    p = sub.add_parser(
        "loop",
        help="학습 루프 — 예측에서 검토 대기 목록을 만들고, 판정된 것을 결함 보관함으로 (v1.x)",
    )
    lsub = p.add_subparsers(dest="loop_cmd", required=True)
    lq = lsub.add_parser(
        "queue",
        help="예측 → 검토 대기 폴더(검수 화면이 그대로 연다)",
        description="불일치 → 경계 → 확신 → 무작위 순으로 담습니다. 예측 폴더를 둘 주면 두 모델이 "
        "어긋난 것이 맨 앞에 옵니다(오류가 독립이라 정보량이 가장 많습니다).",
    )
    lq.add_argument(
        "--pred", required=True, help="예측 폴더(trainer predict 출력 — scores/·masks/)"
    )
    lq.add_argument("--pred-b", help="교차 검증할 다른 모델의 예측 폴더(불일치 우선)")
    lq.add_argument("--images", required=True, help="예측에 쓴 이미지 폴더 또는 목록 .txt")
    lq.add_argument("--out", required=True, help="검토 대기 폴더(비어 있어야 한다)")
    lq.add_argument("-n", type=int, default=DEFAULT_N, help=f"사람이 볼 장수 (기본 {DEFAULT_N})")
    lq.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="운영 임계값 — 이 근처가 '경계'",
    )
    lq.add_argument("--mix", help="경계,확신,무작위 비율 (기본 0.6,0.2,0.2)")
    lq.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"두 예측을 같은 검출로 볼 IoU (기본 {DEFAULT_IOU})",
    )
    lq.add_argument("--seed", type=int, default=0, help="무작위 몫의 시드")
    lq.add_argument(
        "--bank",
        help="결함 보관함 — 주면 **처음 보는 형상**(보관함 어느 조각과도 닮지 않은 것)을 맨 앞에 둔다",
    )
    lq.add_argument(
        "--novelty",
        type=float,
        metavar="T",
        help=f"처음 보는 형상 임계 0~1 (기본 {NOVELTY_THRESHOLD} · 0 = 끔) — --bank 와 함께 씁니다",
    )
    lq.add_argument("--trainer", help="예측을 만든 학습기 이름(사이드카·mask_origin 에 남는다)")
    lq.add_argument("--round", type=int, help="라운드 번호(태그 round-n 으로 붙는다)")
    lq.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로")
    lq.set_defaults(func=cmd_loop_queue)

    la = lsub.add_parser(
        "accept",
        help="판정된 검토 대기 폴더의 채택분 → 결함 보관함",
        description="review.csv 가 채택(accept)한 것만 넣습니다. 마스크는 모델 초안이라 "
        "mask_origin 이 pred:<학습기> 이고, 평가셋(holdout.txt)은 보관함이 거부합니다.",
    )
    la.add_argument("queue", help="검토 대기 폴더")
    la.add_argument("--bank", required=True, help="결함 보관함 폴더(없으면 만든다)")
    la.add_argument("--class", dest="cls", help="클래스 이름 고정(기본: 예측 클래스)")
    la.add_argument("--tags", help="추가 태그 (쉼표 구분)")
    la.add_argument("--round", type=int, help="라운드 번호(태그 round-n · 큐 사이드카보다 우선)")
    la.add_argument("--margin", type=int, help=f"크롭 여유 px (기본 {DEFAULT_MARGIN})")
    la.add_argument("--min-area", type=int, help=f"최소 면적 px (기본 {DEFAULT_MIN_AREA})")
    la.add_argument(
        "--keep-whole",
        action="store_true",
        help="마스크를 성분으로 쪼개지 않고 통째로 하나의 조각으로",
    )
    la.add_argument(
        "--novelty",
        type=float,
        metavar="T",
        help=f"처음 보는 형상 임계 0~1 (기본 {NOVELTY_THRESHOLD} · 0 = 끔) — 이 이상이면 **미분류**로 넣습니다",
    )
    la.add_argument("--um-per-px", type=float, help="픽셀 피치(µm/px)")
    la.add_argument("--json", action="store_true")
    la.add_argument("--verbose", action="store_true")
    la.set_defaults(func=cmd_loop_accept)

    lr = lsub.add_parser(
        "run",
        help="라운드를 다음 단계부터 진행 (predict → 큐 → [사람] → accept → 합성 → 학습 → 승급)",
        description="멱등합니다 — 남은 상태를 읽고 이어서 돕니다. 사람이 판정할 차례면 멈추고 "
        "무엇을 해야 하는지 알려 줍니다. 자동 정지(loop.yaml 의 breaker)가 걸려 있으면 새 라운드를 "
        "열지 않습니다 — 진행 중인 라운드는 그대로 이어 갑니다. 설정은 loop.yaml(예시 loop.example.yaml).",
    )
    lr.add_argument("--config", help="loop.yaml 경로 (기본: ./loop.yaml → ~/.anograft/loop.yaml)")
    lr.add_argument(
        "--accept-partial",
        action="store_true",
        help="검토 대기가 다 판정되지 않아도 판정된 것만 가지고 진행",
    )
    lr.add_argument("--workers", type=int, default=0, help="합성 워커 수 (기본 0 = 단일 프로세스)")
    lr.add_argument(
        "--since",
        type=int,
        metavar="N",
        help="라운드 N 이후의 유입 이력을 무시하고 그 이미지들을 다시 스코어링 (어댑터 교체·버그 수정)",
    )
    lr.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로(도구가 읽는다)")
    lr.set_defaults(func=cmd_loop_run)

    lt = lsub.add_parser(
        "tick",
        help="스케줄러가 부르는 진입점 — 새로 들어온 이미지만 보고 한 라운드를 멱등하게 진행",
        description="Windows 작업 스케줄러·cron 이 인자 없이 부르는 명령입니다. Graft 는 상주하지 "
        "않습니다 — 한 번 돌고 끝납니다. 이미 처리한 이미지는 유입 커서(processed.jsonl)로 걸러지고, "
        "다른 실행이 돌고 있으면 오류가 아니라 '다음에'(종료 코드 0)입니다. 언제 돌지는 loop.yaml 의 "
        "trigger(신규 조각·신규 이미지·최소/최대 간격)가 정하고, 안 돈 사유는 tick.json 과 rounds.jsonl 에 "
        "남습니다. 자동 정지(breaker)가 걸리면 --force 로도 돌지 않습니다 — loop breaker-reset 으로 "
        "무엇을 바꿨는지 적어야 다시 돕니다.",
    )
    lt.add_argument("--config", help="loop.yaml 경로 (기본: ./loop.yaml → ~/.anograft/loop.yaml)")
    lt.add_argument(
        "--accept-partial",
        action="store_true",
        help="검토 대기가 다 판정되지 않아도 판정된 것만 가지고 진행",
    )
    lt.add_argument("--workers", type=int, default=0, help="합성 워커 수 (기본 0 = 단일 프로세스)")
    lt.add_argument(
        "--since",
        type=int,
        metavar="N",
        help="라운드 N 이후의 유입 이력을 무시하고 그 이미지들을 다시 스코어링",
    )
    lt.add_argument(
        "--force",
        action="store_true",
        help="트리거 기준(loop.yaml 의 trigger)을 무시하고 그대로 돈다 — 잠금은 그대로 지킨다",
    )
    lt.add_argument("--json", action="store_true", help="결과를 JSON 한 줄로(스케줄러·상위 도구용)")
    lt.set_defaults(func=cmd_loop_tick)

    lb = lsub.add_parser(
        "baseline-reset",
        help="기준선 재설정 — 이 지점 앞뒤로 점수를 견주지 않는다(클래스 신설 뒤)",
        description="클래스를 새로 만들면 평가셋에 그 클래스 정답이 없어 라운드 점수 비교가 끊깁니다. "
        "평가셋을 다시 만든 뒤 이 명령으로 지점을 남기면, 다음 라운드는 점수를 견주지 않고 현재 모델을 "
        "새로 세웁니다. 원장(rounds.jsonl)에 남으므로 나중에 '왜 여기서 점수가 튀었나'에 답할 수 있습니다.",
    )
    lb.add_argument("--config", help="loop.yaml 경로")
    lb.add_argument("--note", default="", help="왜 재설정하는지 한 줄(원장에 남습니다)")
    lb.add_argument("--json", action="store_true")
    lb.set_defaults(func=cmd_loop_baseline_reset)

    lk = lsub.add_parser(
        "breaker-reset",
        help="자동 정지 해제 — 무엇을 바꿨는지 적고 다시 돌린다",
        description="자동 루프는 망가져도 계속 돕니다. 그래서 몇 라운드째 나아지지 않거나, 모델 초안을 "
        "아무도 다듬지 않거나(사람 수정률 0 수렴), 보관함 클래스 분포가 급히 바뀌면 라운드를 열지 "
        "않습니다. 멈춘 것은 실패가 아니라 데이터 말고 다른 것을 바꿀 때라는 신호입니다(조명·해상도·"
        "모델·평가셋). 무엇을 바꿨는지 --note 로 적으면 그 지점이 원장에 남고, 그 앞 라운드는 다음 "
        "판정에서 빠집니다. 기준은 loop.yaml 의 breaker(기본값은 전부 0 = 제한 없음).",
    )
    lk.add_argument("--config", help="loop.yaml 경로")
    lk.add_argument("--note", default="", help="무엇을 바꿨는지 한 줄(원장에 남습니다)")
    lk.add_argument("--json", action="store_true")
    lk.set_defaults(func=cmd_loop_breaker_reset)

    ls = lsub.add_parser("status", help="지금 어디인가 — champion · 진행 중인 라운드 · 최근 이력")
    ls.add_argument("--config", help="loop.yaml 경로")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=cmd_loop_status)

    p = sub.add_parser(
        "sample",
        help="샘플 YOLO 세트(브러시드 메탈 + scratch/pit/stain) 생성 — 보유 데이터 없이 5분 시작",
    )
    sample_yolo.add_arguments(p)
    p.set_defaults(func=sample_yolo.run_from_args)

    return parser


def _console_fail_soft() -> None:
    """cp949 콘솔에서 `→`·`≈` 같은 기호가 UnicodeEncodeError로 CLI를 죽이지 않게 — 인코딩은 그대로, 못 찍는 글자만 `?`."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # 닫힌 스트림·특수 핸들이면 조용히 넘어간다
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    # PyInstaller(frozen) + spawn: 워커 프로세스는 이 exe를 다시 실행한다 — 여기서 잡아 주지 않으면 워커가 CLI 본체를 돈다.
    # 비-frozen 환경에서는 no-op.
    multiprocessing.freeze_support()
    _console_fail_soft()
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return EXIT_RECIPE_ERROR
    return int(func(args))
