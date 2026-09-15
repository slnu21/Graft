"""anograft CLI (argparse — 의존성 추가 없음). 설계 §9.

메시지는 한국어 1줄(CLI는 내부 도구; GUI 문자열만 ko/en). 종료 코드: 0 정상 · 1 레시피/은행 오류 · 2 전부 skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from anograft import __version__, runner
from anograft.bank import Bank
from anograft.bank.bank import BankError
from anograft.bank.importers import dataset as dataset_importer
from anograft.bank.importers import pairs as pairs_importer
from anograft.bank.importers import yolo as yolo_importer
from anograft.bank.importers.common import DEFAULT_MARGIN, DEFAULT_MIN_AREA
from anograft.bank.mask_from_box import LOW_CONFIDENCE
from anograft.bank.mask_from_box import METHODS as MASK_METHODS
from anograft.core import recipe as R
from anograft.core import registry
from anograft.datasets import DatasetError, adapter_names, get_adapter, info_lines
from anograft.io import imgio
from anograft.io.targets import load_target
from anograft.preview import (
    crop,
    render_compare,
    render_grid,
    render_preview,
    source_tile,
    union_bbox,
)
from anograft.samples import yolo as sample_yolo

EXIT_OK = 0
EXIT_RECIPE_ERROR = 1
EXIT_ALL_SKIPPED = 2


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_recipe(args: argparse.Namespace, **overrides: object) -> R.Recipe | None:
    """레시피 로드 + CLI 오버라이드. 실패하면 stderr에 사유를 쓰고 None."""
    try:
        return R.Recipe.load(args.recipe, **overrides)
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
        print(f"  {i.method.ljust(width)}  {state}{req}")
    print()
    print(f"프리셋: {', '.join(R.preset_names())}")
    return EXIT_OK


def cmd_recipe_init(args: argparse.Namespace) -> int:
    try:
        data = R.init_recipe_dict(
            args.preset,
            name=args.name,
            bank=args.bank,
            targets=args.targets,
            out=args.out,
            seed=args.seed,
            count=args.count,
        )
    except KeyError as e:
        print(str(e.args[0]), file=sys.stderr)
        return EXIT_RECIPE_ERROR
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None)
    header = (
        f"# anograft {__version__} — recipe init --preset {args.preset}\n"
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
        rec = R.Recipe.load(args.recipe)
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
            for w in rec.validate_against(bank):
                _err(f"경고: {w}")
            print(f"은행 대조 OK: {bank.name} — 클래스 {bank.classes}, 소스 {len(bank)}")
        except (BankError, ValueError) as e:
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
        prep = runner.prepare(rec)
    except runner.PrepareError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    if args.dry_run:
        for k, v in runner.dry_run_table(prep):
            print(f"{k:>14}: {v}")
        for w in prep.warnings:
            _err(f"경고: {w}")
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
    return EXIT_ALL_SKIPPED if summary.all_skipped else EXIT_OK


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
        source_tile(s.image, s.mask, s.id, s.mask_origin, tile=args.tile, confidence=s.confidence)
        for s in sources
    ]
    imgio.write_image(Path(args.out), render_grid(tiles, cols=args.cols))
    est = sum(1 for s in sources if s.mask_origin.startswith("yolo-box:"))
    low = sum(1 for s in sources if s.confidence is not None and s.confidence < LOW_CONFIDENCE)
    print(
        f"은행 미리보기: {Path(args.out).as_posix()} — 소스 {len(tiles)}개 (추정 마스크 {est}, 저신뢰 {low} = 빨간 테두리) · "
        f"{args.cols}열 · 타일 {args.tile}px"
    )
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


def cmd_bank_ls(args: argparse.Namespace) -> int:
    try:
        bank = Bank.load(args.bank)
    except BankError as e:
        _err(str(e))
        return EXIT_RECIPE_ERROR
    no_pitch = bank.no_pitch_count()
    print(
        f"은행 {bank.name} ({Path(args.bank).as_posix()}) — 소스 {len(bank)} · "
        f"um_per_px {bank.um_per_px} (미지정 {no_pitch}/{len(bank)}) · 임포트 {len(bank.imports)}회"
    )
    rows = bank.summary()
    if not rows:
        print("  (클래스 없음)")
    width = max((len(r.cls) for r in rows), default=5)
    print(
        f"  {'id':>3}  {'class'.ljust(width)}  {'n':>5}  {'area_med':>9}  {'exact':>5}  {'est':>5}  "
        f"{'lowconf':>7}  {'no_um':>5}  origins · tags"
    )
    for r in rows:
        origins = ", ".join(f"{k}:{v}" for k, v in sorted(r.origins.items()))
        tags = ", ".join(f"{k}:{v}" for k, v in sorted(r.tags.items()))
        print(
            f"  {r.class_id:>3}  {r.cls.ljust(width)}  {r.count:>5}  {r.area_median:>9.0f}  "
            f"{r.exact:>5}  {r.estimated:>5}  {r.low_conf:>7}  {r.no_pitch:>5}  {origins}"
            + (f" · {tags}" if tags else "")
        )
    low = bank.low_confidence()
    if low:
        est = sum(r.estimated for r in rows)
        _err(
            f"참고: 저신뢰(confidence < {LOW_CONFIDENCE}) 추정 마스크 {len(low)}/{est}개 — "
            f"bank preview 로 확인하고 라벨 탭 YOLO 초안으로 다듬거나 --mask-from otsu 로 다시 임포트. "
            f"예: {', '.join(s.id + ' ' + ('/'.join(s.flags) or '-') for s in low[:3])}"
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

    bl = bsub.add_parser("ls", help="클래스 | 소스 수 | 면적 중앙값 | 마스크 출처(정확/추정)")
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

    p = sub.add_parser("dataset", help="표준 데이터셋 안내 (내려받지 않는다)")
    dsub = p.add_subparsers(dest="dataset_command", metavar="<action>")
    di = dsub.add_parser("info", help="이름·라이선스·URL·기대 폴더 구조·카테고리")
    di.add_argument("name", nargs="?", default=None)
    di.set_defaults(func=cmd_dataset_info)

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
