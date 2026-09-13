"""anograft CLI (argparse — 의존성 추가 없음). 설계 §9.

메시지는 한국어 1줄(CLI는 내부 도구; GUI 문자열만 ko/en). 종료 코드: 0 정상 · 1 레시피/은행 오류 · 2 전부 skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from anograft import __version__
from anograft.core import recipe as R
from anograft.core import registry

EXIT_OK = 0
EXIT_RECIPE_ERROR = 1
EXIT_ALL_SKIPPED = 2


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
        Path(args.write).write_text(header + text, encoding="utf-8")
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
        print("실행 불가한 스테이지:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
    if args.resolved:
        sys.stdout.write(rec.to_yaml())
    return EXIT_RECIPE_ERROR if problems else EXIT_OK


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return EXIT_RECIPE_ERROR
    return int(func(args))
