"""정적 번들(빌드된 프론트) 찾기·안전한 경로 해석 — **순수 함수**(소켓·핸들러 의존 0).

번들은 **repo 에 커밋하지 않는다**(`src/anograft/web/static/` 는 gitignore). CI 가 `web/` 을 빌드해
여기에 넣고 wheel·zip 에 실린다. 그래서 소스 체크아웃에서는 번들이 **없는 게 정상**이고,
`serve` 는 그때 죽지 말고 안내를 띄워야 한다(fail-soft — 결정 6 ①).

경로 규칙 셋:
- URL 경로는 **항상 번들 루트 안**으로 해석한다(`..`·절대경로·심볼릭 링크 탈출 차단).
- 디렉터리는 `index.html` 로.
- 파일이 없으면 `index.html` 로 폴백하지 **않는다** — 지금 프론트는 해시 라우팅이라 SPA 폴백이 필요 없고,
  폴백을 두면 오타 난 자원 요청이 200 HTML 로 와서 디버깅이 어려워진다.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

#: 번들이 들어갈 자리 — CI 가 `web/dist/*` 를 그대로 복사한다.
BUNDLE_DIR = Path(__file__).resolve().parent / "static"

_TEXT_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
}


@dataclass(frozen=True)
class BundleStatus:
    """번들이 있는지와, 없다면 사람에게 할 말."""

    root: Path
    present: bool
    reason: str = ""


def bundle_status(root: Path | None = None) -> BundleStatus:
    root = (root or BUNDLE_DIR).resolve()
    if (root / "index.html").is_file():
        return BundleStatus(root, True)
    return BundleStatus(
        root,
        False,
        "화면 파일(웹 번들)이 없습니다 — 소스에서 실행할 때는 정상입니다(번들은 저장소에 두지 않습니다).",
    )


def resolve_static(root: Path, url_path: str) -> Path | None:
    """URL 경로 → 실제 파일. 번들 밖으로 나가거나 없는 파일이면 ``None``.

    `..` 뿐 아니라 심볼릭 링크로 밖을 가리키는 경우까지 막으려고 **해석(resolve) 후에** 포함 관계를 본다.
    """
    root = root.resolve()
    rel = url_path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
    if not rel or rel.endswith("/"):
        rel += "index.html"
    if "\x00" in rel:
        return None
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    if candidate.is_dir():
        candidate = (candidate / "index.html").resolve()
        if root not in candidate.parents and candidate != root:
            return None
    return candidate if candidate.is_file() else None


def content_type(path: Path) -> str:
    """서빙할 MIME. 텍스트류는 **charset=utf-8 을 반드시 붙인다** — 한국어 UI 가 깨진다."""
    suffix = path.suffix.lower()
    if suffix in _TEXT_TYPES:
        base = _TEXT_TYPES[suffix]
        return f"{base}; charset=utf-8" if base.startswith(("text/", "application/json")) else base
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def missing_bundle_html(dev_port: int = 5173) -> str:
    """번들이 없을 때 브라우저에 띄우는 안내 — API 는 그대로 살아 있다는 것을 알려 준다."""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Graft — 화면 파일이 없습니다</title>
<style>
 :root{{color-scheme:light dark}}
 body{{margin:0;padding:40px;font:14px/1.7 "Malgun Gothic","Apple SD Gothic Neo",-apple-system,"Segoe UI",sans-serif}}
 main{{max-width:640px;margin:0 auto}}
 h1{{font-size:19px;margin:0 0 4px}}
 p.sub{{color:#6b7280;margin:0 0 24px}}
 code{{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;background:rgba(127,127,127,.15);padding:1px 5px;border-radius:4px}}
 ol{{padding-left:20px}} li{{margin:6px 0}}
 .ok{{margin-top:24px;padding:12px 14px;border-radius:8px;background:rgba(0,161,136,.12)}}
</style></head>
<body><main>
<h1>화면 파일(웹 번들)이 없습니다</h1>
<p class="sub">소스에서 실행할 때는 정상입니다 — 번들은 저장소에 두지 않고 배포할 때 만듭니다.</p>
<ol>
<li>개발 중이라면 <code>web/</code> 에서 <code>npm install</code> 후 <code>npm run dev</code> 를 켜고
 <a href="http://127.0.0.1:{dev_port}/">http://127.0.0.1:{dev_port}/</a> 로 접속하세요. 이 서버는 그대로 두면 됩니다(API 를 대신 받습니다).</li>
<li>한 번만 확인하려면 <code>npm run build</code> 로 번들을 만든 뒤 이 주소를 새로 고치세요.</li>
<li>배포판(zip·wheel)으로 설치했는데 이 화면이 보인다면 그건 버그입니다 — 번들이 빠진 배포물입니다.</li>
</ol>
<div class="ok">API 는 지금도 동작합니다 — <code>/api/health</code> · <code>/api/doctor</code> · <code>/api/methods</code></div>
</main></body></html>
"""
