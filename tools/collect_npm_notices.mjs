/**
 * npm 의존성 고지 수집 — `THIRD-PARTY-NOTICES.md` 의 npm 절을 만든다.
 *
 * 왜 자동인가: MSIX/Store 계획이 있어 고지가 형식적인 일이 아니고(결정 6 ④), 손으로 적으면
 * 의존성이 바뀔 때 **조용히 낡는다**. 그래서 CI 가 `--check` 로 낡았는지 본다.
 *
 * **`package-lock.json` 만 읽는다** — `npm ls` 를 부르지 않는다:
 *   ① node_modules 설치 없이도 돈다(CI 에서 `npm ci` 순서에 매이지 않는다)
 *   ② lock 이 곧 정본이다(결정 6 ② — 버전을 고정하는 그 파일)
 *   ③ Node 24 는 `npm.cmd` 를 직접 spawn 하지 못하고(EINVAL), `shell: true` 로 우회하면 DEP0190 경고가 난다
 * lockfileVersion 3 은 항목마다 `license` 와 `dev` 를 들고 있어 그대로 쓸 수 있다.
 *
 * 쓰는 법:
 *   node tools/collect_npm_notices.mjs            # 표를 stdout 으로
 *   node tools/collect_npm_notices.mjs --write    # THIRD-PARTY-NOTICES.md 의 마커 사이를 갱신
 *   node tools/collect_npm_notices.mjs --check    # 커밋된 내용과 다르면 exit 1 (CI)
 */

import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const LOCK = resolve(ROOT, 'web', 'package-lock.json')
const NOTICES = resolve(ROOT, 'THIRD-PARTY-NOTICES.md')
const BEGIN = '<!-- npm:begin — tools/collect_npm_notices.mjs 가 만든다. 손으로 고치지 말 것 -->'
const END = '<!-- npm:end -->'

const lock = JSON.parse(readFileSync(LOCK, 'utf8'))
if (lock.lockfileVersion < 3) {
  console.error(`lockfileVersion ${lock.lockfileVersion} 은 license 를 담지 않습니다 — npm 9+ 로 다시 만드세요.`)
  process.exit(2)
}

/** 루트('') 를 뺀 모든 항목. 이름은 `node_modules/` 접두를 벗기고 중첩(`a/node_modules/b`)은 마지막 것. */
const entries = Object.entries(lock.packages)
  .filter(([path]) => path)
  .map(([path, meta]) => ({
    name: path.slice(path.lastIndexOf('node_modules/') + 'node_modules/'.length),
    version: meta.version ?? '(버전 없음)',
    license: meta.license ?? '(확인 필요)',
    dev: meta.dev === true,
  }))

// 배포물에 코드가 들어가는 것 = dev 가 아닌 것. 지금은 react 계열뿐이고, 늘어나면 확인 게이트에 걸린다.
const runtime = entries.filter((e) => !e.dev)
const tooling = entries.filter((e) => e.dev)

function table(pkgs) {
  const rows = [...pkgs]
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((p) => `| ${p.name} | ${p.version} | ${p.license} |`)
  return ['| 구성 요소 Component | 버전 Version | 라이선스 License |', '|---|---|---|', ...rows].join('\n')
}

/** 빌드 도구는 **코드가 배포물에 들어가지 않는다** → 100줄짜리 표 대신 분포만. 전체는 명령으로 뽑는다. */
function summary(pkgs) {
  const counts = new Map()
  for (const p of pkgs) counts.set(p.license, (counts.get(p.license) ?? 0) + 1)
  const parts = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([lic, n]) => `${lic} ${n}`)
  return `총 ${pkgs.length}개 · ${parts.join(' · ')}`
}

const section = [
  BEGIN,
  '',
  '### 웹 UI 번들에 포함 / Bundled in the web UI',
  '',
  '`anograft serve` 가 내보내는 화면 파일에 코드가 들어갑니다. 폰트·아이콘은 받지 않습니다(시스템 폰트 + 인라인 SVG).',
  '',
  table(runtime),
  '',
  '### 웹 빌드 도구 (배포물에 코드가 포함되지 않음) / Web build tooling (no code shipped)',
  '',
  `Vite·TypeScript·Babel·vitest 등 **빌드 때만** 쓰는 것들입니다 — ${summary(tooling)}.`,
  '플랫폼별 선택 의존성까지 포함한 목록이라 실제로 설치되는 것은 이보다 적습니다.',
  '전체 목록은 `node tools/collect_npm_notices.mjs` 로 뽑을 수 있고, 버전은 `web/package-lock.json` 이 고정합니다.',
  '',
  END,
].join('\n')

const mode = process.argv[2] ?? '--print'
if (mode === '--print') {
  console.log(section)
  console.log(`\n(빌드 도구 전체)\n${table(tooling)}`)
} else {
  const current = readFileSync(NOTICES, 'utf8')
  const start = current.indexOf(BEGIN)
  const stop = current.indexOf(END)
  if (start < 0 || stop < 0) {
    console.error(`THIRD-PARTY-NOTICES.md 에 마커가 없습니다 — ${BEGIN}`)
    process.exit(2)
  }
  const next = current.slice(0, start) + section + current.slice(stop + END.length)
  if (mode === '--check') {
    if (next !== current) {
      console.error('THIRD-PARTY-NOTICES.md 가 낡았습니다 — `npm run notices -- --write` 를 돌리고 커밋하세요.')
      process.exit(1)
    }
    console.log('THIRD-PARTY-NOTICES.md 최신입니다.')
  } else if (mode === '--write') {
    writeFileSync(NOTICES, next, 'utf8')
    console.log('THIRD-PARTY-NOTICES.md 갱신했습니다.')
  } else {
    console.error(`모르는 옵션: ${mode} (--print | --write | --check)`)
    process.exit(2)
  }
}
