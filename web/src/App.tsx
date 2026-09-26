import { useEffect, useState } from 'react'

import {
  type Doctor,
  type Health,
  type MethodInfo,
  fetchDoctor,
  fetchHealth,
  fetchMethods,
} from './api'
import { type StageInfo } from './format'
import { BankScreen } from './screens/BankScreen'
import { Batch } from './screens/Batch'
import { Label } from './screens/Label'
import { Loop } from './screens/Loop'
import { Review } from './screens/Review'
import { Start } from './screens/Start'
import { Studio } from './screens/Studio'
import { Soon } from './screens/Soon'

/** 화면 순서 = 목업 v2 의 흐름(①보관함 ②결함 표시 …). 아직 안 옮긴 화면은 `ready: false`(예정 배지). */
const SCREENS = [
  { id: 'start', num: '', label: '시작', ready: true },
  { id: 'bank', num: '①', label: '결함 보관함', ready: true },
  { id: 'label', num: '②', label: '결함 표시', ready: true },
  { id: 'studio', num: '③', label: '미리보기', ready: true },
  { id: 'batch', num: '④', label: '일괄 생성', ready: true },
  { id: 'review', num: '⑤', label: '검수', ready: true },
  { id: 'loop', num: '⑥', label: '학습 루프', ready: true },
] as const

type ScreenId = (typeof SCREENS)[number]['id']

function currentScreen(): ScreenId {
  const id = location.hash.replace(/^#\/?/, '') as ScreenId
  return SCREENS.some((s) => s.id === id) ? id : 'start'
}

export function App() {
  const [screen, setScreen] = useState<ScreenId>(currentScreen)
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (document.documentElement.dataset.theme as 'light' | 'dark') ?? 'light',
  )
  const [health, setHealth] = useState<Health | null>(null)
  const [doctor, setDoctor] = useState<Doctor | null>(null)
  const [methods, setMethods] = useState<MethodInfo[]>([])
  const [stages, setStages] = useState<StageInfo[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setScreen(currentScreen())
    addEventListener('hashchange', onHash)
    return () => removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('graft.theme', theme)
    } catch {
      // 저장 못 해도 이번 세션 동안은 적용된다
    }
  }, [theme])

  useEffect(() => {
    let alive = true
    Promise.all([fetchHealth(), fetchDoctor(), fetchMethods()])
      .then(([h, d, m]) => {
        if (!alive) return
        setHealth(h)
        setDoctor(d)
        setMethods(m.methods)
        setStages(m.stages ?? [])
      })
      .catch((err: Error) => alive && setError(err.message))
    return () => {
      alive = false
    }
  }, [])

  const go = (id: ScreenId) => {
    location.hash = `#/${id}`
    setScreen(id)
  }

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <i className="dot" />
          Graft
        </span>
        <span className="muted mono">{health ? health.version : '…'}</span>
        <span className="muted">· 오프라인 · CPU</span>
        <span className="spacer" />
        <span className="chip">로컬 전용 127.0.0.1</span>
        <button
          className="icon-btn"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          title={theme === 'dark' ? '밝은 화면으로' : '어두운 화면으로'}
          aria-label="화면 테마 바꾸기"
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </header>

      <div className="body">
        <nav className="rail" aria-label="화면">
          {SCREENS.map((s) => (
            <button
              key={s.id}
              className={s.id === screen ? 'on' : ''}
              onClick={() => go(s.id)}
              aria-current={s.id === screen ? 'page' : undefined}
            >
              {s.num && <span className="num">{s.num}</span>}
              {s.label}
              {!s.ready && <span className="soon">예정</span>}
            </button>
          ))}
        </nav>

        <main className="main">
          {screen === 'start' ? (
            <Start
              health={health}
              doctor={doctor}
              methods={methods}
              stages={stages}
              error={error}
            />
          ) : screen === 'review' ? (
            <Review />
          ) : screen === 'bank' ? (
            <BankScreen />
          ) : screen === 'label' ? (
            <Label />
          ) : screen === 'studio' ? (
            <Studio />
          ) : screen === 'batch' ? (
            <Batch />
          ) : screen === 'loop' ? (
            <Loop />
          ) : (
            <Soon label={SCREENS.find((s) => s.id === screen)!.label} />
          )}
        </main>
      </div>

      <footer className="statusbar">
        <span>{error ? error : health ? '서버 연결됨' : '서버에 묻는 중…'}</span>
        <span className="spacer" />
        <span className="mono">anograft serve</span>
      </footer>
    </div>
  )
}
