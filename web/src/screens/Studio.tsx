import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ApiError,
  type PresetCard,
  type PreviewMeta,
  type StudioState,
  fetchPresets,
  fetchStudioState,
  openStudio,
  presetImageUrl,
  runPreview,
  saveRecipe,
  setPreset,
  setSeed,
  studioImageUrl,
  studioThumbUrl,
} from '../api'
import {
  cleanSeed,
  scaleNote,
  statusLine,
  suspicions,
  targetLabel,
  wipeClip,
  wipeFromPointer,
} from '../studio'

/**
 * 미리보기 — 바탕 한 장에 결함을 붙여 **원본 | 합성** 을 와이프로 견준다.
 *
 * 합성은 전부 파이썬이 한다(`runner`·`studio.jobs.run_preview`): 화면은 파라미터가 바뀔 때마다
 * `preview` 를 POST 하고 결과 PNG 를 `<img>` 로 받는다. WebSocket·SSE 는 두지 않았다 — 로컬 왕복은
 * 수 ms 라 단순 요청/응답으로 충분하고, "최신만 살리기"(Qt `LatestOnlyQueue`)는 **이전 요청을
 * `AbortController` 로 취소**하는 것으로 갈음한다.
 */
export function Studio() {
  const [state, setState] = useState<StudioState | null>(null)
  const [meta, setMeta] = useState<PreviewMeta | null>(null)
  const [cards, setCards] = useState<PresetCard[]>([])
  const [gallery, setGallery] = useState(false)
  /** 이 입력으로 합성이 안 되는 프리셋 — 그림 요청이 404 로 오면 빈 칸을 둔다(fail-soft). */
  const [noThumb, setNoThumb] = useState<string[]>([])
  const [recipePath, setRecipePath] = useState('')
  const [seedText, setSeedText] = useState('')
  const [savePath, setSavePath] = useState('')
  const [wipe, setWipe] = useState(0.5)
  const [showGt, setShowGt] = useState(true)
  const [showRoi, setShowRoi] = useState(false)
  const [busy, setBusy] = useState(false)
  const [rendering, setRendering] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const stageRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  /** 진행 중인 미리보기 요청 — 새 요청이 오면 취소한다(낡은 결과가 나중에 도착해 화면을 되돌리지 않게). */
  const pending = useRef<AbortController | null>(null)

  const open = state?.open === true

  const preview = useCallback(async (body: { targetIndex?: number; variant?: number } = {}) => {
    pending.current?.abort()
    const ctrl = new AbortController()
    pending.current = ctrl
    setRendering(true)
    try {
      const m = await runPreview(body, ctrl.signal)
      setMeta(m)
      setError(null)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return // 최신 요청만 살린다
      setMeta(null)
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      if (pending.current === ctrl) {
        pending.current = null
        setRendering(false)
      }
    }
  }, [])

  useEffect(() => {
    fetchStudioState()
      .then((s) => {
        setState(s)
        if (!s.open) return
        setRecipePath(s.recipePath)
        setSeedText(String(s.recipe.seed))
        setSavePath(s.recipePath || 'recipes/studio.yaml')
        // 서버가 이미 세션을 들고 있으면(화면을 다시 열었을 때) 바로 한 장 보여 준다
        void preview()
      })
      .catch((err: Error) => setError(err.message))
    fetchPresets()
      .then((p) => setCards(p.presets))
      .catch(() => setCards([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 처음 한 번만
  }, [])

  const doOpen = async () => {
    setBusy(true)
    setError(null)
    try {
      const s = await openStudio({ recipe: recipePath.trim() })
      setState(s)
      setNoThumb([])
      if (s.open) {
        setSeedText(String(s.recipe.seed))
        setSavePath(s.recipePath || 'recipes/studio.yaml')
      }
      setMessage(null)
      await preview({ targetIndex: 0 })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  /** 레시피를 바꾸는 편집은 상태를 받아 두고 곧바로 다시 합성한다(지난 그림은 서버가 이미 버렸다). */
  const edit = async (fn: () => Promise<StudioState>) => {
    try {
      const s = await fn()
      setState(s)
      if (s.open) setSeedText(String(s.recipe.seed))
      setMeta(null)
      await preview()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  // ------------------------------------------------------------ 와이프
  const moveWipe = (clientX: number) => {
    const el = stageRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    setWipe(wipeFromPointer(clientX - rect.left, rect.width))
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return
      // 목업은 Tab 이지만 웹에서 Tab 을 가로채면 **키보드 초점 이동**이 막힌다 → G 로.
      if (e.key.toLowerCase() === 'g') {
        setShowGt((v) => !v)
      } else if (e.key.toLowerCase() === 'r') {
        setShowRoi((v) => !v)
      } else if (e.key === ' ') {
        setWipe((w) => (w > 0.5 ? 0 : 1))
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (!state?.open) return
        const n = state.targets.length
        if (!n) return
        const next = (state.targetIndex + (e.key === 'ArrowRight' ? 1 : n - 1)) % n
        setState({ ...state, targetIndex: next })
        void preview({ targetIndex: next })
      } else {
        return
      }
      e.preventDefault()
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [preview, state])

  const version = meta?.version ?? 0

  return (
    <>
      <div className="page-head">
        <h1>미리보기</h1>
        <p>바탕 이미지 한 장에 결함을 붙여 봅니다. 마음에 들면 같은 레시피로 일괄 생성합니다.</p>
      </div>

      <div className="openbar">
        <label htmlFor="studio-recipe">레시피</label>
        <input
          id="studio-recipe"
          value={recipePath}
          spellCheck={false}
          placeholder="예: recipes/sample-poisson.yaml"
          onChange={(e) => setRecipePath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void doOpen()}
        />
        <button
          className="btn primary"
          onClick={() => void doOpen()}
          disabled={busy || !recipePath.trim()}
        >
          {busy ? '여는 중…' : '열기'}
        </button>
      </div>

      {error && (
        <div className="notice bad" role="alert" style={{ marginBottom: 12 }}>
          <span>{error}</span>
        </div>
      )}
      {message && (
        <div className="notice" style={{ marginBottom: 12 }}>
          <span>{message}</span>
        </div>
      )}

      {!open ? (
        <div className="cards">
          <section className="card">
            <h2>아직 연 레시피가 없습니다</h2>
            <p className="sub">
              레시피 한 장을 여세요. 없으면{' '}
              <code>anograft sample --out samples/ring --shape ring --quickstart</code> 가 샘플
              데이터·보관함·레시피를 한 번에 만듭니다.
            </p>
          </section>
        </div>
      ) : (
        <div className="studio">
          <aside className="studio-targets">
            <section className="card">
              <h2>바탕 이미지 {state.targets.length}</h2>
              <p className="sub">
                ←→ 로 옮깁니다 · 보관함 {state.bankName || '(없음)'} · 결함 조각 {state.bankN}개
              </p>
              <div className="tglist">
                {state.targets.map((t) => (
                  <button
                    key={t.path}
                    className={`tgitem${t.index === state.targetIndex ? ' on' : ''}`}
                    title={t.path}
                    onClick={() => {
                      setState({ ...state, targetIndex: t.index })
                      void preview({ targetIndex: t.index })
                    }}
                  >
                    <img src={studioThumbUrl(t.index)} alt="" loading="lazy" draggable={false} />
                    <span>{targetLabel(t)}</span>
                  </button>
                ))}
              </div>
            </section>
          </aside>

          <section className="studio-main">
            <div
              className={`stagewrap${rendering ? ' busy' : ''}`}
              ref={stageRef}
              onPointerDown={(e) => {
                dragging.current = true
                e.currentTarget.setPointerCapture(e.pointerId)
                moveWipe(e.clientX)
              }}
              onPointerMove={(e) => dragging.current && moveWipe(e.clientX)}
              onPointerUp={() => (dragging.current = false)}
              onPointerCancel={() => (dragging.current = false)}
              style={meta ? { aspectRatio: `${meta.width} / ${meta.height}` } : undefined}
            >
              {meta ? (
                <>
                  <img src={studioImageUrl('base', version)} alt="원본" draggable={false} />
                  {meta.status === 'ok' && (
                    <img
                      className="synth"
                      src={studioImageUrl('synth', version)}
                      alt="합성"
                      draggable={false}
                      style={{ clipPath: wipeClip(wipe) }}
                    />
                  )}
                  {showGt && meta.hasGt && (
                    <img
                      className="layer"
                      src={studioImageUrl('gt', version)}
                      alt=""
                      draggable={false}
                      style={{ clipPath: wipeClip(wipe) }}
                    />
                  )}
                  {showRoi && meta.hasRoi && (
                    <img
                      className="layer"
                      src={studioImageUrl('roi', version)}
                      alt=""
                      draggable={false}
                    />
                  )}
                  <i className="wipe" style={{ left: `${wipe * 100}%` }} />
                  <span className="cornertag l">원본</span>
                  <span className="cornertag r">합성</span>
                </>
              ) : (
                <div className="stage-empty">
                  {rendering ? '합성 중…' : '왼쪽에서 바탕 이미지를 고르세요'}
                </div>
              )}
            </div>

            <div className="legend">
              <span className="it">
                <i className="sw gt" />
                정답 영역
              </span>
              <span className="it">
                <i className="sw roi" />
                붙일 수 있는 영역
              </span>
              <span className="spacer" />
              <span className="mono">{statusLine(meta)}</span>
            </div>
            <p className="muted scale-note">{scaleNote(meta)}</p>
          </section>

          <aside className="studio-side">
            <section className="card">
              <h2>레시피</h2>
              <p className="sub">
                {state.recipe.name} · 결함 조각 {state.bankN}개 · 클래스{' '}
                {state.classes.join(', ') || '(없음)'}
              </p>
              <label className="field">
                <span>프리셋</span>
                <select
                  value={state.recipe.preset}
                  onChange={(e) => void edit(() => setPreset(e.target.value))}
                >
                  {state.presets.map((p) => (
                    <option key={p} value={p}>
                      {cards.find((c) => c.name === p)?.title ?? p}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>시드</span>
                <input
                  className="text-in"
                  value={seedText}
                  spellCheck={false}
                  inputMode="numeric"
                  onChange={(e) => setSeedText(e.target.value)}
                  onBlur={() => {
                    const n = cleanSeed(seedText)
                    if (n === null) setSeedText(String(state.recipe.seed))
                    else if (n !== state.recipe.seed) void edit(() => setSeed(n))
                  }}
                  onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                />
              </label>
              <div className="row-btns">
                <button className="btn" onClick={() => setGallery(!gallery)}>
                  {gallery ? '갤러리 닫기' : '프리셋 고르기…'}
                </button>
                <button
                  className="btn"
                  onClick={() => void edit(() => setSeed(state.recipe.seed + 1))}
                  title="시드를 1 올려 다른 배치를 봅니다"
                >
                  다른 시드
                </button>
                <button className="btn" disabled={rendering} onClick={() => void preview()}>
                  다시 합성
                </button>
              </div>
            </section>

            <section className="card">
              <h2>보기</h2>
              <label className="check">
                <input
                  type="checkbox"
                  checked={showGt}
                  onChange={(e) => setShowGt(e.target.checked)}
                />
                정답 영역 (G)
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={showRoi}
                  onChange={(e) => setShowRoi(e.target.checked)}
                />
                붙일 수 있는 영역 (R)
              </label>
              <label className="field" style={{ marginTop: 8 }}>
                <span>와이프</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={Math.round(wipe * 100)}
                  onChange={(e) => setWipe(Number(e.target.value) / 100)}
                />
              </label>
              <p className="sub">스페이스로 원본 ↔ 합성 전체 보기</p>
            </section>

            <section className="card">
              <h2>파이프라인</h2>
              <p className="sub">단계별 알고리즘 — 값 편집은 다음 단위에서 붙습니다.</p>
              {state.stages.map((s) => (
                <div className="stage-row" key={s.stage}>
                  <span className="name" title={`${s.stage} · ${s.method}`}>
                    {s.label}
                  </span>
                  <span className="methods">{s.methodLabel}</span>
                </div>
              ))}
            </section>

            {(state.warnings.length > 0 || (meta?.warnings.length ?? 0) > 0) && (
              <section className="card">
                <h2>경고</h2>
                {[...new Set([...(meta?.warnings ?? state.warnings)])].map((w) => (
                  <div className="notice" key={w}>
                    <span>{w}</span>
                  </div>
                ))}
              </section>
            )}

            {suspicions(meta).length > 0 && (
              <section className="card">
                <h2>의심스러운 것</h2>
                {suspicions(meta).map((s) => (
                  <div className="notice" key={s}>
                    <span>{s}</span>
                  </div>
                ))}
              </section>
            )}

            <section className="card">
              <h2>일괄 생성으로</h2>
              <p className="sub">레시피를 저장하면 그대로 명령줄에서 돌릴 수 있습니다.</p>
              <label className="field">
                <span>경로</span>
                <input
                  className="text-in"
                  value={savePath}
                  spellCheck={false}
                  onChange={(e) => setSavePath(e.target.value)}
                />
              </label>
              <button
                className="btn primary"
                disabled={!savePath.trim()}
                onClick={() => {
                  saveRecipe(savePath.trim())
                    .then((r) => setMessage(`저장했습니다 — ${r.runCommand}`))
                    .catch((err: Error) => setError(err.message))
                }}
              >
                레시피 저장
              </button>
              <p className="sub mono" style={{ marginTop: 6 }}>
                {state.runCommand}
              </p>
            </section>
          </aside>
        </div>
      )}

      {open && gallery && (
        <div className="cards" style={{ marginTop: 12 }}>
          <section className="card">
            <h2>프리셋 고르기</h2>
            <p className="sub">
              지금 고른 바탕으로 합성한 그림입니다. 돌릴 수 없는 프리셋은 그림이 비어 있습니다.
            </p>
            <div className="grid">
              {cards.map((c) => (
                <button
                  key={c.name}
                  className={`tile${c.name === state.recipe.preset ? ' on' : ''}`}
                  title={`${c.name} · ${c.useFor}`}
                  onClick={() => {
                    setGallery(false)
                    void edit(() => setPreset(c.name))
                  }}
                >
                  {noThumb.includes(c.name) ? (
                    <span className="tile-empty">이 입력으로는 못 돌립니다</span>
                  ) : (
                    <img
                      src={presetImageUrl(c.name, version)}
                      alt=""
                      loading="lazy"
                      onError={() => setNoThumb((v) => (v.includes(c.name) ? v : [...v, c.name]))}
                    />
                  )}
                  <span className="tile-cap">
                    <b>{c.title}</b>
                    <em>{c.name}</em>
                  </span>
                  <span className="tile-note">{c.summary}</span>
                </button>
              ))}
            </div>
          </section>
        </div>
      )}
    </>
  )
}
