import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ApiError,
  type BatchSettings,
  type BatchState,
  fetchBatchState,
  openBatch,
  openReview,
  setBatchSettings,
  startBatch,
  stopBatch,
} from '../api'
import {
  PHASE_LABELS,
  canReview,
  logTail,
  percent,
  perClassText,
  phaseOf,
  pollDelay,
  progressText,
  summaryLine,
} from '../batch'

/**
 * 일괄 생성 — 미리보기에서 맞춘 레시피를 원본 해상도로 돌려 데이터셋을 만든다.
 *
 * 오래 걸리는 작업이지만 **스트리밍은 두지 않는다**(U5 확정): 서버가 스레드에서 돌리고 화면은
 * `state` 를 0.4 초마다 묻는다. 끝나면 폴링을 멈춘다(`pollDelay`).
 */
export function Batch() {
  const [state, setState] = useState<BatchState | null>(null)
  const [recipePath, setRecipePath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  const open = state?.open === true
  const run = state?.open ? state.run : null
  const phase = phaseOf(run ?? null)
  const summary = run?.summary ?? null

  useEffect(() => {
    fetchBatchState()
      .then((s) => {
        setState(s)
        if (s.open) setRecipePath(s.recipePath)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  // 폴링 — 도는 동안만. 끝나면 `pollDelay` 가 0 이라 타이머를 걸지 않는다.
  useEffect(() => {
    const delay = pollDelay(phase)
    if (!delay) return
    const id = window.setTimeout(() => {
      fetchBatchState()
        .then(setState)
        .catch(() => undefined) // 한 번 실패해도 다음 주기에 다시 묻는다(fail-soft)
    }, delay)
    return () => clearTimeout(id)
  }, [phase, state])

  // 로그는 새 줄이 오면 끝으로
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [run?.log.length])

  const call = useCallback(async (fn: () => Promise<BatchState>) => {
    setBusy(true)
    setError(null)
    try {
      setState(await fn())
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }, [])

  const patch = (settings: Partial<BatchSettings>) => void call(() => setBatchSettings(settings))

  const doStart = () => {
    if (!state?.open) return
    const s = state.settings
    const ok = window.confirm(
      `${s.count}장을 ${s.out} 에 만듭니다. 이미 있는 파일은 덮어씁니다. 계속할까요?`,
    )
    if (ok) void call(startBatch)
  }

  const goReview = () => {
    if (!summary) return
    openReview(summary.root)
      .then(() => {
        location.hash = '#/review'
      })
      .catch((err: Error) => setError(err.message))
  }

  return (
    <>
      <div className="page-head">
        <h1>일괄 생성</h1>
        <p>
          미리보기에서 맞춘 레시피를 <b>원본 해상도</b>로 돌려 학습용 데이터셋을 만듭니다.
        </p>
      </div>

      <div className="openbar">
        <label htmlFor="batch-recipe">레시피</label>
        <input
          id="batch-recipe"
          value={recipePath}
          spellCheck={false}
          placeholder="예: recipes/sample-poisson.yaml"
          onChange={(e) => setRecipePath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void call(() => openBatch(recipePath.trim()))}
        />
        <button
          className="btn primary"
          disabled={busy || !recipePath.trim() || phase === 'running'}
          onClick={() => void call(() => openBatch(recipePath.trim()))}
        >
          열기
        </button>
      </div>

      {error && (
        <div className="notice bad" role="alert" style={{ marginBottom: 12 }}>
          <span>{error}</span>
        </div>
      )}

      {!open ? (
        <div className="cards">
          <section className="card">
            <h2>아직 연 레시피가 없습니다</h2>
            <p className="sub">
              미리보기 화면에서 <b>레시피 저장</b> 한 파일을 여세요. 같은 레시피를 명령줄에서
              <code> anograft run &lt;recipe&gt;</code> 로 돌려도 결과가 같습니다.
            </p>
          </section>
        </div>
      ) : (
        <div className="review">
          <aside className="review-side">
            <section className="card">
              <h2>무엇을 만드나</h2>
              <p className="sub">
                {state.recipe.name} · {state.recipe.preset}
                <br />
                보관함 {state.recipe.bank || '(없음)'} · 바탕 {state.recipe.targets}
                <br />
                이미지당 결함 {state.recipe.defectsPerImage[0]}–{state.recipe.defectsPerImage[1]}개
                · 정상 이미지 {state.recipe.includeNormals ? '포함' : '제외'}
              </p>
            </section>

            <section className="card batch-settings">
              <h2>설정</h2>
              <label className="field">
                <span>출력 폴더</span>
                <input
                  className="text-in"
                  value={state.settings.out}
                  spellCheck={false}
                  disabled={phase === 'running'}
                  onChange={(e) =>
                    setState({ ...state, settings: { ...state.settings, out: e.target.value } })
                  }
                  onBlur={(e) => patch({ out: e.target.value })}
                />
              </label>
              <label className="field">
                <span>장수</span>
                <input
                  className="text-in"
                  inputMode="numeric"
                  value={state.settings.count}
                  disabled={phase === 'running'}
                  onChange={(e) =>
                    setState({
                      ...state,
                      settings: { ...state.settings, count: Number(e.target.value) || 0 },
                    })
                  }
                  onBlur={(e) => patch({ count: Number(e.target.value) || 0 })}
                />
              </label>
              <label className="field">
                <span>시드</span>
                <input
                  className="text-in"
                  inputMode="numeric"
                  value={state.settings.seed}
                  disabled={phase === 'running'}
                  onChange={(e) =>
                    setState({
                      ...state,
                      settings: { ...state.settings, seed: Number(e.target.value) || 0 },
                    })
                  }
                  onBlur={(e) => patch({ seed: Number(e.target.value) || 0 })}
                />
              </label>
              <label className="field">
                <span title="0 이면 한 프로세스에서 — 작은 배치(20장 미만)는 이쪽이 빠릅니다">
                  워커
                </span>
                <input
                  className="text-in"
                  inputMode="numeric"
                  value={state.settings.workers}
                  disabled={phase === 'running'}
                  onChange={(e) =>
                    setState({
                      ...state,
                      settings: { ...state.settings, workers: Number(e.target.value) || 0 },
                    })
                  }
                  onBlur={(e) => patch({ workers: Number(e.target.value) || 0 })}
                />
              </label>
              <label className="field">
                <span>출력 형식</span>
                <select
                  value={state.settings.writer}
                  disabled={phase === 'running'}
                  onChange={(e) => patch({ writer: e.target.value })}
                >
                  {state.writerFormats.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </label>
              {state.settings.writer === 'mvtec' && (
                <label className="field">
                  <span>카테고리</span>
                  <input
                    className="text-in"
                    value={state.settings.mvtecCategory}
                    spellCheck={false}
                    disabled={phase === 'running'}
                    onChange={(e) =>
                      setState({
                        ...state,
                        settings: { ...state.settings, mvtecCategory: e.target.value },
                      })
                    }
                    onBlur={(e) => patch({ mvtecCategory: e.target.value })}
                  />
                </label>
              )}
              <p className="sub mono" style={{ marginTop: 6 }}>
                {state.runCommand}
              </p>
            </section>

            <section className="card">
              <h2>실행</h2>
              <p className="sub">{PHASE_LABELS[phase]}</p>
              <div className="row-btns">
                <button
                  className="btn primary"
                  disabled={busy || phase === 'running' || phase === 'stopping'}
                  onClick={doStart}
                >
                  일괄 생성 시작
                </button>
                <button
                  className="btn danger"
                  disabled={phase !== 'running'}
                  onClick={() => void call(stopBatch)}
                >
                  중지
                </button>
              </div>
              {canReview(summary) && (
                <button className="btn" style={{ marginTop: 6 }} onClick={goReview}>
                  검수 화면에서 열기 →
                </button>
              )}
            </section>
          </aside>

          <section className="review-main">
            <div className="card">
              <div className="run-head">
                <b>{progressText(run ?? null)}</b>
                <span className="spacer" />
                <span className="muted">{PHASE_LABELS[phase]}</span>
              </div>
              <div className="progress">
                <i style={{ width: `${percent(run?.done ?? 0, run?.total ?? 0)}%` }} />
              </div>
              {summary && (
                <>
                  <p className="sub" style={{ marginTop: 8 }}>
                    {summaryLine(summary)}
                  </p>
                  {perClassText(summary) && (
                    <p className="sub">클래스별 인스턴스: {perClassText(summary)}</p>
                  )}
                  {Object.entries(summary.files).map(([name, rel]) => (
                    <p className="sub mono" key={name}>
                      {name}: {rel}
                    </p>
                  ))}
                </>
              )}
              {run?.error && (
                <div className="notice bad" style={{ marginTop: 8 }}>
                  <span>{run.error}</span>
                </div>
              )}
              <pre className="runlog" ref={logRef}>
                {logTail(run?.log ?? []).join('\n') || '실행하면 여기에 기록이 남습니다.'}
              </pre>
            </div>
          </section>
        </div>
      )}
    </>
  )
}
