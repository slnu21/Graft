import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  ApiError,
  type HistogramData,
  type ReviewDetail,
  type ReviewItem,
  type ReviewState,
  fetchHistogram,
  fetchReviewDetail,
  fetchReviewItems,
  fetchReviewState,
  imageUrl,
  openReview,
  pruneDataset,
  setVerdict,
  writeReport,
} from '../api'
import { Histogram } from '../components/Histogram'
import { VERDICT_LABEL, VERDICT_MARK, countsText, judgeable, nextIndexAfter, progress } from '../review'

/**
 * 검수 — 웹으로 옮기는 첫 화면. 출력 폴더만 읽으면 되어 독립적이고, 루프 화면(U6)이 이걸 그대로 확장한다.
 *
 * 기본 뷰는 **모아 보기(그리드)**(사용자 결정 2026-09-23) — 다만 훑기가 목적이므로 타일에 결함이 보여야 한다.
 * 판정은 키보드 우선: `A` 채택 · `R` 반려 · `U` 판정 취소 · `←/→` 이동.
 */
export function Review() {
  const [state, setState] = useState<ReviewState | null>(null)
  const [root, setRoot] = useState('')
  const [items, setItems] = useState<ReviewItem[]>([])
  const [filter, setFilter] = useState('all')
  const [cls, setCls] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<ReviewDetail | null>(null)
  const [hist, setHist] = useState<HistogramData | null>(null)
  const [histKey, setHistKey] = useState('area')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const gridRef = useRef<HTMLDivElement>(null)

  const open = state?.open === true

  useEffect(() => {
    fetchReviewState()
      .then((s) => {
        setState(s)
        if (s.open) setRoot(s.root)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  const reload = useCallback(async () => {
    if (!open) return
    const res = await fetchReviewItems(filter, cls)
    setItems(res.items)
    const ok = judgeable(res.items)
    setSelected((cur) => (cur && ok.some((i) => i.index === cur) ? cur : (ok[0]?.index ?? null)))
  }, [open, filter, cls])

  useEffect(() => {
    reload().catch((err: Error) => setError(err.message))
  }, [reload])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      return
    }
    let alive = true
    fetchReviewDetail(selected)
      .then((d) => alive && setDetail(d))
      .catch((err: Error) => alive && setError(err.message))
    return () => {
      alive = false
    }
  }, [selected])

  useEffect(() => {
    if (!open) return
    let alive = true
    fetchHistogram(histKey, cls)
      .then((h) => alive && setHist(h))
      .catch(() => alive && setHist(null)) // 은행이 없으면 실제 분포가 없다 — 화면은 그대로 뜬다
    return () => {
      alive = false
    }
  }, [open, histKey, cls, items])

  const doOpen = async () => {
    setBusy(true)
    setError(null)
    try {
      const s = await openReview(root.trim())
      setState(s)
      setMessage(null)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const judge = useCallback(
    async (verdict: string) => {
      if (!selected) return
      const goTo = verdict ? nextIndexAfter(items, selected) : null
      try {
        const res = await setVerdict(selected, verdict)
        setItems((cur) => cur.map((it) => (it.index === selected ? res.item : it)))
        setState((cur) => (cur?.open ? { ...cur, counts: res.counts } : cur))
        if (goTo) setSelected(goTo)
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err))
      }
    },
    [selected, items],
  )

  const move = useCallback(
    (delta: number) => {
      if (!items.length) return
      const at = items.findIndex((it) => it.index === selected)
      const next = items[Math.min(items.length - 1, Math.max(0, (at < 0 ? 0 : at) + delta))]
      if (next) setSelected(next.index)
    },
    [items, selected],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return
      const key = e.key.toLowerCase()
      if (key === 'a') void judge('accept')
      else if (key === 'r') void judge('reject')
      else if (key === 'u') void judge('')
      else if (e.key === 'ArrowRight') move(1)
      else if (e.key === 'ArrowLeft') move(-1)
      else return
      e.preventDefault()
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [judge, move])

  const done = useMemo(() => progress(items), [items])
  const hints = state?.open ? state.contrastHints : []

  return (
    <>
      <div className="page-head">
        <h1>검수</h1>
        <p>만든 데이터셋을 눈으로 훑고 채택·반려합니다. 반려한 것을 뺀 정리본을 내보낼 수 있습니다.</p>
      </div>

      <div className="openbar">
        <label htmlFor="review-root">출력 폴더</label>
        <input
          id="review-root"
          value={root}
          spellCheck={false}
          placeholder="예: out/set-A"
          onChange={(e) => setRoot(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void doOpen()}
        />
        <button className="btn primary" onClick={() => void doOpen()} disabled={busy || !root.trim()}>
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
            <h2>아직 연 폴더가 없습니다</h2>
            <p className="sub">`anograft run` 이 만든 출력 폴더를 여세요 — manifest.csv 가 있는 그 폴더입니다.</p>
            <ol className="steps">
              <li>
                데이터셋 만들기 — <code>anograft run recipes/x.yaml</code>
              </li>
              <li>여기에 그 <code>output.root</code> 경로를 넣고 열기</li>
            </ol>
          </section>
        </div>
      ) : (
        <div className="review">
          <aside className="review-side">
            <section className="card">
              <h2>고르기</h2>
              <p className="sub">{countsText(state.counts)}</p>
              <label className="field">
                <span>필터</span>
                <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                  {state.filters.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>클래스</span>
                <select value={cls} onChange={(e) => setCls(e.target.value)}>
                  <option value="">전체</option>
                  {state.classes.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
              <div className="progress" title={`${Math.round(done * 100)}% 판정함`}>
                <i style={{ width: `${done * 100}%` }} />
              </div>
              <p className="sub" style={{ margin: '6px 0 0' }}>
                합성 {judgeable(items).length}장 중 {judgeable(items).filter((i) => i.verdict).length}
                장 판정
              </p>
            </section>

            {hist && (
              <section className="card">
                <h2>분포</h2>
                <p className="sub">합성과 실제를 같은 구간에서 셉니다.</p>
                <label className="field">
                  <span>기준</span>
                  <select value={histKey} onChange={(e) => setHistKey(e.target.value)}>
                    <option value="area">면적</option>
                    <option value="long">긴 변</option>
                    <option value="contrast">밝기 차(결함 − 주변)</option>
                    <option value="texture">거칠기</option>
                    <option value="sharpness">선명도</option>
                  </select>
                </label>
                <Histogram data={hist} />
              </section>
            )}

            {hints.length > 0 && (
              <div className="notice">
                <span>
                  <b>합성 대비가 실제의 절반 이하인 클래스가 있습니다.</b>{' '}
                  {hints.map((h) => h.cls).join(' · ')} — 이 클래스는 `relative-paste` 로 갈라
                  만들어 보세요.
                </span>
              </div>
            )}

            <section className="card">
              <h2>내보내기</h2>
              <p className="sub">반려한 것을 뺀 사본을 만듭니다. 원본은 그대로 둡니다.</p>
              <div className="row-btns">
                <button
                  className="btn"
                  disabled={busy}
                  onClick={() => {
                    const out = `${state.root}-pruned`
                    setBusy(true)
                    pruneDataset(out, false)
                      .then((r) =>
                        setMessage(`정리본을 만들었습니다 — ${r.out} (남김 ${r.kept} · 뺌 ${r.dropped})`),
                      )
                      .catch((err: Error) => setError(err.message))
                      .finally(() => setBusy(false))
                  }}
                >
                  반려 빼고 내보내기
                </button>
                <button
                  className="btn"
                  disabled={busy}
                  onClick={() => {
                    setBusy(true)
                    writeReport()
                      .then((r) => setMessage(`리포트를 만들었습니다 — ${r.path}`))
                      .catch((err: Error) => setError(err.message))
                      .finally(() => setBusy(false))
                  }}
                >
                  리포트 만들기
                </button>
              </div>
            </section>
          </aside>

          <section className="review-main">
            {detail && (
              <div className="detail">
                <div className="detail-img">
                  <img src={imageUrl(detail.index, 'detail')} alt={`${detail.index} 합성 결과`} />
                </div>
                <div className="detail-side">
                  <div className="verdict-row">
                    <button
                      className={`btn${detail.verdict === 'accept' ? ' primary' : ''}`}
                      onClick={() => void judge('accept')}
                    >
                      채택 <kbd>A</kbd>
                    </button>
                    <button
                      className={`btn${detail.verdict === 'reject' ? ' danger' : ''}`}
                      onClick={() => void judge('reject')}
                    >
                      반려 <kbd>R</kbd>
                    </button>
                    <button className="btn" onClick={() => void judge('')}>
                      판정 취소 <kbd>U</kbd>
                    </button>
                  </div>
                  <dl className="kv">
                    <dt>번호</dt>
                    <dd>{detail.index}</dd>
                    <dt>클래스</dt>
                    <dd>{detail.classes.join(', ') || '—'}</dd>
                    <dt>바탕 이미지</dt>
                    <dd>{detail.target || '—'}</dd>
                    <dt>붙이기</dt>
                    <dd>
                      {detail.blend}
                      {detail.fallback ? ' (대체 처리됨)' : ''}
                    </dd>
                    <dt>면적</dt>
                    <dd>{detail.areaPx.toLocaleString('ko-KR')} px</dd>
                    <dt>결함 조각</dt>
                    <dd>{detail.sourceIds.join(', ') || '—'}</dd>
                  </dl>
                  {detail.warnings.length > 0 && (
                    <div className="notice" style={{ marginTop: 10 }}>
                      <span>{detail.warnings.join(' · ')}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            <div className="grid" ref={gridRef}>
              {items.map((it) => (
                <button
                  key={it.key}
                  className={`tile${it.status === 'ok' && it.index === selected ? ' on' : ''} v-${
                    it.verdict || 'none'
                  }${it.status === 'ok' ? '' : ' plain'}`}
                  onClick={() => it.status === 'ok' && setSelected(it.index)}
                  disabled={it.status !== 'ok'}
                  title={
                    it.status === 'ok'
                      ? `${it.index} · ${VERDICT_LABEL[it.verdict] ?? ''}`
                      : it.status === 'normal'
                        ? '정상 이미지 — 판정 대상이 아닙니다'
                        : `건너뜀 — ${it.reason || '사유 없음'}`
                  }
                >
                  {it.status === 'ok' ? (
                    <img src={imageUrl(it.index)} alt="" loading="lazy" decoding="async" />
                  ) : (
                    <span className="tile-empty">{it.status === 'normal' ? '정상' : '건너뜀'}</span>
                  )}
                  {/* 캡션은 판정 대상에만 — 정상·건너뜀은 타일 안 글자로 이미 구분된다 */}
                  {it.status === 'ok' && (
                    <span className="tile-cap">
                      <b>{VERDICT_MARK[it.verdict] ?? '·'}</b> {it.index}
                      {it.classes.length > 0 && <em>{it.classes.join(',')}</em>}
                    </span>
                  )}
                </button>
              ))}
              {!items.length && <p className="muted">이 필터에 걸린 것이 없습니다.</p>}
            </div>

          </section>
        </div>
      )}
    </>
  )
}
