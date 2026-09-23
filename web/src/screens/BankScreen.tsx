import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  ApiError,
  type BankSourceRow,
  type BankState,
  bankImageUrl,
  deleteBankSources,
  fetchBankSources,
  fetchBankState,
  openBank,
} from '../api'
import { SORT_LABELS, bankCountsText, confidenceText, toggleId } from '../bank'

/**
 * 결함 보관함 — 모아 둔 결함 조각을 훑고, 못 쓸 것을 지운다.
 *
 * U3 검수 화면과 같은 뼈대(좌측 고르기 + 그리드 + 상세)를 쓴다. 다른 점 둘:
 * 여러 개를 **함께 고를 수 있고**(일괄 삭제), 타일을 캐시하지 않는다(마스크를 다듬으면 그림이 바뀐다).
 */
export function BankScreen() {
  const [state, setState] = useState<BankState | null>(null)
  const [root, setRoot] = useState('')
  const [rows, setRows] = useState<BankSourceRow[]>([])
  const [cls, setCls] = useState('')
  const [tag, setTag] = useState('')
  const [sort, setSort] = useState('id')
  const [onlyLow, setOnlyLow] = useState(false)
  const [onlyEstimated, setOnlyEstimated] = useState(false)
  const [text, setText] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const open = state?.open === true

  useEffect(() => {
    fetchBankState()
      .then((s) => {
        setState(s)
        if (s.open) setRoot(s.root)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  const reload = useCallback(async () => {
    if (!open) return
    const res = await fetchBankSources({
      cls,
      tag,
      low: onlyLow,
      estimated: onlyEstimated,
      q: text,
      sort,
    })
    setRows(res.sources)
    setPicked((cur) => cur.filter((id) => res.sources.some((r) => r.id === id)))
  }, [open, cls, tag, onlyLow, onlyEstimated, text, sort])

  useEffect(() => {
    reload().catch((err: Error) => setError(err.message))
  }, [reload])

  const doOpen = async () => {
    setBusy(true)
    setError(null)
    try {
      setState(await openBank(root.trim()))
      setMessage(null)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const doDelete = async () => {
    if (!picked.length) return
    // 되돌릴 수 없다 — 화면이 먼저 확인을 받는다(브라우저 기본 대화상자는 로컬 도구에 충분하다)
    if (!confirm(`결함 조각 ${picked.length}개를 지웁니다. 되돌릴 수 없습니다.`)) return
    setBusy(true)
    try {
      const res = await deleteBankSources(picked)
      setState(res.state)
      setPicked([])
      setMessage(`${res.removed}개를 지웠습니다. (클래스 목록은 그대로 둡니다 — 출력 class id 순서가 바뀌면 안 됩니다)`)
      await reload()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const detail = useMemo(
    () => (picked.length === 1 ? rows.find((r) => r.id === picked[0]) : undefined),
    [picked, rows],
  )

  return (
    <>
      <div className="page-head">
        <h1>결함 보관함</h1>
        <p>모아 둔 결함 조각을 훑어보고, 마스크가 엉뚱한 것은 지웁니다.</p>
      </div>

      <div className="openbar">
        <label htmlFor="bank-root">보관함 폴더</label>
        <input
          id="bank-root"
          value={root}
          spellCheck={false}
          placeholder="예: bank/metal_nut"
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
            <h2>아직 연 보관함이 없습니다</h2>
            <p className="sub">`bank.yaml` 이 있는 폴더를 여세요.</p>
            <ol className="steps">
              <li>
                보유 YOLO 라벨에서 — <code>anograft bank import-yolo …</code>
              </li>
              <li>
                데이터가 없다면 — <code>anograft sample --out s --quickstart</code>
              </li>
            </ol>
          </section>
        </div>
      ) : (
        <div className="review">
          <aside className="review-side">
            <section className="card">
              <h2>고르기</h2>
              <p className="sub">{bankCountsText(state.total, rows.length, picked.length)}</p>
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
              {state.tags.length > 0 && (
                <label className="field">
                  <span>태그</span>
                  <select value={tag} onChange={(e) => setTag(e.target.value)}>
                    <option value="">전체</option>
                    {state.tags.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label className="field">
                <span>정렬</span>
                <select value={sort} onChange={(e) => setSort(e.target.value)}>
                  {state.sortKeys.map((k) => (
                    <option key={k} value={k}>
                      {SORT_LABELS[k] ?? k}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>찾기</span>
                <input
                  className="text-in"
                  value={text}
                  spellCheck={false}
                  placeholder="이름·원본"
                  onChange={(e) => setText(e.target.value)}
                />
              </label>
              <label className="check">
                <input type="checkbox" checked={onlyLow} onChange={(e) => setOnlyLow(e.target.checked)} />
                마스크 신뢰도 낮음만
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={onlyEstimated}
                  onChange={(e) => setOnlyEstimated(e.target.checked)}
                />
                박스에서 추정한 마스크만
              </label>
            </section>

            {state.holdout.leaked.length > 0 && (
              <div className="notice bad">
                <span>
                  <b>평가셋이 보관함에 들어와 있습니다 ({state.holdout.leaked.length}개).</b> 이대로
                  학습하면 성능 비교가 무의미해집니다. 아래 목록에서 골라 지우세요 —{' '}
                  {state.holdout.leaked.slice(0, 3).join(' · ')}
                  {state.holdout.leaked.length > 3 ? ' …' : ''}
                </span>
              </div>
            )}

            <section className="card">
              <h2>고른 것</h2>
              <p className="sub">
                {picked.length ? `${picked.length}개 골랐습니다.` : '타일을 눌러 고르세요.'}
              </p>
              <div className="row-btns">
                <button className="btn danger" disabled={!picked.length || busy} onClick={() => void doDelete()}>
                  지우기
                </button>
                <button className="btn" disabled={!picked.length} onClick={() => setPicked([])}>
                  선택 해제
                </button>
                <button className="btn" disabled={!rows.length} onClick={() => setPicked(rows.map((r) => r.id))}>
                  모두 고르기
                </button>
              </div>
            </section>
          </aside>

          <section className="review-main">
            {detail && (
              <div className="detail">
                <div className="detail-img">
                  <img src={bankImageUrl(detail.id, 'detail')} alt={`${detail.id} 결함 조각`} />
                </div>
                <div className="detail-side">
                  <dl className="kv">
                    <dt>번호</dt>
                    <dd>{detail.id}</dd>
                    <dt>크기</dt>
                    <dd>
                      {detail.size[1]} × {detail.size[0]} px
                    </dd>
                    <dt>면적</dt>
                    <dd>{detail.areaPx.toLocaleString('ko-KR')} px</dd>
                    <dt>마스크</dt>
                    <dd>{detail.estimated ? '박스에서 추정' : '정확'}</dd>
                    <dt>신뢰도</dt>
                    <dd>{confidenceText(detail.confidence)}</dd>
                    <dt>픽셀 크기</dt>
                    <dd>{detail.umPerPx ? `${detail.umPerPx} µm/px` : '미지정'}</dd>
                    <dt>원본</dt>
                    <dd>{detail.origin || '—'}</dd>
                    {detail.tags.length > 0 && (
                      <>
                        <dt>태그</dt>
                        <dd>{detail.tags.join(', ')}</dd>
                      </>
                    )}
                  </dl>
                  {detail.flags.length > 0 && (
                    <div className="notice" style={{ marginTop: 10 }}>
                      <span>{detail.flags.join(' · ')}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="grid">
              {rows.map((r) => (
                <button
                  key={r.id}
                  className={`tile${picked.includes(r.id) ? ' on' : ''}${r.lowConfidence ? ' low' : ''}`}
                  onClick={() => setPicked((cur) => toggleId(cur, r.id))}
                  title={`${r.id} · ${confidenceText(r.confidence)}`}
                >
                  <img src={bankImageUrl(r.id)} alt="" loading="lazy" decoding="async" />
                  <span className="tile-cap">
                    <b>{picked.includes(r.id) ? '✓' : '·'}</b> {r.name}
                    <em>{r.cls}</em>
                  </span>
                </button>
              ))}
              {!rows.length && <p className="muted">이 조건에 걸린 결함 조각이 없습니다.</p>}
            </div>
          </section>
        </div>
      )}
    </>
  )
}
