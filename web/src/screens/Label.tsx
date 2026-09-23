import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ApiError,
  type LabelState,
  clearMask,
  fetchLabelState,
  labelImageUrl,
  openLabel,
  saveToBank,
  sendPolygon,
  sendStroke,
  undoLabel,
} from '../api'
import { TOOL_LABELS, type Tool, canvasToImage, statsText } from '../label'

/**
 * 결함 표시 — 결함 사진에 마스크를 그려 보관함에 넣는다.
 *
 * **그리기 연산은 전부 서버(파이썬)가 한다**(규약: 프론트는 API 호출만). 화면은 포인터를 따라
 * 미리보기 선만 긋고, 획이 끝날 때 점 목록을 한 번 보낸다 — 점마다 보내면 요청이 폭주한다.
 * 그래서 브러시 결과가 Qt 라벨 탭과 **정확히 같다**.
 */
export function Label() {
  const [state, setState] = useState<LabelState | null>(null)
  const [path, setPath] = useState('')
  const [tool, setTool] = useState<Tool>('brush')
  const [radius, setRadius] = useState(10)
  const [cls, setCls] = useState('')
  const [bank, setBank] = useState('')
  const [version, setVersion] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const wrapRef = useRef<HTMLDivElement>(null)
  const drawRef = useRef<HTMLCanvasElement>(null)
  const points = useRef<[number, number][]>([])
  const drawing = useRef(false)

  const open = state?.open === true

  useEffect(() => {
    fetchLabelState()
      .then((s) => {
        setState(s)
        if (s.open) setPath(s.path)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  const apply = useCallback((next: LabelState) => {
    setState(next)
    setVersion((v) => v + 1) // 마스크가 바뀌었으니 그림을 다시 받는다
  }, [])

  const doOpen = async () => {
    setBusy(true)
    setError(null)
    try {
      apply(await openLabel(path.trim()))
      setMessage(null)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const run = useCallback(
    async (fn: () => Promise<LabelState>) => {
      try {
        apply(await fn())
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err))
      }
    },
    [apply],
  )

  // ------------------------------------------------------------ 포인터
  const toImage = (e: React.PointerEvent) => {
    const el = wrapRef.current
    if (!el || !state?.open) return null
    const rect = el.getBoundingClientRect()
    return canvasToImage(
      e.clientX - rect.left,
      e.clientY - rect.top,
      rect.width,
      rect.height,
      state.width,
      state.height,
    )
  }

  const onDown = (e: React.PointerEvent) => {
    const pt = toImage(e)
    if (!pt) return
    drawing.current = true
    points.current = [pt]
    e.currentTarget.setPointerCapture(e.pointerId)
    paintPreview()
  }

  const onMove = (e: React.PointerEvent) => {
    if (!drawing.current) return
    const pt = toImage(e)
    if (!pt) return
    points.current.push(pt)
    paintPreview()
  }

  const onUp = async () => {
    if (!drawing.current) return
    drawing.current = false
    const pts = points.current
    points.current = []
    clearPreview()
    if (!pts.length || !state?.open) return
    if (tool === 'polygon') {
      if (pts.length < 3) return
      await run(() => sendPolygon(pts))
    } else {
      await run(() => sendStroke(pts, radius, tool === 'eraser'))
    }
  }

  /** 미리보기는 **화면에만** 그린다 — 진짜 마스크는 서버가 들고 있다. */
  const paintPreview = () => {
    const canvas = drawRef.current
    const el = wrapRef.current
    if (!canvas || !el || !state?.open) return
    const rect = el.getBoundingClientRect()
    canvas.width = Math.round(rect.width)
    canvas.height = Math.round(rect.height)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const sx = rect.width / state.width
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.strokeStyle = tool === 'eraser' ? '#ffffff' : '#D8294A'
    ctx.fillStyle = ctx.strokeStyle
    ctx.globalAlpha = 0.55
    ctx.lineWidth = Math.max(1, radius * 2 * sx)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    points.current.forEach(([x, y], i) => {
      const px = x * sx
      const py = y * sx
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    })
    if (tool === 'polygon') {
      ctx.closePath()
      ctx.fill()
    } else {
      ctx.stroke()
    }
  }

  const clearPreview = () => {
    const canvas = drawRef.current
    const ctx = canvas?.getContext('2d')
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return
      const k = e.key.toLowerCase()
      if (k === 'b') setTool('brush')
      else if (k === 'e') setTool('eraser')
      else if (k === 'p') setTool('polygon')
      else if (k === 'z' && (e.ctrlKey || e.metaKey)) void run(() => undoLabel(e.shiftKey))
      else if (k === '[') setRadius((r) => Math.max(1, r - 2))
      else if (k === ']') setRadius((r) => Math.min(80, r + 2))
      else return
      e.preventDefault()
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [run])

  return (
    <>
      <div className="page-head">
        <h1>결함 표시</h1>
        <p>결함 사진에 영역을 칠해 보관함에 넣습니다. 칠한 모양이 그대로 정답 영역이 됩니다.</p>
      </div>

      <div className="openbar">
        <label htmlFor="label-path">결함 사진</label>
        <input
          id="label-path"
          value={path}
          spellCheck={false}
          placeholder="예: defects/img_0007.png"
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && void doOpen()}
        />
        <button className="btn primary" onClick={() => void doOpen()} disabled={busy || !path.trim()}>
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
            <h2>아직 연 사진이 없습니다</h2>
            <p className="sub">결함이 찍힌 사진 한 장을 여세요. 마스크가 이미 있으면 함께 불러옵니다.</p>
          </section>
        </div>
      ) : (
        <div className="review">
          <aside className="review-side">
            <section className="card">
              <h2>도구</h2>
              <p className="sub">단축키 B 붓 · E 지우개 · P 다각형 · Ctrl+Z 되돌리기</p>
              <div className="row-btns">
                {(['brush', 'eraser', 'polygon'] as Tool[]).map((t) => (
                  <button
                    key={t}
                    className={`btn${tool === t ? ' primary' : ''}`}
                    onClick={() => setTool(t)}
                  >
                    {TOOL_LABELS[t]}
                  </button>
                ))}
              </div>
              <label className="field" style={{ marginTop: 8 }}>
                <span>붓 굵기</span>
                <input
                  type="range"
                  min={1}
                  max={80}
                  value={radius}
                  onChange={(e) => setRadius(Number(e.target.value))}
                />
                <span className="mono">{radius}px</span>
              </label>
              <div className="row-btns" style={{ marginTop: 6 }}>
                <button className="btn" disabled={!state.canUndo} onClick={() => void run(() => undoLabel(false))}>
                  되돌리기
                </button>
                <button className="btn" disabled={!state.canRedo} onClick={() => void run(() => undoLabel(true))}>
                  다시 하기
                </button>
                <button className="btn" onClick={() => void run(clearMask)}>
                  전부 지우기
                </button>
              </div>
            </section>

            <section className="card">
              <h2>지금 칠한 것</h2>
              <p className="sub">{statsText(state.stats)}</p>
            </section>

            <section className="card">
              <h2>보관함에 저장</h2>
              <p className="sub">칠한 영역이 결함 조각 하나로 들어갑니다.</p>
              <label className="field">
                <span>보관함</span>
                <input
                  className="text-in"
                  value={bank}
                  spellCheck={false}
                  placeholder="bank/mine"
                  onChange={(e) => setBank(e.target.value)}
                />
              </label>
              <label className="field">
                <span>클래스</span>
                <input
                  className="text-in"
                  value={cls}
                  spellCheck={false}
                  placeholder="scratch"
                  onChange={(e) => setCls(e.target.value)}
                />
              </label>
              <button
                className="btn primary"
                disabled={busy || !bank.trim() || !cls.trim() || !state.stats.areaPx}
                onClick={() => {
                  setBusy(true)
                  saveToBank(bank.trim(), cls.trim())
                    .then((r) =>
                      setMessage(
                        `보관함에 넣었습니다 — ${r.added.map((a) => a.id).join(', ') || '(없음)'}${
                          r.warnings.length ? ` · ${r.warnings[0]}` : ''
                        }`,
                      ),
                    )
                    .catch((err: Error) => setError(err.message))
                    .finally(() => setBusy(false))
                }}
              >
                보관함에 저장
              </button>
            </section>
          </aside>

          <section className="review-main">
            <div
              className="canvas-wrap"
              ref={wrapRef}
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={() => void onUp()}
              onPointerCancel={() => void onUp()}
              style={{ aspectRatio: `${state.width} / ${state.height}` }}
            >
              <img src={labelImageUrl('base')} alt="결함 사진" draggable={false} />
              <img
                className="mask-layer"
                src={labelImageUrl('overlay', version)}
                alt=""
                draggable={false}
              />
              <canvas className="preview-layer" ref={drawRef} />
            </div>
            <p className="muted" style={{ marginTop: 6, fontSize: 11.5 }}>
              {state.width} × {state.height} px · 칠한 모양은 서버가 들고 있습니다(명령줄·기존 창과 같은 마스크).
            </p>
          </section>
        </div>
      )}
    </>
  )
}
