/**
 * 학습 루프 화면의 **표시용 순수 함수**(U8). 판정은 전부 파이썬이 냈다 — 승급·트리거·자동 정지·사람
 * 수정률·표 행·추이 점은 `loop/policy.py`·`loop/board.py` 가 이미 답을 준다. 여기서 하는 일은
 * 그 숫자를 **좌표와 문장**으로 바꾸는 것뿐이다(합성·판정 로직 0).
 *
 * 그래서 vitest 가 보는 것도 그 범위다: 좌표 계산 · 빈 값 표기 · 폴링 주기.
 */

import type { LoopPoint, LoopRoundRow, LoopState } from './api'

/** 값이 없을 때 쓰는 표시 — **0 과 구분되어야 한다**(모르는 것과 0 은 다르다). */
export const NONE = '—'

/**
 * 폴링 주기(ms). 0 = 폴링하지 않음.
 *
 * 루프는 이 서버 밖(스케줄러·CLI)에서 돌기 때문에 화면이 자동으로 따라가야 하는 때가 있다 —
 * **돌고 있는 동안**(잠금이 잡혀 있다)이다. 그 밖에는 사람이 새로고침하거나 화면을 다시 열 때
 * 읽으면 된다(상태 한 번에 보관함을 통째로 세므로 조용할 때까지 5초마다 묻지 않는다).
 */
export function pollDelay(state: LoopState | null): number {
  if (!state || !state.open) return 0
  return state.lock ? 3000 : 0
}

/** 지표 한 칸. 없으면 `—`(0 으로 그리지 않는다 — 학습 전에 멈춘 라운드가 "점수 0" 이 된다). */
export function metricText(value: number | null | undefined, digits = 4): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : NONE
}

/** 사람 수정률 — 분모가 0 이면 **모름**이다. */
export function rateText(rate: number | null | undefined): string {
  return typeof rate === 'number' ? `${Math.round(rate * 100)}%` : '모름'
}

/** 부호를 붙인 증감. `null` 이면 빈 문자열(모르는 것을 0 으로 쓰지 않는다). */
export function deltaText(delta: number | null | undefined, digits = 4): string {
  if (typeof delta !== 'number' || !Number.isFinite(delta)) return ''
  const sign = delta > 0 ? '▲' : delta < 0 ? '▼' : '='
  return `${sign} ${Math.abs(delta).toFixed(digits)}`
}

/**
 * 마지막 점과 그 앞 점의 차이 — **같은 구간 안에서만**. 지점(기준선 재설정·자동 정지 해제)을 넘어
 * 견주면 화면이 "떨어졌다"는 거짓말을 한다(설계 §2b.5(3)).
 */
export function latestDelta(points: LoopPoint[]): number | null {
  if (points.length < 2) return null
  const last = points[points.length - 1]
  const prev = points[points.length - 2]
  return prev.segment === last.segment ? last.metric - prev.metric : null
}

/** 판정 진행률 0~100. 총 장수를 모르면 0. */
export function progressPercent(judged: number, total: number): number {
  if (total <= 0) return 0
  return Math.min(100, Math.round((judged / total) * 100))
}

/** 라운드 한 줄의 결과 표시 — 지점이 찍힌 라운드는 그걸 먼저 말한다(그 뒤로 비교가 끊긴다). */
export function outcomeLabel(row: LoopRoundRow): string {
  if (row.markerLabel) return row.markerLabel
  if (row.bootstrap) return '기준선'
  return row.promoted ? '승급' : '유지'
}

export function outcomeTone(row: LoopRoundRow): 'ok' | 'hold' | 'mark' {
  if (row.markerLabel) return 'mark'
  return row.promoted ? 'ok' : 'hold'
}

/** 다음 할 일 줄의 강조. `kind` 는 서버가 정한다 — 여기서 다시 판단하지 않는다. */
export function actionTone(kind: string): 'ok' | 'warn' | 'bad' | 'muted' {
  if (kind === 'breaker') return 'bad'
  if (kind === 'review') return 'warn'
  if (kind === 'run') return 'ok'
  return 'muted'
}

/** 해시·스냅샷 파일명은 앞 몇 자만 — 표가 가로로 터진다. */
export function shortHash(value: string, n = 8): string {
  const text = (value ?? '').split('/').pop() ?? ''
  return text.length <= n ? text : text.slice(0, n)
}

/** ISO 시각 → `09-26 14:03`. 못 읽으면 원문 그대로(서버가 준 것을 버리지 않는다). */
export function whenText(iso: string): string {
  if (!iso) return NONE
  const at = new Date(iso)
  if (Number.isNaN(at.getTime())) return iso
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(at.getMonth() + 1)}-${pad(at.getDate())} ${pad(at.getHours())}:${pad(at.getMinutes())}`
}

// ---------------------------------------------------------------- 추이 그래프

export type ChartBox = {
  width: number
  height: number
  left: number
  right: number
  top: number
  bottom: number
}

export const CHART: ChartBox = { width: 640, height: 190, left: 44, right: 16, top: 16, bottom: 30 }

export type ChartLine = { segment: number; points: string }
export type ChartDot = {
  round: number
  metric: number
  promoted: boolean
  marker: string
  x: number
  y: number
}
export type ChartTick = { value: number; y: number }

export type Chart = {
  lines: ChartLine[]
  dots: ChartDot[]
  ticks: ChartTick[]
  min: number
  max: number
}

/**
 * 점 목록 → 좌표. x 는 **순서**로 고르게 놓는다(라운드 번호로 놓으면 지표 없는 라운드가 빈칸을 만든다).
 *
 * 선은 `segment` 가 같은 이웃끼리만 잇는다. 값이 하나뿐이거나 전부 같으면 위아래로 조금 벌려
 * 가운데 선으로 그린다(0 높이 그래프는 축이 무너진다).
 */
export function chart(points: LoopPoint[], box: ChartBox = CHART): Chart {
  const values = points.map((p) => p.metric)
  let min = values.length ? Math.min(...values) : 0
  let max = values.length ? Math.max(...values) : 1
  if (max - min < 1e-9) {
    min -= 0.05
    max += 0.05
  } else {
    const pad = (max - min) * 0.15
    min -= pad
    max += pad
  }
  const innerW = box.width - box.left - box.right
  const innerH = box.height - box.top - box.bottom
  const x = (i: number) =>
    points.length < 2 ? box.left + innerW / 2 : box.left + (innerW * i) / (points.length - 1)
  const y = (v: number) => box.top + innerH * (1 - (v - min) / (max - min))

  const dots: ChartDot[] = points.map((p, i) => ({
    round: p.round,
    metric: p.metric,
    promoted: p.promoted,
    marker: p.marker,
    x: x(i),
    y: y(p.metric),
  }))
  const lines: ChartLine[] = []
  let run: ChartDot[] = []
  let segment = points.length ? points[0].segment : 0
  const flush = () => {
    if (run.length >= 2) {
      lines.push({ segment, points: run.map((d) => `${round2(d.x)},${round2(d.y)}`).join(' ') })
    }
    run = []
  }
  points.forEach((p, i) => {
    if (p.segment !== segment) {
      flush()
      segment = p.segment
    }
    run.push(dots[i])
  })
  flush()

  const ticks: ChartTick[] = [min, (min + max) / 2, max].map((v) => ({ value: v, y: y(v) }))
  return { lines, dots, ticks, min, max }
}

function round2(v: number): number {
  return Math.round(v * 100) / 100
}
