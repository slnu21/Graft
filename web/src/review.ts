/**
 * 검수 화면의 **표시용 순수 함수** — 판정하지 않는다(그건 파이썬이 한다).
 * 여기 있는 것은 "막대를 어디에 그릴까", "다음으로 갈 항목은 무엇인가" 같은 화면 계산뿐이다.
 */

import type { HistogramData, ReviewItem } from './api'

/** 판정 표시 — 색각을 고려해 색만이 아니라 기호도 함께(사전 §3.3 채택/반려/미검수). */
export const VERDICT_MARK: Record<string, string> = {
  accept: '✓',
  reject: '✗',
  '': '·',
}

export const VERDICT_LABEL: Record<string, string> = {
  accept: '채택',
  reject: '반려',
  '': '미검수',
}

/** 판정 대상 — 합성 결과(`ok`)만. 정상 이미지·건너뜀은 눈으로만 보는 것이라 진행률에도 안 넣는다. */
export function judgeable(items: ReviewItem[]): ReviewItem[] {
  return items.filter((it) => it.status === 'ok')
}

/**
 * 판정 뒤 어디로 갈까 — **같은 자리에 머무르지 않고 다음 미판정으로** 간다(키보드 연타로 훑는 화면).
 * 뒤에 남은 게 없으면 앞쪽을 다시 본다. 전부 판정됐으면 `null`(머무른다).
 */
export function nextIndexAfter(items: ReviewItem[], current: string): string | null {
  const list = judgeable(items)
  const at = list.findIndex((it) => it.index === current)
  if (at < 0) return list[0]?.index ?? null
  const after = list.slice(at + 1).find((it) => !it.verdict)
  if (after) return after.index
  const before = list.slice(0, at).find((it) => !it.verdict)
  return before?.index ?? null
}

/** 진행률 — 미검수를 뺀 비율. 0~1. 항목이 없으면 0. */
export function progress(items: ReviewItem[]): number {
  const list = judgeable(items)
  if (!list.length) return 0
  return list.filter((it) => it.verdict).length / list.length
}

export type Bar = { x: number; w: number; hA: number; hB: number; label: string }

/**
 * 히스토그램 막대 좌표 — 두 계열을 **같은 구간**에서 세었으므로 같은 스케일로 그린다.
 * 높이는 0~1 비율이고 SVG 치수는 컴포넌트가 곱한다(뷰포트가 바뀌어도 이 함수는 그대로).
 */
export function bars(hist: HistogramData): Bar[] {
  const n = hist.synthetic.length
  if (!n) return []
  const peak = Math.max(1, ...hist.synthetic, ...hist.real)
  const w = 1 / n
  return hist.synthetic.map((a, i) => ({
    x: i * w,
    w,
    hA: a / peak,
    hB: (hist.real[i] ?? 0) / peak,
    label: formatEdge(hist.edges[i], hist.log),
  }))
}

/** 축 눈금 글자 — 로그 구간이면 반올림이 커야 읽힌다. */
export function formatEdge(value: number, log: boolean): string {
  if (!Number.isFinite(value)) return ''
  const abs = Math.abs(value)
  if (log || abs >= 1000) return Math.round(value).toLocaleString('ko-KR')
  if (abs >= 10) return value.toFixed(0)
  return value.toFixed(1)
}

/** 한 줄 요약 — 카드 부제에 쓴다. */
export function countsText(counts: Record<string, number>): string {
  const parts: string[] = []
  const add = (key: string, label: string) => {
    const n = counts[key]
    if (n) parts.push(`${label} ${n}`)
  }
  add('ok', '합성')
  add('normal', '정상')
  add('skipped', '건너뜀')
  add('accept', '채택')
  add('reject', '반려')
  return parts.join(' · ') || '없음'
}
