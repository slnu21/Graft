/**
 * 표시용 순수 함수 — 값을 **보여 주는 모양**으로만 바꾼다. 여기에 판단(무엇이 정상인가, 무엇을 경고할까)을
 * 두지 않는다. 그건 파이썬(`core`)이 하고 API 가 실어 나른다.
 *
 * vitest 가 검사하는 범위가 정확히 이 파일 같은 것들이다(비즈니스 로직은 계속 pytest).
 */

import type { MethodInfo } from './api'

export type StageInfo = { stage: string; label: string; desc: string; en: string }

export type StageGroup = StageInfo & { items: MethodInfo[]; unusable: number }

/** 스테이지 순서는 **서버가 준 순서**를 그대로 따른다(파이프라인 1~7). 프론트가 순서를 알 필요가 없다. */
export function groupByStage(stages: StageInfo[], methods: MethodInfo[]): StageGroup[] {
  return stages.map((s) => {
    const items = methods.filter((m) => m.stage === s.stage)
    return { ...s, items, unusable: items.filter((m) => !m.usable).length }
  })
}

/** doctor 값 한 칸 — 배열·불린·null 을 한국어 한 줄로. */
export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return value ? '예' : '아니요'
  if (Array.isArray(value)) return value.length ? value.map(String).join(' · ') : '(없음)'
  return String(value)
}

/** 긴 경로는 가운데를 접는다 — 카드 폭을 넘기면 레이아웃이 무너진다. */
export function shortenPath(path: string, max = 48): string {
  if (path.length <= max) return path
  const head = Math.ceil((max - 1) / 2)
  const tail = Math.floor((max - 1) / 2)
  return `${path.slice(0, head)}…${path.slice(path.length - tail)}`
}
