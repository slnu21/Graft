/**
 * 보관함 화면의 표시용 순수 함수. 필터·정렬은 **서버(파이썬)** 가 하고, 여기서는 고른 목록과 문구만 다룬다.
 */

/** 정렬 키 이름 — 서버가 키 목록(`sortKeys`)을 주고 화면이 한국어를 붙인다. */
export const SORT_LABELS: Record<string, string> = {
  id: '번호',
  confidence: '마스크 신뢰도',
  area: '면적',
  class: '클래스',
}

/** 고르기 토글 — 이미 있으면 빼고, 없으면 넣는다(순서 유지). */
export function toggleId(picked: string[], id: string): string[] {
  return picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]
}

/** "전체 120개 · 지금 보는 것 18개 · 고른 것 3개" */
export function bankCountsText(total: number, shown: number, picked: number): string {
  const parts = [`전체 ${total}개`]
  if (shown !== total) parts.push(`지금 보는 것 ${shown}개`)
  if (picked) parts.push(`고른 것 ${picked}개`)
  return parts.join(' · ')
}

/** 신뢰도 — `null` 은 "정확 마스크"라서 점수가 없는 것이지 나쁜 게 아니다. */
export function confidenceText(confidence: number | null): string {
  if (confidence === null || confidence === undefined) return '정확 마스크(점수 없음)'
  return `${confidence.toFixed(2)}${confidence < 0.5 ? ' — 낮음' : ''}`
}
