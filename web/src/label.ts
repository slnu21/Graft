/**
 * 결함 표시 화면의 순수 함수 — **좌표 변환과 문구만**. 마스크 연산은 전부 서버(파이썬)가 한다.
 */

export type Tool = 'brush' | 'eraser' | 'polygon'

export const TOOL_LABELS: Record<Tool, string> = {
  brush: '붓',
  eraser: '지우개',
  polygon: '다각형',
}

/**
 * 화면 좌표 → **원본 픽셀** 좌표. 캔버스는 원본 비율 그대로 늘어나므로 한 배율이면 된다.
 * 서버는 원본 기준 좌표만 받는다 — 화면 크기를 서버가 알 필요가 없다(창을 줄여도 결과가 같다).
 */
export function canvasToImage(
  x: number,
  y: number,
  viewW: number,
  viewH: number,
  imageW: number,
  imageH: number,
): [number, number] {
  if (viewW <= 0 || viewH <= 0) return [0, 0]
  const sx = imageW / viewW
  const sy = imageH / viewH
  return [clamp(x * sx, 0, imageW - 1), clamp(y * sy, 0, imageH - 1)]
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

export type LabelStats = {
  areaPx: number
  areaRatio: number
  components: number
  lengthPx: number
  contrast: number | null
  lightingDeg: number | null
  lightingWord: string
}

/** 한 줄 요약 — 아직 아무것도 안 칠했으면 그걸 먼저 말한다. */
export function statsText(stats: LabelStats): string {
  if (!stats.areaPx) return '아직 칠한 곳이 없습니다.'
  const parts = [
    `${stats.areaPx.toLocaleString('ko-KR')} px`,
    `${(stats.areaRatio * 100).toFixed(2)}%`,
    `긴 변 ${Math.round(stats.lengthPx)} px`,
  ]
  if (stats.components > 1) parts.push(`덩어리 ${stats.components}개`)
  if (stats.contrast !== null) parts.push(`밝기 차 ${stats.contrast.toFixed(0)}`)
  if (stats.lightingWord) parts.push(`밝은 쪽 ${stats.lightingWord}`)
  return parts.join(' · ')
}
