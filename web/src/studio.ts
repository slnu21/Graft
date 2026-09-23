/**
 * 미리보기 화면의 **표시용 순수 함수**. 합성·판정은 전부 파이썬이 한다 — 여기 있는 것은 문구와 좌표뿐이고
 * vitest 가 검사하는 범위도 정확히 이만큼이다(비즈니스 로직은 계속 pytest).
 */

import type { PreviewMeta, StudioTarget } from './api'

/** 와이프 손잡이 위치(0~1) — 캔버스 폭 기준. 밖으로 나가면 끝에 붙인다. */
export function wipeFromPointer(x: number, width: number): number {
  if (width <= 0) return 0.5
  return Math.min(1, Math.max(0, x / width))
}

/**
 * 오른쪽(합성)만 보이게 자르는 `clip-path`. 왼쪽은 원본이 그대로 드러난다 —
 * **합성을 원본 위에 얹고 잘라내는** 방식이라 두 장의 크기가 어긋날 수 없다.
 */
export function wipeClip(wipe: number): string {
  const pct = Math.round(Math.min(1, Math.max(0, wipe)) * 1000) / 10
  return `inset(0 0 0 ${pct}%)`
}

/**
 * 미리보기 해상도 문구 — **상시 표시**가 규약이다(축소본이라 배치 좌표·Poisson 결과가 원본과 다르다).
 */
export function scaleNote(meta: PreviewMeta | null): string {
  if (!meta) return '미리보기는 긴 변 1024px 축소본입니다'
  const size = `${meta.width} × ${meta.height} px`
  if (meta.scale >= 1) return `${size} · 원본 크기 그대로`
  return `${size} · 원본의 ${Math.round(meta.scale * 100)}% 축소본 — 실제 결과는 일괄 생성(원본 해상도)에서`
}

/** 상태 한 줄: 무엇이 붙었나 · 얼마나 걸렸나. 건너뛴 장은 이유를 먼저 말한다. */
export function statusLine(meta: PreviewMeta | null): string {
  if (!meta) return '아직 만들지 않았습니다'
  if (meta.status !== 'ok') return `건너뜀 — ${meta.reason ?? '이유 없음'}`
  const what = meta.defects.map((d) => `${d.cls} ${d.sourceId} (${d.blend})`).join(' · ')
  return `${what || '결함 없음'} · ${Math.round(meta.elapsedMs)} ms`
}

/** 조각 신뢰도·빛 방향처럼 "그림은 그럴듯한데 의심스러운" 것들 — 경고와 따로 보여 준다. */
export function suspicions(meta: PreviewMeta | null): string[] {
  if (!meta || meta.status !== 'ok') return []
  const out: string[] = []
  if (meta.lowConfidence.length)
    out.push(
      `마스크 신뢰도가 낮은 조각: ${meta.lowConfidence.join(', ')} — 결함 보관함에서 다듬으세요`,
    )
  if (meta.flipped.length)
    out.push(
      `빛 방향이 뒤집힌 듯한 결함: ${meta.flipped
        .map((f) => `${f.cls}#${f.n}`)
        .join(', ')} — 프리셋 dent-graft(찍힘·덴트) 를 써 보세요`,
    )
  return out
}

/** 바탕 이름 한 줄 — 긴 파일명은 앞을 살린다(뒤 숫자만 다른 이름이 많다). */
export function targetLabel(t: StudioTarget): string {
  return t.name.length <= 22 ? t.name : `${t.name.slice(0, 20)}…`
}

/** 시드 입력 — 음수·빈칸은 서버가 거절하므로 화면에서 먼저 0 으로 잡는다. */
export function cleanSeed(text: string): number | null {
  if (!text.trim()) return null
  const n = Number(text)
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) return null
  return n
}
