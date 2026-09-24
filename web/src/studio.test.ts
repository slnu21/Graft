import { describe, expect, it } from 'vitest'

import type { PreviewMeta } from './api'
import { cleanSeed, scaleNote, statusLine, suspicions, wipeClip, wipeFromPointer } from './studio'

const meta = (over: Partial<PreviewMeta> = {}): PreviewMeta => ({
  version: 1,
  status: 'ok',
  reason: null,
  elapsedMs: 132.4,
  scale: 0.5,
  width: 1024,
  height: 683,
  longSide: 1024,
  target: { index: 0, name: 'plate_000.png' },
  variant: 0,
  seed: 7,
  defects: [{ cls: 'stain', sourceId: 'a1', blend: 'poisson' }],
  instances: [],
  warnings: [],
  stageWarnings: {},
  lowConfidence: [],
  flipped: [],
  hasGt: true,
  hasRoi: true,
  ...over,
})

describe('wipeFromPointer', () => {
  it('폭 기준 0~1 로 자른다', () => {
    expect(wipeFromPointer(50, 200)).toBe(0.25)
    expect(wipeFromPointer(-10, 200)).toBe(0)
    expect(wipeFromPointer(999, 200)).toBe(1)
  })

  it('폭이 0 이면 가운데', () => {
    expect(wipeFromPointer(10, 0)).toBe(0.5)
  })
})

describe('wipeClip', () => {
  it('왼쪽을 잘라 오른쪽(합성)만 남긴다', () => {
    expect(wipeClip(0.5)).toBe('inset(0 0 0 50%)')
    expect(wipeClip(0)).toBe('inset(0 0 0 0%)')
    expect(wipeClip(2)).toBe('inset(0 0 0 100%)')
  })
})

describe('scaleNote', () => {
  it('축소본이면 배율과 "실제 결과는 일괄 생성에서" 를 말한다', () => {
    expect(scaleNote(meta())).toBe(
      '1024 × 683 px · 원본의 50% 축소본 — 실제 결과는 일괄 생성(원본 해상도)에서',
    )
  })

  it('원본 크기면 축소 문구를 빼고, 아직 없으면 규약 문구만', () => {
    expect(scaleNote(meta({ scale: 1 }))).toBe('1024 × 683 px · 원본 크기 그대로')
    expect(scaleNote(null)).toContain('1024px 축소본')
  })
})

describe('statusLine', () => {
  it('무엇이 붙었는지와 걸린 시간', () => {
    expect(statusLine(meta())).toBe('stain a1 (poisson) · 132 ms')
  })

  it('건너뛴 장은 이유를 먼저', () => {
    expect(statusLine(meta({ status: 'skipped', reason: 'ROI 없음' }))).toBe('건너뜀 — ROI 없음')
  })
})

describe('suspicions', () => {
  it('저신뢰 조각과 빛 뒤집힘을 각각 한 줄로', () => {
    const s = suspicions(meta({ lowConfidence: ['s1'], flipped: [{ n: 1, cls: 'dent' }] }))
    expect(s).toHaveLength(2)
    expect(s[0]).toContain('s1')
    expect(s[1]).toContain('dent#1')
  })

  it('건너뛴 장에는 붙이지 않는다', () => {
    expect(suspicions(meta({ status: 'skipped', lowConfidence: ['s1'] }))).toEqual([])
  })
})

describe('cleanSeed', () => {
  it('음수·소수·빈칸은 null', () => {
    expect(cleanSeed('7')).toBe(7)
    expect(cleanSeed('-1')).toBeNull()
    expect(cleanSeed('1.5')).toBeNull()
    expect(cleanSeed('')).toBeNull()
  })
})
