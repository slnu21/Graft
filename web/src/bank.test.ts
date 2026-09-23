import { describe, expect, it } from 'vitest'

import { SORT_LABELS, bankCountsText, confidenceText, toggleId } from './bank'

describe('toggleId', () => {
  it('없으면 넣고 있으면 뺀다', () => {
    expect(toggleId([], 'a')).toEqual(['a'])
    expect(toggleId(['a', 'b'], 'a')).toEqual(['b'])
  })

  it('고른 순서를 지킨다', () => {
    expect(toggleId(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
  })
})

describe('bankCountsText', () => {
  it('필터가 없으면 전체만 말한다', () => {
    expect(bankCountsText(120, 120, 0)).toBe('전체 120개')
  })

  it('필터·선택이 있으면 덧붙인다', () => {
    expect(bankCountsText(120, 18, 3)).toBe('전체 120개 · 지금 보는 것 18개 · 고른 것 3개')
  })
})

describe('confidenceText', () => {
  it('null 은 정확 마스크 — 나쁜 점수가 아니다', () => {
    expect(confidenceText(null)).toBe('정확 마스크(점수 없음)')
  })

  it('낮으면 말해 준다', () => {
    expect(confidenceText(0.42)).toBe('0.42 — 낮음')
    expect(confidenceText(0.8)).toBe('0.80')
  })
})

describe('SORT_LABELS', () => {
  it('서버가 주는 키를 모두 덮는다 — 빠지면 화면에 영어가 샌다', () => {
    for (const key of ['id', 'confidence', 'area', 'class']) {
      expect(SORT_LABELS[key]).toBeTruthy()
    }
  })
})
