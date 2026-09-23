import { describe, expect, it } from 'vitest'

import type { HistogramData, ReviewItem } from './api'
import { bars, countsText, formatEdge, judgeable, nextIndexAfter, progress } from './review'

const item = (index: string, verdict = '', status = 'ok'): ReviewItem => ({
  key: index,
  index,
  status,
  verdict,
  note: '',
  classes: ['scratch'],
  areaPx: 100,
  blend: 'poisson',
  fallback: false,
  reason: '',
  target: 't.png',
  warnings: [],
})

describe('nextIndexAfter', () => {
  it('뒤쪽의 다음 미검수로 간다', () => {
    const items = [item('1', 'accept'), item('2'), item('3'), item('4')]
    expect(nextIndexAfter(items, '2')).toBe('3')
  })

  it('뒤가 다 판정됐으면 앞쪽 미검수로 돌아온다', () => {
    const items = [item('1'), item('2'), item('3', 'reject')]
    expect(nextIndexAfter(items, '2')).toBe('1')
  })

  it('전부 판정됐으면 null — 머무른다', () => {
    const items = [item('1', 'accept'), item('2', 'reject')]
    expect(nextIndexAfter(items, '1')).toBeNull()
  })

  it('목록에 없는 index 면 첫 항목으로', () => {
    expect(nextIndexAfter([item('1'), item('2')], '없음')).toBe('1')
    expect(nextIndexAfter([], '없음')).toBeNull()
  })
})

describe('judgeable', () => {
  it('정상·건너뜀은 판정 대상이 아니다 — 진행률도 그것으로 센다', () => {
    const items = [item('1', 'accept'), item('', '', 'normal'), item('2', '', 'skipped')]
    expect(judgeable(items).map((i) => i.index)).toEqual(['1'])
    expect(progress(items)).toBe(1)
    expect(nextIndexAfter(items, '1')).toBeNull()
  })
})

describe('progress', () => {
  it('판정한 비율', () => {
    expect(progress([item('1', 'accept'), item('2')])).toBe(0.5)
    expect(progress([])).toBe(0)
    expect(progress([item('1', 'reject')])).toBe(1)
  })
})

describe('bars', () => {
  const hist: HistogramData = {
    key: 'area',
    cls: '',
    edges: [0, 10, 20, 30],
    synthetic: [2, 4, 0],
    real: [1, 0, 8],
    log: false,
    realLabel: '실제',
    classOptions: [],
  }

  it('두 계열을 같은 최댓값으로 정규화한다 — 따로 정규화하면 비교가 거짓말이 된다', () => {
    const out = bars(hist)
    expect(out).toHaveLength(3)
    expect(out[1].hA).toBeCloseTo(4 / 8)
    expect(out[2].hB).toBe(1)
  })

  it('막대 폭이 1을 채운다', () => {
    const out = bars(hist)
    expect(out[0].x).toBe(0)
    expect(out.at(-1)!.x + out.at(-1)!.w).toBeCloseTo(1)
  })

  it('빈 히스토그램은 빈 배열', () => {
    expect(bars({ ...hist, synthetic: [], real: [], edges: [] })).toEqual([])
  })

  it('전부 0 이어도 나눗셈이 터지지 않는다', () => {
    const out = bars({ ...hist, synthetic: [0, 0, 0], real: [0, 0, 0] })
    expect(out.every((b) => b.hA === 0 && b.hB === 0)).toBe(true)
  })
})

describe('formatEdge', () => {
  it('크기에 따라 자릿수를 바꾼다', () => {
    expect(formatEdge(1234, false)).toBe('1,234')
    expect(formatEdge(42, false)).toBe('42')
    expect(formatEdge(4.25, false)).toBe('4.3')
    expect(formatEdge(Number.NaN, false)).toBe('')
  })
})

describe('countsText', () => {
  it('0 인 항목은 빼고 잇는다', () => {
    expect(countsText({ ok: 4, normal: 2, skipped: 0, accept: 1 })).toBe('합성 4 · 정상 2 · 채택 1')
    expect(countsText({})).toBe('없음')
  })
})
