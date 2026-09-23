import { describe, expect, it } from 'vitest'

import type { MethodInfo } from './api'
import { formatValue, groupByStage, shortenPath, type StageInfo } from './format'

const stages: StageInfo[] = [
  { stage: 'source', label: '결함 고르기', desc: '', en: 'Source' },
  { stage: 'blend', label: '붙이기', desc: '', en: 'Blend' },
]

const methods: MethodInfo[] = [
  { stage: 'source', method: 'bank', usable: true, reason: null },
  { stage: 'blend', method: 'poisson', usable: true, reason: null },
  { stage: 'blend', method: 'multiband', usable: false, reason: 'scipy 없음' },
]

describe('groupByStage', () => {
  it('서버가 준 스테이지 순서를 그대로 지킨다', () => {
    expect(groupByStage(stages, methods).map((g) => g.stage)).toEqual(['source', 'blend'])
  })

  it('못 쓰는 method 수를 센다', () => {
    const [source, blend] = groupByStage(stages, methods)
    expect(source.unusable).toBe(0)
    expect(blend.unusable).toBe(1)
    expect(blend.items).toHaveLength(2)
  })

  it('method 가 없는 스테이지도 빈 채로 남긴다 — 화면에서 사라지면 안 된다', () => {
    const extra = [...stages, { stage: 'gtmask', label: '정답 영역', desc: '', en: 'GT mask' }]
    const groups = groupByStage(extra, methods)
    expect(groups).toHaveLength(3)
    expect(groups[2].items).toEqual([])
  })
})

describe('formatValue', () => {
  it('null·불린·배열을 한국어로', () => {
    expect(formatValue(null)).toBe('—')
    expect(formatValue(undefined)).toBe('—')
    expect(formatValue(true)).toBe('예')
    expect(formatValue(false)).toBe('아니요')
    expect(formatValue([])).toBe('(없음)')
    expect(formatValue(['a', 'b'])).toBe('a · b')
    expect(formatValue(4)).toBe('4')
  })

  it('0·빈 문자열을 없는 값으로 취급하지 않는다', () => {
    expect(formatValue(0)).toBe('0')
    expect(formatValue('')).toBe('')
  })
})

describe('shortenPath', () => {
  it('짧으면 그대로', () => {
    expect(shortenPath('C:/graft/out')).toBe('C:/graft/out')
  })

  it('길면 가운데를 접고 길이를 지킨다', () => {
    const long = 'C:/Users/raltl/Documents/Workspace/claude/Graft/out/set-A/images/plate_000.png'
    const short = shortenPath(long, 30)
    expect(short).toHaveLength(30)
    expect(short).toContain('…')
    expect(short.startsWith('C:/Users/')).toBe(true)
    expect(short.endsWith('.png')).toBe(true)
  })
})
