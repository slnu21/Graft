import { describe, expect, it } from 'vitest'

import type { FieldSpec, PerClassRow } from './api'
import {
  inBounds,
  inputText,
  modifiedText,
  parseInput,
  rotateOf,
  splitAdvanced,
  updateRow,
  variantKeys,
  widgetOf,
  withRangeEnd,
} from './params'

const spec = (over: Partial<FieldSpec> = {}): FieldSpec => ({
  name: 'feather_px',
  kind: 'int',
  value: 3,
  optional: false,
  enabled: true,
  choices: [],
  lo: 0,
  hi: null,
  loOpen: false,
  hiOpen: false,
  onDefault: null,
  title: '가장자리 흐림 (px)',
  tooltip: '가장자리 흐림 · feather_px',
  desc: '',
  advanced: false,
  baseline: 3,
  hasBaseline: true,
  modified: false,
  step: 1,
  slider: false,
  ...over,
})

describe('widgetOf', () => {
  it('kind 마다 위젯 하나', () => {
    expect(widgetOf(spec())).toBe('number')
    expect(widgetOf(spec({ kind: 'float' }))).toBe('number')
    expect(widgetOf(spec({ kind: 'range' }))).toBe('range')
    expect(widgetOf(spec({ kind: 'int_range' }))).toBe('range')
    expect(widgetOf(spec({ kind: 'bool' }))).toBe('bool')
    expect(widgetOf(spec({ kind: 'choice' }))).toBe('choice')
  })

  it('모르는 종류는 텍스트로 — 새 필드가 생겨도 폼이 죽지 않는다', () => {
    expect(widgetOf(spec({ kind: 'weird' as FieldSpec['kind'] }))).toBe('text')
  })
})

describe('inputText', () => {
  it('null 은 빈칸, 목록은 쉼표', () => {
    expect(inputText(null)).toBe('')
    expect(inputText(['a', 'b'])).toBe('a, b')
    expect(inputText(0.5)).toBe('0.5')
  })
})

describe('parseInput', () => {
  it('정수는 반올림, 실수는 그대로, 나머지는 문자열', () => {
    expect(parseInput(spec(), '3.6')).toBe(4)
    expect(parseInput(spec({ kind: 'float' }), '0.35')).toBe(0.35)
    expect(parseInput(spec({ kind: 'list' as FieldSpec['kind'] }), 'a, b')).toBe('a, b')
  })
})

describe('withRangeEnd', () => {
  it('한 쪽만 바꾼다', () => {
    const s = spec({ kind: 'range', value: [0.8, 1.25] })
    expect(withRangeEnd(s, 0, '0.9')).toEqual([0.9, 1.25])
    expect(withRangeEnd(s, 1, '1.5')).toEqual([0.8, 1.5])
  })

  it('숫자가 아니면 그 쪽을 그대로 둔다', () => {
    const s = spec({ kind: 'range', value: [0.8, 1.25] })
    expect(withRangeEnd(s, 1, '')).toEqual([0.8, 1.25])
  })

  it('정수 범위는 반올림', () => {
    const s = spec({ kind: 'int_range', value: [60, 95] })
    expect(withRangeEnd(s, 0, '70.4')).toEqual([70, 95])
  })
})

describe('inBounds', () => {
  it('닫힌 구간과 열린 구간', () => {
    expect(inBounds(spec({ lo: 0, hi: 10 }), 0)).toBe(true)
    expect(inBounds(spec({ lo: 0, hi: 10 }), -1)).toBe(false)
    expect(inBounds(spec({ lo: 0, hi: 10, loOpen: true }), 0)).toBe(false)
    expect(inBounds(spec({ lo: null, hi: null }), 1e9)).toBe(true)
    expect(inBounds(spec(), Number.NaN)).toBe(false)
  })
})

describe('splitAdvanced', () => {
  it('고급은 따로 모은다', () => {
    const { basic, advanced } = splitAdvanced([spec(), spec({ name: 'x', advanced: true })])
    expect(basic.map((f) => f.name)).toEqual(['feather_px'])
    expect(advanced.map((f) => f.name)).toEqual(['x'])
  })
})

describe('modifiedText', () => {
  it('0 이면 빈 문자열', () => {
    expect(modifiedText(0)).toBe('')
    expect(modifiedText(2)).toBe('바뀜 2')
  })
})

describe('variantKeys', () => {
  it('0 부터 n-1', () => {
    expect(variantKeys(3)).toEqual([0, 1, 2])
    expect(variantKeys(0)).toEqual([])
    expect(variantKeys(-1)).toEqual([])
  })
})

describe('per_class 표', () => {
  const rows: PerClassRow[] = [
    { cls: 'scratch', on: false, rotate: [-15, 15], flip: '', scale: null },
    { cls: 'dent', on: true, rotate: [-15, 15], flip: 'none', scale: [0.9, 1.1] },
  ]

  it('한 줄만 바꾼 새 목록을 만든다(제자리 수정 없음)', () => {
    const next = updateRow(rows, 'dent', { flip: 'horizontal' })
    expect(next[1].flip).toBe('horizontal')
    expect(rows[1].flip).toBe('none')
    expect(next[0]).toBe(rows[0])
  })

  it('회전은 한 쪽만, 숫자가 아니면 그대로', () => {
    expect(rotateOf(rows[1], 0, '-30')).toEqual([-30, 15])
    expect(rotateOf(rows[1], 1, 'x')).toEqual([-15, 15])
    expect(rotateOf(rows[1], 1, '')).toEqual([-15, 15]) // 빈칸이 0 이 되면 안 된다
  })
})
