import { describe, expect, it } from 'vitest'

import type { BatchRun, BatchSummary } from './api'
import {
  PHASE_LABELS,
  canReview,
  logTail,
  percent,
  perClassText,
  phaseOf,
  pollDelay,
  progressText,
  summaryLine,
} from './batch'

const summary = (over: Partial<BatchSummary> = {}): BatchSummary => ({
  root: 'out/set-a',
  nOk: 12,
  nSkipped: 1,
  nNormals: 4,
  nFallback: 0,
  perClass: { scratch: 9, pit: 3 },
  files: {},
  warnings: [],
  cancelled: false,
  done: 13,
  count: 13,
  text: '완료: ok 12 · skipped 1',
  ...over,
})

const run = (over: Partial<BatchRun> = {}): BatchRun => ({
  running: true,
  stopping: false,
  done: 3,
  total: 10,
  log: [],
  error: null,
  summary: null,
  ...over,
})

describe('phaseOf', () => {
  it('실행 상태를 한 단계로 접는다', () => {
    expect(phaseOf(null)).toBe('idle')
    expect(phaseOf(run())).toBe('running')
    expect(phaseOf(run({ stopping: true }))).toBe('stopping')
    expect(phaseOf(run({ running: false, summary: summary() }))).toBe('done')
    expect(phaseOf(run({ running: false, summary: summary({ cancelled: true }) }))).toBe(
      'cancelled',
    )
    expect(phaseOf(run({ running: false, error: '실패' }))).toBe('failed')
  })

  it('모든 단계에 한국어 이름이 있다', () => {
    for (const p of ['idle', 'running', 'stopping', 'done', 'cancelled', 'failed'] as const) {
      expect(PHASE_LABELS[p]).toBeTruthy()
    }
  })
})

describe('pollDelay', () => {
  it('도는 동안만 묻는다 — 끝나면 0(폴링 중지)', () => {
    expect(pollDelay('running')).toBe(400)
    expect(pollDelay('stopping')).toBe(400)
    expect(pollDelay('done')).toBe(0)
    expect(pollDelay('idle')).toBe(0)
    expect(pollDelay('failed')).toBe(0)
  })
})

describe('percent / progressText', () => {
  it('총 장수가 0 이면 0 — 나누지 않는다', () => {
    expect(percent(0, 0)).toBe(0)
    expect(percent(3, 10)).toBe(30)
    expect(percent(99, 10)).toBe(100)
  })

  it('진행 문장', () => {
    expect(progressText(run())).toBe('3 / 10 장 · 30%')
    expect(progressText(null)).toBe('아직 돌리지 않았습니다')
  })
})

describe('summaryLine / perClassText', () => {
  it('숫자는 서버가 센 것 그대로', () => {
    expect(summaryLine(summary())).toBe('성공 12 · 건너뜀 1 · 정상 4 → out/set-a')
    expect(summaryLine(summary({ nFallback: 2 }))).toContain('폴백 2')
    expect(summaryLine(null)).toBe('')
  })

  it('클래스별은 이름순, 없으면 빈 문자열', () => {
    expect(perClassText(summary())).toBe('pit 3 · scratch 9')
    expect(perClassText(summary({ perClass: {} }))).toBe('')
  })
})

describe('logTail', () => {
  it('마지막 n 줄만', () => {
    const lines = Array.from({ length: 10 }, (_, i) => `l${i}`)
    expect(logTail(lines, 3)).toEqual(['l7', 'l8', 'l9'])
    expect(logTail(lines, 99)).toHaveLength(10)
  })
})

describe('canReview', () => {
  it('쓴 장이 있어야 검수로 넘긴다', () => {
    expect(canReview(summary())).toBe(true)
    expect(canReview(summary({ nOk: 0 }))).toBe(false)
    expect(canReview(null)).toBe(false)
  })
})
