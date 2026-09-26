import { describe, expect, it } from 'vitest'

import type { LoopPoint, LoopRoundRow, LoopState } from './api'
import {
  CHART,
  NONE,
  actionTone,
  chart,
  deltaText,
  latestDelta,
  metricText,
  outcomeLabel,
  outcomeTone,
  pollDelay,
  progressPercent,
  rateText,
  shortHash,
  whenText,
} from './loop'

const point = (round: number, metric: number, over: Partial<LoopPoint> = {}): LoopPoint => ({
  round,
  metric,
  promoted: true,
  segment: 0,
  marker: '',
  ...over,
})

const row = (over: Partial<LoopRoundRow> = {}): LoopRoundRow => ({
  round: 3,
  at: '2026-09-26T14:03:00+09:00',
  metric: 0.41,
  metricName: 'mAP50',
  promoted: true,
  reason: '개선',
  intake: 6,
  sources: 137,
  snapshot: 'round-003/bank.snapshot.json',
  pipelineHash: 'aa40f71b9c',
  bootstrap: false,
  classes: 3,
  perClass: { blowhole: 41 },
  correctionRate: 0.34,
  drafted: 50,
  corrected: 17,
  marker: '',
  markerLabel: '',
  markerNote: '',
  ...over,
})

const openState = (over: Partial<Extract<LoopState, { open: true }>> = {}) =>
  ({
    open: true,
    configPath: 'loop.yaml',
    out: 'out/loop',
    bankPath: 'bank/mine',
    recipePath: 'r.yaml',
    fieldPath: 'field',
    trainer: 'yolo',
    metricName: 'mAP50',
    champion: null,
    round: null,
    review: { judged: 0, total: 0, waiting: false, queueDir: '' },
    trigger: null,
    breaker: null,
    tick: null,
    lock: null,
    lockText: '',
    processed: 0,
    action: { kind: 'run', text: '지금 돌 때입니다', command: 'anograft loop run' },
    corrections: { rate: null, drafted: 0, corrected: 0, text: '' },
    bank: { sources: 0, classes: [], perClass: {}, rows: [] },
    failures: [],
    warnings: [],
    ...over,
  }) as LoopState

describe('폴링', () => {
  it('돌고 있는 동안만 묻는다', () => {
    expect(pollDelay(openState({ lock: { pid: 1 } }))).toBe(3000)
    expect(pollDelay(openState())).toBe(0)
    expect(pollDelay(null)).toBe(0)
    expect(pollDelay({ open: false, suggest: '', example: '' })).toBe(0)
  })
})

describe('빈 값', () => {
  it('없는 지표는 0 이 아니라 —', () => {
    expect(metricText(null)).toBe(NONE)
    expect(metricText(0)).toBe('0.0000')
    expect(metricText(0.4123456, 3)).toBe('0.412')
  })

  it('분모가 없는 수정률은 모름', () => {
    expect(rateText(null)).toBe('모름')
    expect(rateText(0)).toBe('0%')
    expect(rateText(0.336)).toBe('34%')
  })

  it('모르는 증감은 빈 문자열', () => {
    expect(deltaText(null)).toBe('')
    expect(deltaText(0.02, 2)).toBe('▲ 0.02')
    expect(deltaText(-0.02, 2)).toBe('▼ 0.02')
    expect(deltaText(0, 2)).toBe('= 0.00')
  })
})

describe('마지막 증감', () => {
  it('같은 구간 안에서만 견준다', () => {
    expect(latestDelta([point(1, 0.3), point(2, 0.34)])).toBeCloseTo(0.04)
  })

  it('지점을 넘으면 견주지 않는다', () => {
    expect(latestDelta([point(2, 0.34), point(3, 0.21, { segment: 1 })])).toBeNull()
    expect(latestDelta([point(1, 0.3)])).toBeNull()
  })
})

describe('라운드 줄', () => {
  it('지점이 찍힌 라운드는 그것을 먼저 말한다', () => {
    expect(outcomeLabel(row({ markerLabel: '기준선 재설정' }))).toBe('기준선 재설정')
    expect(outcomeTone(row({ markerLabel: '기준선 재설정' }))).toBe('mark')
    expect(outcomeLabel(row({ promoted: false }))).toBe('유지')
    expect(outcomeTone(row({ promoted: false }))).toBe('hold')
    expect(outcomeLabel(row({ bootstrap: true }))).toBe('기준선')
  })

  it('해시는 앞 몇 자 · 스냅샷은 파일 이름만', () => {
    expect(shortHash('aa40f71b9c3d')).toBe('aa40f71b')
    expect(shortHash('round-003/bank.snapshot.json', 5)).toBe('bank.')
    expect(shortHash('')).toBe('')
  })

  it('시각은 짧게, 못 읽으면 원문 그대로', () => {
    expect(whenText('2026-09-26T14:03:00')).toBe('09-26 14:03')
    expect(whenText('언젠가')).toBe('언젠가')
    expect(whenText('')).toBe(NONE)
  })
})

describe('강조', () => {
  it('자동 정지가 가장 세다', () => {
    expect(actionTone('breaker')).toBe('bad')
    expect(actionTone('review')).toBe('warn')
    expect(actionTone('run')).toBe('ok')
    expect(actionTone('running')).toBe('muted')
  })

  it('판정 진행률', () => {
    expect(progressPercent(2, 5)).toBe(40)
    expect(progressPercent(0, 0)).toBe(0)
    expect(progressPercent(9, 5)).toBe(100)
  })
})

describe('추이 그래프', () => {
  it('x 는 순서로 고르게 · y 는 값이 클수록 위', () => {
    const c = chart([point(1, 0.3), point(2, 0.4), point(5, 0.5)])
    expect(c.dots.map((d) => d.round)).toEqual([1, 2, 5])
    expect(c.dots[0].x).toBeCloseTo(CHART.left)
    expect(c.dots[2].x).toBeCloseTo(CHART.width - CHART.right)
    expect(c.dots[0].y).toBeGreaterThan(c.dots[2].y)
    expect(c.lines).toHaveLength(1)
  })

  it('지점에서 선이 끊긴다', () => {
    const c = chart([
      point(1, 0.3),
      point(2, 0.34),
      point(3, 0.21, { segment: 1 }),
      point(4, 0.25, { segment: 1 }),
    ])
    expect(c.lines.map((l) => l.segment)).toEqual([0, 1])
    expect(c.lines[0].points.split(' ')).toHaveLength(2)
  })

  it('점이 하나면 선이 없고 가운데에 찍힌다', () => {
    const c = chart([point(1, 0.3)])
    expect(c.lines).toHaveLength(0)
    expect(c.dots[0].x).toBeCloseTo((CHART.left + CHART.width - CHART.right) / 2)
    expect(c.max).toBeGreaterThan(c.min) // 0 높이 그래프는 축이 무너진다
  })

  it('전부 같은 값이어도 그릴 수 있다', () => {
    const c = chart([point(1, 0.4), point(2, 0.4)])
    expect(c.max - c.min).toBeGreaterThan(0)
    expect(c.dots[0].y).toBeCloseTo(c.dots[1].y)
    expect(c.ticks).toHaveLength(3)
  })

  it('점이 없어도 죽지 않는다', () => {
    const c = chart([])
    expect(c.dots).toHaveLength(0)
    expect(c.lines).toHaveLength(0)
  })
})
