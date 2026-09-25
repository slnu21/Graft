import { describe, expect, it } from 'vitest'

import { canvasToImage, saveAction, saveHint, saveTitle, statsText } from './label'

describe('canvasToImage', () => {
  it('화면 좌표를 원본 픽셀로 되돌린다', () => {
    expect(canvasToImage(50, 25, 100, 50, 400, 200)).toEqual([200, 100])
  })

  it('원본 밖으로 나가지 않는다 — 서버가 받는 좌표는 언제나 그림 안이다', () => {
    expect(canvasToImage(1000, -10, 100, 50, 400, 200)).toEqual([399, 0])
  })

  it('아직 그려지지 않은 칸(폭 0)은 원점으로', () => {
    expect(canvasToImage(10, 10, 0, 0, 400, 200)).toEqual([0, 0])
  })
})

describe('statsText', () => {
  const base = {
    areaPx: 0,
    areaRatio: 0,
    components: 1,
    lengthPx: 0,
    contrast: null,
    lightingDeg: null,
    lightingWord: '',
  }

  it('아무것도 안 칠했으면 그것부터 말한다', () => {
    expect(statsText(base)).toBe('아직 칠한 곳이 없습니다.')
  })

  it('덩어리가 하나면 개수를 말하지 않는다', () => {
    const one = statsText({ ...base, areaPx: 120, areaRatio: 0.01, lengthPx: 20 })
    expect(one).toContain('120 px')
    expect(one).not.toContain('덩어리')
    expect(statsText({ ...base, areaPx: 120, components: 3 })).toContain('덩어리 3개')
  })
})

describe('보관함 조각 다듬기 (U7)', () => {
  const edit = { root: 'bank/mine', id: 'scratch/img0007-01' }

  it('편집 중이 아니면 새 조각을 더한다고 말한다', () => {
    expect(saveTitle(null)).toBe('보관함에 저장')
    expect(saveAction(null)).toBe('보관함에 저장')
    expect(saveHint(null)).toContain('결함 조각 하나로 들어갑니다')
  })

  it('편집 중이면 덮어쓴다는 것과 어느 조각인지를 말한다', () => {
    expect(saveTitle(edit)).toBe('보관함 조각 갱신')
    expect(saveAction(edit)).toBe('조각 갱신')
    const hint = saveHint(edit)
    expect(hint).toContain('scratch/img0007-01')
    expect(hint).toContain('덮어씁니다')
    // 되돌릴 수 없는 결과(추정 신뢰도가 지워진다)를 숨기지 않는다
    expect(hint).toContain('지워집니다')
  })
})
