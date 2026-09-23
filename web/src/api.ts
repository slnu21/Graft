/**
 * 백엔드 호출만 모아 둔다 — 화면 컴포넌트는 `fetch` 를 직접 쓰지 않는다.
 *
 * 규약: **프론트는 API 를 호출만 한다(합성·판정 로직 0).** 여기서 하는 일은 호출과 모양 확인뿐이고,
 * 판단이 필요하면 그건 파이썬 쪽(`core`)에 순수 함수로 들어가야 한다.
 */

export type Health = { name: string; version: string; ok: boolean }

export type MethodInfo = {
  stage: string
  method: string
  usable: boolean
  reason: string | null
  /** 한국어 표시명 — 용어의 한 원천은 `core/help.py` 라 서버가 실어 준다(프론트에 사전 사본을 두지 않는다). */
  label?: string
  summary?: string
}

export type StageMeta = { stage: string; label: string; desc: string; en: string }

export type Doctor = Record<string, unknown>

/** 서버가 준 오류를 화면에 그대로 보일 수 있게 감싼다(fail-soft — 한 카드가 못 떠도 나머지는 뜬다). */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { headers: { Accept: 'application/json' } })
  } catch {
    throw new ApiError('서버에 연결하지 못했습니다 — anograft serve 가 떠 있는지 확인하세요.', 0)
  }
  const body = (await res.json().catch(() => null)) as { error?: string } | null
  if (!res.ok) {
    throw new ApiError(body?.error ?? `요청이 실패했습니다 (${res.status})`, res.status)
  }
  return body as T
}

/**
 * 쓰기. `X-Graft-Request` 헤더가 CSRF 방어다 — 커스텀 헤더가 붙으면 브라우저가 단순 요청으로
 * 보내지 못하고 preflight 를 먼저 쏘며, 서버는 거기에 CORS 허용을 주지 않는다.
 */
async function post<T>(path: string, payload: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Graft-Request': '1' },
      body: JSON.stringify(payload ?? {}),
    })
  } catch {
    throw new ApiError('서버에 연결하지 못했습니다 — anograft serve 가 떠 있는지 확인하세요.', 0)
  }
  const body = (await res.json().catch(() => null)) as { error?: string } | null
  if (!res.ok) {
    throw new ApiError(body?.error ?? `요청이 실패했습니다 (${res.status})`, res.status)
  }
  return body as T
}

export const fetchHealth = () => get<Health>('/api/health')
export const fetchDoctor = () => get<Doctor>('/api/doctor')
export const fetchMethods = () =>
  get<{ methods: MethodInfo[]; stages: StageMeta[] }>('/api/methods')

// ---------------------------------------------------------------- 검수(U3)

export type FilterOption = { id: string; label: string }
export type ContrastHint = { cls: string; synthetic: number; real: number; n: number }

export type ReviewState =
  | { open: false }
  | {
      open: true
      root: string
      summary: string
      counts: Record<string, number>
      classes: string[]
      filters: FilterOption[]
      contrastHints: ContrastHint[]
      directional: string[]
    }

export type ReviewItem = {
  /** 목록에서 행을 구분하는 안정된 키 — 정상(normal) 행은 `index` 가 비어 있어서 따로 필요하다. */
  key: string
  index: string
  status: string
  verdict: string
  note: string
  classes: string[]
  areaPx: number
  blend: string
  fallback: boolean
  reason: string
  target: string
  warnings: string[]
}

export type ReviewDetail = ReviewItem & {
  image: string
  mask: string
  sidecar: string
  sourceIds: string[]
  instances: Record<string, unknown>[]
  appearance: Record<string, number[]>
  appearanceClasses: Record<string, string[]>
}

export type HistogramData = {
  key: string
  cls: string
  edges: number[]
  synthetic: number[]
  real: number[]
  log: boolean
  realLabel: string
  classOptions: string[]
}

export type PruneResult = {
  out: string
  kept: number
  dropped: number
  normals: number
  skipped: number
  files: number
  warnings: string[]
}

export const fetchReviewState = () => get<ReviewState>('/api/review/state')
export const openReview = (root: string) => post<ReviewState>('/api/review/open', { root })
export const fetchReviewItems = (filter: string, cls: string) =>
  get<{ items: ReviewItem[]; total: number }>(
    `/api/review/items?filter=${encodeURIComponent(filter)}&class=${encodeURIComponent(cls)}`,
  )
export const fetchReviewDetail = (index: string) =>
  get<ReviewDetail>(`/api/review/item?index=${encodeURIComponent(index)}`)
export const fetchHistogram = (key: string, cls: string) =>
  get<HistogramData>(
    `/api/review/histogram?key=${encodeURIComponent(key)}&class=${encodeURIComponent(cls)}`,
  )
export const setVerdict = (index: string, verdict: string) =>
  post<{ item: ReviewItem; counts: Record<string, number> }>('/api/review/verdict', {
    index,
    verdict,
  })
export const pruneDataset = (out: string, dropUnreviewed: boolean) =>
  post<PruneResult>('/api/review/prune', { out, dropUnreviewed })
export const writeReport = () => post<{ path: string }>('/api/review/report', {})

/** 이미지는 `<img src>` 로 직접 받는다 — fetch 로 가져와 base64 로 만들 이유가 없다. */
export const imageUrl = (index: string, kind: 'thumb' | 'detail' | 'mask' = 'thumb') =>
  `/api/review/image?index=${encodeURIComponent(index)}&kind=${kind}`

// ---------------------------------------------------------------- 결함 보관함(U4)

export type BankSourceRow = {
  id: string
  cls: string
  name: string
  areaPx: number
  maskOrigin: string
  estimated: boolean
  confidence: number | null
  lowConfidence: boolean
  flags: string[]
  tags: string[]
  umPerPx: number | null
  origin: string
  size: [number, number]
}

export type BankState =
  | { open: false }
  | {
      open: true
      root: string
      summary: string
      classes: string[]
      tags: string[]
      sortKeys: string[]
      total: number
      /** T4 — 평가셋이 은행에 섞였는지. `leaked` 가 있으면 라운드 비교가 무의미해진다. */
      holdout: { listed: number; leaked: string[] }
      directional: { cls: string; r: number }[]
    }

export type BankQuery = {
  cls?: string
  tag?: string
  low?: boolean
  estimated?: boolean
  q?: string
  sort?: string
  desc?: boolean
}

export const fetchBankState = () => get<BankState>('/api/bank/state')
export const openBank = (root: string) => post<BankState>('/api/bank/open', { root })
export const fetchBankSources = (query: BankQuery) => {
  const p = new URLSearchParams()
  if (query.cls) p.set('class', query.cls)
  if (query.tag) p.set('tag', query.tag)
  if (query.low) p.set('low', '1')
  if (query.estimated) p.set('estimated', '1')
  if (query.q) p.set('q', query.q)
  if (query.sort) p.set('sort', query.sort)
  if (query.desc) p.set('desc', '1')
  return get<{ sources: BankSourceRow[]; total: number }>(`/api/bank/sources?${p}`)
}
export const deleteBankSources = (ids: string[]) =>
  post<{ removed: number; state: BankState }>('/api/bank/delete', { ids })

/** 보관함 타일은 캐시하지 않는다(마스크를 다듬으면 그림이 바뀐다) — 버전 쿼리로 강제 갱신한다. */
export const bankImageUrl = (id: string, kind: 'tile' | 'detail' | 'mask' = 'tile', v = 0) =>
  `/api/bank/image?id=${encodeURIComponent(id)}&kind=${kind}${v ? `&v=${v}` : ''}`

// ---------------------------------------------------------------- 결함 표시(U4)

import type { LabelStats } from './label'

export type LabelState =
  | { open: false }
  | {
      open: true
      path: string
      width: number
      height: number
      canUndo: boolean
      canRedo: boolean
      stats: LabelStats
    }

export type SaveResult = {
  added: { id: string; cls: string; areaPx: number }[]
  warnings: string[]
  bank: string
}

export const fetchLabelState = () => get<LabelState>('/api/label/state')
export const openLabel = (path: string, mask = '') =>
  post<LabelState>('/api/label/open', { path, mask })
/** 좌표는 **원본 픽셀** 기준. 획이 끝날 때 한 번만 보낸다(점마다 보내면 요청이 폭주한다). */
export const sendStroke = (points: [number, number][], radius: number, erase: boolean) =>
  post<LabelState>('/api/label/stroke', { points, radius, erase })
export const sendPolygon = (points: [number, number][], erase = false) =>
  post<LabelState>('/api/label/polygon', { points, erase })
export const autoSelect = (box: [number, number, number, number], method = 'grabcut') =>
  post<LabelState & { method: string }>('/api/label/auto', { box, method })
export const undoLabel = (redo = false) => post<LabelState>('/api/label/undo', { redo })
export const clearMask = () => post<LabelState>('/api/label/clear', {})
export const saveToBank = (bank: string, cls: string, tags: string[] = [], umPerPx?: number) =>
  post<SaveResult>('/api/label/save', { bank, cls, tags, umPerPx })

/** 마스크는 그릴 때마다 바뀐다 — `v` 로 캐시를 깬다. */
export const labelImageUrl = (kind: 'base' | 'mask' | 'overlay', v = 0) =>
  `/api/label/image?kind=${kind}${v ? `&v=${v}` : ''}`
