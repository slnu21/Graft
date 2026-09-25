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

export type StageMeta = {
  stage: string
  label: string
  desc: string
  en: string
}

export type Doctor = Record<string, unknown>

/**
 * 서버가 준 오류를 화면에 그대로 보일 수 있게 감싼다(fail-soft — 한 카드가 못 떠도 나머지는 뜬다).
 * `body` 에는 응답 JSON 이 그대로 들어 있다 — 카드 편집 실패는 `stage`·`name` 으로 **어느 행인지**를 알려준다.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body: { stage?: string; name?: string } = {},
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
  const body = (await res.json().catch(() => null)) as {
    error?: string
    stage?: string
    name?: string
  } | null
  if (!res.ok) {
    throw new ApiError(body?.error ?? `요청이 실패했습니다 (${res.status})`, res.status, body ?? {})
  }
  return body as T
}

/**
 * 쓰기. `X-Graft-Request` 헤더가 CSRF 방어다 — 커스텀 헤더가 붙으면 브라우저가 단순 요청으로
 * 보내지 못하고 preflight 를 먼저 쏘며, 서버는 거기에 CORS 허용을 주지 않는다.
 */
async function post<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Graft-Request': '1' },
      body: JSON.stringify(payload ?? {}),
      signal,
    })
  } catch (err) {
    // 취소는 오류가 아니다 — 부른 쪽이 `AbortError` 를 보고 조용히 넘긴다(최신 요청만 살린다).
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError('서버에 연결하지 못했습니다 — anograft serve 가 떠 있는지 확인하세요.', 0)
  }
  const body = (await res.json().catch(() => null)) as {
    error?: string
    stage?: string
    name?: string
  } | null
  if (!res.ok) {
    throw new ApiError(body?.error ?? `요청이 실패했습니다 (${res.status})`, res.status, body ?? {})
  }
  return body as T
}

export const fetchHealth = () => get<Health>('/api/health')
export const fetchDoctor = () => get<Doctor>('/api/doctor')
export const fetchMethods = () =>
  get<{ methods: MethodInfo[]; stages: StageMeta[] }>('/api/methods')

// ---------------------------------------------------------------- 검수(U3)

export type FilterOption = { id: string; label: string }
export type ContrastHint = {
  cls: string
  synthetic: number
  real: number
  n: number
}

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

// ---------------------------------------------------------------- 미리보기(U5a)

export type StageRow = {
  stage: string
  label: string
  method: string
  methodLabel: string
}

export type StudioRecipe = {
  name: string
  preset: string
  seed: number
  bank: string
  targets: string
  out: string
  count: number
  writer: string
  defectsPerImage: number[]
}

export type StudioTarget = { index: number; name: string; path: string }

export type StudioState =
  | { open: false; presets: string[] }
  | {
      open: true
      recipe: StudioRecipe
      recipePath: string
      summary: Record<string, string>
      stages: StageRow[]
      presets: string[]
      warnings: string[]
      pathNotes: string[]
      longSide: number
      targetIndex: number
      targets: StudioTarget[]
      bankName: string
      bankN: number
      classes: string[]
      pipelineHash: string
      runCommand: string
      version: number
    }

export type PreviewDefect = { cls: string; sourceId: string; blend: string }
export type PreviewInstance = {
  cls: string
  classId: number
  bbox: number[]
  areaPx: number
}

export type PreviewMeta = {
  version: number
  status: 'ok' | 'skipped'
  reason: string | null
  elapsedMs: number
  /** 축소 배율(1 = 원본). **미리보기는 긴 변 1024 축소본**이라 원본 해상도 `run` 과 결과가 다르다. */
  scale: number
  width: number
  height: number
  longSide: number
  target: { index: number; name: string }
  variant: number
  seed: number
  defects: PreviewDefect[]
  instances: PreviewInstance[]
  warnings: string[]
  /** 카드별로 나눈 경고 — 고르는 규칙(`roi:` 는 배치 카드)은 서버가 안다. */
  stageWarnings: Record<string, string[]>
  lowConfidence: string[]
  flipped: { n: number; cls: string }[]
  hasGt: boolean
  hasRoi: boolean
}

export type PresetCard = {
  name: string
  title: string
  summary: string
  useFor: string
  avoid: string
  evidence: string
  stages: { label: string; method: string; methodLabel: string }[]
}

export const fetchStudioState = () => get<StudioState>('/api/studio/state')
export const fetchPresets = () =>
  get<{ presets: PresetCard[]; current: string }>('/api/studio/presets')
export const openStudio = (body: { recipe?: string; bank?: string; targets?: string }) =>
  post<StudioState>('/api/studio/open', body)

/**
 * 합성 한 장. `signal` 로 **이전 요청을 취소**하는 것이 Qt `LatestOnlyQueue`("최신만 살리기")의 웹판이다
 * — 파라미터를 빠르게 돌릴 때 낡은 결과가 나중에 도착해 화면을 되돌리는 일을 막는다.
 */
export const runPreview = (
  body: { targetIndex?: number; variant?: number; longSide?: number },
  signal?: AbortSignal,
) => post<PreviewMeta>('/api/studio/preview', body, signal)

export const setPreset = (name: string) => post<StudioState>('/api/studio/preset', { name })
export const setSeed = (seed: number) => post<StudioState>('/api/studio/seed', { seed })
export const saveRecipe = (path: string) =>
  post<{ path: string; runCommand: string }>('/api/studio/save', { path })

/** 그림은 `<img src>` 로 받는다 — `v`(미리보기 판 번호)가 캐시를 깬다. */
export const studioImageUrl = (kind: 'base' | 'synth' | 'gt' | 'roi', v = 0) =>
  `/api/studio/image?kind=${kind}${v ? `&v=${v}` : ''}`
export const studioThumbUrl = (index: number) => `/api/studio/image?kind=thumb&index=${index}`
export const presetImageUrl = (name: string, v = 0) =>
  `/api/studio/preset-image?name=${encodeURIComponent(name)}${v ? `&v=${v}` : ''}`

// ---------------------------------------------------------------- 파이프라인 카드(U5b)

/**
 * 설정 한 칸의 스펙 — **파이썬 스키마에서 그대로 온다**(`studio/params.py`). 라벨·툴팁·단위·제약·기본값이
 * 전부 여기 실려 있으므로 폼은 그리기만 하면 된다("스키마가 곧 UI").
 */
export type FieldSpec = {
  name: string
  kind: 'int' | 'float' | 'int_range' | 'range' | 'bool' | 'choice' | 'text' | 'list' | 'path'
  value: unknown
  optional: boolean
  /** `X | None` 필드가 켜져 있는가(꺼짐 = 값이 null). */
  enabled: boolean
  choices: string[]
  lo: number | null
  hi: number | null
  loOpen: boolean
  hiOpen: boolean
  onDefault: unknown
  /** 한국어 라벨 + 단위 — 원천은 `core/help.py` 다(프론트에 사전 사본을 두지 않는다). */
  title: string
  tooltip: string
  desc: string
  advanced: boolean
  baseline: unknown
  hasBaseline: boolean
  /** 프리셋 값과 다른가 — ● 표시와 ↺ 되돌리기의 기준. */
  modified: boolean
  step: number
  slider: boolean
}

export type MethodChoice = {
  method: string
  label: string
  usable: boolean
  reason: string
  summary: string
}

export type StageCard = {
  stage: string
  /** 1~7 (roi 는 0 — 배치 카드의 하위 블록). */
  no: number
  label: string
  tip: string
  method: string
  methodLabel: string
  summary: string
  modified: number
  methods: MethodChoice[]
  fields: FieldSpec[]
}

export type PerClassRow = {
  cls: string
  on: boolean
  rotate: [number, number]
  flip: string
  /** 표에 없는 값 — 그대로 들고 다니다 저장할 때 보존한다. */
  scale: [number, number] | null
}

export type PerClassTable = {
  classes: string[]
  flipChoices: { label: string; value: string }[]
  rows: PerClassRow[]
  text: string
}

export type CardsPayload = {
  cards: StageCard[]
  perClass: PerClassTable
  longSides: { label: string; px: number }[]
  modified: number
  version: number
}

export type EditResult = { state: StudioState; cards: CardsPayload }

export const fetchCards = () => get<CardsPayload>('/api/studio/cards')
export const setField = (stage: string, name: string, value: unknown) =>
  post<EditResult>('/api/studio/field', { stage, name, value })
export const setMethod = (stage: string, method: string) =>
  post<EditResult>('/api/studio/method', { stage, method })
/** `name` 을 빼면 그 카드의 바뀐 행 전부를 프리셋 값으로. */
export const resetStage = (stage: string, name?: string) =>
  post<EditResult>('/api/studio/reset', name ? { stage, name } : { stage })
export const setPerClass = (rows: PerClassRow[]) =>
  post<EditResult>('/api/studio/per-class', { rows })

/** 단계별 중간 결과 — 마지막 미리보기의 추적에서(미리보기가 바뀌면 `v` 로 캐시를 깬다). */
export const stageImageUrl = (stage: string, v = 0) =>
  `/api/studio/stage-image?stage=${encodeURIComponent(stage)}${v ? `&v=${v}` : ''}`
/** 시드 변형 k — 캔버스의 미리보기는 건드리지 않는다. */
export const variantImageUrl = (k: number, v = 0) =>
  `/api/studio/variant-image?k=${k}${v ? `&v=${v}` : ''}`

// ---------------------------------------------------------------- 일괄 생성(U6)

export type BatchSummary = {
  root: string
  nOk: number
  nSkipped: number
  nNormals: number
  nFallback: number
  perClass: Record<string, number>
  files: Record<string, string>
  warnings: string[]
  /** 중지 요청으로 멈췄나 — 그때까지의 파일·manifest 는 남아 있다. */
  cancelled: boolean
  done: number
  count: number
  text: string
}

/** 실행 한 번의 스냅샷. 스트리밍이 아니라 **폴링**으로 받는다(`pollDelay`). */
export type BatchRun = {
  running: boolean
  stopping: boolean
  done: number
  total: number
  log: string[]
  error: string | null
  summary: BatchSummary | null
}

export type BatchSettings = {
  out: string
  count: number
  seed: number
  workers: number
  writer: string
  mvtecCategory: string
}

export type BatchState =
  | { open: false; writerFormats: string[]; run?: null }
  | {
      open: true
      recipePath: string
      writerFormats: string[]
      settings: BatchSettings
      runCommand: string
      run: BatchRun | null
      recipe: {
        name: string
        preset: string
        bank: string
        targets: string
        defectsPerImage: number[]
        includeNormals: boolean
      }
    }

export const fetchBatchState = () => get<BatchState>('/api/batch/state')
export const openBatch = (recipe: string) => post<BatchState>('/api/batch/open', { recipe })
export const setBatchSettings = (settings: Partial<BatchSettings>) =>
  post<BatchState>('/api/batch/settings', { settings })
/** 시작 — **디스크에 쓴다**. 화면이 먼저 확인을 받는다. */
export const startBatch = () => post<BatchState>('/api/batch/start', {})
export const stopBatch = () => post<BatchState>('/api/batch/stop', {})
