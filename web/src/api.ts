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

export const fetchHealth = () => get<Health>('/api/health')
export const fetchDoctor = () => get<Doctor>('/api/doctor')
export const fetchMethods = () =>
  get<{ methods: MethodInfo[]; stages: StageMeta[] }>('/api/methods')
