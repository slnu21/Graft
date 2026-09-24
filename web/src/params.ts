/**
 * 파이프라인 카드 폼의 **표시용 순수 함수**. "스키마가 곧 UI" 의 프론트 쪽 절반이다.
 *
 * 규칙: 폼은 서버가 준 스펙(`FieldSpec`)대로 **그리기만** 한다 — 라벨·툴팁·단위·기본값·제약은 전부
 * 파이썬(`core/help.py`·`studio/params.py`)이 정한다. 여기 있는 것은 위젯 종류를 고르고, 입력 문자열을
 * JSON 값으로 되돌리고, 접기/문구를 만드는 것뿐이다(값 검증은 서버가 다시 한다).
 */

import type { FieldSpec, PerClassRow } from './api'

export type Widget = 'number' | 'range' | 'bool' | 'choice' | 'text'

/** 스펙의 `kind` → 위젯 한 종류. 모르는 종류는 텍스트로(새 필드가 생겨도 폼이 죽지 않는다). */
export function widgetOf(spec: FieldSpec): Widget {
  switch (spec.kind) {
    case 'int':
    case 'float':
      return 'number'
    case 'int_range':
    case 'range':
      return 'range'
    case 'bool':
      return 'bool'
    case 'choice':
      return 'choice'
    default:
      return 'text'
  }
}

/** 입력칸에 보일 문자열. 목록은 쉼표로, 범위는 각 칸이 따로 받으므로 여기서는 다루지 않는다. */
export function inputText(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) return value.join(', ')
  return String(value)
}

/**
 * 입력 → 서버로 보낼 JSON 값. 숫자는 숫자로, 목록·경로·문자열은 문자열 그대로 보낸다
 * (서버가 `params.coerce` 로 다시 정리하므로 여기서는 **형식만** 맞춘다).
 */
export function parseInput(spec: FieldSpec, raw: string): unknown {
  if (spec.kind === 'int') return Math.round(Number(raw))
  if (spec.kind === 'float') return Number(raw)
  return raw
}

/** 범위 한 쪽만 바꾼 새 값 `[lo, hi]`. 정수 범위는 반올림한다. */
export function withRangeEnd(spec: FieldSpec, side: 0 | 1, raw: string): [number, number] {
  const cur = Array.isArray(spec.value) ? (spec.value as number[]) : [0, 0]
  const out: [number, number] = [Number(cur[0] ?? 0), Number(cur[1] ?? 0)]
  const n = numberOf(raw) // 빈칸은 NaN — `Number('')` 은 0 이라 그대로 쓰면 값을 0 으로 지운다
  if (Number.isFinite(n)) out[side] = spec.kind === 'int_range' ? Math.round(n) : n
  return out
}

/** 빈칸·공백은 숫자가 아니다(`Number('')` 이 0 인 것에 기대면 값이 조용히 0 이 된다). */
function numberOf(raw: string): number {
  return raw.trim() === '' ? Number.NaN : Number(raw)
}

/** 숫자 입력이 스펙의 범위 안인가 — 서버에 보내기 전에 화면에서 먼저 잡는다(열린 구간도 본다). */
export function inBounds(spec: FieldSpec, value: number): boolean {
  if (!Number.isFinite(value)) return false
  if (spec.lo !== null && (spec.loOpen ? value <= spec.lo : value < spec.lo)) return false
  if (spec.hi !== null && (spec.hiOpen ? value >= spec.hi : value > spec.hi)) return false
  return true
}

/** 고급 옵션은 접어 둔다 — 카드가 길어지면 7단계가 한눈에 안 들어온다(Qt 폼과 같은 규칙). */
export function splitAdvanced(fields: FieldSpec[]): { basic: FieldSpec[]; advanced: FieldSpec[] } {
  return {
    basic: fields.filter((f) => !f.advanced),
    advanced: fields.filter((f) => f.advanced),
  }
}

/** "바뀜 2" — 프리셋과 다른 행의 수. 0 이면 빈 문자열(배지를 숨긴다). */
export function modifiedText(n: number): string {
  return n ? `바뀜 ${n}` : ''
}

/** 시드 변형 그리드의 k 목록 — `image_rng(seed, k)` 의 k 다(0 = 지금 캔버스). */
export function variantKeys(n: number): number[] {
  return Array.from({ length: Math.max(0, n) }, (_, i) => i)
}

/** 표의 한 줄만 바꾼 새 목록(불변) — React 상태를 제자리에서 고치지 않는다. */
export function updateRow(
  rows: PerClassRow[],
  cls: string,
  patch: Partial<PerClassRow>,
): PerClassRow[] {
  return rows.map((r) => (r.cls === cls ? { ...r, ...patch } : r))
}

/** 회전 범위 한 쪽만 바꾼 행 값. 숫자가 아니면 그대로 둔다. */
export function rotateOf(row: PerClassRow, side: 0 | 1, raw: string): [number, number] {
  const n = numberOf(raw)
  const out: [number, number] = [row.rotate[0], row.rotate[1]]
  if (Number.isFinite(n)) out[side] = n
  return out
}
