/**
 * 일괄 생성 화면의 **표시용 순수 함수**. 실행·진행·요약은 전부 파이썬이 내고, 여기서는 그것을 문장과
 * 숫자로 바꾸기만 한다(합성·판정 로직 0).
 */

import type { BatchRun, BatchSummary } from './api'

export type Phase = 'idle' | 'running' | 'stopping' | 'done' | 'cancelled' | 'failed'

/** 지금 어느 단계인가 — 버튼 활성·문구·폴링 주기가 전부 이 하나에서 갈린다. */
export function phaseOf(run: BatchRun | null): Phase {
  if (!run) return 'idle'
  if (run.running) return run.stopping ? 'stopping' : 'running'
  if (run.error) return 'failed'
  if (run.summary?.cancelled) return 'cancelled'
  return 'done'
}

export const PHASE_LABELS: Record<Phase, string> = {
  idle: '대기',
  running: '생성 중',
  stopping: '중지 요청 — 현재 이미지까지 기록하고 멈춥니다',
  done: '완료',
  cancelled: '취소됨',
  failed: '실패',
}

/**
 * 폴링 주기(ms). 스트리밍을 두지 않기로 했으므로 **도는 동안만 자주** 묻고 아니면 멈춘다
 * (0 = 폴링하지 않음). 로컬 왕복은 수 ms 라 0.4 초면 진행률이 매끄럽다.
 */
export function pollDelay(phase: Phase): number {
  return phase === 'running' || phase === 'stopping' ? 400 : 0
}

/** 0~100. 총 장수를 모르면 0(진행바는 불확정으로 그린다). */
export function percent(done: number, total: number): number {
  if (total <= 0) return 0
  return Math.min(100, Math.round((done / total) * 100))
}

/** "12 / 40 장 · 30%" */
export function progressText(run: BatchRun | null): string {
  if (!run) return '아직 돌리지 않았습니다'
  return `${run.done} / ${run.total} 장 · ${percent(run.done, run.total)}%`
}

/** 완료 요약 한 줄 — 숫자는 서버가 센 것 그대로. */
export function summaryLine(summary: BatchSummary | null): string {
  if (!summary) return ''
  const parts = [`성공 ${summary.nOk}`, `건너뜀 ${summary.nSkipped}`, `정상 ${summary.nNormals}`]
  if (summary.nFallback) parts.push(`폴백 ${summary.nFallback}`)
  return `${parts.join(' · ')} → ${summary.root}`
}

/** 클래스별 인스턴스 수 — 없으면 빈 문자열. */
export function perClassText(summary: BatchSummary | null): string {
  const per = summary?.perClass ?? {}
  const keys = Object.keys(per).sort()
  if (!keys.length) return ''
  return keys.map((k) => `${k} ${per[k]}`).join(' · ')
}

/** 로그는 끝쪽이 중요하다 — 화면에는 마지막 n 줄만. */
export function logTail(log: string[], n = 200): string[] {
  return log.length <= n ? log : log.slice(log.length - n)
}

/** 검수로 넘길 수 있나 — 실제로 쓴 장이 있어야 의미가 있다. */
export function canReview(summary: BatchSummary | null): boolean {
  return !!summary && summary.nOk > 0
}
