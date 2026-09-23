import type { HistogramData } from '../api'
import { bars, formatEdge } from '../review'

/**
 * 합성 vs 실제 막대 — **계열색은 고정**이다(합성 `#00A188` / 실제 `#C8841C`, 색각 검증 통과).
 * 좌표 계산은 `review.ts` 의 순수 함수가 하고(vitest), 여기서는 SVG 로만 옮긴다.
 */
export function Histogram({ data, height = 96 }: { data: HistogramData; height?: number }) {
  const list = bars(data)
  if (!list.length) return <p className="muted">분포를 만들 값이 없습니다.</p>

  const W = 100 // viewBox 폭 — 반응형이라 퍼센트로 그린다
  const pad = 2
  const plot = height - 16

  return (
    <figure className="hist">
      <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" role="img" aria-label="분포">
        {list.map((b, i) => {
          const x = pad + b.x * (W - pad * 2)
          const w = b.w * (W - pad * 2)
          const half = w / 2
          return (
            <g key={i}>
              <rect
                x={x}
                y={plot - b.hA * plot}
                width={Math.max(0.6, half - 0.3)}
                height={b.hA * plot}
                fill="#00A188"
              />
              <rect
                x={x + half}
                y={plot - b.hB * plot}
                width={Math.max(0.6, half - 0.3)}
                height={b.hB * plot}
                fill="#C8841C"
              />
            </g>
          )
        })}
        <line x1={pad} y1={plot} x2={W - pad} y2={plot} stroke="currentColor" strokeWidth="0.3" opacity="0.35" />
      </svg>
      <figcaption>
        <span className="key">
          <i style={{ background: '#00A188' }} /> 합성
        </span>
        <span className="key">
          <i style={{ background: '#C8841C' }} /> {data.realLabel || '실제'}
        </span>
        <span className="range">
          {formatEdge(data.edges[0], data.log)} – {formatEdge(data.edges.at(-1) ?? 0, data.log)}
          {data.log ? ' (로그 구간)' : ''}
        </span>
      </figcaption>
    </figure>
  )
}
