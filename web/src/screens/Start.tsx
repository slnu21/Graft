import type { Doctor, Health, MethodInfo } from '../api'
import { formatValue, groupByStage, shortenPath, type StageInfo } from '../format'

/**
 * 시작 화면 — **환경 점검 + 다음에 할 일.** 목업 검토가 "모든 화면이 데이터가 다 있는 상태만 그린다"고
 * 지적한 자리이고, 웹에서 가장 먼저 필요한 화면이기도 하다(서버가 떴는지, 무엇을 쓸 수 있는지).
 *
 * 값은 전부 API 가 준 것을 그대로 보인다 — 여기서 판단하지 않는다.
 */
export function Start({
  health,
  doctor,
  methods,
  stages,
  error,
}: {
  health: Health | null
  doctor: Doctor | null
  methods: MethodInfo[]
  stages: StageInfo[]
  error: string | null
}) {
  const groups = groupByStage(stages, methods)
  const unusable = methods.filter((m) => !m.usable)

  return (
    <>
      <div className="page-head">
        <h1>시작</h1>
        <p>이 PC에서 무엇이 준비됐는지 먼저 봅니다. 화면은 차례로 늘어납니다.</p>
      </div>

      {error && (
        <div className="notice bad" style={{ marginBottom: 12 }} role="alert">
          <span>{error}</span>
        </div>
      )}

      <div className="cards">
        <section className="card">
          <h2>환경</h2>
          <p className="sub">문제를 알릴 때 이 표를 그대로 붙이면 됩니다.</p>
          {doctor ? (
            <dl className="kv">
              {ENV_KEYS.filter((k) => k.key in doctor).map((k) => (
                <div key={k.key} style={{ display: 'contents' }}>
                  <dt>{k.label}</dt>
                  <dd>
                    {k.path
                      ? shortenPath(formatValue(doctor[k.key]))
                      : formatValue(doctor[k.key])}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <Skeleton rows={5} />
          )}
        </section>

        <section className="card">
          <h2>쓸 수 있는 알고리즘</h2>
          <p className="sub">
            {methods.length ? `${methods.length - unusable.length} / ${methods.length}개` : '…'} ·
            단계마다 고를 수 있는 방법입니다.
          </p>
          {groups.length ? (
            groups.map((g) => (
              <div className="stage-row" key={g.stage}>
                <span className="name" title={g.desc}>
                  {g.label}
                </span>
                <span className="methods">
                  {g.items.map((m, i) => (
                    <span key={m.method} className={m.usable ? '' : 'off'} title={m.reason ?? ''}>
                      {i > 0 && ' · '}
                      {m.label ?? m.method}
                    </span>
                  ))}
                </span>
              </div>
            ))
          ) : (
            <Skeleton rows={6} />
          )}
          {unusable.length > 0 && (
            <div className="notice" style={{ marginTop: 10 }}>
              <span>
                {unusable.length}개는 지금 못 씁니다 — {unusable[0].reason}. 나머지는 그대로
                동작합니다.
              </span>
            </div>
          )}
        </section>

        <section className="card">
          <h2>다음에 할 일</h2>
          <p className="sub">아직은 명령줄이 더 많은 일을 합니다. 화면은 차례로 옮겨 옵니다.</p>
          <ol className="steps">
            <li>
              결함 보관함 만들기 — <code>anograft bank import-yolo</code> 또는{' '}
              <code>anograft sample --quickstart</code>
            </li>
            <li>
              레시피 만들기 — <code>anograft recipe init --preset poisson-graft</code>
            </li>
            <li>
              데이터셋 생성 — <code>anograft run recipes/x.yaml</code>
            </li>
            <li>
              검수 — ⑤ 검수 화면, 또는 <code>anograft dataset report out</code>
            </li>
            <li>
              (선택) 학습 루프 — <code>anograft loop run</code> 으로 한 바퀴, 현황은 ⑥ 학습 루프
            </li>
          </ol>
        </section>
      </div>

      {health && (
        <p className="muted" style={{ marginTop: 14, fontSize: 11.5 }}>
          이 화면은 {health.name} {health.version} 이 여는 로컬 서버입니다 — 바깥으로 나가는 통신은
          없습니다.
        </p>
      )}
    </>
  )
}

const ENV_KEYS: { key: string; label: string; path?: boolean }[] = [
  { key: 'anograft', label: '버전' },
  { key: 'python', label: '파이썬' },
  { key: 'platform', label: '플랫폼' },
  { key: 'cpu_count', label: 'CPU 수' },
  { key: 'opencv', label: 'OpenCV' },
  { key: 'numpy', label: 'NumPy' },
  { key: 'gui', label: '기존 창(GUI)' },
  { key: 'pyside6', label: 'PySide6' },
  { key: 'cwd', label: '작업 폴더', path: true },
]

function Skeleton({ rows }: { rows: number }) {
  return (
    <div style={{ display: 'grid', gap: 7 }}>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${100 - i * 7}%` }} />
      ))}
    </div>
  )
}
