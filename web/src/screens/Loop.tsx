import { useCallback, useEffect, useState } from 'react'

import {
  type LoopRounds,
  type LoopState,
  fetchLoopRounds,
  fetchLoopState,
  openLoop,
  openReview,
} from '../api'
import { PathField } from '../components/PathField'
import { shortenPath } from '../format'
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
} from '../loop'

/**
 * ⑥ 학습 루프 — **지금 어디인가**를 보여 주고, 사람이 할 차례면 그 자리로 보낸다.
 *
 * 여기서 라운드를 돌리지 않는다(설계 §2b.1). 한 바퀴는 학습기를 몇 시간 돌리는 일이고 그걸 부르는
 * 자리는 이미 `anograft loop run` 과 스케줄러가 부르는 `loop tick` 이다 — 화면은 그 명령을 그대로
 * 보여 주고(칠 수 있게), 검토 대기가 생기면 **⑤ 검수 화면으로 넘긴다**(큐 폴더가 곧 검수 입력이다).
 */
export function Loop() {
  const [state, setState] = useState<LoopState | null>(null)
  const [rounds, setRounds] = useState<LoopRounds | null>(null)
  const [configPath, setConfigPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [recentKey, setRecentKey] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const open = state?.open === true

  const load = useCallback(async () => {
    const s = await fetchLoopState()
    setState(s)
    if (s.open) {
      setConfigPath(s.configPath)
      setRounds(await fetchLoopRounds())
    } else if (s.suggest && !configPath) {
      setConfigPath(s.suggest)
    }
  }, [configPath])

  useEffect(() => {
    load().catch((err: Error) => setError(err.message))
    // 첫 로드만 — 그 뒤 갱신은 아래 폴링과 "열기"가 맡는다
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 돌고 있는 동안만 따라간다(루프는 이 서버 밖에서도 돈다). 조용할 때는 묻지 않는다.
  const delay = pollDelay(state)
  useEffect(() => {
    if (!delay) return
    const timer = setTimeout(() => void load().catch(() => undefined), delay)
    return () => clearTimeout(timer)
  }, [delay, state, load])

  const doOpen = () => {
    setBusy(true)
    setError(null)
    openLoop(configPath.trim())
      .then(async (s) => {
        setState(s)
        setRecentKey((k) => k + 1)
        setRounds(await fetchLoopRounds())
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setBusy(false))
  }

  const goReview = (queueDir: string) => {
    openReview(queueDir)
      .then(() => {
        location.hash = '#/review'
      })
      .catch((err: Error) => setError(err.message))
  }

  const points = rounds?.points ?? []
  const c = chart(points)
  const delta = latestDelta(points)
  const lastRow = rounds?.rounds[0] ?? null

  return (
    <>
      <div className="page-head">
        <h1>학습 루프</h1>
        <p>
          라운드가 <b>어디까지 왔는지</b>와 <b>왜 도는지/안 도는지</b>를 봅니다. 한 바퀴는 여기서
          돌리지 않습니다 — 명령줄이나 스케줄러가 돌리고, 사람이 판정할 차례면 검수 화면으로 넘깁니다.
        </p>
      </div>

      <div className="openbar">
        <label htmlFor="loop-config">루프 설정</label>
        <PathField
          id="loop-config"
          kind="loop"
          value={configPath}
          onChange={setConfigPath}
          onEnter={doOpen}
          placeholder="예: loop.yaml (비우면 loop.yaml → ~/.anograft/loop.yaml 순서로 찾습니다)"
          reloadKey={recentKey}
        />
        <button className="btn primary" disabled={busy} onClick={doOpen}>
          열기
        </button>
      </div>

      {error && (
        <div className="notice bad" role="alert" style={{ marginBottom: 12 }}>
          <span>{error}</span>
        </div>
      )}

      {!open || !state || state.open !== true ? (
        <div className="cards">
          <section className="card">
            <h2>아직 연 루프 설정이 없습니다</h2>
            <p className="sub">
              루프는 <b>선택 계층</b>입니다 — 합성·검수만 쓰신다면 필요 없습니다. 쓰시려면{' '}
              <code>loop.example.yaml</code> 을 복사해 <code>loop.yaml</code> 을 만들고(학습기 이름 ·
              보관함 · 레시피 · 동결 평가셋 · 현장 이미지 폴더) 한 바퀴를 돌리세요:
            </p>
            <pre className="cmd">anograft loop run --config loop.yaml</pre>
            <p className="sub">
              스케줄러(작업 스케줄러·cron)가 부르는 자리는 <code>anograft loop tick</code> 입니다.
            </p>
          </section>
        </div>
      ) : (
        <div className="loop">
          <div className={`notice loop-action ${actionTone(state.action.kind)}`}>
            <div className="tx">
              <b>{state.action.text}</b>
              {state.action.command && <pre className="cmd">{state.action.command}</pre>}
            </div>
            {state.review.waiting && state.review.queueDir && (
              <button className="btn primary" onClick={() => goReview(state.review.queueDir)}>
                검토 대기로 가기 →
              </button>
            )}
          </div>

          <div className="loop-stats">
            <section className="card">
              <h2>현재 모델</h2>
              <div className="sub">
                {state.champion
                  ? `라운드 ${state.champion.round} · ${shortenPath(state.champion.model, 38) || '(이름 없음)'}`
                  : '아직 승급한 모델이 없습니다'}
              </div>
              <div className="stat mono" title={state.metricHint}>
                {metricText(state.champion?.metric ?? null)}
                <span className="u"> {state.metricName}</span>
              </div>
              <div className="sub mono">
                {deltaText(delta) || '지난 라운드와 견줄 것이 없습니다'}
              </div>
            </section>

            <section className="card">
              <h2>라운드</h2>
              <div className="sub">
                {state.round ? (state.round.bootstrap ? '부트스트랩' : '수집 포함') : '아직 없음'}
              </div>
              <div className="stat mono">
                {state.round ? state.round.number : NONE}
                <span className="u"> 번째</span>
              </div>
              <div className="sub">
                {state.round
                  ? state.round.finished
                    ? '완료 — 다음 라운드를 기다립니다'
                    : `다음 단계: ${state.round.nextLabel || state.round.next}`
                  : '첫 라운드가 열리면 채워집니다'}
              </div>
            </section>

            <section className="card">
              <h2>결함 조각</h2>
              <div className="sub">지금 보관함 · 클래스 {state.bank.classes.length}</div>
              <div className="stat mono">
                {state.bank.sources}
                <span className="u"> 조각</span>
              </div>
              <div className="sub mono">
                {lastRow ? `마지막 라운드 ${lastRow.sources} 조각` : '라운드 기록 없음'}
              </div>
            </section>

            <section className="card">
              <h2>사람 수정률</h2>
              <div className="sub">모델 초안 중 사람이 다듬은 비율</div>
              <div className="stat mono">{rateText(state.corrections.rate)}</div>
              <div className="sub">{state.corrections.text}</div>
            </section>
          </div>

          {state.round && (
            <section className="card">
              <h2>이 라운드의 단계</h2>
              <div className="sub">
                {state.round.started && `시작 ${whenText(state.round.started)}`}
                {state.round.updated && ` · 마지막 갱신 ${whenText(state.round.updated)}`}
              </div>
              <ol className="loop-steps">
                {state.round.steps.map((s) => (
                  <li key={s.phase} className={s.state}>
                    <i />
                    <span>{s.label}</span>
                    <em className="mono">{s.phase}</em>
                  </li>
                ))}
              </ol>
              {state.review.total > 0 && (
                <>
                  <div className="progress" aria-label="판정 진행률">
                    <i
                      style={{
                        width: `${progressPercent(state.review.judged, state.review.total)}%`,
                      }}
                    />
                  </div>
                  <div className="sub">
                    검토 대기 {state.review.total}장 중 {state.review.judged}장 판정됨
                  </div>
                  {state.review.queueDir && <div className="sub mono">{state.review.queueDir}</div>}
                </>
              )}
            </section>
          )}

          <section className="card">
            <h2>라운드별 {rounds?.metricLabel ?? state.metricLabel}</h2>
            <div className="sub">
              {state.metricHint && <>{state.metricHint}. </>}지점(기준선 재설정 · 자동 정지 해제)
              앞뒤로는 <b>선을 잇지 않습니다</b> — 그 구간은 견주지 않기로 한 자리입니다.
            </div>
            {c.dots.length === 0 ? (
              <p className="sub">아직 점수가 있는 라운드가 없습니다.</p>
            ) : (
              <svg className="loop-chart" viewBox={`0 0 ${CHART.width} ${CHART.height}`} role="img">
                <g stroke="var(--line)" strokeWidth="1">
                  <line
                    x1={CHART.left}
                    y1={CHART.top}
                    x2={CHART.left}
                    y2={CHART.height - CHART.bottom}
                  />
                  <line
                    x1={CHART.left}
                    y1={CHART.height - CHART.bottom}
                    x2={CHART.width - CHART.right}
                    y2={CHART.height - CHART.bottom}
                  />
                  {c.ticks.map((t) => (
                    <line
                      key={t.value}
                      x1={CHART.left}
                      y1={t.y}
                      x2={CHART.width - CHART.right}
                      y2={t.y}
                      strokeDasharray="2 4"
                    />
                  ))}
                </g>
                <g fill="var(--tx3)" fontSize="10" fontFamily="var(--mono)">
                  {c.ticks.map((t) => (
                    <text key={t.value} x={CHART.left - 6} y={t.y + 3} textAnchor="end">
                      {t.value.toFixed(2)}
                    </text>
                  ))}
                  {c.dots.map((d) => (
                    <text
                      key={d.round}
                      x={d.x}
                      y={CHART.height - CHART.bottom + 14}
                      textAnchor="middle"
                    >
                      R{d.round}
                    </text>
                  ))}
                </g>
                {c.lines.map((l) => (
                  <polyline
                    key={l.segment}
                    points={l.points}
                    fill="none"
                    stroke="var(--teal)"
                    strokeWidth="2"
                  />
                ))}
                {c.dots.map((d) => (
                  <g key={d.round}>
                    <circle
                      cx={d.x}
                      cy={d.y}
                      r={d.promoted ? 3.5 : 2.5}
                      fill={d.promoted ? 'var(--teal)' : 'var(--tx3)'}
                    />
                    {d.marker && (
                      <line
                        x1={d.x}
                        y1={CHART.top}
                        x2={d.x}
                        y2={CHART.height - CHART.bottom}
                        stroke="var(--warn)"
                        strokeWidth="1"
                        strokeDasharray="3 3"
                      />
                    )}
                    <title>
                      R{d.round} · {metricText(d.metric)} · {d.promoted ? '승급' : '유지'}
                    </title>
                  </g>
                ))}
              </svg>
            )}
          </section>

          <div className="loop-cols">
            <section className="card">
              <h2>다음 라운드</h2>
              <div className="sub">기준은 loop.yaml 의 trigger · breaker — 기본값은 제한 없음입니다.</div>
              <dl className="kv">
                <dt>트리거</dt>
                <dd>
                  {state.trigger
                    ? `${state.breaker?.tripped ? '(참고) ' : state.trigger.start ? '지금 돌 때입니다 — ' : '지금은 돌지 않습니다 — '}${state.trigger.reason}`
                    : NONE}
                </dd>
                <dt>자동 정지</dt>
                <dd className={state.breaker?.tripped ? 'bad' : undefined}>
                  {state.breaker
                    ? state.breaker.tripped
                      ? state.breaker.reason
                      : state.breaker.reason || '이상 없음'
                    : NONE}
                </dd>
                <dt>마지막 tick</dt>
                <dd>
                  {state.tick
                    ? `${whenText(state.tick.at)} — ${state.tick.ran ? '돌았습니다' : '돌지 않았습니다'}${
                        state.tick.reason ? `: ${state.tick.reason}` : ''
                      }`
                    : '스케줄러가 부른 적이 없습니다'}
                </dd>
                <dt>처리 이력</dt>
                <dd className="mono">이미 스코어링한 이미지 {state.processed}장</dd>
                <dt>지금</dt>
                <dd>{state.lockText || '도는 중인 실행 없음'}</dd>
              </dl>
              {state.failures.length > 0 && (
                <div className="notice bad">
                  <div className="tx">
                    {state.failures.map((f, i) => (
                      <div key={i}>
                        라운드 {f.round} {f.phase}: {f.reason}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <section className="card">
              <h2>보관함 분포</h2>
              <div className="sub">
                증가분은 <b>마지막 라운드 이후</b> 늘어난 조각입니다(미분류는 합성·출력에서 빠집니다).
              </div>
              <table className="loop-table">
                <thead>
                  <tr>
                    <th>클래스</th>
                    <th className="num">조각</th>
                    <th className="num">이후 추가</th>
                    <th className="num">비중</th>
                  </tr>
                </thead>
                <tbody>
                  {state.bank.rows.map((r) => (
                    <tr key={r.name} className={r.unsorted ? 'muted-row' : undefined}>
                      <td>{r.unsorted ? '미분류' : r.name}</td>
                      <td className="num mono">{r.count}</td>
                      <td className="num mono">
                        {r.delta === null ? NONE : r.delta > 0 ? `+${r.delta}` : r.delta}
                      </td>
                      <td className="num mono">{Math.round(r.share * 100)}%</td>
                    </tr>
                  ))}
                  {state.bank.rows.length === 0 && (
                    <tr>
                      <td colSpan={4} className="sub">
                        보관함을 읽지 못했거나 비어 있습니다.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </section>
          </div>

          <section className="card">
            <h2>라운드 기록</h2>
            <div className="sub">
              {rounds?.ledger ?? ''} — 더하기만 하는 기록입니다(보관함 스냅샷과 모델이 함께 묶입니다).
            </div>
            <table className="loop-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>끝난 때</th>
                  <th className="num" title={state.metricHint}>
                    {rounds?.metricName ?? state.metricName}
                  </th>
                  <th className="num">편입</th>
                  <th className="num">조각</th>
                  <th className="num">사람 수정률</th>
                  <th>설정 지문</th>
                  <th>결과</th>
                </tr>
              </thead>
              <tbody>
                {(rounds?.rounds ?? []).map((r) => (
                  <tr key={r.round} className={r.marker ? 'mark-row' : undefined}>
                    <td className="mono">R{r.round}</td>
                    <td className="mono">{whenText(r.at)}</td>
                    <td className="num mono">{metricText(r.metric)}</td>
                    <td className="num mono">{r.intake}</td>
                    <td className="num mono">{r.sources}</td>
                    <td className="num mono">{rateText(r.correctionRate)}</td>
                    <td className="mono">{shortHash(r.pipelineHash)}</td>
                    <td className="wrap">
                      <span className={`pill ${outcomeTone(r)}`}>{outcomeLabel(r)}</span>
                      {r.reason && <em className="why"> {r.reason}</em>}
                    </td>
                  </tr>
                ))}
                {(rounds?.rounds.length ?? 0) === 0 && (
                  <tr>
                    <td colSpan={8} className="sub">
                      아직 끝난 라운드가 없습니다.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            {(rounds?.warnings ?? []).map((w) => (
              <div className="sub" key={w}>
                경고: {w}
              </div>
            ))}
          </section>

          <section className="card">
            <h2>이 루프가 보는 것</h2>
            <dl className="kv">
              <dt>설정</dt>
              <dd className="mono">{state.configPath}</dd>
              <dt>학습기</dt>
              <dd className="mono">{state.trainer}</dd>
              <dt>라운드 폴더</dt>
              <dd className="mono">{state.out}</dd>
              <dt>보관함</dt>
              <dd className="mono">{state.bankPath}</dd>
              <dt>레시피</dt>
              <dd className="mono">{state.recipePath}</dd>
              <dt>현장 이미지</dt>
              <dd className="mono">{state.fieldPath || '(없음 — 합성만 도는 라운드)'}</dd>
            </dl>
          </section>
        </div>
      )}
    </>
  )
}
