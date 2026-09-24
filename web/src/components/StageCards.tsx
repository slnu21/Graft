import { useEffect, useRef, useState } from 'react'

import type { CardsPayload, FieldSpec, PerClassRow, StageCard } from '../api'
import { stageImageUrl } from '../api'
import {
  inBounds,
  inputText,
  modifiedText,
  parseInput,
  rotateOf,
  splitAdvanced,
  updateRow,
  widgetOf,
  withRangeEnd,
} from '../params'

/**
 * 파이프라인 카드 — **스키마가 곧 UI**. 서버가 준 `FieldSpec` 대로 위젯을 놓기만 하고,
 * 값 검증·기본값·라벨은 전부 파이썬이 정한다(Qt `param_form.py` 의 웹판).
 *
 * 숫자·텍스트는 350 ms 디바운스 뒤에 보낸다(화살표 연타·타이핑 중에 합성이 돌면 안 된다).
 * 콤보·체크는 즉시. Qt 폼과 같은 규칙이다.
 */
const DEBOUNCE_MS = 350

type Props = {
  cards: CardsPayload
  /** 편집 — 보낸 뒤 화면이 다시 합성한다. 실패하면 `error` 로 어느 행인지 돌아온다. */
  onField: (stage: string, name: string, value: unknown) => void
  onMethod: (stage: string, method: string) => void
  onReset: (stage: string, name?: string) => void
  onPerClass: (rows: PerClassRow[]) => void
  /** 마지막 재검증 오류 — 그 행을 붉게. */
  error: { stage: string; name?: string; message: string } | null
  /** 스테이지 경고(`<stage>: …`) — 카드마다 자기 것만. */
  warningsOf: (stage: string) => string[]
  /** 미리보기 판 번호 — 단계별 썸네일 캐시를 깬다. 0 이면 썸네일을 숨긴다. */
  version: number
}

export function StageCards({
  cards,
  onField,
  onMethod,
  onReset,
  onPerClass,
  error,
  warningsOf,
  version,
}: Props) {
  return (
    <>
      {cards.cards.map((card) => (
        <StageCardView
          key={card.stage}
          card={card}
          onField={onField}
          onMethod={onMethod}
          onReset={onReset}
          error={error}
          warnings={warningsOf(card.stage)}
          version={version}
          perClass={card.stage === 'geometry' ? cards.perClass : null}
          onPerClass={onPerClass}
        />
      ))}
    </>
  )
}

function StageCardView({
  card,
  onField,
  onMethod,
  onReset,
  error,
  warnings,
  version,
  perClass,
  onPerClass,
}: {
  card: StageCard
  onField: Props['onField']
  onMethod: Props['onMethod']
  onReset: Props['onReset']
  error: Props['error']
  warnings: string[]
  version: number
  perClass: CardsPayload['perClass'] | null
  onPerClass: Props['onPerClass']
}) {
  const [openAdvanced, setOpenAdvanced] = useState(false)
  const { basic, advanced } = splitAdvanced(card.fields)
  const sub = card.no === 0 // roi — 배치 카드의 하위 블록

  return (
    <section className={`card stage${sub ? ' sub' : ''}`}>
      <div className="stage-head">
        <h2 title={card.tip}>
          {card.no > 0 && <i className="no">{card.no}</i>}
          {card.label}
        </h2>
        <span className="spacer" />
        {card.modified > 0 && (
          <>
            <span className="chg" title="프리셋 값과 다른 설정">
              {modifiedText(card.modified)}
            </span>
            <button
              className="icon-btn"
              title="이 카드를 프리셋 값으로 되돌리기"
              onClick={() => onReset(card.stage)}
            >
              ↺
            </button>
          </>
        )}
      </div>

      <select
        className="method"
        value={card.method}
        title={card.summary}
        onChange={(e) => onMethod(card.stage, e.target.value)}
      >
        {card.methods.map((m) => (
          <option key={m.method} value={m.method} disabled={!m.usable}>
            {m.usable ? m.label : `${m.label} — ${m.reason}`}
          </option>
        ))}
      </select>

      {version > 0 && (
        <img
          className="stage-thumb"
          src={stageImageUrl(card.stage, version)}
          alt=""
          loading="lazy"
          onError={(e) => (e.currentTarget.style.display = 'none')}
        />
      )}

      {warnings.map((w) => (
        <div className="notice" key={w}>
          <span>{w}</span>
        </div>
      ))}
      {error && error.stage === card.stage && (
        <div className="notice bad" role="alert">
          <span>{error.message}</span>
        </div>
      )}

      <div className="form">
        {basic.map((f) => (
          <FieldRow
            key={f.name}
            spec={f}
            stage={card.stage}
            onField={onField}
            onReset={onReset}
            bad={!!error && error.stage === card.stage && error.name === f.name}
          />
        ))}
      </div>

      {advanced.length > 0 && (
        <>
          <button className="more" onClick={() => setOpenAdvanced(!openAdvanced)}>
            {openAdvanced ? '고급 옵션 접기' : `고급 옵션 (${advanced.length})`}
          </button>
          {openAdvanced && (
            <div className="form">
              {advanced.map((f) => (
                <FieldRow
                  key={f.name}
                  spec={f}
                  stage={card.stage}
                  onField={onField}
                  onReset={onReset}
                  bad={!!error && error.stage === card.stage && error.name === f.name}
                />
              ))}
            </div>
          )}
        </>
      )}

      {perClass && <PerClassTable table={perClass} onChange={onPerClass} />}
    </section>
  )
}

/** 한 행 — [끔/켬] 라벨 | 위젯 | ↺. 위젯 종류는 `widgetOf`(스펙의 `kind`)가 고른다. */
function FieldRow({
  spec,
  stage,
  onField,
  onReset,
  bad,
}: {
  spec: FieldSpec
  stage: string
  onField: Props['onField']
  onReset: Props['onReset']
  bad: boolean
}) {
  const widget = widgetOf(spec)
  const send = useDebounced((value: unknown) => onField(stage, spec.name, value))
  const off = spec.optional && !spec.enabled

  return (
    <div className={`row${bad ? ' bad' : ''}${spec.modified ? ' chg' : ''}`}>
      <label title={spec.tooltip}>
        {spec.optional && (
          <input
            type="checkbox"
            checked={spec.enabled}
            title={spec.enabled ? '끄면 이 효과를 쓰지 않습니다' : '켜기'}
            onChange={(e) => onField(stage, spec.name, e.target.checked ? spec.onDefault : null)}
          />
        )}
        <span className="lb">{spec.title}</span>
      </label>

      <div className="in">
        {off ? (
          <span className="muted">끔</span>
        ) : widget === 'bool' ? (
          <input
            type="checkbox"
            checked={!!spec.value}
            onChange={(e) => onField(stage, spec.name, e.target.checked)}
          />
        ) : widget === 'choice' ? (
          <select
            value={String(spec.value ?? '')}
            onChange={(e) => onField(stage, spec.name, e.target.value)}
          >
            {spec.choices.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        ) : widget === 'range' ? (
          <>
            <NumberInput
              spec={spec}
              raw={(spec.value as number[])?.[0]}
              onCommit={(v) => send(withRangeEnd(spec, 0, v))}
            />
            <span className="tilde">~</span>
            <NumberInput
              spec={spec}
              raw={(spec.value as number[])?.[1]}
              onCommit={(v) => send(withRangeEnd(spec, 1, v))}
            />
          </>
        ) : widget === 'number' ? (
          <>
            {spec.slider && (
              <input
                type="range"
                min={spec.lo ?? 0}
                max={spec.hi ?? 1}
                step={spec.step}
                value={Number(spec.value ?? 0)}
                onChange={(e) => send(parseInput(spec, e.target.value))}
              />
            )}
            <NumberInput
              spec={spec}
              raw={spec.value as number}
              onCommit={(v) => send(parseInput(spec, v))}
            />
          </>
        ) : (
          <input
            className="text-in"
            value={inputText(spec.value)}
            spellCheck={false}
            onChange={(e) => send(e.target.value)}
          />
        )}
      </div>

      <button
        className={`icon-btn${spec.modified ? '' : ' hidden'}`}
        title="이 값을 프리셋 값으로"
        onClick={() => onReset(stage, spec.name)}
      >
        ↺
      </button>
    </div>
  )
}

/** 숫자 칸 — 타이핑 중에는 화면 값만 바꾸고, 범위를 벗어난 값은 보내지 않는다(서버가 또 막지만 깜빡임이 는다). */
function NumberInput({
  spec,
  raw,
  onCommit,
}: {
  spec: FieldSpec
  raw: number | undefined
  onCommit: (value: string) => void
}) {
  const [text, setText] = useState(inputText(raw))
  const typing = useRef(false)

  useEffect(() => {
    if (!typing.current) setText(inputText(raw))
  }, [raw])

  return (
    <input
      className={`num${text !== '' && !inBounds(spec, Number(text)) ? ' bad' : ''}`}
      inputMode="decimal"
      value={text}
      spellCheck={false}
      onChange={(e) => {
        typing.current = true
        setText(e.target.value)
        if (e.target.value !== '' && inBounds(spec, Number(e.target.value)))
          onCommit(e.target.value)
      }}
      onBlur={() => {
        typing.current = false
        setText(inputText(raw))
      }}
    />
  )
}

/** 값 변경을 350 ms 모았다 한 번 보낸다 — 타이핑·슬라이더가 합성을 매 글자 돌리지 않게. */
function useDebounced(fn: (value: unknown) => void, ms = DEBOUNCE_MS) {
  const timer = useRef<number | null>(null)
  const latest = useRef(fn)
  latest.current = fn
  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), [])
  return (value: unknown) => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = window.setTimeout(() => latest.current(value), ms)
  }
}

/** `geometry.per_class` — 클래스마다 회전·뒤집기를 따로(크기는 YAML 에서, 값은 보존된다). */
function PerClassTable({
  table,
  onChange,
}: {
  table: CardsPayload['perClass']
  onChange: (rows: PerClassRow[]) => void
}) {
  if (!table.rows.length) return null
  const set = (cls: string, patch: Partial<PerClassRow>) =>
    onChange(updateRow(table.rows, cls, patch))

  return (
    <div className="perclass">
      <p className="sub" title="geometry.per_class — 찍힘만 ±15° 로 좁히는 식">
        클래스별 예외 — 회전·뒤집기 (크기는 YAML 에서)
      </p>
      {table.rows.map((r) => (
        <div className="pc-row" key={r.cls}>
          <label>
            <input
              type="checkbox"
              checked={r.on}
              onChange={(e) => set(r.cls, { on: e.target.checked })}
            />
            {r.cls}
          </label>
          <div className="pc-in">
            <span className="muted">회전</span>
            <input
              className="num"
              inputMode="decimal"
              disabled={!r.on}
              value={r.rotate[0]}
              onChange={(e) => set(r.cls, { rotate: rotateOf(r, 0, e.target.value) })}
            />
            <span className="tilde">~</span>
            <input
              className="num"
              inputMode="decimal"
              disabled={!r.on}
              value={r.rotate[1]}
              onChange={(e) => set(r.cls, { rotate: rotateOf(r, 1, e.target.value) })}
            />
            <select
              disabled={!r.on}
              value={r.flip}
              title="뒤집기 — 빛이 위·아래에서 오면 좌우는 안전합니다"
              onChange={(e) => set(r.cls, { flip: e.target.value })}
            >
              {table.flipChoices.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      ))}
    </div>
  )
}
