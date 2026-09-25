import { useEffect, useState } from 'react'

import { type RecentKind, fetchRecent } from '../api'

type Props = {
  id: string
  /** 칸의 **뜻**(화면이 아니라) — ③ 에서 저장한 레시피가 ④ 목록에도 뜨는 이유. */
  kind: RecentKind
  value: string
  onChange: (value: string) => void
  onEnter?: () => void
  /** 칸을 떠날 때 값을 서버에 넘기는 칸(일괄 생성 설정)이 쓴다. */
  onBlur?: (value: string) => void
  placeholder?: string
  className?: string
  disabled?: boolean
  /** 무언가를 연 직후 이 값을 바꾸면 목록을 다시 받는다(방금 연 것이 맨 위로). */
  reloadKey?: number
}

/**
 * 경로 입력칸 + **최근 목록**(U7).
 *
 * 목록은 서버(`~/.anograft/recent.json`)가 들고 있고 화면은 받아서 보여 주기만 한다 —
 * 브라우저를 바꿔도 남고, "화면은 API 호출만 한다"는 규약에 예외를 만들지 않는다.
 * 고르개는 브라우저 기본 `<datalist>` 다: 새 의존성 0 이고 키보드도 그냥 된다.
 * 목록을 못 받아도(파일이 없거나 홈이 읽기 전용) 칸은 평소처럼 쓴다(fail-soft).
 */
export function PathField({
  id,
  kind,
  value,
  onChange,
  onEnter,
  onBlur,
  placeholder,
  className,
  disabled,
  reloadKey = 0,
}: Props) {
  const [items, setItems] = useState<string[]>([])
  const listId = `${id}-recent`

  useEffect(() => {
    let alive = true
    fetchRecent()
      .then((r) => alive && setItems(r.recent[kind] ?? []))
      .catch(() => alive && setItems([]))
    return () => {
      alive = false
    }
  }, [kind, reloadKey])

  return (
    <>
      <input
        id={id}
        className={className}
        list={items.length ? listId : undefined}
        value={value}
        spellCheck={false}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        onBlur={(e) => onBlur?.(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && onEnter?.()}
      />
      {items.length > 0 && (
        <datalist id={listId}>
          {items.map((v) => (
            <option key={v} value={v} />
          ))}
        </datalist>
      )}
    </>
  )
}
