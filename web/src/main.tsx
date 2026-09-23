import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './App'
import './theme.css'

// 테마는 React 가 그리기 전에 정한다 — `:root` 가 라이트라 다크 사용자만 깜빡일 수 있다.
// `?theme=dark` 가 저장값보다 우선한다(목업과 같은 규칙 — 헤드리스 렌더로 양쪽을 찍을 수 있어야 한다).
try {
  const asked = new URLSearchParams(location.search).get('theme')
  const saved = asked ?? localStorage.getItem('graft.theme')
  if (saved === 'dark' || saved === 'light') {
    document.documentElement.dataset.theme = saved
  }
} catch {
  // 사생활 보호 모드 등에서 localStorage 가 막힌다 — 기본(라이트)으로 그냥 간다.
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
