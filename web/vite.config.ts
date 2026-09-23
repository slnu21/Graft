/// <reference types="vitest" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 빌드 산출물은 파이썬 패키지 안(`src/anograft/web/static/`)으로 바로 들어간다 — 그 폴더는 gitignore 이고
// CI 가 만들어 wheel·zip 에 싣는다(결정 6 ①). 소스 체크아웃에 번들이 없는 것이 정상이다.
//
// 오프라인 규약: 폰트·아이콘은 vendoring 하지 않고 **시스템 폰트 스택 + 인라인 SVG** 만 쓴다(목업과 같은 방식) →
// 받을 자원이 없으니 CDN 금지가 저절로 지켜진다. 소스맵은 배포에서 뺀다.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../src/anograft/web/static',
    emptyOutDir: true,
    sourcemap: false,
    target: 'es2022',
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    // 개발은 vite(5173) ↔ `anograft serve --api-only`(8000) 프록시로 핫리로드(결정 6 ⑤).
    proxy: { '/api': { target: 'http://127.0.0.1:8000' } },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
