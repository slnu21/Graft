/**
 * 아직 안 옮긴 화면 — 빈 화면 대신 **어디서 그 일을 할 수 있는지**를 말해 준다.
 * (목업 검토: "모든 화면이 데이터가 다 있는 상태만 그린다" — 빈 상태를 처음부터 둔다.)
 */
export function Soon({ label }: { label: string }) {
  return (
    <>
      <div className="page-head">
        <h1>{label}</h1>
        <p>아직 이 화면은 옮기는 중입니다.</p>
      </div>
      <div className="cards">
        <section className="card">
          <h2>지금은 이렇게 합니다</h2>
          <p className="sub">같은 일을 명령줄과 기존 창(PySide6)에서 할 수 있습니다.</p>
          <ol className="steps">
            <li>
              기존 화면: <code>python -m anograft.gui</code>
            </li>
            <li>
              명령줄 전체 목록: <code>anograft --help</code>
            </li>
          </ol>
        </section>
      </div>
    </>
  )
}
