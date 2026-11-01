import { useState } from 'react'
import { tokenizeText } from './api/tokenization'
import TokenList from './components/TokenList'
import ModelNavigator from './components/ModelNavigator'
import type { TokenizeResponse } from './types/tokenization'

const sampleText = 'Xin chào Việt Nam!\nTôi học AI 😊.'

export default function App() {
  const [text, setText] = useState(sampleText)
  const [data, setData] = useState<TokenizeResponse | null>(null)
  const [selectedPosition, setSelectedPosition] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleTokenize() {
    setLoading(true)
    setError(null)
    setData(null)
    setSelectedPosition(null)

    try {
      const result = await tokenizeText(text)
      setData(result)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể gọi API tokenize')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page">
      <header className="page-header">
        <p className="eyebrow">Visualize LLM</p>
        <h1>Khám phá từ token đến head</h1>
        <p>Xem cấu trúc model, chọn layer và head. Nhập text để kiểm tra token thật, ID và vị trí.</p>
      </header>

      <ModelNavigator />

      <section className="input-panel" aria-label="Nhập text">
        <h2>Tokenization</h2>
        <label htmlFor="source-text">Text đầu vào</label>
        <textarea
          id="source-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={4}
          maxLength={1000}
          spellCheck={false}
        />
        <div className="input-actions">
          <span>{Array.from(text).length} ký tự</span>
          <button type="button" onClick={() => void handleTokenize()} disabled={loading || text.length === 0}>
            {loading ? 'Đang tokenize…' : 'Tokenize'}
          </button>
        </div>
      </section>

      {error && <p className="error" role="alert">{error}. Kiểm tra FastAPI đang chạy ở cổng 8000.</p>}
      {data && <TokenList data={data} selectedPosition={selectedPosition} onSelect={setSelectedPosition} />}
    </main>
  )
}
