import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { tokenizeText } from './api/tokenization'
import { createTrace, getLayerQKV } from './api/trace'
import { getModels } from './api/model'
import type { ModelOption } from './types/model'
import TokenList from './components/TokenList'
import ModelNavigator from './components/ModelNavigator'
import type { TokenizeResponse } from './types/tokenization'
import type { QKVLayer, TraceCreated } from './types/trace'

const QKVHeatmaps = lazy(() => import('./components/QKVHeatmaps'))
const sampleText = 'Xin chào Việt Nam!\nTôi học AI 😊.'

export default function App() {
  const [models, setModels] = useState<ModelOption[]>([])
  const [modelId, setModelId] = useState('')
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [catalogRetry, setCatalogRetry] = useState(0)
  const [text, setText] = useState(sampleText)
  const [data, setData] = useState<TokenizeResponse | null>(null)
  const [trace, setTrace] = useState<TraceCreated | null>(null)
  const [selectedPosition, setSelectedPosition] = useState<number | null>(null)
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [queryHead, setQueryHead] = useState(0)
  const [loading, setLoading] = useState<'tokenize' | 'trace' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [layerData, setLayerData] = useState<QKVLayer | null>(null)
  const [layerError, setLayerError] = useState<string | null>(null)
  const [loadingLayer, setLoadingLayer] = useState(false)
  const [retryLayer, setRetryLayer] = useState(0)
  const cache = useRef(new Map<string, QKVLayer>())

  useEffect(() => {
    const controller = new AbortController()
    setCatalogError(null)
    getModels(controller.signal)
      .then((options) => {
        if (controller.signal.aborted) return
        if (!options.length) throw new Error('Danh sách model đang trống.')
        setModels(options)
        setModelId(options[0].model_id)
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setCatalogError(caught instanceof Error ? caught.message : 'Không đọc được danh sách model.')
      })
    return () => controller.abort()
  }, [catalogRetry])

  function handleModelChange(nextModel: string) {
    setModelId(nextModel)
    setData(null)
    setTrace(null)
    setLayerData(null)
    setError(null)
    setLayerError(null)
    setSelectedPosition(null)
    setSelectedLayer(0)
    setQueryHead(0)
    cache.current.clear()
  }

  useEffect(() => {
    const controller = new AbortController()
    setLayerData(null)
    setLayerError(null)
    setLoadingLayer(false)
    if (!trace) return () => controller.abort()

    const key = `${trace.trace_id}:${selectedLayer}`
    const cached = cache.current.get(key)
    if (cached) {
      setLayerData(cached)
      return () => controller.abort()
    }

    setLoadingLayer(true)
    getLayerQKV(trace.trace_id, selectedLayer, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return
        cache.current.set(key, result)
        setLayerData(result)
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setLayerError(caught instanceof Error ? caught.message : 'Không lấy được QKV.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingLayer(false)
      })

    return () => controller.abort()
  }, [trace, selectedLayer, retryLayer])

  async function handleRun(mode: 'tokenize' | 'trace') {
    setLoading(mode)
    setError(null)
    setData(null)
    setTrace(null)
    setSelectedPosition(null)
    cache.current.clear()

    try {
      if (mode === 'tokenize') {
        setData(await tokenizeText(text, modelId))
      } else {
        const result = await createTrace(text, modelId)
        setData(result.tokenization)
        setSelectedLayer((previous) => previous < result.model_info.num_layers ? previous : 0)
        setQueryHead((previous) => previous < result.model_info.num_query_heads ? previous : 0)
        setSelectedPosition(0)
        setTrace(result)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể gọi API.')
    } finally {
      setLoading(null)
    }
  }

  return (
    <main className="page">
      <header className="page-header">
        <p className="eyebrow">Visualize LLM · QKV đầu vào attention</p>
        <h1>Khám phá từ token đến head</h1>
        <p>Chạy một input, duyệt layer và head từ trace đã thu. Mỗi heatmap hiển thị activation thật của một head.</p>
      </header>

      <section className="input-panel" aria-label="Nhập text">
        <h2>Input · Batch size 1</h2>
        <label htmlFor="model-id">Model ID</label>
        <select id="model-id" className="model-selector" value={modelId} onChange={(event) => handleModelChange(event.target.value)} disabled={loading !== null || models.length === 0}>
          {!models.length && <option value="">Đang đọc danh sách model…</option>}
          {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.label} — {model.model_id}</option>)}
        </select>
        <p className="muted">Đổi lựa chọn sẽ xóa kết quả đang xem. Trọng số chỉ được nạp khi bấm chạy model; lần đầu có thể cần tải từ Hugging Face.</p>
        {catalogError && <div role="alert" className="error"><p>{catalogError}</p><button type="button" onClick={() => setCatalogRetry((value) => value + 1)}>Thử lại</button></div>}
        <label htmlFor="source-text">Text đầu vào</label>
        <textarea id="source-text" value={text} onChange={(event) => setText(event.target.value)} rows={4} maxLength={1000} spellCheck={false} disabled={loading !== null} />
        <div className="input-actions">
          <span>{Array.from(text).length} ký tự · Trace hỗ trợ 1–32 token</span>
          <div className="run-buttons">
            <button className="tokenize-button" type="button" onClick={() => void handleRun('tokenize')} disabled={loading !== null || !modelId || text.length === 0}>{loading === 'tokenize' ? 'Đang tokenize…' : 'Chỉ tokenize'}</button>
            <button type="button" onClick={() => void handleRun('trace')} disabled={loading !== null || !modelId || text.length === 0}>{loading === 'trace' ? 'Đang chạy model…' : 'Chạy model, tạo trace'}</button>
          </div>
        </div>
        {loading === 'trace' && <p role="status" className="muted">Đang thu QKV của các layer. Lần đầu có thể cần tải và nạp trọng số lên GPU.</p>}
      </section>

      {error && <p className="error" role="alert">{error}</p>}
      {data && <TokenList data={data} selectedPosition={selectedPosition} onSelect={setSelectedPosition} />}
      {trace && <p className="trace-label">Trace: <code>{trace.trace_id}</code> · Dữ liệu của text trong bảng token phía trên. Chỉnh ô nhập chưa thay đổi trace này.</p>}

      {modelId && <ModelNavigator key={modelId} modelId={modelId}
        layer={selectedLayer} head={queryHead} traceInfo={trace?.model_info}
        onSelectLayer={(layer) => { setSelectedLayer(layer); setQueryHead(0) }}
        onSelectHead={setQueryHead}
      />}

      {trace && loadingLayer && <p role="status" className="muted">Đang đọc QKV layer {selectedLayer} từ trace…</p>}
      {trace && layerError && <div className="error" role="alert"><p>{layerError} Nếu backend đã restart hoặc trace bị thay thế, hãy tạo trace mới.</p><button className="secondary-button" type="button" onClick={() => setRetryLayer((value) => value + 1)}>Đọc lại layer</button></div>}
      {trace && layerData && layerData.trace_id === trace.trace_id && layerData.layer_index === selectedLayer && (
        <Suspense fallback={<p className="muted" role="status">Đang nạp Plotly…</p>}>
          <QKVHeatmaps layer={layerData} info={trace.model_info} tokens={trace.tokenization.tokens} queryHead={queryHead} selectedPosition={selectedPosition} onSelectToken={setSelectedPosition} />
        </Suspense>
      )}
    </main>
  )
}
