import { useEffect, useState } from 'react'
import { getModelInfo } from '../api/model'
import type { ModelInfo } from '../types/model'

type Props = {
  layer: number | null
  head: number | null
  onSelectLayer: (layer: number) => void
  onSelectHead: (head: number) => void
  traceInfo?: ModelInfo
}

export default function ModelNavigator({ layer, head, onSelectLayer, onSelectHead, traceInfo }: Props) {
  const [loadedInfo, setInfo] = useState<ModelInfo | null>(null)
  const info = traceInfo ?? loadedInfo
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setInfo(null)
    setError(null)

    getModelInfo(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setInfo(result)
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof Error ? caught.message : 'Không đọc được config model')
        }
      })

    return () => controller.abort()
  }, [retry])

  const kvHead = info && head !== null ? Math.floor(head / info.queries_per_kv_head) : null

  return (
    <section className="model-panel" aria-labelledby="model-heading">
      <div className="result-heading">
        <div>
          <p className="eyebrow">Cấu trúc từ config</p>
          <h2 id="model-heading">Khám phá model</h2>
        </div>
        {info && <span className="model-tag">{info.model_id}</span>}
      </div>

      {error && !traceInfo ? (
        <div role="alert" className="model-error">
          <p>{error}. Kiểm tra FastAPI đang chạy ở cổng 8000.</p>
          <button type="button" className="secondary-button" onClick={() => setRetry((value) => value + 1)}>Thử lại</button>
        </div>
      ) : !info ? (
        <p role="status" className="muted">Đang đọc thông tin model…</p>
      ) : (
        <>
          <dl className="model-stats">
            <div><dt>Layer</dt><dd>{info.num_layers}</dd></div>
            <div><dt>Query head / layer</dt><dd>{info.num_query_heads}</dd></div>
            <div><dt>KV head / layer</dt><dd>{info.num_kv_heads}</dd></div>
            <div><dt>Chiều mỗi head</dt><dd>{info.head_dim}</dd></div>
            <div><dt>Hidden size</dt><dd>{info.hidden_size}</dd></div>
          </dl>

          <p className="muted">Chọn một layer để xem các head. Chỉ số bắt đầu từ 0; màu viền biểu thị lựa chọn.</p>
          <div className="layer-grid" aria-label="Các layer của model">
            {Array.from({ length: info.num_layers }, (_, index) => (
              <button
                key={index}
                type="button"
                className={`layer-card${layer === index ? ' active' : ''}`}
                aria-pressed={layer === index}
                onClick={() => onSelectLayer(index)}
              >
                <span className="field-label">Layer</span>
                <strong>{index}</strong>
              </button>
            ))}
          </div>

          {layer !== null && (
            <div className="head-section">
              <nav className="model-path" aria-label="Vị trí trong model">
                {info.model_id} / Layer {layer}{head !== null && ` / Query head ${head}`}
              </nav>
              <h3>Query head của layer {layer}</h3>
              <p className="muted">Mỗi nhóm {info.queries_per_kv_head} query head dùng chung một KV head.</p>
              <div className="head-grid" aria-label={`Query head của layer ${layer}`}>
                {Array.from({ length: info.num_query_heads }, (_, index) => (
                  <button
                    key={index}
                    type="button"
                    className={`head-card${head === index ? ' active' : ''}`}
                    aria-pressed={head === index}
                    onClick={() => onSelectHead(index)}
                  >
                    <strong>Q head {index}</strong>
                    <span>KV head {Math.floor(index / info.queries_per_kv_head)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {head !== null && (
            <div className="head-summary" aria-live="polite">
              <h3>Layer {layer} · Query head {head} → KV head {kvHead}</h3>
              <p>Với một input có S token, các heatmap sẽ có shape kỳ vọng:</p>
              <div className="tensor-preview-grid">
                <div><strong>Q · head {head}</strong><code>[S, {info.head_dim}]</code><span>Đầu vào attention</span></div>
                <div><strong>K · head {kvHead}</strong><code>[S, {info.head_dim}]</code><span>Đầu vào attention</span></div>
                <div><strong>V · head {kvHead}</strong><code>[S, Dv]</code><span>Chiều thực tế lấy từ trace</span></div>
              </div>
            </div>
          )}
          <p className="config-note">{traceInfo ? 'Trace đã được tạo. Layer được tải khi chọn và giữ trong cache; đổi head/token không chạy lại model.' : 'Đây là thông tin kiến trúc từ config. Bấm “Chạy model, tạo trace” để thu activation thật.'}</p>
        </>
      )}
    </section>
  )
}
