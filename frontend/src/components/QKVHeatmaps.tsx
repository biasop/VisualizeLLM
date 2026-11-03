import { useMemo, useState, useEffect } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import type { PlotMouseEvent } from 'plotly.js'
import type { ModelInfo } from '../types/model'
import type { TokenItem } from '../types/tokenization'
import type { QKVLayer } from '../types/trace'
import { compareVectors, differenceFromReference, selectHeadViews, tensorColorLimit } from '../visualization/qkv'

const Plot = createPlotlyComponent(Plotly)

type Props = {
  layer: QKVLayer
  info: ModelInfo
  tokens: TokenItem[]
  queryHead: number
  selectedPosition: number | null
  onSelectToken: (position: number) => void
}

export default function QKVHeatmaps({ layer, info, tokens, queryHead, selectedPosition, onSelectToken }: Props) {
  const [cell, setCell] = useState<{ name: string; head: number; token: number; dimension: number; value: number; displayedValue: number } | null>(null)
  const [viewMode, setViewMode] = useState<'absolute' | 'delta'>('delta')
  const [colorMode, setColorMode] = useState<'layer' | 'tensor'>('tensor')
  const [comparisonPosition, setComparisonPosition] = useState(0)
  useEffect(() => setCell(null), [layer, queryHead, viewMode, comparisonPosition])
  useEffect(() => setComparisonPosition(0), [layer.trace_id])
  const positions = tokens.map((token) => token.position)
  const comparisonRow = Math.max(0, positions.indexOf(comparisonPosition))
  const selection = useMemo(() => {
    try { return { views: selectHeadViews(layer, info, tokens.length, queryHead), error: null } }
    catch (error) { return { views: [], error: error instanceof Error ? error.message : 'Trace QKV không hợp lệ.' } }
  }, [layer, info, tokens.length, queryHead])
  const tensorLimits = useMemo(() => Object.fromEntries(layer.tensors.map((tensor) => [tensor.name, tensorColorLimit(tensor, viewMode === 'delta' ? comparisonRow : undefined)])), [layer, viewMode, comparisonRow])
  const commonLimit = viewMode === 'absolute' ? layer.color_limit : Math.max(...Object.values(tensorLimits))

  if (selection.error) return <p role="alert" className="error">{selection.error}</p>

  const dimensions = Array.from({ length: info.head_dim }, (_, index) => index)
  const labels = tokens.map((token) => `#${token.position} ${token.raw_token.replaceAll('Ġ', '␠').replaceAll('Ċ', '↵')}`)
  const selectedRow = Math.max(0, positions.indexOf(selectedPosition ?? 0))

  return (
    <section className="result" aria-label="Heatmap QKV">
      <p className="eyebrow">Activation thật · Batch size 1</p>
      <h2>Layer {layer.layer_index} · Query head {queryHead} → KV head {Math.floor(queryHead / info.queries_per_kv_head)}</h2>
      <p className="muted">Hàng là token, cột là chiều vector. Bấm một ô để xem giá trị và làm nổi hàng tương ứng trên cả ba tensor.</p>
      <div className="heatmap-controls">
        <label>Giá trị biểu diễn <select value={viewMode} onChange={(event) => setViewMode(event.target.value as 'absolute' | 'delta')}>
          <option value="delta">Δ so với token tham chiếu</option>
          <option value="absolute">Activation gốc</option>
        </select></label>
        <label>Thang màu <select value={colorMode} onChange={(event) => setColorMode(event.target.value as 'layer' | 'tensor')}>
          <option value="layer">Chung cho Q/K/V của layer</option>
          <option value="tensor">Riêng từng tensor, cố định qua mọi head</option>
        </select></label>
        <label>Token tham chiếu (cố định khi bấm hàng) <select value={comparisonPosition} onChange={(event) => setComparisonPosition(Number(event.target.value))}>
          {tokens.map((token) => <option value={token.position} key={token.position}>#{token.position} · {token.raw_token.replaceAll('Ġ', '␠')}</option>)}
        </select></label>
      </div>
      <p className="view-mode-note">{viewMode === 'delta'
        ? `ĐANG VẼ CHÊNH LỆCH: ΔX[i,h,d] = X[i,h,d] − X[${positions[comparisonRow]},h,d]. Hàng tham chiếu bằng 0. Đây là phép biến đổi để quan sát khác biệt giữa token, không phải activation gốc hay một bước tính của model. Hover hiển thị cả giá trị gốc và Δ.`
        : 'ĐANG VẼ ACTIVATION GỐC. Các token có thể có nền activation lớn giống nhau nên màu gần nhau dù giá trị khác nhau. Chuyển sang Δ để quan sát chênh lệch.'}</p>
      {viewMode === 'delta' && tokens.length === 1 && <p className="muted">Input chỉ có một token nên toàn bộ Δ bằng 0. Chọn “Activation gốc” để xem vector của token này.</p>}
      <p className="color-note">{colorMode === 'layer'
        ? `Thang chung [−${commonLimit}, +${commonLimit}], lấy max trị tuyệt đối của dữ liệu đang vẽ trên mọi token/head của Q/K/V trong layer.`
        : 'Mỗi tensor có thang riêng, lấy max trị tuyệt đối của dữ liệu đang vẽ trên mọi token/head của tensor đó. Cùng màu ở Q/K/V KHÔNG có nghĩa cùng độ lớn.'} Thang giữ nguyên khi đổi head/token đang chọn; tính lại khi đổi layer, chế độ hoặc token tham chiếu. Xanh: âm · trắng: 0 · đỏ: dương. Không chia chuẩn hóa các giá trị.</p>
      <div className="heatmap-grid">
        {selection.views.map(({ tensor, head, matrix }) => {
          const limit = colorMode === 'layer' ? commonLimit : tensorLimits[tensor.name]
          const displayedMatrix = viewMode === 'delta' ? differenceFromReference(matrix, comparisonRow) : matrix
          const comparison = compareVectors(matrix[selectedRow], matrix[comparisonRow])
          return (
          <article className="heatmap-panel" key={tensor.name}>
            <h3>{viewMode === 'delta' ? 'Δ ' : ''}{tensor.name} · head {head}</h3>
            <p className="muted">{tensor.stage === 'pre_rope' ? 'Sau projection, trước RoPE' : 'Sau projection; V không áp dụng RoPE'}<br />Shape đang xem [{tokens.length}, {info.head_dim}] · nguồn {tensor.source_dtype}</p>
            <p className="vector-comparison">Token #{positions[selectedRow]} ↔ #{positions[comparisonRow]}: <strong>{comparison.differentDimensions}/{info.head_dim}</strong> chiều khác nhau · max |Δ| = <code>{String(comparison.maxAbsDifference)}</code></p>
            <p className="muted">Màu: [−{limit}, +{limit}]</p>
            <Plot
              data={[{
                type: 'heatmap', z: displayedMatrix, x: dimensions, y: positions,
                text: matrix.map((row, rowIndex) => row.map((value, dimension) => `Gốc: ${String(value)}${viewMode === 'delta' ? `<br>Δ so với #${positions[comparisonRow]}: ${String(displayedMatrix[rowIndex][dimension])}` : ''}`)),
                colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmin: -limit, zmax: limit, zmid: 0, zauto: false,
                colorbar: { thickness: 10, len: 0.85 },
                hovertemplate: `${tensor.name} · layer ${layer.layer_index} · head ${head}<br>Token #%{y} · chiều %{x}<br>%{text}<extra></extra>`,
              }]}
              layout={{
                autosize: true, height: Math.max(350, tokens.length * 24 + 130),
                margin: { l: 90, r: 45, t: 15, b: 55 },
                xaxis: { title: { text: 'Chiều vector' }, dtick: 16, range: [-0.5, info.head_dim - 0.5] },
                yaxis: { title: { text: 'Token' }, autorange: 'reversed', tickmode: 'array', tickvals: positions, ticktext: labels, tickfont: { size: 10 } },
                shapes: selectedPosition === null ? [] : [{
                  type: 'rect', xref: 'x', yref: 'y', x0: -0.5, x1: info.head_dim - 0.5,
                  y0: selectedPosition - 0.5, y1: selectedPosition + 0.5,
                  line: { color: '#111827', width: 2 }, fillcolor: 'rgba(0,0,0,0)',
                }],
              }}
              config={{ responsive: true, displaylogo: false, scrollZoom: false }}
              useResizeHandler
              style={{ width: '100%' }}
              onClick={(event: PlotMouseEvent) => {
                const point = event.points[0]
                if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') return
                const row = positions.indexOf(point.y)
                const value = matrix[row]?.[point.x]
                if (value === undefined) return
                onSelectToken(point.y)
                setCell({ name: tensor.name, head, token: point.y, dimension: point.x, value, displayedValue: displayedMatrix[row][point.x] })
              }}
            />
            <details className="vector-values">
              <summary>Xem số từng chiều: token #{positions[selectedRow]} và #{positions[comparisonRow]}</summary>
              <div className="vector-table-scroll"><table>
                <thead><tr><th>Chiều</th><th>#{positions[selectedRow]}</th><th>#{positions[comparisonRow]}</th><th>Δ (trái − phải)</th></tr></thead>
                <tbody>{matrix[selectedRow].map((value, dimension) => <tr key={dimension}><td>{dimension}</td><td>{String(value)}</td><td>{String(matrix[comparisonRow][dimension])}</td><td>{String(comparison.differences[dimension])}</td></tr>)}</tbody>
              </table></div>
            </details>
          </article>
        )})}
      </div>
      {cell && selectedPosition === cell.token && <p className="cell-detail" aria-live="polite"><strong>{cell.name}</strong> · layer {layer.layer_index} · head {cell.head} · token #{cell.token} · chiều {cell.dimension}<br />Gốc: <code>{String(cell.value)}</code>{viewMode === 'delta' && <> · Δ so với token #{positions[comparisonRow]}: <code>{String(cell.displayedValue)}</code></>}</p>}
    </section>
  )
}
