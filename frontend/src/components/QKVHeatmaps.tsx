import { useMemo, useState, useEffect } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import type { PlotMouseEvent } from 'plotly.js'
import type { ModelInfo } from '../types/model'
import type { TokenItem } from '../types/tokenization'
import type { QKVLayer } from '../types/trace'
import { compareVectors, selectHeadViews, tensorColorLimit, transformTokenAxis } from '../visualization/qkv'
import type { QKVViewMode } from '../visualization/qkv'

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
  const [viewMode, setViewMode] = useState<QKVViewMode>('center')
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
  const referenceRow = viewMode === 'delta' ? comparisonRow : 0
  const transformed = useMemo(() => Object.fromEntries(layer.tensors.map((tensor) => [tensor.name, {
    ...tensor, values: transformTokenAxis(tensor.values, viewMode, referenceRow),
  }])), [layer, viewMode, referenceRow])
  const tensorLimits = useMemo(() => Object.fromEntries(Object.values(transformed).map((tensor) => [tensor.name, tensorColorLimit(tensor)])), [transformed])
  const commonLimit = viewMode === 'absolute' ? layer.color_limit : Math.max(...Object.values(tensorLimits))

  if (selection.error) return <p role="alert" className="error">{selection.error}</p>

  const labels = tokens.map((token) => `#${token.position} ${token.raw_token.replaceAll('Ġ', '␠').replaceAll('Ċ', '↵')}`)
  const selectedRow = Math.max(0, positions.indexOf(selectedPosition ?? 0))
  const viewLabels: Record<QKVViewMode, string> = {
    absolute: 'Activation gốc', delta: 'Δ theo token tham chiếu', center: 'Trừ trung bình', zscore: 'Z-score', minmax: 'Min–max',
  }
  const descriptions: Record<QKVViewMode, string> = {
    absolute: 'ĐANG VẼ ACTIVATION GỐC. Giá trị có cùng đơn vị và dấu với tensor đã thu.',
    delta: `ĐANG VẼ Δ: X[i,h,d] − X[${positions[comparisonRow]},h,d]. Hàng tham chiếu bằng 0. Đây là dữ liệu dẫn xuất, không phải activation gốc.`,
    center: 'TRỪ TRUNG BÌNH: X′[i,h,d] = X[i,h,d] − μ[h,d]. μ được tính qua tất cả token, riêng từng head/chiều của mỗi tensor. Đỏ = cao hơn trung bình, xanh = thấp hơn; vẫn giữ đơn vị activation.',
    zscore: 'Z-SCORE: Z[i,h,d] = (X[i,h,d] − μ[h,d]) / σ[h,d]. Thống kê qua trục token, riêng từng head/chiều và từng tensor; σ population (chia S). Chiều không biến thiên được đặt 0. Giá trị là số độ lệch chuẩn, không còn là độ lớn activation. Với ít token, biến thiên rất nhỏ cũng có thể được khuếch đại.',
    minmax: 'MIN–MAX: (X[i,h,d] − min[h,d]) / (max[h,d] − min[h,d]). Min/max tính qua trục token, riêng từng head/chiều và từng tensor. Chiều hằng được đặt 0. Màu cho biết vị trí trong khoảng [0,1], không cho biết dấu hoặc biên độ activation gốc.',
  }

  return (
    <section className="result" aria-label="Heatmap QKV">
      <p className="eyebrow">Activation thật · Batch size 1</p>
      <h2>Layer {layer.layer_index} · Query head {queryHead} → KV head {Math.floor(queryHead / info.queries_per_kv_head)}</h2>
      <p className="muted">Hàng là token, cột là chiều vector. Bấm một ô để xem giá trị và làm nổi hàng tương ứng trên cả ba tensor.</p>
      <div className="heatmap-controls">
        <label>Giá trị biểu diễn <select value={viewMode} onChange={(event) => setViewMode(event.target.value as QKVViewMode)}>
          <option value="center">Trừ trung bình theo token</option>
          <option value="zscore">Z-score theo token</option>
          <option value="minmax">Min–max theo token [0,1]</option>
          <option value="delta">Δ so với token tham chiếu</option>
          <option value="absolute">Activation gốc</option>
        </select></label>
        <label>Thang màu <select value={colorMode} disabled={viewMode === 'minmax'} onChange={(event) => setColorMode(event.target.value as 'layer' | 'tensor')}>
          <option value="layer">Chung cho Q/K/V của layer</option>
          <option value="tensor">Riêng từng tensor, cố định qua mọi head</option>
        </select></label>
        <label>{viewMode === 'delta' ? 'Token tham chiếu Δ' : 'Token so sánh trong bảng số'} <select value={comparisonPosition} onChange={(event) => setComparisonPosition(Number(event.target.value))}>
          {tokens.map((token) => <option value={token.position} key={token.position}>#{token.position} · {token.raw_token.replaceAll('Ġ', '␠')}</option>)}
        </select></label>
      </div>
      <p className="view-mode-note">{descriptions[viewMode]} Hover luôn hiển thị giá trị gốc và giá trị đang vẽ. Các biến đổi chỉ phục vụ visualization, không sửa trace hay dùng thay Q/K/V khi tính attention.</p>
      {viewMode !== 'absolute' && tokens.length === 1 && <p className="muted">Input chỉ có một token nên chế độ này cho toàn bộ giá trị bằng 0. Chọn “Activation gốc” để xem vector của token này.</p>}
      <p className="color-note">{viewMode === 'minmax' ? 'Thang cố định [0,1] cho mọi head/tensor. Trắng = min, xanh đậm = max; không biểu diễn dấu âm/dương.' : colorMode === 'layer'
        ? `Thang chung [−${commonLimit}, +${commonLimit}], lấy max trị tuyệt đối của dữ liệu đang vẽ trên mọi token/head của Q/K/V trong layer.`
        : 'Mỗi tensor có thang riêng, lấy max trị tuyệt đối của dữ liệu đang vẽ trên mọi token/head của tensor đó. Cùng màu ở Q/K/V KHÔNG có nghĩa cùng độ lớn.'} Thống kê và thang giữ nguyên khi đổi head/token đang chọn; tính lại khi đổi layer hoặc chế độ (và token tham chiếu trong chế độ Δ). {viewMode !== 'minmax' && 'Xanh: âm · trắng: 0 · đỏ: dương trong dữ liệu đang vẽ. Không cắt giá trị.'}</p>
      <div className="heatmap-grid">
        {selection.views.map(({ tensor, head, matrix }) => {
          const headDim = tensor.shape[2]
          const dimensions = Array.from({ length: headDim }, (_, index) => index)
          const limit = viewMode === 'minmax' ? 1 : colorMode === 'layer' ? commonLimit : tensorLimits[tensor.name]
          const displayedMatrix = transformed[tensor.name].values.map((row) => [...row[head]])
          const comparison = compareVectors(matrix[selectedRow], matrix[comparisonRow])
          return (
          <article className="heatmap-panel" key={tensor.name}>
            <h3>{tensor.name} · head {head} · {viewLabels[viewMode]}</h3>
            <p className="muted">Đầu vào attention, sau các xử lý của model; không tự áp dụng thêm RoPE.<br />Shape đang xem [{tokens.length}, {headDim}] · nguồn {tensor.source_dtype}</p>
            <p className="vector-comparison">Token #{positions[selectedRow]} ↔ #{positions[comparisonRow]}: <strong>{comparison.differentDimensions}/{headDim}</strong> chiều khác nhau · max |Δ| = <code>{String(comparison.maxAbsDifference)}</code></p>
            <p className="muted">Màu: [{viewMode === 'minmax' ? 0 : -limit}, {limit}]</p>
            <Plot
              data={[{
                type: 'heatmap', z: displayedMatrix, x: dimensions, y: positions,
                text: matrix.map((row, rowIndex) => row.map((value, dimension) => `Gốc: ${String(value)}<br>${viewLabels[viewMode]}: ${String(displayedMatrix[rowIndex][dimension])}`)),
                colorscale: viewMode === 'minmax' ? [[0, '#f7fbff'], [0.5, '#6baed6'], [1, '#08306b']] : [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
                zmin: viewMode === 'minmax' ? 0 : -limit, zmax: limit, zauto: false,
                colorbar: { thickness: 10, len: 0.85 },
                hovertemplate: `${tensor.name} · layer ${layer.layer_index} · head ${head}<br>Token #%{y} · chiều %{x}<br>%{text}<extra></extra>`,
              }]}
              layout={{
                autosize: true, height: Math.max(350, tokens.length * 24 + 130),
                margin: { l: 90, r: 45, t: 15, b: 55 },
                xaxis: { title: { text: 'Chiều vector' }, dtick: 16, range: [-0.5, headDim - 0.5] },
                yaxis: { title: { text: 'Token' }, autorange: 'reversed', tickmode: 'array', tickvals: positions, ticktext: labels, tickfont: { size: 10 } },
                shapes: selectedPosition === null ? [] : [{
                  type: 'rect', xref: 'x', yref: 'y', x0: -0.5, x1: headDim - 0.5,
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
                <thead><tr><th>Chiều</th><th>Gốc #{positions[selectedRow]}</th><th>Gốc #{positions[comparisonRow]}</th><th>Δ gốc</th><th>{viewLabels[viewMode]} #{positions[selectedRow]}</th></tr></thead>
                <tbody>{matrix[selectedRow].map((value, dimension) => <tr key={dimension}><td>{dimension}</td><td>{String(value)}</td><td>{String(matrix[comparisonRow][dimension])}</td><td>{String(comparison.differences[dimension])}</td><td>{String(displayedMatrix[selectedRow][dimension])}</td></tr>)}</tbody>
              </table></div>
            </details>
          </article>
        )})}
      </div>
      {cell && selectedPosition === cell.token && <p className="cell-detail" aria-live="polite"><strong>{cell.name}</strong> · layer {layer.layer_index} · head {cell.head} · token #{cell.token} · chiều {cell.dimension}<br />Gốc: <code>{String(cell.value)}</code> · {viewLabels[viewMode]}: <code>{String(cell.displayedValue)}</code></p>}
    </section>
  )
}
