import type { ModelInfo } from '../types/model'
import type { QKVLayer, TensorPayload } from '../types/trace'

export type HeadView = { tensor: TensorPayload; head: number; matrix: number[][] }
export type QKVViewMode = 'absolute' | 'delta' | 'center' | 'zscore' | 'minmax'

// Axis 0 is token: each (head, dimension) has independent statistics.
// Input and trace metadata are never modified.
export function transformTokenAxis(values: number[][][], mode: QKVViewMode, referenceRow = 0): number[][][] {
  const result = values.map((token) => token.map((vector) => [...vector]))
  if (mode === 'absolute' || values.length === 0) return result
  if (mode === 'delta' && !values[referenceRow]) throw new Error('Token tham chiếu không hợp lệ.')

  for (let head = 0; head < values[0].length; head++) {
    for (let dimension = 0; dimension < values[0][head].length; dimension++) {
      let mean = 0
      let m2 = 0
      let minimum = Infinity
      let maximum = -Infinity
      // Welford's algorithm; population variance (correction=0).
      for (let token = 0; token < values.length; token++) {
        const value = values[token][head][dimension]
        const delta = value - mean
        mean += delta / (token + 1)
        m2 += delta * (value - mean)
        minimum = Math.min(minimum, value)
        maximum = Math.max(maximum, value)
      }
      const std = Math.sqrt(Math.max(0, m2 / values.length))
      const range = maximum - minimum
      for (let token = 0; token < values.length; token++) {
        const value = values[token][head][dimension]
        if (mode === 'delta') result[token][head][dimension] = value - values[referenceRow][head][dimension]
        if (mode === 'center') result[token][head][dimension] = value - mean
        if (mode === 'zscore') result[token][head][dimension] = std === 0 ? 0 : (value - mean) / std
        if (mode === 'minmax') result[token][head][dimension] = range === 0 ? 0 : (value - minimum) / range
      }
    }
  }
  return result
}

export function compareVectors(left: number[], right: number[]) {
  if (left.length !== right.length) throw new Error('Hai vector phải cùng số chiều.')
  const differences = left.map((value, index) => value - right[index])
  return {
    differences,
    differentDimensions: differences.filter((value) => value !== 0).length,
    maxAbsDifference: differences.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0),
  }
}

export function differenceFromReference(matrix: number[][], referenceRow: number): number[][] {
  const reference = matrix[referenceRow]
  if (!reference) throw new Error('Token tham chiếu không hợp lệ.')
  return matrix.map((row) => row.map((value, dimension) => value - reference[dimension]))
}

export function tensorColorLimit(tensor: TensorPayload, referenceRow?: number): number {
  let limit = 0
  for (const token of tensor.values) {
    for (let head = 0; head < token.length; head++) {
      for (let dimension = 0; dimension < token[head].length; dimension++) {
        const reference = referenceRow === undefined ? 0 : tensor.values[referenceRow][head][dimension]
        limit = Math.max(limit, Math.abs(token[head][dimension] - reference))
      }
    }
  }
  return limit || 1
}

// Pure data selection: no fetching and no model execution.
export function selectHeadViews(layer: QKVLayer, info: ModelInfo, tokenCount: number, queryHead: number): HeadView[] {
  if (!Number.isInteger(queryHead) || queryHead < 0 || queryHead >= info.num_query_heads) {
    throw new Error('Query head không hợp lệ.')
  }
  if (!Number.isFinite(layer.color_limit) || layer.color_limit <= 0) {
    throw new Error('Giới hạn màu của trace không hợp lệ.')
  }
  const kvHead = Math.floor(queryHead / info.queries_per_kv_head)
  return [
    { name: 'q_attention_input', heads: info.num_query_heads, head: queryHead, axis: 'query_head', stage: 'attention_input' },
    { name: 'k_attention_input', heads: info.num_kv_heads, head: kvHead, axis: 'kv_head', stage: 'attention_input' },
    { name: 'v_attention_input', heads: info.num_kv_heads, head: kvHead, axis: 'kv_head', stage: 'attention_input' },
  ].map(({ name, heads, head, axis, stage }) => {
    const tensor = layer.tensors.find((candidate) => candidate.name === name)
    if (!tensor || tensor.stage !== stage || tensor.axes.join(',') !== `token,${axis},head_dimension`
      || tensor.shape.length !== 3 || tensor.shape[0] !== tokenCount || tensor.shape[1] !== heads
      || !Number.isInteger(tensor.shape[2]) || tensor.shape[2] <= 0
      || (name !== 'v_attention_input' && tensor.shape[2] !== info.head_dim)
      || tensor.values.length !== tokenCount
      || tensor.values.some((row) => row.length !== heads || row.some((vector) => vector.length !== tensor.shape[2] || vector.some((value) => !Number.isFinite(value))))) {
      throw new Error(`Tensor ${name} không đúng shape, trục, stage hoặc chứa giá trị không hợp lệ.`)
    }
    return { tensor, head, matrix: tensor.values.map((row) => [...row[head]]) }
  })
}
