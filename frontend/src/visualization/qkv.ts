import type { ModelInfo } from '../types/model'
import type { QKVLayer, TensorPayload } from '../types/trace'

export type HeadView = { tensor: TensorPayload; head: number; matrix: number[][] }

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
    { name: 'q_pre_rope', heads: info.num_query_heads, head: queryHead, axis: 'query_head', stage: 'pre_rope' },
    { name: 'k_pre_rope', heads: info.num_kv_heads, head: kvHead, axis: 'kv_head', stage: 'pre_rope' },
    { name: 'v', heads: info.num_kv_heads, head: kvHead, axis: 'kv_head', stage: 'after_projection' },
  ].map(({ name, heads, head, axis, stage }) => {
    const tensor = layer.tensors.find((candidate) => candidate.name === name)
    if (!tensor || tensor.stage !== stage || tensor.axes.join(',') !== `token,${axis},head_dimension`
      || tensor.shape.join(',') !== `${tokenCount},${heads},${info.head_dim}`
      || tensor.values.length !== tokenCount
      || tensor.values.some((row) => row.length !== heads || row.some((vector) => vector.length !== info.head_dim || vector.some((value) => !Number.isFinite(value))))) {
      throw new Error(`Tensor ${name} không đúng shape, trục, stage hoặc chứa giá trị không hợp lệ.`)
    }
    return { tensor, head, matrix: tensor.values.map((row) => [...row[head]]) }
  })
}
