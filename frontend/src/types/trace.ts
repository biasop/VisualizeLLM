import type { ModelInfo } from './model'
import type { TokenizeResponse } from './tokenization'

export type TraceCreated = {
  trace_id: string
  tokenization: TokenizeResponse
  model_info: ModelInfo
}

export type TensorPayload = {
  name: string
  shape: number[]
  axes: string[]
  stage: string
  source_dtype: string
  values: number[][][]
}

export type QKVLayer = {
  trace_id: string
  layer_index: number
  color_limit: number
  tensors: TensorPayload[]
}
