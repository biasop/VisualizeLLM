import type { ModelInfo, ModelOption } from '../types/model'

export async function getModels(signal: AbortSignal): Promise<ModelOption[]> {
  const response = await fetch('/api/models', { signal })
  if (!response.ok) throw new Error(`Danh sách model trả HTTP ${response.status}`)
  return (await response.json()) as ModelOption[]
}

export async function getModelInfo(modelId: string, signal: AbortSignal): Promise<ModelInfo> {
  const response = await fetch(`/api/model-info?model_id=${encodeURIComponent(modelId)}`, { signal })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Model info API trả HTTP ${response.status}`)
  }
  return (await response.json()) as ModelInfo
}
