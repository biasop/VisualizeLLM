import type { ModelInfo } from '../types/model'

export async function getModelInfo(signal: AbortSignal): Promise<ModelInfo> {
  const response = await fetch('/api/model-info', { signal })
  if (!response.ok) {
    throw new Error(`Model info API trả HTTP ${response.status}`)
  }
  return (await response.json()) as ModelInfo
}
