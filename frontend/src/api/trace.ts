import type { QKVLayer, TraceCreated } from '../types/trace'

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch { /* Server may return a plain-text error. */ }
    throw new Error(detail || `API trả HTTP ${response.status}`)
  }
  return (await response.json()) as T
}

export async function createTrace(text: string): Promise<TraceCreated> {
  return readResponse<TraceCreated>(await fetch('/api/traces', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  }))
}

export async function getLayerQKV(traceId: string, layer: number, signal: AbortSignal): Promise<QKVLayer> {
  const result = await readResponse<QKVLayer>(await fetch(
    `/api/traces/${encodeURIComponent(traceId)}/layers/${layer}/qkv`, { signal },
  ))
  if (result.trace_id !== traceId || result.layer_index !== layer) {
    throw new Error('Response QKV không khớp trace/layer đã yêu cầu.')
  }
  return result
}
