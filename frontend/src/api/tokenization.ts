import type { TokenizeResponse } from '../types/tokenization'

export async function tokenizeText(text: string, modelId: string): Promise<TokenizeResponse> {
  const response = await fetch('/api/tokenize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model_id: modelId }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Tokenize API trả HTTP ${response.status}`)
  }

  return (await response.json()) as TokenizeResponse
}
