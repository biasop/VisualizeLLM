import type { TokenizeResponse } from '../types/tokenization'

export async function tokenizeText(text: string): Promise<TokenizeResponse> {
  const response = await fetch('/api/tokenize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) {
    throw new Error(`Tokenize API trả HTTP ${response.status}`)
  }

  return (await response.json()) as TokenizeResponse
}
