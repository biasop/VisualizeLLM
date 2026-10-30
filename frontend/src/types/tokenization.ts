export type TokenItem = {
  position: number
  id: number
  raw_token: string
  start: number | null
  end: number | null
  attention_mask: number
  is_special: boolean
}

export type TokenizeResponse = {
  text: string
  model_id: string
  tokens: TokenItem[]
}
