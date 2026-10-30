import type { TokenItem, TokenizeResponse } from '../types/tokenization'

type TokenListProps = {
  data: TokenizeResponse
  selectedPosition: number | null
  onSelect: (position: number) => void
}

function sourceSpan(text: string, token: TokenItem): string | null {
  if (token.start === null || token.end === null) return null
  return Array.from(text).slice(token.start, token.end).join('')
}

export default function TokenList({ data, selectedPosition, onSelect }: TokenListProps) {
  const selected = data.tokens.find((token) => token.position === selectedPosition)
  const span = selected ? sourceSpan(data.text, selected) : null

  return (
    <section className="result" aria-label="Kết quả tokenization">
      <div className="result-heading">
        <div>
          <p className="eyebrow">Dữ liệu tokenizer</p>
          <h2>{data.tokens.length} token</h2>
        </div>
        <span className="model-tag">{data.model_id}</span>
      </div>

      <div className="original-text">
        <span className="field-label">Text gốc</span>
        <pre>{data.text}</pre>
      </div>

      <p className="token-legend">␠ biểu thị khoảng trắng trong raw token.</p>
      <div className="token-grid" aria-label="Dãy token theo vị trí">
        {data.tokens.map((token) => (
          <button
            key={token.position}
            type="button"
            className={`token-card${selectedPosition === token.position ? ' selected' : ''}`}
            aria-pressed={selectedPosition === token.position}
            onClick={() => onSelect(token.position)}
          >
            <span className="token-position">#{token.position}</span>
            <span className="token-raw">{token.raw_token.replaceAll('Ġ', '␠') || '∅'}</span>
            <span className="token-id">ID {token.id}</span>
          </button>
        ))}
      </div>

      {selected && (
        <div className="detail" aria-live="polite">
          <div>
            <p className="eyebrow">Token được chọn</p>
            <h3>Vị trí {selected.position}</h3>
          </div>
          <dl>
            <div><dt>Token ID</dt><dd>{selected.id}</dd></div>
            <div><dt>Raw token</dt><dd><code>{JSON.stringify(selected.raw_token)}</code></dd></div>
            <div><dt>Text theo offset</dt><dd><code>{span === null ? 'Không có offset' : JSON.stringify(span)}</code></dd></div>
            <div><dt>Offset</dt><dd>{selected.start === null ? '—' : `[${selected.start}, ${selected.end})`}</dd></div>
            <div><dt>Special token</dt><dd>{selected.is_special ? 'Có' : 'Không'}</dd></div>
            <div><dt>Attention mask</dt><dd>{selected.attention_mask}</dd></div>
          </dl>
        </div>
      )}
    </section>
  )
}
