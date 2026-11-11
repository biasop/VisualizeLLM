from transformers import BatchEncoding, PreTrainedTokenizerBase

from backend.schemas import TokenItem, TokenizeResponse


def build_tokenize_response(
    text: str,
    encoding: BatchEncoding,
    tokenizer: PreTrainedTokenizerBase,
    model_id: str,
) -> TokenizeResponse:
    raw_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    special_ids = set(tokenizer.all_special_ids)
    items = []
    for i, token_id in enumerate(encoding["input_ids"]):
        start, end = encoding["offset_mapping"][i]
        items.append(TokenItem(
            position=i, id=token_id, raw_token=raw_tokens[i],
            start=start, end=end, attention_mask=encoding["attention_mask"][i],
            is_special=token_id in special_ids,
        ))
    return TokenizeResponse(text=text, model_id=model_id, tokens=items)
