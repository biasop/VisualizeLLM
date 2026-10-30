from fastapi import FastAPI
from transformers import AutoTokenizer
from backend.schemas import TokenizeRequest, TokenItem, TokenizeResponse
from tokenization import tokenize_text

MODEL_ID = "Qwen/Qwen2.5-0.5B"

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

@app.post("/api/tokenize", response_model=TokenizeResponse)
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    encoding = tokenize_text(request.text, tokenizer)
    raw_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    items: list[TokenItem] = []

    for i, token_id in enumerate(encoding["input_ids"]):
        start, end = encoding["offset_mapping"][i]

        token_item = TokenItem(position=i, id=token_id, raw_token=raw_tokens[i], start=start, end=end, attention_mask=encoding["attention_mask"][i], is_special=token_id in tokenizer.all_special_ids)

        items.append(token_item)

    return TokenizeResponse(text=request.text, model_id=MODEL_ID, tokens=items)


