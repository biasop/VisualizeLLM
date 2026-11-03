from fastapi import FastAPI
from transformers import AutoTokenizer
from backend.schemas import TokenizeRequest, TokenItem, TokenizeResponse
from tokenization import tokenize_text
from backend.model_info import inspect_head_layout
from backend.schemas import ModelInfoResponse, TraceCreatedResponse
from backend.trace_store import trace_lock, traces
from backend.model_runner import get_loaded_model
from backend.qkv_trace import capture_qkv, reshape_layer_qkv
from fastapi import HTTPException
from uuid import uuid4

MODEL_ID = "Qwen/Qwen2.5-0.5B"

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model_info = ModelInfoResponse(
    model_id = MODEL_ID,
    **inspect_head_layout(MODEL_ID)
)

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


@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    return model_info

@app.post("/api/traces", response_model = TraceCreatedResponse, status_code = 201)
def create_trace(request: TokenizeRequest) -> TraceCreatedResponse:
    encoding = tokenize_text(request.text, tokenizer)

    num_tokens = len(encoding["input_ids"])
    if not 1 <= num_tokens <= 32:
        raise HTTPException(
            status_code=422,
            detail= "Input cần có từ 1 đến 32 token."
        )

    raw_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    items = []

    for i, token_id in enumerate(encoding["input_ids"]):
        start, end = encoding["offset_mapping"][i]
        items.append(
            TokenItem(
                position=i,
                id=token_id,
                raw_token=raw_tokens[i],
                start=start,
                end=end,
                attention_mask=encoding["attention_mask"][i],
                is_special=token_id in tokenizer.all_special_ids,
            )
        )

    tokenization = TokenizeResponse(
        text = request.text,
        model_id = MODEL_ID,
        tokens=items
    )

    layout = model_info.model_dump(exclude={"model_id"})

    with trace_lock:
        model = get_loaded_model(MODEL_ID)
        captured = capture_qkv(model, encoding)

        layers = {
            layer_index: reshape_layer_qkv(layer_tensors, layout) 
            for layer_index, layer_tensors in captured.items()
        }

        trace_id = str(uuid4())

        response = TraceCreatedResponse(
            trace_id=trace_id,
            tokenization=tokenization,
            model_info=model_info,
        )

        traces.clear()
        traces[trace_id] = layers

    return response
