from fastapi import FastAPI
from transformers import AutoTokenizer
from backend.schemas import TokenizeRequest, TokenizeResponse
from tokenization import tokenize_text
from backend.model_info import inspect_head_layout
from backend.schemas import ModelInfoResponse, TraceCreatedResponse, QKVLayerResponse, TensorPayload
from backend.trace_store import trace_lock, traces
from backend.model_runner import get_loaded_model
from backend.qkv_trace import capture_qkv, UnsupportedTraceError
from backend.tokenization_service import build_tokenize_response
from fastapi import HTTPException
from uuid import uuid4, UUID
import torch

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
    return build_tokenize_response(request.text, encoding, tokenizer, MODEL_ID)


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

    tokenization = build_tokenize_response(request.text, encoding, tokenizer, MODEL_ID)

    with trace_lock:
        model = get_loaded_model(MODEL_ID)
        try:
            layers = capture_qkv(model, encoding)
        except UnsupportedTraceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        trace_id = str(uuid4())

        response = TraceCreatedResponse(
            trace_id=trace_id,
            tokenization=tokenization,
            model_info=model_info,
        )

        traces.clear()
        traces[trace_id] = layers

    return response

@app.get("/api/traces/{trace_id}/layers/{layer_index}/qkv", response_model=QKVLayerResponse)
def get_layer_qkv(
    trace_id: UUID,
    layer_index: int
) -> QKVLayerResponse:
    with trace_lock:
        layers = traces.get(str(trace_id))

        if layers is None:
            raise HTTPException(
                status_code = 404,
                detail = "Trace không tồn tại hoặc đã được thay thế."
            )

        if layer_index not in layers:
            raise HTTPException(
                status_code=404,
                detail="Layer không tồn tại trong trace.",
            )

        layer = layers[layer_index]

    specifications = [
        ("q_attention_input", "query_head", "attention_input"),
        ("k_attention_input", "kv_head", "attention_input"),
        ("v_attention_input", "kv_head", "attention_input"),
    ]

    payloads = []
    max_abs =0.0

    for name, head_axis, stage in specifications:
        tensor = layer[name]

        if not torch.isfinite(tensor).all().item():
            raise HTTPException(
                status_code=500,
                detail=f"Tensor {name} chứa giá trị không finite.",
            )

        max_abs = max(max_abs, tensor.abs().max().item())

        payloads.append(
            TensorPayload(
                name=name,
                shape=list(tensor.shape),
                axes=["token", head_axis, "head_dimension"],
                stage=stage,
                source_dtype=str(tensor.dtype),
                values=tensor.float().tolist(),
            )
        )
    return QKVLayerResponse(
        trace_id=trace_id,
        layer_index=layer_index,
        color_limit=max_abs if max_abs > 0 else 1.0,
        tensors=payloads,
    )
