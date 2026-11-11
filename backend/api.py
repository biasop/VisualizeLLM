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
from functools import lru_cache
from backend.model_catalog import DEFAULT_MODEL_ID, MODEL_OPTIONS, MODEL_IDS
from backend.schemas import ModelOption
from uuid import uuid4, UUID
import torch

app = FastAPI()


def validate_model_id(model_id: str) -> None:
    if model_id not in MODEL_IDS:
        raise HTTPException(status_code=422, detail="Model ID không nằm trong danh sách hỗ trợ.")


@lru_cache(maxsize=5)
def get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


@lru_cache(maxsize=5)
def model_info_for(model_id: str) -> ModelInfoResponse:
    return ModelInfoResponse(model_id=model_id, **inspect_head_layout(model_id))


def prepare_tokenizer(model_id: str):
    validate_model_id(model_id)
    try:
        return get_tokenizer(model_id)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Không tải được tokenizer {model_id}. Kiểm tra kết nối Hugging Face và thử lại.") from error


@app.get("/api/models", response_model=list[ModelOption])
def get_models():
    return MODEL_OPTIONS

@app.post("/api/tokenize", response_model=TokenizeResponse)
def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    tokenizer = prepare_tokenizer(request.model_id)
    encoding = tokenize_text(request.text, tokenizer)
    return build_tokenize_response(request.text, encoding, tokenizer, request.model_id)


@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info(model_id: str = DEFAULT_MODEL_ID) -> ModelInfoResponse:
    validate_model_id(model_id)
    try:
        return model_info_for(model_id)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Không đọc được config {model_id}: {error}") from error

@app.post("/api/traces", response_model = TraceCreatedResponse, status_code = 201)
def create_trace(request: TokenizeRequest) -> TraceCreatedResponse:
    tokenizer = prepare_tokenizer(request.model_id)
    model_info = get_model_info(request.model_id)
    encoding = tokenize_text(request.text, tokenizer)

    num_tokens = len(encoding["input_ids"])
    if not 1 <= num_tokens <= 32:
        raise HTTPException(
            status_code=422,
            detail= "Input cần có từ 1 đến 32 token."
        )

    tokenization = build_tokenize_response(request.text, encoding, tokenizer, request.model_id)

    with trace_lock:
        try:
            model = get_loaded_model(request.model_id)
            layers = capture_qkv(model, encoding)
        except UnsupportedTraceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except torch.cuda.OutOfMemoryError as error:
            raise HTTPException(status_code=503, detail="GPU không đủ VRAM. Hãy chọn model nhỏ hơn hoặc giảm số token.") from error
        except (OSError, ValueError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=f"Không chạy được {request.model_id}: {error}") from error

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
