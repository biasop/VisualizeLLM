from pydantic import BaseModel
from uuid import UUID
from backend.model_catalog import DEFAULT_MODEL_ID

class TokenizeRequest(BaseModel):
    text: str
    model_id: str = DEFAULT_MODEL_ID


class ModelOption(BaseModel):
    model_id: str
    label: str


class TokenItem(BaseModel):
    position: int
    id: int
    raw_token: str
    start: int | None
    end: int | None
    attention_mask: int
    is_special: bool


class TokenizeResponse(BaseModel):
    text: str
    model_id: str
    tokens: list[TokenItem]


class ModelInfoResponse(BaseModel):
    model_id: str
    num_layers: int
    hidden_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    queries_per_kv_head: int

class TraceCreatedResponse(BaseModel):
    trace_id: UUID
    tokenization: TokenizeResponse
    model_info: ModelInfoResponse

class TensorPayload(BaseModel):
    name: str
    shape: list[int]
    axes: list[str]
    stage: str
    source_dtype: str
    values: list[list[list[float]]]

class QKVLayerResponse(BaseModel):
    trace_id: UUID
    layer_index: int
    color_limit: float
    tensors: list[TensorPayload]
