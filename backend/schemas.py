from pydantic import BaseModel
from uuid import UUID

class TokenizeRequest(BaseModel):
    text: str


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