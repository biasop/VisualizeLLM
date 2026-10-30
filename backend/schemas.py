from pydantic import BaseModel


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


