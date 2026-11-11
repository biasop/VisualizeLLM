import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel
)
import gc

from tokenization import tokenize_text

def load_model(model_id:str) -> PreTrainedModel:
    if not torch.cuda.is_available():
        raise RuntimeError("Pytorch chưa sử dụng được CUDA")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype = torch.float16,
        attn_implementation="sdpa"
    )

    model = model.to("cuda")
    model.eval()

    return model

_loaded_model: PreTrainedModel | None = None
_loaded_model_id: str | None = None


def get_loaded_model(model_id: str) -> PreTrainedModel:
    """Caller must hold trace_lock. Release the old model before loading another."""
    global _loaded_model, _loaded_model_id
    if _loaded_model is None or _loaded_model_id != model_id:
        _loaded_model = None
        _loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _loaded_model = load_model(model_id)
        _loaded_model_id = model_id
    return _loaded_model

@torch.inference_mode()
def forward_once(
    model: PreTrainedModel,
    encoding: BatchEncoding,
    **model_kwargs
) -> torch.Tensor:
    nums_token = len(encoding["input_ids"])
    if not 1 <= nums_token <= 32:
        raise ValueError("Bản tạm thời input cần có từ 1 đến 32 token")
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [encoding["input_ids"]],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.tensor(
        [encoding["attention_mask"]],
        dtype=torch.long,
        device=device
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        **model_kwargs
    )

    return outputs.logits

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from tokenization import tokenize_text

    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoding = tokenize_text("Xin chào Việt Nam!\nTôi học AI 😊.", tokenizer)

    model = load_model(model_id)
    logits = forward_once(model, encoding)

    print("Shape:", logits.shape)
    print("Finite:", torch.isfinite(logits).all().item())
    print("Requires grad:", logits.requires_grad)
