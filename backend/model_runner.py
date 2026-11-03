import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel
)
from functools import lru_cache

from tokenization import tokenize_text

def load_model(model_id:str) -> PreTrainedModel:
    if not torch.cuda.is_available():
        raise RuntimeError("Pytorch chưa sử dụng được CUDA")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype = torch.float16,
        attn_implementation="eager"
    )

    model = model.to("cuda")
    model.eval()

    return model

@lru_cache(maxsize=1)
def get_loaded_model(model_id: str) -> PreTrainedModel:
    return load_model(model_id)

@torch. inference_mode()
def forward_once(
    model: PreTrainedModel,
    encoding: BatchEncoding,
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
        use_cache=False
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