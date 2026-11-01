import torch
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)

from backend.model_runner import load_model, forward_once
from tokenization import tokenize_text


def capture_qkv(
    model: PreTrainedModel,
    encoding: BatchEncoding,
    layer_index:int
) -> dict[str, torch.Tensor]:

    if not 0 <= layer_index < model.config.num_hidden_layers:
        raise ValueError("Layer được chọn không nằm trong model")

    attention = model.model.layers[layer_index].self_attn

    captured = {}

    def save_q(module, inputs, output):
        captured["q_pre_rope_flat"] = output.detach().cpu()

    def save_k(module, inputs, output):
        captured["k_pre_rope_flat"] = output.detach().cpu()

    def save_v(module, inputs, output):
        captured["v_flat"] = output.detach().cpu()

    handles = []

    try:
        handles.append(attention.q_proj.register_forward_hook(save_q))
        handles.append(attention.k_proj.register_forward_hook(save_k))
        handles.append(attention.v_proj.register_forward_hook(save_v))
        forward_once(model=model, encoding=encoding)
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) != 3:
        raise RuntimeError("Chưa thu được đầy đù Q, K, V")

    return captured

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B"
    layer_index = 0
    text = "Xin chào Việt Nam!\nTôi học AI 😊."

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoding = tokenize_text(text, tokenizer)
    model = load_model(model_id)

    tensors = capture_qkv(model, encoding, layer_index)

    print("Layer:", layer_index)
    print("Số token:", len(encoding["input_ids"]))

    for name, tensor in tensors.items():
        print(
            name,
            "shape=", tuple(tensor.shape),
            "dtype=", tensor.dtype,
            "device=", tensor.device,
            "requires_grad=", tensor.requires_grad,
            "finite=", torch.isfinite(tensor).all().item(),
        )
