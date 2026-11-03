import torch
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)

from backend.model_runner import load_model, forward_once
from tokenization import tokenize_text
from backend.model_info import inspect_head_layout


def capture_qkv(
    model: PreTrainedModel,
    encoding: BatchEncoding,
) -> dict[int, dict[str, torch.Tensor]]:

    captured: dict[int, dict[str, torch.Tensor]] = {
        i: {} for i in range(model.config.num_hidden_layers)
    }

    def make_hook(layer_index: int, tensor_name: str):
        def save(module, inputs, output):
            captured[layer_index][tensor_name] = output.detach().cpu()
        return save

    handles = []

    try:
        for i, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            handles.append(attention.q_proj.register_forward_hook(make_hook(i, "q_pre_rope_flat")))
            handles.append(attention.k_proj.register_forward_hook(make_hook(i, "k_pre_rope_flat")))
            handles.append(attention.v_proj.register_forward_hook(make_hook(i, "v_flat")))
        forward_once(model=model, encoding=encoding)
    finally:
        for handle in handles:
            handle.remove()

    expected_keys = {"q_pre_rope_flat", "k_pre_rope_flat", "v_flat"}

    for layer_index, layer_tensors in captured.items():
        if set(layer_tensors) != expected_keys:
            raise RuntimeError(f"Layer {layer_index} chưa có đủ Q/K/V.")

    return captured

def reshape_layer_qkv(
    layer_tensors: dict[str, torch.Tensor],
    layout: dict[str, int]
)->dict[str, torch.Tensor]:
    q = layer_tensors["q_pre_rope_flat"]
    q_heads = q[0].reshape(
        q.shape[1],
        layout["num_query_heads"],
        layout["head_dim"]
    )

    k = layer_tensors["k_pre_rope_flat"]
    k_heads = k[0].reshape(
        k.shape[1],
        layout["num_kv_heads"],
        layout["head_dim"]
    )

    v = layer_tensors["v_flat"]
    v_heads = v[0].reshape(
        v.shape[1],
        layout["num_kv_heads"],
        layout["head_dim"]
    )

    return {
        "q_pre_rope": q_heads,
        "k_pre_rope": k_heads,
        "v": v_heads,
    }
    

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B"
    text = "Xin chào Việt Nam!\nTôi học AI 😊."

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoding = tokenize_text(text, tokenizer)
    model = load_model(model_id)

    tensors = capture_qkv(model, encoding)
    print("Số token:", len(encoding["input_ids"]))

    for layer_index, layer_tensors in tensors.items():
        print("Layer: ", layer_index)
        for name, tensor in layer_tensors.items():
            print(
                name,
                "shape=", tuple(tensor.shape),
                "dtype=", tensor.dtype,
                "device=", tensor.device,
                "requires_grad=", tensor.requires_grad,
                "finite=", torch.isfinite(tensor).all().item(),
            )

    layout = inspect_head_layout(model_id)
    reshaped = reshape_layer_qkv(tensors[0], layout)

    for name, tensor in reshaped.items():
        print(name, tuple(tensor.shape))

    query_head = 8
    kv_head = query_head // layout["queries_per_kv_head"]

    print(reshaped["q_pre_rope"][:, query_head, :].shape)
    print(reshaped["k_pre_rope"][:, kv_head, :].shape)
    print(reshaped["v"][:, kv_head, :].shape)