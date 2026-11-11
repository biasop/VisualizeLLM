import torch
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    BatchEncoding,
    PreTrainedModel,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

from backend.model_runner import forward_once

TRACE_BACKEND = "visualize_trace_sdpa"


class UnsupportedTraceError(ValueError):
    """The model does not expose the attention layout supported by this tracer."""


def tracing_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    trace_buffer=None,
    **kwargs,      
):
    if trace_buffer is not None:
        trace_buffer.append(
            (
                id(module),
                {
                    "q_attention_input": query.detach().to("cpu", copy=True),
                    "k_attention_input": key.detach().to("cpu", copy=True),
                    "v_attention_input": value.detach().to("cpu", copy=True),
                },
            )
        )

    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs
    )

AttentionInterface.register(TRACE_BACKEND, tracing_attention)
AttentionMaskInterface.register(TRACE_BACKEND, sdpa_mask)


def capture_qkv(
    model: PreTrainedModel,
    encoding: BatchEncoding,
) -> dict[int, dict[str, torch.Tensor]]:
    """Run once and return independent CPU tensors shaped [token, head, dimension].

    The caller must serialize access to the shared model (for example, trace_lock).
    Capture happens at the attention backend input, after model-specific processing.
    Only one attention invocation per decoder layer and batch size one are supported.
    """
    if model.training:
        raise ValueError("Hãy gọi model.eval() trước khi capture QKV.")
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise UnsupportedTraceError("Không xác định được số decoder layer của model.")

    token_count = len(encoding["input_ids"])
    buffer = []
    previous_backend = model.config._attn_implementation

    try:
        model.set_attn_implementation(TRACE_BACKEND)

        if model.config._attn_implementation != TRACE_BACKEND:
            raise UnsupportedTraceError(
                "Model không cho phép dùng attention tracer này."
            )
        forward_once(model, encoding, trace_buffer=buffer)
    finally:
        model.set_attn_implementation(previous_backend)

    if len(buffer) != num_layers:
        raise UnsupportedTraceError(
            f"Cần một lần gọi attention cho mỗi layer: nhận {len(buffer)}, "
            f"dự kiến {num_layers}. Model có thể không truyền trace_buffer "
            "qua AttentionInterface."
        )
    if len({module_id for module_id, _ in buffer}) != num_layers:
        raise UnsupportedTraceError("Tracer chưa hỗ trợ attention module dùng chung giữa các layer.")

    layers = {}
    for layer_index, (_, tensors) in enumerate(buffer):
        layer = {}
        for name, tensor in tensors.items():
            if (tensor.ndim != 4 or tensor.shape[0] != 1
                    or tensor.shape[2] != token_count
                    or tensor.shape[1] <= 0 or tensor.shape[3] <= 0):
                raise UnsupportedTraceError(
                    f"Layer {layer_index}, {name}: cần shape [1, head, {token_count}, dimension], "
                    f"nhận {tuple(tensor.shape)}."
                )
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Layer {layer_index}, {name} chứa NaN hoặc infinity.")
            layer[name] = tensor[0].permute(1, 0, 2).contiguous()

        q = layer["q_attention_input"]
        k = layer["k_attention_input"]
        v = layer["v_attention_input"]
        if q.shape[2] != k.shape[2] or k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1]:
            raise UnsupportedTraceError(
                f"Layer {layer_index}: shape Q/K/V không phù hợp với MHA hoặc GQA."
            )
        layers[layer_index] = layer

    return layers


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from backend.model_runner import load_model
    from tokenization import tokenize_text

    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoding = tokenize_text("Xin chào Việt Nam!\nTôi học AI 😊.", tokenizer)
    model = load_model(model_id)
    layers = capture_qkv(model, encoding)
    for layer_index, tensors in layers.items():
        print("Layer:", layer_index)
        for name, tensor in tensors.items():
            print(name, "shape=", tuple(tensor.shape), "dtype=", tensor.dtype,
                  "device=", tensor.device, "requires_grad=", tensor.requires_grad,
                  "finite=", torch.isfinite(tensor).all().item())

