from transformers import AutoConfig, PretrainedConfig

def inspect_head_layout(model_id: str) -> dict[str, int]:
    config = AutoConfig.from_pretrained(model_id)
    return head_layout_from_config(config)


def head_layout_from_config(config: PretrainedConfig) -> dict[str, int]:
    """Metadata for conventional decoder-only MHA/GQA; explicit head_dim wins."""
    hidden_size = config.hidden_size
    query_heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is None:
        kv_heads = query_heads
    num_layers = config.num_hidden_layers
    if any(not isinstance(value, int) or value <= 0 for value in (
        hidden_size, query_heads, kv_heads, num_layers
    )):
        raise ValueError("Config layer/head không hợp lệ.")
    if query_heads % kv_heads:
        raise ValueError("Số query head phải chia hết cho số KV head.")

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if hidden_size % query_heads:
            raise ValueError("Không thể suy ra head_dim từ config này.")
        head_dim = hidden_size // query_heads
    if not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("head_dim không hợp lệ.")

    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_query_heads": query_heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "queries_per_kv_head": query_heads // kv_heads,
    }

if __name__ == "__main__":
    layout = inspect_head_layout("Qwen/Qwen2.5-0.5B")
    print(layout)

    for query_head in range(layout["num_query_heads"]):
        kv_head = query_head // layout["queries_per_kv_head"]
        print(f"query_head {query_head} -> kv_head {kv_head}")
