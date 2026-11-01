from transformers import AutoConfig

def inspect_head_layout(model_id: str) -> dict[str, int]:
    config = AutoConfig.from_pretrained(model_id)

    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError("hidden_size phải chia hết cho num_attention_heads")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads phải chia hết cho num_key_value_heads")

    head_dim = config.hidden_size // config.num_attention_heads
    queries_per_kv_head = (
        config.num_attention_heads // config.num_key_value_heads
    )

    return {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_query_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "queries_per_kv_head": queries_per_kv_head,
    }

if __name__ == "__main__":
    layout = inspect_head_layout("Qwen/Qwen2.5-0.5B")
    print(layout)

    for query_head in range(layout["num_query_heads"]):
        kv_head = query_head // layout["queries_per_kv_head"]
        print(f"query_head {query_head} -> kv_head {kv_head}")