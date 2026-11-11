"""Small public checkpoints; architecture compatibility is tested separately."""

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_OPTIONS = [
    {"model_id": DEFAULT_MODEL_ID, "label": "Qwen2.5 · 0.5B"},
    {"model_id": "Qwen/Qwen3-0.6B", "label": "Qwen3 · 0.6B"},
    {"model_id": "HuggingFaceTB/SmolLM2-360M", "label": "SmolLM2 · 360M"},
    {"model_id": "EleutherAI/pythia-160m", "label": "Pythia · 160M"},
    {"model_id": "openai-community/gpt2", "label": "GPT-2 · 124M"},
]
MODEL_IDS = frozenset(option["model_id"] for option in MODEL_OPTIONS)
