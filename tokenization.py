from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerFast,
)

def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    if not tokenizer.is_fast:
        raise ValueError("Cần fast tokenizer để lấy offset_mapping")

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=True,
        return_offsets_mapping=True)

    return encoding

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        use_fast=True,
    )

    print(type(tokenizer).__name__)
    print("is_fast:", tokenizer.is_fast)

    text = "Xin chào Việt Nam!\nTôi học AI 😊."
    encoding = tokenize_text(text, tokenizer)

    print("IDs:", encoding["input_ids"])
    print("Mask:", encoding["attention_mask"])
    print("Offsets:", encoding["offset_mapping"])

    decoded = tokenizer.decode(
        encoding["input_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    print("Original:", repr(text))
    print("Decoded:", repr(decoded))
    print("Round-trip:", decoded == text)

