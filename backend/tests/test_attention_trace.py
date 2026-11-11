"""Compatibility tests use tiny RANDOM models, not pretrained checkpoints."""
import unittest
from unittest.mock import patch

import torch
from transformers import (
    AutoModelForCausalLM, BatchEncoding, GPT2Config, GPTNeoXConfig,
    LlamaConfig, OPTConfig, Qwen2Config, Qwen3Config,
)

from backend.model_info import head_layout_from_config
from backend.model_runner import forward_once
from backend.qkv_trace import capture_qkv, TRACE_BACKEND


def configs():
    shared = dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  bos_token_id=1, eos_token_id=2)
    return [
        Qwen2Config(**shared, num_key_value_heads=2),
        Qwen3Config(**shared, num_key_value_heads=2, head_dim=16),
        LlamaConfig(**shared, num_key_value_heads=2),
        GPTNeoXConfig(**shared, rotary_pct=0.5),
        GPT2Config(vocab_size=32, n_embd=32, n_layer=2, n_head=4,
                   n_positions=16, bos_token_id=1, eos_token_id=2),
        OPTConfig(vocab_size=32, hidden_size=32, ffn_dim=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  word_embed_proj_dim=32, max_position_embeddings=16,
                  bos_token_id=1, eos_token_id=2),
    ]


class AttentionTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_families_shapes_masks_and_restoration(self):
        for config in configs():
            with self.subTest(model_type=config.model_type):
                model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
                encoding = BatchEncoding({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]})
                before = forward_once(model, encoding).clone()
                with patch("backend.qkv_trace.forward_once", wraps=forward_once) as run:
                    captured = capture_qkv(model, encoding)
                    self.assertEqual(run.call_count, 1)
                self.assertEqual(model.config._attn_implementation, "sdpa")
                torch.testing.assert_close(before, forward_once(model, encoding))
                layout = head_layout_from_config(config)
                self.assertEqual(len(captured), 2)
                for tensors in captured.values():
                    for name, tensor in tensors.items():
                        heads = layout["num_query_heads"] if name.startswith("q_") else layout["num_kv_heads"]
                        self.assertEqual(tuple(tensor.shape), (4, heads, layout["head_dim"]))
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertFalse(tensor.requires_grad)
                        self.assertTrue(torch.isfinite(tensor).all())

                # The recorder must receive the same causal/padding behavior as SDPA.
                model.set_attn_implementation(TRACE_BACKEND)
                future_changed = BatchEncoding({"input_ids": [1, 2, 3, 9], "attention_mask": [1, 1, 1, 1]})
                traced = forward_once(model, encoding)
                torch.testing.assert_close(traced, before)
                torch.testing.assert_close(traced[:, :3], forward_once(model, future_changed)[:, :3])
                padding = BatchEncoding({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 0, 1, 1]})
                changed_padding = BatchEncoding({"input_ids": [1, 9, 3, 4], "attention_mask": [1, 0, 1, 1]})
                torch.testing.assert_close(forward_once(model, padding)[:, 2:], forward_once(model, changed_padding)[:, 2:])

    def test_qwen_trace_matches_actual_rope_output(self):
        from transformers.models.qwen2 import modeling_qwen2
        model = AutoModelForCausalLM.from_config(configs()[0], attn_implementation="sdpa").eval()
        encoding = BatchEncoding({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
        original = modeling_qwen2.apply_rotary_pos_emb
        rotated = []

        def observe_rope(*args, **kwargs):
            q, k = original(*args, **kwargs)
            rotated.append((q.detach().clone(), k.detach().clone()))
            return q, k

        with patch.object(modeling_qwen2, "apply_rotary_pos_emb", side_effect=observe_rope):
            captured = capture_qkv(model, encoding)
        for index, (q, k) in enumerate(rotated):
            torch.testing.assert_close(captured[index]["q_attention_input"], q[0].permute(1, 0, 2))
            torch.testing.assert_close(captured[index]["k_attention_input"], k[0].permute(1, 0, 2))

    def test_backend_restored_after_forward_error(self):
        model = AutoModelForCausalLM.from_config(configs()[0], attn_implementation="sdpa").eval()
        encoding = BatchEncoding({"input_ids": [1], "attention_mask": [1]})
        with patch("backend.qkv_trace.forward_once", side_effect=RuntimeError("test error")):
            with self.assertRaisesRegex(RuntimeError, "test error"):
                capture_qkv(model, encoding)
        self.assertEqual(model.config._attn_implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
