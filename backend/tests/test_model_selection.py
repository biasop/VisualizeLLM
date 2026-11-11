"""Offline tests: tokenizers and models here are tiny synthetic fixtures."""
import unittest
import weakref
from unittest.mock import patch

import torch
from fastapi import HTTPException
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast, Qwen2Config

from backend import api, model_runner
from backend.model_info import head_layout_from_config


class ModelSelectionTests(unittest.TestCase):
    def test_requests_keep_tokenizer_config_and_trace_with_selected_model(self):
        vocab = {"[UNK]": 0, "hello": 1, "world": 2}
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
        configs = {
            "Qwen/Qwen2.5-0.5B": Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1),
            "openai-community/gpt2": GPT2Config(vocab_size=8, n_embd=24, n_layer=1, n_head=3, n_positions=16,
                bos_token_id=1, eos_token_id=2),
        }
        models = {key: AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
                  for key, config in configs.items()}
        def info(model_id):
            return api.ModelInfoResponse(model_id=model_id, **head_layout_from_config(configs[model_id]))

        with patch.object(api, "get_tokenizer", return_value=tokenizer) as tokenizer_loader, \
             patch.object(api, "model_info_for", side_effect=info), \
             patch.object(api, "get_loaded_model", side_effect=models.__getitem__), \
             patch.dict(api.traces, {}, clear=True):
            self.assertEqual(len(api.get_models()), 5)
            with self.assertRaises(HTTPException) as caught:
                api.tokenize(api.TokenizeRequest(text="hello", model_id="unknown"))
            self.assertEqual(caught.exception.status_code, 422)
            previous_trace = None
            for model_id, config in configs.items():
                self.assertEqual(api.get_model_info(model_id).model_id, model_id)
                payload = api.TokenizeRequest(text="hello world", model_id=model_id)
                self.assertEqual(api.tokenize(payload).model_id, model_id)
                trace = api.create_trace(payload)
                self.assertEqual(trace.model_info.model_id, model_id)
                self.assertEqual(trace.tokenization.model_id, model_id)
                tensors = api.get_layer_qkv(trace.trace_id, 0).tensors
                self.assertEqual(tensors[0].shape, [2, config.num_attention_heads, config.hidden_size // config.num_attention_heads])
                if previous_trace:
                    with self.assertRaises(HTTPException) as caught:
                        api.get_layer_qkv(previous_trace, 0)
                    self.assertEqual(caught.exception.status_code, 404)
                previous_trace = trace.trace_id
                tokenizer_loader.assert_called_with(model_id)

    def test_switch_releases_old_model_before_loading_new_one(self):
        reference = None
        def load(model_id):
            nonlocal reference
            if reference is not None:
                self.assertIsNone(reference())
            model = torch.nn.Linear(1, 1)
            reference = weakref.ref(model)
            return model

        with patch.object(model_runner, "_loaded_model", None), \
             patch.object(model_runner, "_loaded_model_id", None), \
             patch.object(model_runner, "load_model", side_effect=load) as loader, \
             patch.object(torch.cuda, "is_available", return_value=False):
            model_runner.get_loaded_model("first")
            model_runner.get_loaded_model("first")
            self.assertEqual(loader.call_count, 1)
            model_runner.get_loaded_model("second")
            self.assertEqual(loader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
