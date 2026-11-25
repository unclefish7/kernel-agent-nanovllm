"""独立的 CPU 验收回归。所有模型使用本地随机权重或可手算的词元历史。"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nanovllm_selector.port import evaluate as ev


class Window:
    def __init__(self, total=0, length=0):
        self.total = total
        self.length = length

    def crop(self, length):
        # 模拟只截短位置计数、遗漏序列状态的缓存。
        self.length = length


class HistoryModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self, *, bad_chunk=False, bad_single=False):
        super().__init__()
        self.bad_chunk = bad_chunk
        self.bad_single = bad_single

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        state = past_key_values if past_key_values is not None else Window()
        if past_key_values is not None:
            if (input_ids.numel() > 1 and self.bad_chunk) or (input_ids.numel() == 1 and self.bad_single):
                state.total += 1000
        rows = []
        for token in input_ids[0].tolist():
            state.total += token
            state.length += 1
            rows.append([state.total, state.length])
        return SimpleNamespace(logits=torch.tensor([rows], dtype=torch.float32), past_key_values=state)


class TestReference(unittest.TestCase):
    def test_shape_sensitive_kv_probes_preserve_the_original_segmented_oracle(self):
        from transformers.cache_utils import DynamicCache

        class SegmentedKV(torch.nn.Module):
            device = torch.device("cpu")

            def forward(self, input_ids, past_key_values=None, use_cache=True):
                cache = DynamicCache() if past_key_values is None else past_key_values
                count = input_ids.shape[1]
                # 放大分段舍入效应：同一词元在不同长度的投影中会写入不同的 KV 值。
                kv = (input_ids.float() + 10 * count).reshape(1, 1, count, 1)
                keys, _ = cache.update(kv, kv, 0)
                sums = keys.flatten().cumsum(0)[-count:]
                return SimpleNamespace(past_key_values=cache, logits=torch.stack((sums, -sums), -1)[None])

        prompts = [[1, 2, 3, 4, 5], [1, 2, 3, 7, 9], [1, 2, 3, 7, 6]]
        prefill, decode, info = self.trace(SegmentedKV(), prompts, [0, 3, 4], [8])
        self.assertFalse(info["crop"]["ok"])
        self.assertFalse(info["multi_token"]["ok"])
        self.assertEqual([v[0].item() for v in prefill + decode], [265, 212, 199, 217])
        self.assertFalse(ev.compare_logits(torch.tensor([218., -217.]), decode[0], exact=True)["ok"])

    def test_pure_kv_cannot_pass_by_turning_off_eligible_prefix_hits(self):
        cls = type("FakeForCausalLM", (), {})
        row = torch.tensor([3., 2.])
        engine = {"prefill": [row, row], "prefix_lengths": [0, 0], "decode": {"ids": [], "logits": []},
                  "prefix_reuse": {"enabled": False, "reason": "窗口", "block_size": 4}}
        tokenizer = lambda *a, **kw: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))

        def hf(*args, diagnostics):
            diagnostics["prefix_strategy"] = "full_kv_snapshot"
            return [row.clone(), row.clone()], []

        with patch.object(ev, "load_candidate"), patch.object(ev, "pick_model_class", return_value=cls), patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ), patch.object(ev, "engine_run", return_value=engine), patch.object(ev, "reference_steps", hf), patch.dict("os.environ"):
            report = ev.evaluate(".", Path(__file__), lengths=(4, 5), decode_steps=0)
            self.assertEqual(report["outcome"], "candidate_failed")
            self.assertEqual(report["cached_prefix_tokens"], [0, 0])

    def test_non_finite_candidate_is_a_serializable_failure(self):
        got = torch.tensor([1., float("nan")])
        report = ev.compare_logits(got, torch.ones(2), exact=True)
        self.assertFalse(report["ok"])
        json.dumps(report, allow_nan=False)

    def test_demo_helper_does_not_substitute_another_prompt_of_equal_length(self):
        def independent(model_dir, ids, positions):
            return [sum(ids[:p + 1]) for p in positions]
        with patch.object(ev, "reference_logits", independent):
            self.assertEqual(ev.reference_logits_multi(".", [[1, 2], [1, 7]]), [3, 8])

    def test_extra_state_on_an_ordinary_dynamic_cache_uses_conservative_path(self):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        cache.update(torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4), 0)
        cache.window = torch.ones(1, 4)
        self.assertFalse(ev._full_kv(cache))

    def test_score_only_reference_cannot_silently_drop_continuation_history(self):
        class NoCache(HistoryModel):
            def forward(self, *args, **kwargs):
                out = super().forward(*args, **kwargs)
                out.past_key_values = None
                out.logits.fill_(1.0)
                return out
        with self.assertRaisesRegex(ValueError, "HF"):
            self.trace(NoCache(), [[1, 2, 3]], [0], [1])

    def test_prefix_report_checks_disabled_reuse_and_eligible_kv_hit(self):
        prompts = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]
        for enabled, lengths, valid in [(False, [0, 0], True), (False, [0, 4], False),
                                        (True, [0, 0], False), (True, [0, 4], True)]:
            policy = {"enabled": enabled, "block_size": 4, "reason": None if enabled else "窗口"}
            with self.subTest(enabled=enabled, lengths=lengths):
                if valid:
                    ev.check_prefix_hits(prompts, lengths, policy)
                else:
                    with self.assertRaises(ValueError):
                        ev.check_prefix_hits(prompts, lengths, policy)

    def trace(self, model, prompts, prefixes, continuation):
        info = {}
        prefill, decode = ev.reference_trace(model, prompts, prefixes, continuation, info)
        return prefill, decode, info

    def test_unsafe_crop_and_chunk_have_independent_evidence(self):
        for bad_chunk in (False, True):
            with self.subTest(bad_chunk=bad_chunk):
                a, b, info = self.trace(HistoryModel(bad_chunk=bad_chunk),
                                        [[1, 2, 3, 4, 5], [1, 2, 3, 8, 9]], [0, 3], [6])
                self.assertEqual([x.tolist() for x in a], [[15, 5], [23, 5]])
                self.assertEqual(b[0].tolist(), [29, 6])
                self.assertFalse(info["crop"]["ok"])
                self.assertEqual(info["multi_token"]["ok"], not bad_chunk)
                self.assertEqual(info["prefix_strategy"], "rebuild_and_step")

    def test_unreliable_serial_continuation_is_diagnosed_without_candidate(self):
        with self.assertRaisesRegex(ValueError, "HF"):
            self.trace(HistoryModel(bad_single=True), [[1, 2, 3, 4]], [0], [5])

    def test_standard_kv_snapshot_preserves_exact_segmented_scores(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            model = LlamaForCausalLM(LlamaConfig(vocab_size=37, hidden_size=24, intermediate_size=40,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
        prompts = [[1, 2, 3, 4], [1, 2, 3, 8, 9], [1, 2, 3, 8, 6]]
        with torch.inference_mode():
            prior = model(torch.tensor([prompts[0]]), use_cache=True, logits_to_keep=1)
            prior.past_key_values.crop(3)
            middle = model(torch.tensor([[8, 9]]), past_key_values=prior.past_key_values, logits_to_keep=1)
            middle.past_key_values.crop(4)
            last = model(torch.tensor([[6]]), past_key_values=middle.past_key_values, logits_to_keep=1)
            after = model(torch.tensor([[7]]), past_key_values=last.past_key_values, logits_to_keep=1)
        a, b, info = self.trace(model, prompts, [0, 3, 4], [7])
        self.assertEqual(info["prefix_strategy"], "full_kv_snapshot")
        for got, expected in zip(a + b, [prior, middle, last, after], strict=True):
            self.assertTrue(torch.equal(got, expected.logits[0, -1]))

    def test_real_lfm_reference_matches_clean_history_after_branch(self):
        from transformers import Lfm2Config, Lfm2ForCausalLM
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            config = Lfm2Config(vocab_size=47, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                layer_types=["conv", "full_attention"], block_multiple_of=8)
            model = Lfm2ForCausalLM(config).eval()
        branch = [11, 23, 37, 6, 7]
        a, b, info = self.trace(model, [[11, 23, 37, 41, 43], branch], [0, 3], [8])
        with torch.inference_mode():
            full = model(torch.tensor([branch + [8]]), use_cache=False).logits[0]
        self.assertFalse(info["crop"]["ok"])
        self.assertFalse(info["multi_token"]["ok"])
        torch.testing.assert_close(a[-1], full[-2], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(b[0], full[-1], atol=1e-5, rtol=1e-5)

    def test_generation_initialized_caches_match_independent_forced_histories(self):
        from transformers import (
            JambaConfig, JambaForCausalLM, MambaConfig, MambaForCausalLM, ZambaConfig, ZambaForCausalLM,
        )

        def generate_rows(model, prompt, suffix):
            ids = torch.tensor([prompt])
            forced = suffix + [1]  # 最后一个采样用于取得最后一个输入位置的原始分数。
            generated = model.generate(
                ids, attention_mask=torch.ones_like(ids), use_cache=True, do_sample=False,
                max_new_tokens=len(forced), return_dict_in_generate=True, output_logits=True,
                prefix_allowed_tokens_fn=lambda _, history: [forced[len(history) - len(prompt)]],
            )
            self.assertEqual(generated.sequences[0].tolist(), prompt + forced)
            return [row[0] for row in generated.logits]

        jamba = JambaConfig(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, attn_layer_period=2, attn_layer_offset=1,
                            num_experts=1, num_experts_per_tok=1, mamba_d_state=4, use_mamba_kernels=False)
        mamba = MambaConfig(vocab_size=37, hidden_size=16, state_size=4, num_hidden_layers=2)
        zamba = ZambaConfig(vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, mamba_d_state=4, use_mamba_kernels=False)
        first, branch = [5, 7, 11, 13, 17], [5, 7, 11, 19, 23]
        continuation = [29, 8, 3, 21, 4, 12, 6, 10]
        histories = [first, branch] + [branch + continuation[:i] for i in range(1, len(continuation) + 1)]
        for cls, config, cache_name in [(JambaForCausalLM, jamba, "past_key_values"),
                                        (ZambaForCausalLM, zamba, "past_key_values"),
                                        (MambaForCausalLM, mamba, "cache_params")]:
            with self.subTest(model=config.model_type), torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.manual_seed(41)
                model = cls(config).eval()
                first_rows = generate_rows(model, first, [])
                full_rows = [model(torch.tensor([history]), use_cache=False).logits[0, -1] for history in histories]
                for hit in (0, 3):
                    with self.subTest(hit=hit):
                        start = hit or len(branch)
                        suffix = branch[start:]
                        native = generate_rows(model, branch[:start], suffix + continuation)
                        expected = first_rows + native[len(suffix):]
                        prefill, decode, info = self.trace(model, [first, branch], [0, hit], continuation)
                        self.assertEqual((len(prefill), len(decode)), (2, 8))
                        self.assertEqual(info["cache_name"], cache_name)
                        for row, cached, full in zip(prefill + decode, expected, full_rows, strict=True):
                            self.assertEqual(row.shape, (37,))
                            self.assertTrue(torch.isfinite(row).all())
                            torch.testing.assert_close(row, cached, atol=0, rtol=0)
                            torch.testing.assert_close(row, full, atol=1e-5, rtol=1e-5)

    def test_verdict_distinguishes_candidate_and_reference_failures(self):
        # 仅替代引擎和权重装载，保留 evaluate 的完整判定。
        cls = type("FakeForCausalLM", (), {})
        engine = {"prefill": [torch.tensor([4., 2.])], "prefix_lengths": [0],
                  "decode": {"ids": [], "logits": []}, "prefix_reuse": {"enabled": False, "reason": "窗口"}}
        tokenizer = lambda *a, **kw: SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
        for bad_reference in (True, False):
            with self.subTest(bad_reference=bad_reference), patch.object(ev, "load_candidate"), patch.object(
                ev, "pick_model_class", return_value=cls
            ), patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch.object(
                ev, "engine_run", return_value=engine
            ), patch.object(ev, "reference_steps", side_effect=ValueError("HF") if bad_reference else None,
                            return_value=([torch.tensor([3., 2.])], [])), patch.dict("os.environ"):
                report = ev.evaluate(".", Path(__file__), lengths=(2,), decode_steps=0)
                self.assertFalse(report["ok"])
                self.assertEqual(report.get("outcome"), "reference_unreliable" if bad_reference else "candidate_failed")
                self.assertEqual(report["prefix_reuse"]["reason"], "窗口")
