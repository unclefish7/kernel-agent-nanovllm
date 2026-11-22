"""用真实 CPU 调度器与确定性模型检查并发验收工具。"""

import unittest
from types import SimpleNamespace

import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler

from nanovllm_selector.outputs.lfm25_1_2b import check_batch_order
from nanovllm_selector.port.test_evaluate_reference import HistoryModel


class CpuRunner:
    def __init__(self, damage):
        self.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
        self.damage = damage

    def run(self, seqs, is_prefill):
        rows = torch.tensor([[sum(seq.token_ids), len(seq)] for seq in seqs], dtype=torch.float32)
        if self.damage == "score" and not is_prefill and len(seqs) == 2:
            # 最高分仍是第 0 项，只有完整词表检查能发现第二项被污染。
            rows[0, 1] += 1
        ids = self.model.compute_logits(rows).argmax(dim=-1).tolist()
        if self.damage == "token" and not is_prefill and len(seqs) == 2:
            ids[0] = 1
        return ids

    def call(self, method, *args):
        return getattr(self, method)(*args)


class TestBatchOrder(unittest.TestCase):
    def test_joint_decode_reorders_live_requests_and_compares_each_row_to_hf(self):
        for damage in (None, "score", "token"):
            with self.subTest(damage=damage):
                engine = LLMEngine.__new__(LLMEngine)
                engine.model_runner = CpuRunner(damage)
                config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=64,
                                         eos=None, kvcache_block_size=256, num_kvcache_blocks=8,
                                         prefix_cache_disabled_reason=None)
                engine.scheduler = Scheduler(config)
                prompts = {"A": [1, 2, 3], "B": [9, 6, 8, 2]}
                for order in (["A", "B"], ["B", "A"]):
                    captured = check_batch_order.capture_round(engine, prompts, order)
                    report = check_batch_order.compare_round(HistoryModel(), captured, exact=True)
                    self.assertEqual(report["ok"], not damage)
                    batches = [b["requests"] for b in report["batches"] if b["phase"] == "decode"]
                    self.assertEqual(batches[:2], [order, order[::-1]])
                    self.assertEqual(batches[-1], ["B"])
                    self.assertEqual(len(report["requests"]["A"]["steps"]), 3)
                    self.assertEqual(len(report["requests"]["B"]["steps"]), 5)
                    self.assertEqual(report["requests"][order[0]]["steps"][1]["exact"], damage != "score")
