"""双进程验收的离线 CPU 回归。测试权重、分词器和试验产物均在仓库 tmp/ 内。"""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from nanovllm_selector.port import evaluate as ev


def temporary_directory():
    root = Path(__file__).resolve().parents[3] / "tmp"
    root.mkdir(exist_ok=True)
    return TemporaryDirectory(dir=root, prefix="judge-test-")


def save_checkpoint(path, dtype):
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(41)
        model = LlamaForCausalLM(config).to(dtype)
        model.lm_head.weight[0, 0] = 1 + 2 ** -10
        model.save_pretrained(path)


class ReferenceDtypeTests(unittest.TestCase):
    def test_reference_uses_the_checkpoint_dtype_and_engine_device_fallback(self):
        with temporary_directory() as tmp, patch.object(torch.nn.Module, "cuda", lambda model: model):
            for dtype, capability, expected in [
                (torch.float32, (9, 0), torch.float32),
                (torch.bfloat16, (9, 0), torch.bfloat16),
                (torch.bfloat16, (7, 5), torch.float32),
            ]:
                with self.subTest(dtype=dtype, capability=capability), patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ):
                    save_checkpoint(tmp, dtype)
                    model = ev.load_reference_model(tmp)
                    self.assertEqual(model.dtype, expected)
                    if dtype == torch.float32:
                        self.assertEqual(model.lm_head.weight[0, 0].item(), 1 + 2 ** -10)
                    rows, _ = ev.reference_steps(tmp, [[2, 3, 4]], [0], [], 0)
                    self.assertEqual(rows[0].dtype, expected)

    def test_exact_rejects_identical_values_in_different_dtypes(self):
        scores = torch.tensor([2., 1.])
        self.assertFalse(ev.compare_logits(scores.bfloat16(), scores, exact=True)["ok"])
        self.assertTrue(ev.compare_logits(scores.bfloat16(), scores, exact=False)["ok"])

    def test_large_finite_logit_errors_have_serializable_metrics(self):
        limit = torch.finfo(torch.float32).max
        report = ev.compare_logits(torch.tensor([limit, 0.]), torch.tensor([-limit, 0.]), exact=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["max_abs_diff"], 2 * limit)
        json.dumps(report, allow_nan=False)


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = temporary_directory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        cls.model = root / "model"
        save_checkpoint(cls.model, torch.float32)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "The": 1, "quick": 2, "fox": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(cls.model)
        cls.candidate = root / "candidate.py"
        cls.candidate.write_text("class LlamaForCausalLM: pass\n", encoding="utf-8")

    def setUp(self):
        self.trial = self.new_trial()

    def new_trial(self):
        temp = temporary_directory()
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    @staticmethod
    @torch.inference_mode()
    def cpu_engine(cls, model_dir, sequences, arch, **options):
        # 手动按串行请求留下的 KV 裁出完整页，独立产生引擎夹具，不调用参照打分函数。
        # 真实引擎在 inference_mode 外构造参数，CPU 的矩阵乘法也会区分这两类张量。
        with torch.inference_mode(False):
            model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).eval()
        def sample(row):
            return int(torch.multinomial((row == row.max()).float(), 1).item())
        prompts = [*sequences, sequences[-1]]
        hits = [0, 0, 256, 512] if len(prompts) == 4 else [0, 0]
        past = None
        prefill, sampled = [], []
        for prompt, hit in zip(prompts, hits, strict=True):
            if hit:
                past = copy.deepcopy(past)
                past.crop(hit)
            else:
                past = None
            out = model(torch.tensor([prompt[hit:]]), past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            row = out.logits[0, -1].clone()
            prefill.append(row)
            sampled.append(sample(row))
        generated = [sampled.pop()]
        decode = []
        for _ in range(8):
            out = model(torch.tensor([[generated[-1]]]), past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            decode.append(out.logits[0, -1].clone())
            generated.append(sample(decode[-1]))
        return {"prefill": prefill[:-1], "prefill_generated": sampled, "prefix_lengths": hits,
                "prefix_reuse": {"enabled": True, "block_size": 256, "reason": "完整 KV 页"},
                "decode": {"prefill": prefill[-1], "logits": decode, "generated": generated,
                           "ids": prompts[-1] + generated}}

    def invoke(self, phase, *args, success=True):
        from nanovllm_selector.port.explore import judge

        stdout = io.StringIO()
        with redirect_stdout(stdout), patch.object(ev, "engine_run", self.cpu_engine), patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), patch.object(torch.nn.Module, "cuda", lambda model: model):
            rc = judge.main([phase, "--model", str(self.model), "--trial", str(self.trial), *map(str, args)])
        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        report = json.loads(lines[0])
        self.assertEqual(rc, 0 if success else 1, report)
        self.assertIs(report["ok"], success, report)
        return report

    def capture(self, *args):
        self.invoke("engine", "--candidate", self.candidate, *args)
        self.assertEqual({p.name for p in self.trial.iterdir()}, {"engine.json", "engine.safetensors"})

    def read_meta(self):
        return json.loads((self.trial / "engine.json").read_bytes())

    def write_meta(self, meta):
        (self.trial / "engine.json").write_text(json.dumps(meta), encoding="utf-8")

    def write_rows(self, rows):
        path = self.trial / "engine.safetensors"
        save_file(rows, path)
        meta = self.read_meta()
        meta["logits_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_meta(meta)

    def test_real_hf_replays_all_default_and_independent_holdout_rows(self):
        self.capture()
        default = self.invoke("reference", "--candidate-sha256", "a" * 64)
        self.assertEqual((default["matched"], default["total"], default["lengths"]), (12, 12, [128, 256, 768]))
        self.assertEqual(default["cached_prefix_tokens"], [0, 0, 256, 512])
        self.assertEqual(default["candidate_sha256"], "a" * 64)
        self.assertEqual(default["reference"]["dtype"], "torch.float32")
        self.assertTrue(default["reference"]["attention_backend"])
        self.assertEqual([item["step"] for item in default["decode"]["items"]], list(range(1, 9)))
        self.trial = self.new_trial()
        self.capture("--holdout")
        holdout = self.invoke("reference", "--holdout")
        self.assertEqual((holdout["matched"], holdout["total"], holdout["lengths"]), (10, 10, [255]))
        self.assertNotEqual(default["prompt_sha256"], holdout["prompt_sha256"])

    def test_reference_only_reads_and_hashes_candidate_and_ignores_claimed_verdict(self):
        self.capture("--holdout")
        trap = self.trial / "attachment.py"
        trap.write_text("raise AssertionError('参照导入了候选')\n", encoding="utf-8")
        meta = self.read_meta()
        meta.update(ok=False, candidate_sha256="f" * 64)
        self.write_meta(meta)
        before = {p.name: p.read_bytes() for p in self.trial.iterdir()}
        # 新解释器只将硬件和搬运替换成 CPU，按模块的真实入口读取已完成的试验。
        entry = """import runpy, torch
from unittest.mock import patch
with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), patch.object(torch.nn.Module, "cuda", lambda m: m):
    runpy.run_module("nanovllm_selector.port.explore.judge", run_name="__main__")
"""
        proc = subprocess.run([sys.executable, "-B", "-c", entry, "reference", "--model", str(self.model),
                               "--trial", str(self.trial), "--holdout", "--candidate", str(trap)],
                              env={**os.environ, "NANOVLLM_MODEL_FILE": str(trap)},
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        report = json.loads(proc.stdout)
        self.assertEqual((report["matched"], report["total"]), (10, 10))
        self.assertEqual(report["candidate_sha256"], hashlib.sha256(before["attachment.py"]).hexdigest())
        self.assertNotIn("candidate_sha256", self.invoke("reference", "--holdout"))
        self.assertEqual({p.name: p.read_bytes() for p in self.trial.iterdir()}, before)

    def test_tied_maxima_replay_the_same_nonfirst_samples(self):
        path = self.model
        self.model = self.new_trial()
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        with torch.no_grad():
            model.lm_head.weight.zero_()
        model.save_pretrained(self.model)
        PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True).save_pretrained(self.model)
        histories = []
        with torch.random.fork_rng(devices=[]):
            for seed in (71, 72):
                torch.manual_seed(seed)
                self.trial = self.new_trial()
                self.capture("--holdout")
                histories.append(self.invoke("reference", "--holdout")["generated_ids"])
        self.assertTrue(any(histories[0]))
        self.assertEqual(*histories)

    def test_engine_errors_include_the_candidate_line_and_operation(self):
        candidate = self.trial / "candidate.py"
        candidate.write_text("def project():\n    return 1 // 0\nproject()\n", encoding="utf-8")
        report = self.invoke("engine", "--candidate", candidate, success=False)
        self.assertEqual(report["stage"], "engine")
        self.assertIn('candidate.py", line 2', report["error"])
        self.assertIn("return 1 // 0", report["error"])
        self.assertIn("ZeroDivisionError", report["error"])
        self.assertLessEqual(len(report["error"]), 1800)

    def test_reference_rejects_changed_request_history_counters_and_sampled_ids(self):
        self.capture()
        original = self.read_meta()
        for label, edit in [
            ("输入", lambda x: x["requests"][0]["prompt_ids"].__setitem__(5, 11)),
            ("输入类型", lambda x: x["requests"][0]["prompt_ids"].__setitem__(0, True)),
            ("请求数", lambda x: x["requests"].pop()),
            ("长度", lambda x: x["lengths"].reverse()),
            ("步数", lambda x: x.update(decode_steps=1)),
            ("页大小", lambda x: x["prefix_reuse"].update(block_size=4096)),
            ("虚假命中", lambda x: x["requests"][0].update(cached_prefix_tokens=256)),
            ("省掉可复用页", lambda x: x["requests"][2].update(cached_prefix_tokens=0)),
            ("少一个词元", lambda x: x["requests"][-1]["generated_ids"].pop()),
            ("词元越界", lambda x: x["requests"][-1]["generated_ids"].__setitem__(8, 16)),
            ("词元负数", lambda x: x["requests"][-1]["generated_ids"].__setitem__(0, -1)),
            ("布尔值", lambda x: x["requests"][0].update(cached_prefix_tokens=False)),
            ("prefill 采样", lambda x: x["requests"][0]["generated_ids"].__setitem__(
                0, (x["requests"][0]["generated_ids"][0] + 1) % 16)),
            ("最后采样", lambda x: x["requests"][-1]["generated_ids"].__setitem__(
                8, (x["requests"][-1]["generated_ids"][8] + 1) % 16)),
        ]:
            with self.subTest(label=label):
                meta = copy.deepcopy(original)
                edit(meta)
                meta["ok"] = True
                self.write_meta(meta)
                self.assertEqual(self.invoke("reference", success=False)["outcome"], "candidate_failed")
        meta = copy.deepcopy(original)
        meta["prefix_reuse"]["enabled"] = False
        for req in meta["requests"]:
            req["cached_prefix_tokens"] = 0
        self.write_meta(meta)
        self.assertEqual(self.invoke("reference", success=False)["outcome"], "candidate_failed")

    def test_scores_require_every_vocab_element_row_and_native_dtype(self):
        self.capture("--holdout")
        rows = load_file(self.trial / "engine.safetensors")
        for name in ("prefill_0", "generation_prefill", "decode_8"):
            changed = dict(rows)
            row = changed[name] = rows[name].clone()
            row[row.argmin()] -= 0.01
            self.write_rows(changed)
            report = self.invoke("reference", "--holdout", success=False)
            self.assertEqual((report["matched"], report["total"]), (9, 10))
        for label, edit in [
            ("少一行", lambda x: x.pop("decode_8")),
            ("多一行", lambda x: x.update(extra=x["decode_1"].clone())),
            ("多一维", lambda x: x.update(decode_1=x["decode_1"][None, :])),
            ("词表截断", lambda x: x.update(decode_1=x["decode_1"][:-1])),
            ("整数分数", lambda x: x.update(decode_1=x["decode_1"].int())),
            ("精度", lambda x: x.update(decode_1=x["decode_1"].bfloat16())),
            ("非有限分数", lambda x: x["decode_1"].__setitem__(0, float("nan"))),
        ]:
            with self.subTest(label=label):
                changed = {k: v.clone() for k, v in rows.items()}
                edit(changed)
                self.write_rows(changed)
                self.assertEqual(self.invoke("reference", "--holdout", success=False)["outcome"], "candidate_failed")

    def test_reference_rejects_oversized_files_and_symlinks_before_loading_weights(self):
        self.capture("--holdout")
        path = self.trial / "engine.safetensors"
        original = path.read_bytes()
        path.unlink()
        path.symlink_to(self.candidate)
        with patch.object(ev, "load_reference_model", side_effect=AssertionError("不应装载权重")):
            for target in ("链接", "张量大小", "元数据大小"):
                with self.subTest(target=target):
                    if target == "张量大小":
                        path.unlink()
                        with path.open("wb") as stream:
                            stream.truncate(65 * 1024 * 1024)
                    if target == "元数据大小":
                        path.write_bytes(original)
                        (self.trial / "engine.json").write_bytes(b" " * (128 * 1024))
                    self.assertEqual(self.invoke("reference", "--holdout", success=False)["stage"], "trial")


if __name__ == "__main__":
    unittest.main()
