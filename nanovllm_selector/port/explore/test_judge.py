"""双进程验收的离线 CPU 回归。测试权重、分词器和试验产物均在仓库 tmp/ 内。"""

import copy
import hashlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors import safe_open
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


class CheckpointIdentityTests(unittest.TestCase):
    def test_shards_use_content_and_unchanged_files_are_read_once_per_process(self):
        from nanovllm_selector.port.explore.judge import checkpoint_sha256

        with temporary_directory() as tmp:
            model = Path(tmp) / "checkpoint"
            model.mkdir()
            (model / "config.json").write_text("{}")
            parts = {"a": "part-1.safetensors", "b": "part-2.safetensors"}
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": parts}))
            for key, filename in parts.items():
                save_file({key: torch.ones(1)}, model / filename)
            reads = []
            path_open = Path.open

            def observe(path, *args, **kwargs):
                if path.suffix == ".safetensors":
                    reads.append(path.name)
                return path_open(path, *args, **kwargs)

            with patch.object(Path, "open", observe):
                original = checkpoint_sha256(model)
                self.assertEqual(original, checkpoint_sha256(model))
                self.assertCountEqual(reads, parts.values())
                (model / "config.json").write_text('{"use_cache": true}')
                self.assertNotEqual(original, checkpoint_sha256(model))
                self.assertCountEqual(reads, parts.values())
                (model / "config.json").write_text("{}")
                save_file({"a": torch.zeros(1)}, model / parts["a"])
                updated = checkpoint_sha256(model)
                self.assertNotEqual(updated, original)
                self.assertCountEqual(reads, [*parts.values(), parts["a"]])
            # 宿主能在没有任何第三方包的解释器中求出与 worker 相同的摘要。
            script = """import sys
from pathlib import Path
from nanovllm_selector.port.explore.judge import checkpoint_sha256
print(checkpoint_sha256(Path(sys.argv[1])))
"""
            proc = subprocess.run([sys.executable, "-BS", "-c", script, str(model)],
                                  capture_output=True, text=True, check=True)
            self.assertEqual(proc.stdout.strip(), updated)

    def test_reference_fallback_weights_also_contribute_to_checkpoint_identity(self):
        from nanovllm_selector.port.explore.judge import checkpoint_sha256

        with temporary_directory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}")
            save_file({"engine": torch.zeros(1)}, model / "engine.safetensors")
            # 此布局中 nano-vLLM 读 safetensors，Transformers 选择标准 PyTorch 权重。
            torch.save({"hf": torch.zeros(1)}, model / "pytorch_model.bin")
            before = checkpoint_sha256(model)
            torch.save({"hf": torch.ones(1)}, model / "pytorch_model.bin")
            self.assertNotEqual(checkpoint_sha256(model), before)


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
        cls.candidate.write_text("import os\nif os.getenv('JUDGE_TEST_REFERENCE_ONLY'):\n"
                                 "    raise AssertionError('参照导入了候选')\n"
                                 "class LlamaForCausalLM: pass\n", encoding="utf-8")

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
        if cls.__name__ != arch:
            raise AssertionError("引擎用不同的架构名登记了候选")
        with torch.inference_mode(False):
            model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True).eval()
        keep = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}

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
            out = model(torch.tensor([prompt[hit:]]), past_key_values=past, use_cache=True, **keep)
            past = out.past_key_values
            row = out.logits[0, -1].clone()
            prefill.append(row)
            sampled.append(sample(row))
        generated = [sampled.pop()]
        decode = []
        for _ in range(8):
            out = model(torch.tensor([[generated[-1]]]), past_key_values=past, use_cache=True, **keep)
            past = out.past_key_values
            decode.append(out.logits[0, -1].clone())
            generated.append(sample(decode[-1]))
        return {"prefill": prefill[:-1], "prefill_generated": sampled, "prefix_lengths": hits,
                "prefix_reuse": {"enabled": True, "block_size": 256, "reason": "完整 KV 页"},
                "decode": {"prefill": prefill[-1], "logits": decode, "generated": generated,
                           "ids": prompts[-1] + generated}}

    def invoke(self, phase, *args, success=True, trusted_candidate=True):
        from nanovllm_selector.port.explore import judge

        if phase == "reference" and trusted_candidate and not {"--candidate", "--candidate-sha256"}.intersection(args):
            args = (*args, "--candidate-sha256", hashlib.sha256(self.candidate.read_bytes()).hexdigest())
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
        result = self.invoke("engine", "--candidate", self.candidate, *args)
        self.assertEqual({p.name for p in self.trial.iterdir()}, {"engine.json", "engine.safetensors"})
        return result

    def read_meta(self):
        return json.loads((self.trial / "engine.json").read_bytes())

    def write_meta(self, meta):
        (self.trial / "engine.json").write_text(json.dumps(meta), encoding="utf-8")

    def write_rows(self, rows):
        path = self.trial / "engine.safetensors"
        with safe_open(path, framework="pt") as handle:
            binding = handle.metadata()
        save_file(rows, path, metadata=binding)
        meta = self.read_meta()
        meta["logits_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_meta(meta)

    def test_real_hf_replays_all_default_and_independent_holdout_rows(self):
        self.capture()
        digest = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        default = self.invoke("reference", "--candidate-sha256", digest)
        self.assertEqual((default["matched"], default["total"], default["lengths"]), (12, 12, [128, 256, 768]))
        self.assertEqual(default["cached_prefix_tokens"], [0, 0, 256, 512])
        self.assertEqual(default["candidate_sha256"], digest)
        self.assertEqual(default["reference"]["dtype"], "torch.float32")
        self.assertTrue(default["reference"]["attention_backend"])
        self.assertEqual([item["step"] for item in default["decode"]["items"]], list(range(1, 9)))
        self.trial = self.new_trial()
        self.capture("--holdout")
        holdout = self.invoke("reference", "--holdout")
        self.assertEqual((holdout["matched"], holdout["total"], holdout["lengths"]), (10, 10, [255]))
        self.assertNotEqual(default["prompt_sha256"], holdout["prompt_sha256"])

    def cli(self, phase, *args, success=True):
        # 两阶段均在独立解释器运行真实入口；仅用 CPU 等价夹具代替引擎设备。
        entry = """import runpy, torch
from unittest.mock import patch
from nanovllm_selector.port import evaluate
from nanovllm_selector.port.explore.test_judge import JudgeTests
with (
    patch.object(evaluate, "engine_run", JudgeTests.cpu_engine),
    patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
    patch.object(torch.nn.Module, "cuda", lambda m: m),
):
    runpy.run_module("nanovllm_selector.port.explore.judge", run_name="__main__")
"""
        env = os.environ.copy()
        if phase == "reference":
            env.update(JUDGE_TEST_REFERENCE_ONLY="1", NANOVLLM_MODEL_FILE=str(self.candidate))
        proc = subprocess.run([sys.executable, "-B", "-c", entry, phase, "--model", str(self.model),
                               "--trial", str(self.trial), *map(str, args)],
                              env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0 if success else 1, proc.stdout + proc.stderr)
        self.assertEqual(len(proc.stdout.splitlines()), 1)
        return json.loads(proc.stdout)

    def host_binding(self, engine):
        return {"version": 1, "candidate_sha256": hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
                "checkpoint_sha256": engine["checkpoint_sha256"], "holdout": True,
                "engine_json_sha256": hashlib.sha256((self.trial / "engine.json").read_bytes()).hexdigest(),
                "logits_sha256": hashlib.sha256((self.trial / "engine.safetensors").read_bytes()).hexdigest()}

    def test_real_cli_rejects_candidate_b_for_a_and_accepts_the_frozen_candidate(self):
        engine = self.cli("engine", "--holdout", "--candidate", self.candidate)
        other = self.new_trial() / "candidate.py"
        other.write_text("class LlamaForCausalLM:\n    revision = 2\n", encoding="utf-8")
        for args in (("--candidate", other), ("--candidate-sha256", hashlib.sha256(other.read_bytes()).hexdigest())):
            with self.subTest(args=args):
                failed = self.cli("reference", "--holdout", *args, success=False)
                self.assertEqual(failed["stage"], "trial")
                self.assertIn("candidate_sha256", failed["error"])
        self.invoke("reference", "--holdout", trusted_candidate=False, success=False)
        binding = self.trial / "binding.json"
        binding.write_text(json.dumps(self.host_binding(engine)), encoding="utf-8")
        before = {p.name: p.read_bytes() for p in self.trial.iterdir()}
        report = self.cli("reference", "--holdout", "--candidate", self.candidate, "--binding", binding)
        self.assertEqual((report["matched"], report["total"]), (10, 10))
        for key, expected in self.host_binding(engine).items():
            if key != "version":
                self.assertEqual(engine[key], expected)
                self.assertEqual(report[key], expected)
        self.assertEqual(engine["request_sha256"], report["request_sha256"])
        self.assertEqual({p.name: p.read_bytes() for p in self.trial.iterdir()}, before)

    def test_reference_binds_checkpoint_contents_request_and_both_artifacts_before_scoring(self):
        engine = self.capture("--holdout")
        original = self.read_meta()
        for field in ("candidate_sha256", "checkpoint_sha256", "request_sha256"):
            with self.subTest(field=field):
                self.write_meta({**original, field: "f" * 64})
                result = self.invoke("reference", "--holdout", success=False)
                self.assertIn(field, result["error"])
        self.write_meta(original)
        host = self.host_binding(engine)
        binding = self.trial / "binding.json"
        for field in ("candidate_sha256", "checkpoint_sha256", "holdout", "engine_json_sha256", "logits_sha256"):
            with self.subTest(host_field=field):
                binding.write_text(json.dumps({**host, field: False if field == "holdout" else "0" * 64}))
                result = self.invoke("reference", "--holdout", "--binding", binding, success=False)
                self.assertIn(field, result["error"])
        # 语义完全相同的 JSON 也必须与宿主收集时的实际字节一致。
        binding.write_text(json.dumps(host))
        with (self.trial / "engine.json").open("ab") as stream:
            stream.write(b"\n")
        self.assertIn("engine_json_sha256", self.invoke("reference", "--holdout", "--binding", binding,
                                                       success=False)["error"])
        self.model = Path(shutil.copytree(self.model, self.new_trial() / "copy"))
        path = self.model / "model.safetensors"
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with patch.object(ev, "reference_steps", side_effect=AssertionError("不应开始数值验收")):
            self.assertIn("checkpoint_sha256", self.invoke("reference", "--holdout", success=False)["error"])

    def test_relabeling_only_the_manifest_does_not_rebind_the_tensor_payload(self):
        self.capture("--holdout")
        other = self.new_trial() / "candidate.py"
        other.write_text("class LlamaForCausalLM: revision = 2\n")
        self.write_meta({**self.read_meta(), "candidate_sha256": hashlib.sha256(other.read_bytes()).hexdigest()})
        failed = self.invoke("reference", "--holdout", "--candidate", other, success=False)
        self.assertIn("candidate_sha256", failed["error"])

    def test_checkpoint_changed_during_capture_does_not_publish_a_trial(self):
        self.model = Path(shutil.copytree(self.model, self.new_trial() / "model"))
        run = self.cpu_engine

        def changed_after_loading(*args, **kwargs):
            result = run(*args, **kwargs)
            path = self.model / "model.safetensors"
            with path.open("r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                byte = stream.read(1)[0]
                stream.seek(-1, os.SEEK_END)
                stream.write(bytes([byte ^ 1]))
            return result

        self.cpu_engine = changed_after_loading
        result = self.invoke("engine", "--holdout", "--candidate", self.candidate, success=False)
        self.assertIn("checkpoint_sha256", result["error"])
        self.assertEqual(list(self.trial.iterdir()), [])

    def test_strict_engine_requires_a_locally_defined_declared_architecture(self):
        src = self.new_trial() / "candidate.py"
        for code in ("class OtherForCausalLM: pass\n",
                     "class OtherForCausalLM: pass\nLlamaForCausalLM = OtherForCausalLM\n",
                     "from transformers import LlamaForCausalLM\n"):
            with self.subTest(code=code):
                src.write_text(code)
                result = self.invoke("engine", "--candidate", src, "--holdout", success=False)
                self.assertIn("architectures", result["error"])
                self.assertFalse((self.trial / "engine.json").exists())
        self.model = Path(shutil.copytree(self.model, self.new_trial() / "model"))
        path = self.model / "config.json"
        config = json.loads(path.read_bytes())
        config["architectures"].insert(0, "OtherForCausalLM")
        path.write_text(json.dumps(config))
        self.capture("--holdout")
        self.assertEqual(self.read_meta()["architecture"], "LlamaForCausalLM")
        # 历史评测与诊断继续允许从候选中推断未登记的类。
        src.write_text("class AdaptedForCausalLM: pass\n")
        legacy = ev.pick_model_class(ev.load_candidate(src), "LlamaForCausalLM")
        self.assertEqual(legacy.__name__, "AdaptedForCausalLM")

    def test_gpt2_checkpoint_architecture_without_a_causal_lm_suffix_passes_both_phases(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model, local_files_only=True)
        self.model = self.new_trial()
        config = GPT2Config(vocab_size=16, n_positions=512, n_embd=8, n_layer=1, n_head=2,
                            bos_token_id=0, eos_token_id=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            GPT2LMHeadModel(config).save_pretrained(self.model)
        tokenizer.save_pretrained(self.model)
        self.candidate = self.new_trial() / "candidate.py"
        self.candidate.write_text("from torch import nn\nclass GPT2LMHeadModel(nn.Module): pass\n")
        engine = self.capture("--holdout")
        report = self.invoke("reference", "--holdout", "--candidate", self.candidate)
        self.assertEqual(engine["architecture"], "GPT2LMHeadModel")
        self.assertEqual((report["architecture"], report["matched"], report["total"]),
                         ("GPT2LMHeadModel", 10, 10))

    def test_legacy_cli_preserves_fallback_selection_and_both_scoring_modes(self):
        src = self.trial / "candidate.py"
        src.write_text("class AdaptedForCausalLM: pass\n")
        path = self.trial / "legacy.json"

        def slightly_changed(cls, model_dir, sequences, arch, **options):
            self.assertEqual((cls.__name__, arch), ("AdaptedForCausalLM", "LlamaForCausalLM"))
            output = self.cpu_engine(cls, model_dir, sequences, cls.__name__, **options)
            row = output["prefill"][0].clone()
            row[row.argmin()] -= 1e-5
            output["prefill"][0] = row
            return output

        for flag, status in (("--exact", 1), ("--no-exact", 0)):
            with (
                self.subTest(flag=flag),
                patch.dict(os.environ),
                patch.object(ev, "engine_run", slightly_changed),
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
                patch.object(torch.nn.Module, "cuda", lambda m: m),
                redirect_stdout(io.StringIO()),
            ):
                rc = ev.main(["--model", str(self.model), "--candidate", str(src), "--arch", "LlamaForCausalLM",
                              "--lengths", "16", "--report", str(path), flag])
                report = json.loads(path.read_bytes())
                self.assertEqual(rc, status, report)
                self.assertEqual(report["stage"], "compare")
                self.assertTrue(report["comparisons"][0]["argmax_match"])

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
