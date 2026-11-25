"""用本地小模型和真实 Python 包验证 read_code 的公开查询协议。"""

import hashlib
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from nanovllm_selector.port.explore.source import read_code


class ReadCodeTests(unittest.TestCase):
    def setUp(self):
        repo_tmp = Path(__file__).resolve().parents[3] / "tmp"
        repo_tmp.mkdir(exist_ok=True)
        temp_root = Path(os.environ.get("TMPDIR", repo_tmp)).resolve()
        if not temp_root.is_relative_to(repo_tmp):
            temp_root = repo_tmp
        directory = tempfile.TemporaryDirectory(prefix="read-code-", dir=temp_root)
        self.addCleanup(directory.cleanup)
        self.model = Path(directory.name) / "model"
        self.model.mkdir()
        self.config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        }
        self.write_json("config.json", self.config)

    def write_json(self, name, value):
        (self.model / name).write_text(json.dumps(value), encoding="utf-8")

    def query(self, target="hf", **kwargs):
        result = read_code(self.model, {"target": target, **kwargs})
        self.assertTrue(result["ok"], result)
        json.dumps(result)
        return result

    def test_config_meta_tree_and_parameter_shapes(self):
        config = self.query(view="config")
        self.assertEqual(config["config"], self.config)
        first = self.query(limit=1)
        self.assertEqual(first["provenance"], "meta_from_config")
        self.assertEqual(first["items"][0]["ref"], "")
        self.assertTrue(first["items"][0]["type"].endswith(".LlamaForCausalLM"))
        self.assertEqual(first["next_offset"], 1)
        rest = self.query(offset=first["next_offset"], limit=first["total"])
        items = {item["ref"]: item for item in rest["items"]}
        ref = "model.layers.1.self_attn.k_proj"
        self.assertEqual(items[ref]["type"], "torch.nn.modules.linear.Linear")
        self.assertIsNone(rest["next_offset"])
        weights = self.query(ref=ref, view="weights")
        self.assertEqual(weights["provenance"], "meta_from_config")
        row, = weights["items"]
        self.assertEqual((row["key"], row["shape"], row["dtype"]),
                         (ref + ".weight", [8, 16], "float32"))

    def test_source_is_the_installed_definition_and_helpers_are_followable(self):
        from transformers.models.llama import modeling_llama

        method = inspect.unwrap(modeling_llama.LlamaAttention.forward)
        expected, line = inspect.getsourcelines(method)
        ref = "model.layers.0.self_attn"
        page = self.query(ref=ref, method="forward", view="source", offset=2, limit=3)
        self.assertEqual(page["source"], "".join(expected[2:5]))
        self.assertEqual((page["line"], page["start_line"], page["next_offset"]),
                         (line, line + 2, 5))
        path = Path(inspect.getsourcefile(method)).resolve()
        self.assertEqual(Path(page["file"]).resolve(), path)
        self.assertEqual(page["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        module = modeling_llama.__name__
        self.assertEqual(page["source_name"], module + ".LlamaAttention.forward")
        refs = {item["ref"] for item in page["related"]}
        self.assertIn(module + ".apply_rotary_pos_emb", refs)
        helper = self.query(view="source", ref=module + ".apply_rotary_pos_emb")
        self.assertIn("def apply_rotary_pos_emb(", helper["source"])
        neighbors = self.query(ref=module, limit=500)["items"]
        self.assertIn(module + ".LlamaMLP", {item["ref"] for item in neighbors})
        cache = self.query(view="source", ref=module + ".Cache")
        self.assertEqual(cache["source_name"], "transformers.cache_utils.Cache")
        torch_method = self.query(view="source", ref="torch.nn.Linear", method="forward")
        self.assertEqual(torch_method["source_name"], "torch.nn.modules.linear.Linear.forward")

    def test_engine_modules_signatures_and_inherited_method(self):
        root = self.query(target="engine")
        module = "nanovllm.layers.linear"
        self.assertIn(module, {item["ref"] for item in root["items"]})
        members = self.query(target="engine", ref=module)["items"]
        cls = module + ".QKVParallelLinear"
        entry = next(item for item in members if item["ref"] == cls)
        self.assertIn("total_num_heads", entry["signature"])
        method = self.query(target="engine", view="source", ref=cls, method="forward")
        self.assertEqual(method["source_name"], module + ".ColumnParallelLinear.forward")
        self.assertIn("return F.linear", method["source"])
        imported = self.query(target="engine", view="source",
                              ref="nanovllm.models.qwen3.QKVParallelLinear")
        self.assertEqual(imported["source_name"], cls)

    def test_public_exports_and_python_protocol_methods(self):
        ref = "transformers.BertModel"
        bert = self.query(ref=ref, view="source", method="forward", limit=2)
        self.assertEqual(bert["source_name"], "transformers.models.bert.modeling_bert.BertModel.forward")
        self.assertIn("input_ids", bert["signature"])
        ctor = self.query(target="engine", view="source", ref="nanovllm.config.Config", method="__post_init__")
        self.assertEqual(ctor["source_name"], "nanovllm.config.Config.__post_init__")

    def test_engine_checkout_provenance_when_package_is_not_installed(self):
        with patch("nanovllm_selector.port.explore.source.find_spec", return_value=None):
            reply = self.query(target="engine", ref="nanovllm.utils.loader.load_model", view="source")
        self.assertEqual(reply["provenance"], "checkout_source")
        self.assertTrue(reply["file"].endswith("/nano-vllm/nanovllm/utils/loader.py"))

    def test_tree_exposes_module_parameters_even_when_disk_keys_differ(self):
        save_file({"packed.weight": torch.zeros(4, dtype=torch.float16)}, self.model / "model.safetensors")
        ref = "model.layers.0.self_attn.k_proj"
        node, = self.query(ref=ref)["items"]
        self.assertEqual(node["parameters"], [{"key": ref + ".weight", "shape": [8, 16], "dtype": "float32"}])
        row, = self.query(view="weights")["items"]
        self.assertEqual((row["key"], row["shape"], row["dtype"]), ("packed.weight", [4], "F16"))

    def test_pytorch_zip_checkpoint_metadata(self):
        torch.save({"embedding": torch.zeros(3, 5, dtype=torch.bfloat16)}, self.model / "pytorch_model.bin")
        save_file({"adapter": torch.zeros(1)}, self.model / "adapter.safetensors")
        weights = self.query(view="weights")
        self.assertEqual(weights["provenance"], "pytorch")
        self.assertEqual(weights["items"], [
            {"key": "embedding", "shape": [3, 5], "dtype": "bfloat16", "file": "pytorch_model.bin"},
        ])
        self.write_json("model.safetensors.index.json", {"weight_map": {"unused": "stale.safetensors"}})
        save_file({"active": torch.zeros(1)}, self.model / "model.safetensors")
        self.assertEqual(self.query(view="weights")["items"][0]["key"], "active")

    def test_sharded_checkpoint_keys_shapes_dtypes_and_paging(self):
        a, b = "model.layers.0.a", "model.layers.1.b"
        shard_a, shard_b = "part-a.safetensors", "part-b.safetensors"
        save_file({a: torch.zeros(2, 3, dtype=torch.float16)}, self.model / shard_a)
        save_file({b: torch.ones(4, dtype=torch.int64)}, self.model / shard_b)
        save_file({"orphan": torch.ones(1)}, self.model / "other.safetensors")
        self.write_json("model.safetensors.index.json", {"weight_map": {b: shard_b, a: shard_a}})
        first = self.query(view="weights", limit=1)
        self.assertEqual((first["provenance"], first["total"], first["next_offset"]),
                         ("safetensors", 2, 1))
        rows = first["items"] + self.query(view="weights", offset=1)["items"]
        self.assertEqual([(r["key"], r["shape"], r["dtype"], r["file"]) for r in rows],
                         [(a, [2, 3], "F16", shard_a), (b, [4], "I64", shard_b)])
        scoped = self.query(view="weights", ref="model.layers.1")
        self.assertEqual(scoped["items"], rows[1:])

    def test_custom_code_cannot_silently_use_a_different_native_class(self):
        self.config.update(architectures=["CustomLM"], auto_map={"AutoModelForCausalLM": "custom.CustomLM"})
        (self.model / "custom.py").write_text("raise AssertionError('远程代码被执行')\n")
        for model_type in ("llama", "custom_model"):
            self.config["model_type"] = model_type
            self.write_json("config.json", self.config)
            with self.subTest(model_type=model_type):
                result = read_code(self.model, {"target": "hf"})
                self.assertFalse(result["ok"])
                self.assertIn("remote_code", result["error"])
                self.assertEqual(self.query(view="config")["config"]["model_type"], model_type)

    def test_only_authorized_packages_metadata_and_references_are_read(self):
        for target, ref in (("hf", ".env"), ("hf", "../config.json"), ("hf", "os.environ"),
                            ("hf", "torch.nn.modules.module.os.environ"), ("hf", "torch.__dict__"),
                            ("engine", "torch.nn.Linear"), ("engine", "nanovllm.config.os")):
            with self.subTest(target=target, ref=ref):
                result = read_code(self.model, {"target": target, "view": "source", "ref": ref})
                self.assertFalse(result["ok"])
        for args in ({"offset": -1}, {"limit": True}, {"method": "__globals__", "view": "source"},
                     {"target": "other"}, {"view": "other"}, {"path": ".env"}):
            with self.subTest(args=args):
                self.assertFalse(read_code(self.model, {"target": "hf", **args})["ok"])
        secret = self.model / ".env"
        save_file({"x": torch.zeros(1)}, secret)
        save_file({"x": torch.zeros(1)}, self.model.parent / "other.safetensors")
        (self.model / "bad.safetensors").symlink_to(secret)
        for filename in ("../other.safetensors", ".env", "bad.safetensors"):
            self.write_json("model.safetensors.index.json", {"weight_map": {"x": filename}})
            result = read_code(self.model, {"target": "hf", "view": "weights"})
            self.assertFalse(result["ok"])
            self.assertNotIn("PRIVATE_VALUE", repr(result))
        (self.model / "config.json").unlink()
        secret.write_text('{"secret": "PRIVATE_VALUE"}', encoding="utf-8")
        (self.model / "config.json").symlink_to(secret)
        self.assertFalse(read_code(self.model, {"target": "hf", "view": "config"})["ok"])


if __name__ == "__main__":
    unittest.main()
