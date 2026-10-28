"""三步接入：prepare 采集真实规格，select 调用 Agent，run 离线安装方案。"""

import argparse
import atexit
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
import triton
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as HFMLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer as HFDecoder

from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.activation import SiluAndMul
from nanovllm.models.qwen3 import Qwen3MLP, Qwen3ForCausalLM
from nanovllm.utils.context import get_context

from agent import select
from candidates import call_report, catalog, catalog_hash, install, restore, validate_plan


SETTINGS = dict(enforce_eager=True, tensor_parallel_size=1, max_model_len=512,
                max_num_batched_tokens=128, max_num_seqs=2, gpu_memory_utilization=0.25)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def binding_for(model_path):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3" or config.hidden_act != "silu":
        raise ValueError("本轮只接 Qwen3 的 SiLU MLP")
    with (model_path / "model.safetensors").open("rb") as handle:
        weights = hashlib.file_digest(handle, "sha256").hexdigest()
    vendor = Path(__file__).parent / "vendor"
    manifest = json.loads((vendor / "manifest.json").read_text())["swiglu_fwd.py"]
    if hashlib.sha256((vendor / "swiglu_fwd.py").read_bytes()).hexdigest() != manifest["vendored_sha256"]:
        raise ValueError("候选代码与来源记录不一致")
    native_source = inspect.getsource(Qwen3MLP) + inspect.getsource(SiluAndMul)
    return {
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "weights_sha256": weights, "model_type": config.model_type,
        "dtype": str(config.dtype), "gpu_arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__, "triton": triton.__version__,
        "native_source_sha256": hashlib.sha256(native_source.encode()).hexdigest(),
        "catalog_sha256": catalog_hash(),
    }


def workload(llm):
    prompts = [llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    ) for text in ["你好，简单介绍一下自己。", "What is 2 + 3? Answer briefly."]]
    params = SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True)
    batch = llm.generate(prompts, params, use_tqdm=False)
    long_prompt = llm.tokenizer.encode("A short story about a cat. " * 40)
    long = llm.generate([long_prompt], params, use_tqdm=False)
    assert len(batch) == 2 and len(long) == 1
    assert all(len(item["token_ids"]) == 8 for item in batch + long)
    return {"batch": batch, "long_prompt_tokens": len(long_prompt), "long_output": long}


def close(llm):
    atexit.unregister(llm.exit)
    llm.exit()


def prepare(model_path, output_dir):
    binding = binding_for(model_path)
    llm = LLM(str(model_path), **SETTINGS)
    samples = {}
    layers = set()
    handles = []

    def capture(path, module, inputs):
        x = inputs[0]
        spec = {"phase": "prefill" if get_context().is_prefill else "decode",
                "shape": list(x.shape), "stride": list(x.stride()),
                "dtype": str(x.dtype), "device": str(x.device),
                "contiguous": x.is_contiguous()}
        key = json.dumps(spec, sort_keys=True)
        samples.setdefault(key, spec | {"calls": 0})["calls"] += 1
        layers.add(path)

    try:
        for i, layer in enumerate(llm.model_runner.model.model.layers):
            path = f"model.layers.{i}.mlp.act_fn"
            handles.append(layer.mlp.act_fn.register_forward_pre_hook(
                lambda module, inputs, path=path: capture(path, module, inputs)))
        workload(llm)
        config = llm.model_runner.config.hf_config
        context = {
            "binding": binding,
            "model": {"layers": config.num_hidden_layers, "hidden_size": config.hidden_size,
                      "intermediate_size": config.intermediate_size},
            "hardware": {"name": torch.cuda.get_device_name(0), "hip": torch.version.hip},
            "target": {"id": "mlp_activation", "paths": sorted(layers),
                       "contract": "输入最后一维前半是 gate、后半是 up；只替换激活，不重做投影。返回 SiLU(gate)*up 的单个 Tensor，保持 dtype/device，不修改输入。"},
            "observed_inputs": list(samples.values()),
            "reference_sources": {"transformers_mlp": inspect.getsource(HFMLP),
                                  "transformers_decoder_forward": inspect.getsource(HFDecoder.forward),
                                  "nanovllm_mlp": inspect.getsource(Qwen3MLP),
                                  "nanovllm_activation": inspect.getsource(SiluAndMul),
                                  "weight_packing": Qwen3ForCausalLM.packed_modules_mapping},
            "candidates": catalog(),
            "validation_status": "仅做接口接入验证，候选尚未完整数值验收，也无性能排名",
        }
        assert len(layers) == config.num_hidden_layers
        assert {spec["phase"] for spec in samples.values()} == {"prefill", "decode"}
        save(output_dir / "context.json", context)
        print(f"采集完成：{len(layers)} 层，{len(samples)} 种输入规格；未调用 Agent。", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        close(llm)


def run(model_path, output_dir):
    plan = json.loads((output_dir / "plan.json").read_text())
    binding = binding_for(model_path)
    validate_plan(plan, binding)
    installed = []
    original_warmup = ModelRunner.warmup_model

    def install_before_warmup(runner):
        # 原构造流程此时已加载真实权重。补丁仅在本次单进程构造期间有效。
        before = [(name, id(param)) for name, param in runner.model.named_parameters()]
        installed.extend(install(runner.model, plan, binding))
        after = [(name, id(param)) for name, param in runner.model.named_parameters()]
        assert before == after, "安装器改变了参数注册或权重对象"
        try:
            original_warmup(runner)
        finally:
            for _, _, wrapper in installed:
                wrapper.warming = False

    with patch.object(ModelRunner, "warmup_model", new=install_before_warmup):
        llm = LLM(str(model_path), **SETTINGS)
    try:
        outputs = workload(llm)
        calls = call_report(installed)
        candidate = plan["choice"]["candidate_id"]
        for row in calls.values():
            assert row["calls"].get(f"prefill/{candidate}", 0) > 0
            assert row["calls"].get(f"decode/{candidate}", 0) > 0
        restore(installed)
        assert all(mlp.act_fn is wrapper.original for _, mlp, wrapper in installed)
        report = {"status": "integration_smoke_passed", "plan": plan,
                  "settings": SETTINGS, "installed_layers": len(installed),
                  "calls": calls, "outputs": outputs, "original_modules_restored": True,
                  "api_requests_during_run": 0, "strict_correctness_checked": False,
                  "performance_measured": False}
        save(output_dir / "integration.json", report)
        print(json.dumps({key: report[key] for key in
                          ("status", "installed_layers", "original_modules_restored")}, ensure_ascii=False), flush=True)
        print("Agent 选择：", plan["choice"], flush=True)
        print("第一层真实调用：", next(iter(calls.values())), flush=True)
    finally:
        restore(installed)
        close(llm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "select", "run"])
    parser.add_argument("--model", type=Path, default=Path("/workspace/tmp/models/Qwen3-0.6B"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/mlp_agent"))
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if args.mode == "prepare":
        prepare(args.model, args.run_dir)
    elif args.mode == "select":
        context = json.loads((args.run_dir / "context.json").read_text())
        print(json.dumps(select(context, args.run_dir)["choice"], ensure_ascii=False), flush=True)
    else:
        run(args.model, args.run_dir)


if __name__ == "__main__":
    main()
