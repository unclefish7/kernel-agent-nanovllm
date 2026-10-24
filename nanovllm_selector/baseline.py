"""真实 checkpoint 的原生推理冒烟测试，不安装候选、不调用 Agent。"""

import argparse
import atexit
import gc
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from nanovllm import LLM, SamplingParams


def run_native(model_path, report):
    settings = dict(enforce_eager=True, tensor_parallel_size=1,
                    max_model_len=512, max_num_batched_tokens=128,
                    max_num_seqs=2, gpu_memory_utilization=0.25)
    llm = LLM(str(model_path), **settings)
    runner = llm.model_runner
    dtype = next(runner.model.parameters()).dtype
    report["settings"] = settings | {"dtype": str(dtype),
                                     "kv_blocks": runner.config.num_kvcache_blocks}
    try:
        prompts = [llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        ) for text in ["你好，简单介绍一下自己。", "What is 2 + 3? Answer briefly."]]
        # 这组请求使用原生随机采样，验证真正的双请求 generate 链路。
        outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=32),
                               use_tqdm=False)
        assert len(outputs) == 2 and all(output["token_ids"] for output in outputs)
        report["batch_generation"] = {"input_lengths": list(map(len, prompts)),
                                      "outputs": outputs}

        # 固定 argmax 只用于对照测试；引擎、模型、调度和缓存仍走原路径。
        params = SamplingParams(max_tokens=8, ignore_eos=True)
        logits = []
        original_model = runner.run_model

        def capture(*args):
            output = original_model(*args)
            logits.append(output.float().cpu())
            return output

        with patch.object(runner.sampler, "forward",
                          side_effect=lambda values, temperatures: values.argmax(-1)):
            with patch.object(runner, "run_model", side_effect=capture):
                short = llm.generate([prompts[1]], params, use_tqdm=False)[0]
            assert len(logits) == len(short["token_ids"]) == 8

            long_prompt = llm.tokenizer.encode("A short story about a cat. " * 40)
            assert 256 < len(long_prompt) < 480
            original_run = runner.run
            traces = []

            def trace(seqs, is_prefill):
                traces.append({"is_prefill": is_prefill, "sequences": [
                    {"cached_tokens": seq.num_cached_tokens,
                     "scheduled_tokens": seq.num_scheduled_tokens} for seq in seqs]})
                return original_run(seqs, is_prefill)

            with patch.object(runner, "run", side_effect=trace):
                first = llm.generate([long_prompt], params, use_tqdm=False)[0]
                boundary = len(traces)
                second = llm.generate([long_prompt], params, use_tqdm=False)[0]
            assert first["token_ids"] == second["token_ids"]
            assert len(first["token_ids"]) == 8
            assert sum(step["is_prefill"] for step in traces[:boundary]) >= 3
            assert sum(not step["is_prefill"] for step in traces[:boundary]) == 7
            assert sum(not step["is_prefill"] for step in traces[boundary:]) == 7
            assert traces[0]["sequences"][0]["cached_tokens"] == 0
            assert traces[boundary]["sequences"][0]["cached_tokens"] >= 256
            report["chunked_prefill_and_prefix_cache"] = {
                "input_length": len(long_prompt), "outputs_match": True,
                "output": first, "first_trace": traces[:boundary],
                "repeated_trace": traces[boundary:],
            }
        report["greedy_generation"] = short
        return prompts[1], short["token_ids"], logits, dtype
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


def compare_transformers(model_path, prompt, tokens, actual, dtype):
    reference = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=dtype,
        attn_implementation="eager").to("cuda").eval()
    results = []
    with torch.inference_mode():
        for step, output in enumerate(actual):
            # 用同一条 token 前缀，不让采样差异污染 logits 对照。
            prefix = torch.tensor([prompt + tokens[:step]], device="cuda")
            expected = reference(prefix, logits_to_keep=1).logits[:, -1].float().cpu()
            assert torch.isfinite(output).all() and torch.isfinite(expected).all()
            error = (output - expected).abs()
            item = {"step": step, "phase": "prefill" if step == 0 else "decode",
                    "max_absolute_error": error.max().item(),
                    "mean_absolute_error": error.mean().item(),
                    "native_top_token": output.argmax(-1).item(),
                    "transformers_top_token": expected.argmax(-1).item()}
            results.append(item)
            assert item["native_top_token"] == item["transformers_top_token"], item
    return {"scope": "单条固定输入的 prefill 和 7 步 decode；不证明全部输入数值等价",
            "strict_allclose_checked": False, "steps": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("/workspace/tmp/models/Qwen3-0.6B"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.json"))
    args = parser.parse_args()
    torch.manual_seed(0)
    weights = args.model / "model.safetensors"
    with weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with safe_open(weights, framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    report = {"model": str(args.model), "weight_sha256": digest,
              "weight_tensor_count": tensor_count, "agent_enabled": False,
              "torch": torch.__version__, "hip": torch.version.hip,
              "transformers": transformers.__version__,
              "gpu": torch.cuda.get_device_name(0),
              "visible_gpu_count": torch.cuda.device_count()}
    prompt, tokens, actual, dtype = run_native(args.model, report)
    gc.collect()
    torch.cuda.empty_cache()
    report["transformers_comparison"] = compare_transformers(
        args.model, prompt, tokens, actual, dtype)
    report["status"] = "smoke_passed_not_strict_numerical_equivalence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
