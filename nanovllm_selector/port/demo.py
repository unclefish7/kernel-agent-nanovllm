"""用 agent 生成出来的模型文件，在真实引擎里生成一段文字，并和 transformers 对照。

这是给人看的入口：不看 logits，直接看两边的输出文本；再用"同一前缀"逐位置比对，
把"文本不同"和"算错"区分开。

    python -m nanovllm_selector.port.demo \
        --model /workspace/tmp/models/llama-3.2-1B \
        --candidate nanovllm_selector/outputs/iter_llama2/attempt_3.py \
        --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import atexit
import gc
from pathlib import Path

import torch

from .evaluate import load_candidate, pick_model_class


def _free_gpu() -> None:
    """把刚放掉的模型显存还给显卡。

    同一个进程里连续构造多个 LLM 时，PyTorch 的缓存分配器不会自动把
    显存退回系统；不清理的话，后一个 LLM 在 allocate_kv_cache 里算出的
    块数会是 0，直接触发 assert。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _strip(prompt_ids: list[int], out_ids: list[int]) -> list[int]:
    if len(out_ids) >= len(prompt_ids) and out_ids[: len(prompt_ids)] == prompt_ids:
        return out_ids[len(prompt_ids):]
    return out_ids


def run_engine(cls, model_dir: str, arch: str, prompt: str,
               max_tokens: int, temperature: float) -> dict:
    import nanovllm.models as models
    from nanovllm import LLM
    from nanovllm.sampling_params import SamplingParams

    models.ARCH[arch] = cls
    llm = LLM(
        model_dir,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.5,
    )
    try:
        prompt_ids = llm.tokenizer(prompt).input_ids
        out = llm.generate(
            [prompt],
            SamplingParams(temperature=temperature, max_tokens=max_tokens),
            use_tqdm=False,
        )[0]
        new_ids = _strip(prompt_ids, out["token_ids"])
        text = llm.tokenizer.decode(new_ids)
    finally:
        atexit.unregister(llm.exit)
        llm.exit()
        del llm
        _free_gpu()
    return {"text": text, "token_ids": new_ids, "prompt_ids": prompt_ids}


def run_hf(model_dir: str, prompt: str, max_tokens: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model = model.cuda().eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    plen = inputs.input_ids.shape[1]
    prompt_ids = inputs.input_ids[0].tolist()
    del model, inputs
    _free_gpu()
    new_ids = out[0][plen:].tolist()
    return {"text": tokenizer.decode(new_ids), "token_ids": new_ids,
            "prompt_ids": prompt_ids}


def teacher_forced(cls, model_dir: str, arch: str, prompt_ids: list[int],
                   ref_ids: list[int], n: int) -> dict:
    """把 HF 自己贪心生成的前缀逐段喂进引擎，看每步 argmax 是否相同。

    自由生成时只要某一步出现近似并列，两边就会分叉，之后比较文本没有意义；
    同一前缀下逐位置比对才是"算得对不对"的判据。
    """
    from .evaluate import engine_logits, reference_logits_multi

    n = min(n, len(ref_ids) - len(prompt_ids))
    starts = len(prompt_ids)
    prefixes = [ref_ids[: starts + k] for k in range(n)]
    eng = engine_logits(cls, model_dir, prefixes, arch,
                        max_model_len=len(ref_ids) + 8)
    _free_gpu()
    ref = reference_logits_multi(model_dir, prefixes)
    _free_gpu()
    same = 0
    first_bad = None
    for k in range(n):
        a, b = int(eng[k].argmax()), int(ref[k].argmax())
        if a == b:
            same += 1
        elif first_bad is None:
            first_bad = k
    return {"same": same, "total": n, "first_bad": first_bad}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用生成的模型文件真跑一段文字")
    parser.add_argument("--model", required=True, help="HF 模型目录（含权重）")
    parser.add_argument("--candidate", required=True, type=Path, help="候选 .py")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1e-8,
                        help="引擎不允许贪心；1e-8 在数值上等价于贪心")
    parser.add_argument("--teacher-forced", type=int, default=24,
                        help="逐位置比对的位置数，0 表示关闭")
    parser.add_argument("--no-hf", action="store_true")
    args = parser.parse_args(argv)

    cls = pick_model_class(load_candidate(args.candidate))
    arch = args.arch or cls.__name__

    print(f"模型：{args.model}")
    print(f"候选：{args.candidate.name}（类 {cls.__name__}）")
    print(f"提示词：{args.prompt!r}")
    print()

    ref = None
    if not args.no_hf:
        ref = run_hf(args.model, args.prompt, args.max_tokens)
        print(f"[HF  ] {ref['text']!r}")

    engine = run_engine(cls, args.model, arch, args.prompt,
                        args.max_tokens, args.temperature)
    print(f"[引擎] {engine['text']!r}")

    if ref is not None:
        n = max(len(engine["token_ids"]), len(ref["token_ids"]), 1)
        hit = sum(1 for a, b in zip(engine["token_ids"], ref["token_ids"]) if a == b)
        diverge = next((i for i, (a, b) in enumerate(
            zip(engine["token_ids"], ref["token_ids"])) if a != b), None)
        print()
        print(f"自由生成：{hit}/{n} 个 token 相同"
              + ("" if diverge is None else f"，从第 {diverge} 个开始分叉"))

    if args.teacher_forced > 0 and ref is not None:
        tf = teacher_forced(cls, args.model, arch,
                            ref["prompt_ids"], ref["token_ids"] + [],
                            args.teacher_forced)
        print(f"同一前缀逐位置 argmax：{tf['same']}/{tf['total']} 相同"
              + ("" if tf["first_bad"] is None
                 else f"（第 {tf['first_bad']} 个不同）"))
        return 0 if tf["same"] == tf["total"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
