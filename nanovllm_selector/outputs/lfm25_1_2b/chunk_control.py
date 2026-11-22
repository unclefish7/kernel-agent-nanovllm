"""在同一 GPU、同一权重和同一批词元上隔离分块时的算子差异。"""

import atexit
import gc
import hashlib
import json
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanovllm import LLM, SamplingParams
from nanovllm.models import ARCH
from nanovllm.utils.context import get_context
from nanovllm_selector.port.evaluate import FILLER_TEXT, GREEDY_TEMPERATURE, load_candidate, pick_model_class


HERE = Path(__file__).parent
MODEL = "tmp/models/LFM2.5-1.2B-Base"
CANDIDATE = HERE / "agent_repair/attempt_1.py"
SDPA = F.scaled_dot_product_attention


def diff(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    return dict(different=(a != b).sum().item(), total=a.numel(),
                max_abs=(a.double() - b.double()).abs().max().item())


def causal_mask(n, length, device):
    return torch.arange(length, device=device) <= (
        torch.arange(n, device=device) + length - n)[:, None]


def repeated(t, heads):
    return t.repeat_interleave(heads // t.shape[1], dim=1)


def split_native(q, k, v, **kw):
    """只拆 HF 的注意力调用。短卷积、线性层、缓存与后续逐词计算仍由 HF 执行。"""
    if q.shape[2] != 24:
        return SDPA(q, k, v, **kw)
    assert k.shape[2] == 24 and kw["is_causal"] and kw.get("attn_mask") is None
    first = SDPA(q[:, :, :16], k[:, :, :16].contiguous(), v[:, :, :16].contiguous(), **kw)
    mask = causal_mask(8, 24, q.device)[None, None]
    second = SDPA(q[:, :, 16:], repeated(k, q.shape[1]), repeated(v, q.shape[1]),
                  **dict(kw, attn_mask=mask, is_causal=False, enable_gqa=False))
    return torch.cat((first, second), dim=2)


def precision_control(q, k, v, scale):
    """固定原始 Q/K/V 的值，仅改变查询行数、因果掩码表达和计算精度。"""
    scores = (q.double() @ repeated(k, q.shape[1]).double().transpose(-1, -2)) * scale
    scores.masked_fill_(~causal_mask(24, 24, q.device), -torch.inf)
    oracle = scores.softmax(-1) @ repeated(v, q.shape[1]).double()
    report = {}
    for dtype in (torch.bfloat16, torch.float32):
        x, y, z = [t.to(dtype) for t in (q, k, v)]
        whole = SDPA(x, y, z, is_causal=True, enable_gqa=True, scale=scale)
        mask = causal_mask(8, 24, q.device)[None, None]
        chunk = SDPA(x[:, :, 16:], repeated(y, x.shape[1]), repeated(z, x.shape[1]),
                     attn_mask=mask, scale=scale)
        report[str(dtype)] = dict(
            whole_vs_chunk=diff(whole[:, :, 16:], chunk),
            whole_vs_fp64_max=(whole[:, :, 16:].double() - oracle[:, :, 16:]).abs().max().item(),
            chunk_vs_fp64_max=(chunk.double() - oracle[:, :, 16:]).abs().max().item())
    return report


def engine_trace(engine, ids, budget, square_attention=False):
    trace = defaultdict(list)
    histories = defaultdict(list)
    state_checks, page_checks, scores = [], [], []
    model = engine.model_runner.model
    engine.scheduler.max_num_batched_tokens = budget
    layer = None

    def record(name, t):
        if get_context().is_prefill:
            trace[name].append(t.reshape(t.shape[0], -1).detach().cpu().clone())

    def enter_attn(index, args):
        nonlocal layer
        layer = index
        q, k, v = args
        for name, value in zip(("q", "k", "v"), args):
            record(f"L{index}.attn.{name}", value)
        histories[index, "k"].append(k)
        histories[index, "v"].append(v)

    def attention(q, k, v, *args, **kw):
        assert not args
        n, length = q.shape[2], k.shape[2]
        expected = [torch.cat(histories[layer, name]).transpose(0, 1)[None]
                    for name in ("k", "v")]
        m = kw.get("attn_mask")
        eq_mask = (torch.equal(m[0, 0], causal_mask(n, length, q.device)) if m is not None
                   else kw["is_causal"] == (n == length and n > 1) and (n == length or n == 1))
        page_checks.append(dict(layer=layer, query=n, keys=length, mask_exact=eq_mask,
                                k=diff(k, repeated(expected[0], k.shape[1])),
                                v=diff(v, repeated(expected[1], v.shape[1]))))
        if square_attention and 1 < n < length:
            # 只替换这一次 SDPA 的调用形状。KV 仍取实际调用已分页读取的值。
            groups = q.shape[1] // expected[0].shape[1]
            key, value = [x[:, ::groups].contiguous() for x in (k, v)]
            prefix = q.new_zeros((1, q.shape[1], length - n, q.shape[-1]))
            out = SDPA(torch.cat((prefix, q), dim=2), key, value, is_causal=True,
                       enable_gqa=True, scale=kw["scale"])[:, :, -n:]
        else:
            out = SDPA(q, k, v, **kw)
        record(f"L{layer}.attn.sdpa", out[0].transpose(0, 1))
        return out

    def conv_in(index, y):
        b, _, x = y.chunk(3, dim=-1)
        histories[index, "bx"].append(b * x)

    def conv_end(index, mod):
        actual = mod._request_state_buffers["conv_cache"][get_context().request_slots[0]]
        window = torch.cat(histories[index, "bx"])[-actual.shape[-1]:].T
        state_checks.append(dict(layer=index, tokens=sum(t.size(0) for t in histories[index, "bx"]),
                                 **diff(actual, window)))

    with ExitStack() as stack:
        def hook(module, fn):
            stack.callback(module.register_forward_hook(fn).remove)

        for name, module in model.model.named_modules():
            parts = name.split(".")
            if len(parts) < 3 or parts[0] != "layers":
                continue
            i, suffix = int(parts[1]), ".".join(parts[2:])
            if suffix == "self_attn.attn":
                stack.callback(module.register_forward_pre_hook(lambda m, a, i=i: enter_attn(i, a)).remove)
            if suffix == "conv.in_proj":
                hook(module, lambda m, a, y, i=i: conv_in(i, y))
            if suffix == "conv":
                hook(module, lambda m, a, y, i=i: conv_end(i, m))
            if suffix in {"operator_norm", "conv.in_proj", "conv.out_proj", "self_attn.out_proj",
                          "feed_forward.w2", "ffn_norm"}:
                hook(module, lambda m, a, y, name=name: record(name, y[0] if isinstance(y, tuple) else y))
        original = model.compute_logits

        def logits(hidden):
            out = original(hidden)
            scores.append(out[0].detach().cpu().clone())
            return out

        stack.enter_context(patch.object(model, "compute_logits", logits))
        stack.enter_context(patch.object(F, "scaled_dot_product_attention", attention))
        generated = engine.generate([ids], SamplingParams(temperature=GREEDY_TEMPERATURE,
                                    max_tokens=3, ignore_eos=True), use_tqdm=False)[0]["token_ids"]
    return (scores[-3:], generated, {name: torch.cat(ts) for name, ts in trace.items()},
            dict(window_checks=state_checks, page_checks=page_checks))


def hf_trace(model, ids, continuation, split, precision):
    def attention(q, k, v, **kw):
        if q.shape[2] == 24 and not precision:
            precision.update(precision_control(q, k, v, kw["scale"]))
        return split_native(q, k, v, **kw) if split else SDPA(q, k, v, **kw)

    scores, cache = [], None
    with patch.object(F, "scaled_dot_product_attention", attention):
        for tokens in (ids, *([t] for t in continuation)):
            out = model(torch.tensor([tokens], device=model.device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            scores.append(out.logits[0, -1].detach().cpu().clone())
    return scores


@torch.inference_mode()
def main():
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2" and torch.cuda.device_count() == 1
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenizer(FILLER_TEXT).input_ids[:24]
    cls = pick_model_class(load_candidate(CANDIDATE))
    with patch.dict(ARCH, {cls.__name__: cls}):
        engine = LLM(MODEL, max_num_seqs=1, max_model_len=32, max_num_batched_tokens=32,
                     gpu_memory_utilization=0.35, enforce_eager=True)
        try:
            whole, tokens, full_trace, _ = engine_trace(engine, ids, 32)
            chunk, chunk_tokens, chunk_trace, state = engine_trace(engine, ids, 16)
            square, square_tokens, _, _ = engine_trace(engine, ids, 16, square_attention=True)
        finally:
            atexit.unregister(engine.exit)
            engine.exit()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    assert tokens == chunk_tokens == square_tokens
    boundary = {k: diff(v, chunk_trace[k]) for k, v in full_trace.items()}
    first = next((k for k, d in boundary.items() if d["different"]), None)
    precision = {}
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    native = hf_trace(hf, ids, tokens[:2], False, precision)
    controlled = hf_trace(hf, ids, tokens[:2], True, precision)
    def rows(a, b):
        return [diff(x, y) for x, y in zip(a, b, strict=True)]
    report = dict(device=str(torch.cuda.get_device_properties(0)), hip_visible_devices="2",
                  torch=torch.__version__, transformers=transformers.__version__, dtype="bfloat16",
                  candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                  prompt=ids, generated=tokens, chunk_generated=chunk_tokens, square_generated=square_tokens,
                  engine_full_vs_hf=rows(whole, native), engine_chunk_vs_hf=rows(chunk, native),
                  hf_split_attention_vs_hf=rows(controlled, native),
                  engine_chunk_vs_hf_split_attention=rows(chunk, controlled),
                  engine_chunk_square_attention_vs_hf=rows(square, native), first_boundary=first,
                  boundary_differences=boundary, first_hf_attention_precision=precision, **state)
    (HERE / "chunk_control.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k: report[k] for k in ("first_boundary", "engine_chunk_vs_hf",
               "engine_chunk_vs_hf_split_attention", "engine_chunk_square_attention_vs_hf")}, flush=True)


if __name__ == "__main__":
    main()
