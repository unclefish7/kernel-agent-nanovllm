"""从实测实参生成 JSON 配方，在隔离的张量上执行显式支持的原语。"""

import math

import torch
from torch.nn import functional as F


def _encode(value, tid):
    if isinstance(value, torch.Tensor):
        return {"tensor": tid(value)}
    if isinstance(value, (list, tuple)):
        return [_encode(x, tid) for x in value]
    if isinstance(value, dict):
        return {k: _encode(v, tid) for k, v in value.items()}
    if isinstance(value, (torch.dtype, torch.layout, torch.memory_format)):
        return {"torch": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"device": str(value)}
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    return {"unsupported_type": type(value).__qualname__}


def instruction(dispatch, tid):
    return dict(seq=dispatch.seq, op=dispatch.op, args=_encode(dispatch.args, tid),
                kwargs=_encode(dispatch.kwargs, tid), result=_encode(dispatch.result, tid),
                writes=[tid(t) for t in dispatch.writes])


def _decode(value, values):
    if isinstance(value, list):
        return [_decode(x, values) for x in value]
    if isinstance(value, dict):
        if "tensor" in value:
            return values[value["tensor"]]
        if "torch" in value:
            return getattr(torch, value["torch"])
        if "device" in value:
            return torch.device(value["device"])
        if "unsupported_type" in value:
            raise ValueError(f"参数类型缺少重放规则 {value}")
        return {k: _decode(v, values) for k, v in value.items()}
    return value


def call_arguments(step, values):
    return (_decode(step["args"], values),
            {name: _decode(arg, values) for name, arg in step["kwargs"].items()})


def _conv(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    rank = weight.ndim - 2
    if transposed:
        fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[rank]
        return fn(x, weight, bias, stride, padding, output_padding, groups, dilation)
    if any(output_padding):
        raise ValueError("普通卷积不支持 output_padding")
    return {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank](
        x, weight, bias, stride, padding, dilation, groups)


def _index_put(x, indices, values, accumulate=False):
    # None 是完整轴。index_put_ 同时支持重复索引累加及非连续的高级索引轴。
    if accumulate:
        if any(i is None for i in indices):
            raise ValueError("带完整轴的累加索引写缺少独立实现")
        return x.index_put_(tuple(indices), values, accumulate=True)
    x[tuple(slice(None) if i is None else i for i in indices)] = values
    return x


def _slice(x, dim=0, start=None, end=None, step=1):
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start, end, step)
    return x[tuple(slices)]


def _cpu_attention(q, k, v, dropout_p=0., is_causal=False, *, attn_mask=None, scale=None):
    if dropout_p:
        raise ValueError("随机 dropout 需要 RNG 状态")
    repeats = q.shape[-3] // k.shape[-3]
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=-3)
        v = v.repeat_interleave(repeats, dim=-3)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale)
    work = q if q.dtype == torch.float64 else q.float()
    scores = (work @ k.to(work.dtype).transpose(-1, -2)) * (
        1 / math.sqrt(q.size(-1)) if scale is None else scale)
    if is_causal:
        mask = torch.ones_like(scores, dtype=torch.bool).tril()
        scores.masked_fill_(~mask, -torch.inf)
    if attn_mask is not None:
        scores = scores + attn_mask
    return out, scores.logsumexp(-1)


def _to_copy(x, *, dtype=None, device=None, non_blocking=False,
             memory_format=torch.preserve_format, layout=None, pin_memory=None):
    if pin_memory or layout not in (None, x.layout):
        raise ValueError("设备布局变换缺少独立实现")
    return x.to(dtype=dtype, device=device, copy=True, non_blocking=non_blocking,
                memory_format=memory_format)


def _embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return F.embedding(indices, weight, padding_idx, None, 2., scale_grad_by_freq, sparse)


# 不按名字动态调用 torch.ops。未登记的重载，包括返回值恰好无消费者的算子，
# 都会在该条指令报告缺口。
PRIMITIVES = {
    "aten.view.default": torch.reshape,
    "aten._unsafe_view.default": torch.reshape,
    "aten.reshape.default": torch.reshape,
    "aten.transpose.int": torch.transpose,
    "aten.t.default": torch.t,
    "aten.permute.default": torch.permute,
    "aten.squeeze.default": torch.squeeze,
    "aten.squeeze.dim": torch.squeeze,
    "aten.squeeze.dims": torch.squeeze,
    "aten.unsqueeze.default": torch.unsqueeze,
    "aten.slice.Tensor": _slice,
    "aten.select.int": torch.select,
    "aten.narrow.default": torch.narrow,
    "aten.expand.default": lambda x, size, *, implicit=False: x.expand(size),
    "aten.expand_as.default": torch.Tensor.expand_as,
    "aten.contiguous.default": torch.Tensor.contiguous,
    "aten.clone.default": torch.clone,
    "aten.detach.default": torch.detach,
    "aten.alias.default": lambda x: x,
    "aten.lift_fresh.default": lambda x: x,
    "aten._to_copy.default": _to_copy,
    "aten.as_strided.default": torch.as_strided,
    "aten.split_with_sizes.default": torch.split,
    "aten.split.Tensor": torch.split,
    "aten.chunk.default": torch.chunk,
    "aten.unbind.int": torch.unbind,
    "aten.arange.default": torch.arange,
    "aten.arange.start": torch.arange,
    "aten.arange.start_step": torch.arange,
    "aten.zeros.default": torch.zeros,
    "aten.ones.default": torch.ones,
    "aten.full.default": torch.full,
    "aten.convolution.default": _conv,
    "aten.roll.default": torch.roll,
    "aten.copy_.default": torch.Tensor.copy_,
    "aten.index_put_.default": _index_put,
    "aten.constant_pad_nd.default": lambda x, pad, value=0: F.pad(x, pad, value=value),
    "aten._local_scalar_dense.default": torch.Tensor.item,
    "aten.gt.Scalar": torch.gt,
    "aten.clamp.default": torch.clamp,
    "aten.bmm.default": torch.bmm,
    "aten.mm.default": torch.mm,
    "aten.matmul.default": torch.matmul,
    "aten.addmm.default": torch.addmm,
    "aten.baddbmm.default": torch.baddbmm,
    "aten.add.Tensor": torch.add,
    "aten.add_.Tensor": torch.Tensor.add_,
    "aten.mul.Tensor": torch.mul,
    "aten.neg.default": torch.neg,
    "aten.sum.dim_IntList": torch.sum,
    "aten.sum.default": torch.sum,
    "aten.mean.dim": torch.mean,
    "aten.mean.default": torch.mean,
    "aten.pow.Tensor_Scalar": torch.pow,
    "aten.rsqrt.default": torch.rsqrt,
    "aten.cat.default": torch.cat,
    "aten.cos.default": torch.cos,
    "aten.sin.default": torch.sin,
    "aten.silu.default": F.silu,
    "aten.gelu.default": F.gelu,
    "aten.embedding.default": _embedding,
    "aten._scaled_dot_product_flash_attention_for_cpu.default": _cpu_attention,
}


def _compare(spec, got, values, expected, rtol, atol):
    if isinstance(spec, list):
        if not isinstance(got, (tuple, list)) or len(got) != len(spec):
            raise ValueError("返回数量不符")
        for item, tensor in zip(spec, got):
            _compare(item, tensor, values, expected, rtol, atol)
    elif isinstance(spec, dict) and "tensor" in spec:
        tid = spec["tensor"]
        actual = expected[tid]
        if not isinstance(got, torch.Tensor) or actual.shape != got.shape or actual.dtype != got.dtype:
            raise ValueError(f"{tid} 返回形状或类型不符")
        same_view = actual.is_set_to(got)
        if not same_view:
            if actual.is_floating_point() or actual.is_complex():
                equal = torch.allclose(actual, got, rtol=rtol, atol=atol, equal_nan=True)
            else:
                equal = torch.equal(actual, got)
            if not equal:
                raise ValueError(f"{tid} 返回数值不符")
        values[tid] = got
    elif type(spec) is not type(got) or spec != got:
        raise ValueError(f"标量返回值不符，实测 {spec!r}，重算 {got!r}")


def replay_program(node, dispatches, builder, rtol, atol):
    """只从边界取输入；内部临时量、视图、写目标均使用前序重算结果。"""
    program = node.attrs["program"]
    status = node.attrs["program_replay"] = dict(ok=False, checked=0, total=len(dispatches))
    if not program or len(program) != len(dispatches):
        raise ValueError("配方与实测 dispatch 数量不符")
    bindings = node.attrs.get("program_inputs")
    if bindings != builder.program_input_witnesses[node.id]:
        raise ValueError("配方的边界输入与实测来源不符")
    writable = {t.untyped_storage()._cdata for d in dispatches for t in d.writes}
    private = {}
    values = {}
    expected = {}
    for step, d in zip(program, dispatches):
        try:
            if step != instruction(d, builder.tid):
                raise ValueError("配方与实测参数、返回或顺序不符")
            impl = PRIMITIVES.get(step["op"])
            if impl is None:
                raise ValueError("缺少独立重放实现")
            for t in d.inputs:
                tid = builder.tid(t)
                if tid in values:
                    continue
                snap = d.before[id(t)]
                binding = bindings[tid]
                if binding["kind"] == "literal":
                    snap = torch.tensor(binding["value"], dtype=snap.dtype, device=snap.device).reshape(snap.shape)
                key = t.untyped_storage()._cdata
                if key in writable:
                    if key not in private:
                        private[key] = snap.untyped_storage().clone()
                    snap = torch.empty(0, dtype=snap.dtype, device=snap.device).set_(
                        private[key], snap.storage_offset(), snap.shape, snap.stride())
                values[tid] = snap
            args, kwargs = call_arguments(step, values)
            input_values = [values[builder.tid(t)] for t in d.inputs]
            got = impl(*args, **kwargs)
            for t in d.outputs + d.writes:
                expected[builder.tid(t)] = d.after[id(t)]
            _compare(step["result"], got, values, expected, rtol, atol)
            for output in d.outputs:
                computed = values[builder.tid(output)]
                for source, operand in zip(d.inputs, input_values):
                    alias = output.untyped_storage()._cdata == source.untyped_storage()._cdata
                    if alias != (computed.untyped_storage()._cdata == operand.untyped_storage()._cdata):
                        raise ValueError("返回值与输入的存储别名不符")
            for tid in step["writes"]:
                _compare({"tensor": tid}, values[tid], values, expected, rtol, atol)
            status["checked"] += 1
        except Exception as exc:
            raise ValueError(f"seq={d.seq} {d.op}：{exc}") from exc
    status["ok"] = True
    return values, expected
