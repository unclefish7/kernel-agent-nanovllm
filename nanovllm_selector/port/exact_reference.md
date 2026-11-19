# 单卡逐项相等的实现依据

验收逐步比较完整词表，允许的最大绝对差为零。以下运算在本项目 ROCm 单卡的 Llama 上经过实测。用计算图、配置和权重判断哪些运算适用于目标模型。保留每个算子的输入类型及舍入位置。

**位置编码。** 引擎在构造模型前将默认设备设为 GPU、默认类型设为模型类型。HF 参照先在 CPU 初始化频率再移到 GPU。在创建频率的 `torch.arange` 上显式给 `device="cpu", dtype=torch.float32`，在 CPU 完成配置指定的频率缩放。移动频率到目标设备后，用 `(inv_freq.view(1,-1,1) @ positions.float().view(1,1,-1))[0].T` 生成相位。余弦与正弦分别转回 Q/K 的类型，再做旋转。直接在 GPU 初始化频率即使只差一个末位，也可能在后续各层放大。

**低精度残差与归一化。** 保留参考图的两个计算步骤。先用模型类型执行 `residual = hidden + residual`，然后才将它转为 float32 计算均方。归一化结果转回输入类型，再乘权重。实现为独立的 PyTorch 操作，避免编译器合并跨越舍入位置的运算。若用引擎的 `RMSNorm` 子类，请覆盖调用编译函数的路径。

**投影与激活。** 打包装载权重后，按输出宽度切出权重和偏置，用独立的 `F.linear(x, weight, bias)` 执行图中每次投影。查询、键、值和门控、上投影分别对应三次和两次计算。单卡复用引擎装载器和行并行输出层。下面两段分别放入对应的注意力与前馈网络的 forward：

```python
widths = (self.q_size, self.kv_size, self.kv_size)
weights = self.qkv_proj.weight.split(widths)
biases = self.qkv_proj.bias.split(widths) if self.qkv_proj.bias is not None else (None,) * 3
q, k, v = (F.linear(hidden_states, w, b) for w, b in zip(weights, biases))
```

```python
gate_w, up_w = self.gate_up_proj.weight.chunk(2, dim=0)
biases = self.gate_up_proj.bias.chunk(2) if self.gate_up_proj.bias is not None else (None, None)
gate = F.linear(x, gate_w, biases[0])
up = F.linear(x, up_w, biases[1])
return self.down_proj(F.silu(gate) * up)
```

**分页缓存与注意力。** 对图中使用分组查询注意力的模型，可继承引擎的 `Attention` 并实现 `forward_torch(self, q, k, v)`。父类的 `forward` 已用 `slot_mapping` 将本步 K/V 写入引擎分配的页。下列代码展示单个请求的 PyTorch 回退路径。优先保留调用形状及遮罩类型；例如前缀复用时的遮罩会影响底层计算路径。

```python
ctx = get_context()
if ctx.is_prefill:
    offsets = ctx.cu_seqlens_q_list
    ends = ctx.cu_seqlens_k_list
    lengths = [b - a for a, b in zip(ends, ends[1:])]
else:
    lengths = ctx.context_lens_list
    offsets = list(range(len(lengths) + 1))
outputs = []
for i, length in enumerate(lengths):
    start, stop = offsets[i:i + 2]
    count = stop - start
    query = q[start:stop].transpose(0, 1).unsqueeze(0)
    if count == 1:
        query = q[start:stop].view(1, self.num_heads, 1, self.head_dim)
    if ctx.block_tables is None:
        key, value = k[start:stop], v[start:stop]
    else:
        size = self.k_cache.size(1)
        blocks = ctx.block_tables[i, :(length + size - 1) // size].long()
        key = self.k_cache[blocks].flatten(0, 1)[:length]
        value = self.v_cache[blocks].flatten(0, 1)[:length]
    key = key.transpose(0, 1).unsqueeze(0).contiguous()
    value = value.transpose(0, 1).unsqueeze(0).contiguous()
    mask = None
    if 1 < count < length:
        pos = torch.arange(count, device=q.device) + length - count
        mask = (torch.arange(length, device=q.device) <= pos[:, None])[None, None]
        repeats = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    y = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=self.scale,
        is_causal=count > 1 and count == length, enable_gqa=mask is None,
    )
    outputs.append(y[0].transpose(0, 1))
result = torch.cat(outputs)
return result if ctx.is_prefill else result.unsqueeze(1)
```

**共享权重。** 配置启用 `tie_word_embeddings` 时，用 `self.lm_head.weight = self.model.embed_tokens.weight` 共享参数对象。对 `.weight.data` 赋值仍会留下两个独立的 Parameter，装载器会对第二个对象报告缺少权重。
