# 算子库索引

给 agent **查阅**用的代码库。这里只解决一件事：想写某个算子时，能快速找到几份可以照着看的实现。

- 不提供 import，不提供调用接口，不写用法说明。
- 每个算子一个目录，文件名保留上游原名。
- agent 按需读源码，参考完自己写；写出来的东西和这里像不像都无所谓。

找法：先 `ls kernels/` 看有哪些算子；没有的话在整个目录里 `grep -rn "<关键字>" kernels/`。

## 目录

| 算子 | 文件 | 这份实现是什么 |
| --- | --- | --- |
| rmsnorm | `rmsnorm/rmsnorm_triton.py` | 一行一个 program，整行 load 进寄存器做归约；入口 `rmsnorm_triton_wrapper(x, w, eps)` |
| rmsnorm | `rmsnorm/fast_rms_layernorm.py` | 带反向和 gemma 变体，用 `torch.autograd.Function` 封装 |
| rmsnorm | `rmsnorm/rmsnorm_fused_llama.py` | llama 风格融合前向；入口 `rmsnorm_forward(x, weight, eps)` |
| layernorm | `layernorm/layernorm_fwd_triton.py` | 标准 LayerNorm 前向，一行一个 program |
| layernorm | `layernorm/fused_layernorm_triton.py` | torch.compile 生成的两段式：先归约求 mean/var，再归一化 |
| swiglu | `swiglu/swiglu_fwd.py` | 输入是打包张量，kernel 里切两半；带 autotune |
| swiglu | `swiglu/swiglu_triton.py` | 两个输入分开传，含反向 kernel 与 `is_hip()` 分支 |
| rope | `rope/rope_embedding.py` | 只处理 Q，cos/sin 从外面传入，前后两半配对旋转 |
| rope | `rope/fast_rope_embedding.py` | Q、K 一起处理；入口 `fast_rope_embedding(Q, K, cos, sin)` |
| attention | `attention/attention_fwd_triton1.py` | flash 风格前向，块级 online softmax，只有前向 |
| attention | `attention/attention_llama.py` | llama 风格，带 `start_position`，可以直接看增量解码怎么摆 |
| attention | `attention/triton_attention.py` | 前向 + 反向的完整实现 |
| softmax | `softmax/softmax_triton1.py` | 按行 softmax，一行一个 program |
| softmax | `softmax/token_softmax_llama.py` | 变长 batch（`B_Start_Loc` / `B_Seqlen`）的 token 级 softmax |
| matmul | `matmul/matmul_triton1.py` | 分块 GEMM，grid 用 lambda 算 |
| matmul | `matmul/matmul_kernel.py` | 最基础的分块 GEMM，显式传 M/N/K |
| embedding | `embedding/embedding_triton_kernel.py` | 带词表分片（`vob_start_id` / `vob_end_id`）的查表 |
| elementwise | `elementwise/add_example.py` | 最简 elementwise add，适合当新 kernel 的骨架 |

## 缺口

还没有的：paged attention / KV cache 写入、fp8 与量化、fused linear + activation、GQA 专用实现。后续从 vLLM、flash-attn 里补。

## 来源

`references/reps/GEAK-eval` → `geak_eval/data/TritonBench/data/TritonBench_G_v1`，commit `453b021`，Apache-2.0。每个文件头保留上游版权声明。
