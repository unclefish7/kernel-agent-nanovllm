# ModelPort：给新模型自动适配 nano-vLLM

- 版本：v3.0，2026-11-13
- 状态：① 参考图已完成；② 引擎规约已写；③ 未开始
- 读者：项目作者 + 写 `models/<model>.py` 的 agent
- v2.0 里的 contract 一步已删除，见 §3 末尾。

---

## 0. 一句话

> 给一个 HuggingFace 模型目录，产出一份能在 nano-vLLM 里跑的 `models/<model>.py`。

产出只有那一份模型文件。其余都是递进去的材料。

---

## 1. 主线

```
HF 模型目录 ──① 抽图──▶ 参考图（它该算什么）──┐
                                              ├──③ agent ──▶ nanovllm/models/<model>.py
nano-vLLM 源码 ─② 读规约─▶ 引擎规约（引擎要求什么）┘
                              ▲
                     qwen3.py 当同构模板
```

| 步骤 | 做什么 | 产物 | 状态 |
| --- | --- | --- | --- |
| ① | 跑一遍 HF 模型，记录它实际做了哪些计算 | `nanovllm_selector/outputs/<model>/graph.json` | 完成（Qwen3） |
| ② | 读 nano-VLLM 源码，记录它对模型提了哪些要求 | `contracts/nanovllm/spec.yaml` | 完成 |
| ③ | 把两份材料和模板一起交给 agent | `nanovllm/models/<model>.py` | 待做 |

**没有 contract 这一步。** 哪个节点用哪个实现、哪几个节点合并，由 agent 现场决定。

---

## 2. ① 参考图（已完成）

**做什么**：加载 HF 模型，跑一次真实前向，记录这一次前向里执行了哪些算子、谁把结果喂给谁。

**输入**：模型目录，只需要 `config.json`，不加载权重。

**输出**：一份 JSON + 一份给人看的图（终端文本 / Mermaid / SVG / HTML）。

每个节点带：

| 字段 | 内容 |
| --- | --- |
| `op` | 语义判定（matmul / rmsnorm / attention / silu …），由 aten 组合 + 规则得出 |
| `inputs[]` | 每个输入的形状、来源、`arg_name`、`path`（`args[0]` / `kwargs['x']`） |
| `outputs[]` | 输出形状与去向 |
| `state[]` | 权重 / buffer / 运行时张量 / 标量 |
| `impl.source` | HF 里的类与 `file:line` + sha256 |
| `evidence[]` | 每条结论的证据（hook 实测 / dispatch / 参数形状 / 源码） |

**它回答**：这个模型需要哪些算子、每个算子吃什么吐什么、输入摆在第几个参数位。

**它不回答**：这些算子在 nano-VLLM 里该怎么实现。

**Qwen3-0.6B 实测**：22 个模板节点（28 层折叠）、28 条边、展开 453 个实例节点；10 类算子——matmul 196、rmsnorm 113、add 56、silu 28、mul 28、attention 28、embedding 1、lm_head 1、rope 1、arange 1。

**用法**：

```bash
python -m nanovllm_selector.extract.cli --model <模型目录> --outdir <输出目录>
```

---

## 3. ② 引擎规约

**做什么**：读 nano-VLLM 的代码，把"这个引擎对模型提了哪些要求"写成一份文件。

**产物**：`contracts/nanovllm/spec.yaml`，三段，互不索引。

### must：必须成立的条件

五条。写的是**条件**，不是动作——不规定用哪个类，只规定什么必须为真。

| # | 条件 |
| --- | --- |
| M1 | `model(input_ids, positions)` 进、`compute_logits(hidden)` 出；没有 mask / past_kv / use_cache |
| M2 | 跨 step 的 K/V 只能存在于引擎分配的显存里，位置由 `slot_mapping` 给、读由 `block_tables` 给 |
| M3 | 一轮前向 = 整个 batch 拼起来的序列；边界只能从 `cu_seqlens_*` / `context_lens` 取 |
| M4 | 参数名对得上 checkpoint，或用 `packed_modules_mapping` 声明映射；要切分的参数自带 `weight_loader` |
| M5 | 除 KV cache 外不留跨 step 状态；prefill 与 decode 同一条前向 |

每条都带 `evidence`（`file:line`）。

### toolbox：引擎 SDK

十八个条目。作用是"帮你管理和使用引擎侧的资源"，其中只有少部分是纯算子。

| 类别 | 条目 | 给的是什么 |
| --- | --- | --- |
| 运行时状态 | `get_context` | 本轮批信息 |
| 显存资源 | `store_kvcache`、`Attention` | KV cache 怎么读写 |
| 共享只读 | `get_rope`、`RotaryEmbedding` | 跨层共享的旋转表 |
| 带约定的组件 | `RMSNorm`、`SiluAndMul`、`MergedColumnParallelLinear`、`QKVParallelLinear`、`Column/RowParallelLinear`、`VocabParallelEmbedding`、`ParallelLMHead` | 打包格式、权重切分、残差时序 |
| 纯算子 | `apply_rotary_emb`、`ReplicatedLinear`、`Sampler` | 只有数学 |

每个条目给：签名、`file:line`、**它替你处理了什么**、以及坑（如 `SiluAndMul` 前半必须是 gate）。

**算子本身怎么写，这一段不回答。** toolbox 只说引擎里有什么、它替你处理了什么；某个算子具体长什么样，去 `kernels/` 翻（见 §4）。

### context：前向时能读到什么

源码里看不出来的三样——shape、什么时候有值、字段到底什么意思——列成一张 8 行表：

| 字段 | 类型 / shape | 何时有值 |
| --- | --- | --- |
| `is_prefill` | bool | 总是 |
| `cu_seqlens_q` / `cu_seqlens_k` | int32[N+1] | 仅 prefill |
| `max_seqlen_q` / `max_seqlen_k` | int | 仅 prefill |
| `slot_mapping` | int64[T] | prefill / decode 都有 |
| `context_lens` | int32[N] | 仅 decode |
| `block_tables` | int32[N, 块数] | decode 必有 |

另给 `access`（怎么取）和 `built_by`（谁写的，`model_runner.py:138` / `:181`）两行。不写怎么用——`qwen3.py` 已经演示过。

### 为什么没有 contract

contract 要做的是"把 must 按 when 条件挂到参考图的节点上"。但"哪条规则命中哪个节点"本身是语义判断，人工做不动、常规自动化也做不准；交给 agent 正是它该干的活。手工维护一张决定表，模型一多就废了。

---

## 4. ③ agent 生成

**输入四样**

| 材料 | 用途 |
| --- | --- |
| 参考图 JSON | 模型算了什么、什么顺序、什么形状 |
| `contracts/nanovllm/spec.yaml` | 引擎要求什么、能用什么 |
| `kernels/` | 算子写法参考：按算子分目录的现成实现，只读不调 |
| `nanovllm/models/qwen3.py` | 同构模板：类怎么分、Context 怎么用、权重映射怎么写 |
| `nanovllm/layers/*.py` | 需要细节时按需读 |

**`kernels/` 怎么用**：目录名就是算子名，`kernels/index.md` 一行一个文件。agent 先扫索引，需要哪份读哪份；库里没有的，自己写。库里只有代码，没有接口约定——写出来的东西和它像不像都无所谓。

**输出一份**：`nanovllm/models/<model>.py`，与 `qwen3.py` 同构。

**agent 自己决定**

1. 哪些节点合并（如 `rmsnorm + add` 合成 `add_rms_forward`，`gate/up/act/mul` 合成 `gate_up_proj + SiluAndMul`，`q/k/v` 合成 `qkv_proj`）
2. 每个节点用哪个模块（toolbox 里的，或自己写的）
3. 权重名怎么映射

**不做的事**：不要求这一版写 Triton、不要求 TP 正确、不要求性能。这一版只要能在单卡上跑出与 HF 一致的 token。

**落地时还要改一行引擎**：`engine/model_runner.py:10` 目前写死 `from nanovllm.models.qwen3 import Qwen3ForCausalLM`，`:40` 写死实例化。需要按 `config.model_type` 查表。这是引擎侧改动，在子模块里单独提交。

---

## 5. 当前状态与下一步

| 环节 | 状态 |
| --- | --- |
| ① 参考图 | 完成。`nanovllm_selector/extract/`，19 项单测，Qwen3 对账 16/16 |
| ② 引擎规约 | 完成。`contracts/nanovllm/spec.yaml` |
| ③ agent 生成 | 未开始 |

**下一步**：准备 Llama-3.2-1B 的参考图，把四样材料递给 agent，让它写第一版 `models/llama.py`。

Llama 与 Qwen3 的三处硬差异（图上看不出来，写代码时会撞上）：

1. 没有 `q_norm` / `k_norm`——`qwen3.py:68` 用 `if not self.qkv_bias` 判断，照抄会凭空多出两个参数
2. RoPE 是 llama3 型缩放（`factor` / `high_freq_factor` / `low_freq_factor`），`qwen3.py:54` 只读了 `rope_theta`
3. `o_proj` / `down_proj` 用行并行，`gate_up_proj` 用列并行

---

## 6. 术语

| 术语 | 含义 |
| --- | --- |
| 参考图 | 从 HF 模型抽出的计算图，描述"这个模型该算什么" |
| 引擎规约 | nano-vLLM 对模型的要求，描述"这个引擎要求怎么算" |
| must | 规约里必须成立的条件 |
| toolbox | 规约里引擎提供的 SDK，可用可不用 |
| context | 一次前向里引擎通过全局 `Context` 提供的批信息 |
| 模板节点 | 28 层折叠后的一层代表，带 `instances` 列表 |
| `packed_modules_mapping` | checkpoint 权重名到引擎参数名的映射（打包的 gate/up、qkv） |
