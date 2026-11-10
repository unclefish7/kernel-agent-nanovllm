# ModelPort：新模型的自动移植方案

- 版本：v1.1，2026-11-10
- 状态：设计定稿，按 M1–M5 分阶段实现。N0（参考图抽取）已完成并有可运行代码。
- 读者：项目作者（理解全貌与取舍）+ 编码 agent（在明确边界内实现）
- 前置阅读：`docs/graph-extraction-plan.md`（抽图模块的实现细节与踩坑记录）
- 被本文取代的旧文档：`docs/deprecated-kernel-selector-plan.md`（已弃用）
- v1.1 修订依据：2026-11-10 与导师的讨论（6 条结论），逐条对照见附录 B

---

## 0. 一句话

> 给一个 HuggingFace 模型目录和一个目标推理引擎，自动产出一张带证据的参考图、
> 一份融合与实现计划、一个能在该引擎里跑起来并与参考实现数值一致的模型文件。

三件事不能省：**图**（该算什么）、**融合**（哪些算子该并成一个）、**实现**（每个算子用什么写）。
挑 kernel 只是"实现"这一步在多个候选并存时的排序策略。

---

## 1. 名字与定位

### 1.1 为什么不再叫 kernel selector

| 名字隐含的假设 | 实际情况 |
| --- | --- |
| 模型已经能在目标引擎里跑 | 往往一行都没有，要先造出来 |
| 工作是"在若干实现里挑一个" | 工作是"给每个节点造一个实现，并让它们拼得起来" |
| 优化目标是性能 | 当前阶段的验收标准里**没有性能**，只有正确性 |
| 引擎是固定的 | 引擎是可替换的目标端，nano-vLLM 只是第一个 |

导师的原话可以这样拆：**"选算子"和"适配"是同一个动作**——适配就是给图里每个节点选一个
实现再拼起来。所以 selector 不在适配之后，它就在适配里面。

### 1.2 新名字与三个核心对象

**ModelPort**（模型移植）。动词用法：port a model to an engine。

```
reference graph   参考图：这个模型该算什么（HF/transformers 侧抽出来）
target contract   目标契约：目标引擎要求怎么算（接口、上下文、布局、切分）
plan              计划：把参考图的节点分组（融合），每组选一个实现
```

v1.0 里第三项叫 `binding`，v1.1 改名为 `plan`，因为它现在**同时包含分组与选择**两件事。

### 1.3 节点 ≠ kernel（v1.1 新增）

导师原话："不是计算图上一个节点就用一个 kernel"。

计算图上的节点是**语义单元**：`matmul`、`rmsnorm`、`silu`。它不是 kernel 的边界。
一个 kernel 可以覆盖多个节点（`matmul + silu + mul` → 一个 SwiGLU kernel），
一个节点也可以被拆到多个 kernel 里（一个 attention 节点在实现里是 qkv 投影 + rope + sdpa）。

所以流水线里必须有一步**显式决定"怎么分组"**，而不是默认一节点一实现。
这一步在 v1.0 里没有，是 v1.1 最大的结构改动。

### 1.4 三个角色

| 角色 | 做什么 | 是否用 LLM |
| --- | --- | --- |
| 抽图器 `extract/` | 跑一次真实前向，产出一张带证据的图 | 否，确定性 |
| 语义标注 | 给节点定 `op` / `formula` / 输入绑定 | 规则优先，LLM 只做兜底且必须带证据 |
| 规划器 `port/` | 分组（融合）+ 选实现 + 生成代码 | **是，这是 agent 的正式位置** |

---

## 2. 问题定义

### 2.1 输入输出

**输入**

```python
port(
    model_path="/path/to/Llama-3.2-1B",   # HF 格式目录（config.json + 权重）
    target="nanovllm",                     # 目标引擎
    runtime_options={"tp_size": 1, "dtype": "bfloat16"},
    output_dir="nanovllm/models/",
)
```

**输出**

1. `reference.json`：参考图（v1.0 已有）
2. `plan.json`：节点分组 + 每组的实现选择 + 依据
3. `nanovllm/models/llama.py`：可直接 import 的模型实现（约 5–6 个类）
4. `nanovllm/engine/model_runner.py` 的一行改动建议（`ARCH` 注册）
5. `report.md`：逐节点数值对账、未覆盖清单、首个不一致的节点

### 2.2 什么叫"移植成功"

三项正确性，**全部在目标引擎内验证**，任何一项不过就不算完成：

| 级别 | 问题 | 怎么验 | 在哪跑 |
| --- | --- | --- | --- |
| **V1 节点级** | 每个节点算的是不是同一个函数 | 参考侧（HF）与目标侧（生成的模型）各挂一次 hook，跑同一 prompt，逐节点比对输出 | 两侧各一次前向 |
| **V2 接口级** | 是否满足引擎契约 | 能构造、能 prefill、能 decode，不抛异常 | 引擎内 |
| **V3 端到端** | 适配是否合理 | 同一 prompt 贪心解码 8 步，top-1 token 全一致 | 引擎内 |

V1 的比较方式是**逐节点对齐 + 首错定位**：不注入中间值，而是按拓扑序比较两侧对应节点的输出，
**第一个不一致的节点就是 bug 的位置**。融合节点做例外处理——被融合的那几个参考节点没有独立输出，
只比较融合后的输出。

**本阶段不要求快。** 性能对比放在 M5，且必须在三项正确性全绿之后才有意义。

### 2.3 为什么不在引擎外验证（v1.1 新增）

这是被明确质疑过的一点，单独写清楚。

把一个生成的模型**脱离引擎单独跑**，能验证什么、不能验证什么：

| 能验证 | 不能验证 |
| --- | --- |
| 代码语法与 import | 是否满足引擎的 Context 约定（`cu_seqlens` / `slot_mapping` / `context_lens`） |
| 数学语义（这一段算的是不是同一个函数） | KV cache 的布局与读写时序 |
| 不依赖引擎那部分张量形状 | 权重名与 `packed_modules_mapping` 是否对得上 |
| | CUDA graph 捕获期是否合法、TP 切分是否正确 |

而"适配"出问题，绝大多数出在右边那一列。所以：

> **脱离引擎运行不是验收项，只是调试开关。**
> 它回答"这段数学对不对"，不回答"适配对不对"。
> 验收只看 V1–V3，全部在引擎内完成。

因此本方案里唯一的"引擎外运行"是**参考侧**——用 HF 跑同一 prompt 取对照值。
那不是"离开引擎验证适配"，那是必需的对照组。

### 2.4 第一版的简化边界

导师的意见："先减小难度，你可以不管 kv cache 管理、tp 这些，只要单卡跑出来就行，后面再优化。"

落到设计上：

| 项 | 第一版 | 后续 |
| --- | --- | --- |
| 张量并行 | 只支持 `tp_size=1`（契约预留字段，代码里断言） | 多卡、all-reduce |
| KV cache | **复用框架现成的 `Attention` 模块**，不自研 paged 实现 | agent 自己写 |
| prefill / decode | 都要能跑；图只抽 prefill | 分别抽图 |
| 融合实现 | 用 torch 组合（`F.silu(x1) * x2` 这种） | 换成 triton kernel |
| 性能 | 不验收 | 才引入 benchmark |

**"复用框架现成的模块"是一个有意的取舍**：第一版证明"图 → 计划 → 代码"这条链是通的，
而不是证明"我们能写一个 paged attention"。后者放到 M5。

### 2.5 明确不做的事

- 不做量化、不做投机解码、不做 MoE
- 不追求一次支持所有模型：先 Llama-3.2-1B，再 Phi-3-mini，再 Gemma-2-2B
- 不做运行时装配（见 R10 的切换条件）

---

## 3. 现状盘点

### 3.1 已完成：N0 参考图抽取

代码在 `nanovllm_selector/extract/`，入口：

```bash
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --check
```

对 Qwen3-0.6B（CPU / fp32 / 8 token prefill）的实测结果：

| 指标 | 数值 |
| --- | --- |
| 模块对象 / 命名参数 | 427 / 310 |
| 模块调用（一次前向） | 426 |
| aten 算子 | 2412 |
| 实例节点 / 边 | 453 / 620 |
| 模板节点 | 22（层内 16 + 层外 6） |
| 对账 | 16 / 16 通过 |

产物：`graph.json`（机器读）、`graph.md` / `graph.mmd` / `graph.dot` / `graph.svg` / `graph.html`（人读）。

**关键性质**：整张图来自**一次真实 forward 的观测**，不解析源码、不依赖类名。
每个节点带 `evidence`（hook 实测 / dispatch 记录 / 参数形状 / 源码行 + sha256）。

### 3.2 缺什么

| 环节 | 状态 | 缺的具体东西 |
| --- | --- | --- |
| 图的信息完整性 | ⚠️ 有缺陷 | **输入的顺序与形参名没记**（见 6.1） |
| 参考图 | ✅ 有（Qwen3） | 换模型要验证泛化（Llama） |
| 融合决策 | ❌ 没有 | 分组规则 + 合法性校验 |
| 目标契约 | ❌ 没有 | 10 种 op 的 `requires` |
| 实现目录 | ⚠️ 只有 swiglu | 覆盖图里每一种 op，且可检索 |
| 规划器 | ❌ 没有 | 图 + 契约 + 目录 → `plan.json` |
| 代码生成器 | ❌ 没有 | `plan.json` → `llama.py` |
| 验证 | ⚠️ 只覆盖 MLP 激活 | 节点级 + 接口级 + 端到端 |

---

## 4. 两条核心洞察

### 4.1 节点划分随框架变

同一层 Qwen3，两个框架的**节点数都不一样**：

| 参考图（HF） | 目标图（nano-vLLM） | 关系 |
| --- | --- | --- |
| `rmsnorm` + `add`（2 个节点） | `add_rms_forward`（1 个节点） | 2 → 1 融合 |
| `q_proj` / `k_proj` / `v_proj` | `qkv_proj` | 3 → 1 融合 |
| `gate_proj` / `up_proj` | `gate_up_proj` | 2 → 1 融合 |
| `silu` + `mul` | `SiluAndMul` | 2 → 1 融合 |
| `sdpa` + `mask` | `flash_attn_varlen` | 1 → 1 替换 |
| `cache.update` | `store_kvcache` | 1 → 1 替换 |
| `nn.Linear` | `Column/Row/QKVParallelLinear` | 1 → 1 替换（带通信） |

由此得到三条硬约束，写代码时必须遵守：

1. **两边都必须有图。** 只有参考图，绑定就是靠人读代码猜。
2. **计划不是一对一。** `plan.groups` 必须允许一组吃掉参考图里的多个节点；
   反之也要允许一个参考节点落到多个实现里（M1 的 attention 就是合成节点）。
3. **对齐不能按节点逐个匹配，必须做序列对齐 + 合并检测。**

### 4.2 节点是语义单元，不是 kernel 边界

4.1 说的是"两边节点数可能不同"，4.2 说的是"同一张图内也该主动改变分组"。

融合的收益是省掉中间张量的往返。举例（Qwen3 的一层）：

```
gate_proj ─┐
           ├─ silu ─┐
           │        ├─ mul ─┐
up_proj ───┘        │       ├─ down_proj
                    └───────┘
```

不融合：4 次 kernel 启动，4 次显存往返（gate 输出、up 输出、silu 输出、mul 输出）。
融合成 SwiGLU：一次投影出 `[N, 6144]`，一次 kernel 内完成 `silu(x[:3072]) * x[3072:]`，
中间张量只剩一个。

所以要支持的三类融合（按优先级）：

| 模式 | 参考图节点 | 结果 |
| --- | --- | --- |
| `matmul + silu + mul` | gate/up 投影 + 激活 + 乘 | SwiGLU（导师点名的第一个） |
| `rmsnorm + add` | 归一化 + 残差 | `add_rms`（双入双出） |
| `matmul × 3` | q/k/v 投影 | `qkv_proj`（一个矩阵乘 + 分段切） |

**融合是"决定"，不是"发现"**：不是图里恰好有这段就连上，而是规划器主动提出"这三个可以并"，
再由确定性检查器验证合法性（语义等价、形状匹配、不违反目标契约）。

---

## 5. 流水线

```
HF 模型 ──①extract──▶ 参考图 ──②fuse──▶ 分组过的图 ──③plan──▶ plan.json ──④emit──▶ llama.py
                                                                               │
                                                                               ▼
                                              HF 参考实现 ──⑤verify──▶ report.md
```

| 步 | 做什么 | 输入 | 输出 | 判据 |
| --- | --- | --- | --- | --- |
| ① | 抽参考图 | HF 模型目录 | `reference.json` | 对账全过 |
| ② | 融合决策 | 参考图 + 契约 + 目录 | 分组 | 每组语义等价、形状可匹配 |
| ③ | 实现选择 | 分组 + 目录 | `plan.json` | 无未覆盖组 |
| ④ | 生成代码 | plan + config | `llama.py` | `py_compile` + import |
| ⑤ | 验证 | 生成代码 + HF | `report.md` | V1–V3 全过 |

### 5.1 diff 的定位调整（v1.1）

v1.0 把"抽目标图 + diff"放在第一步（N1）。v1.1 把它降为**辅助步骤**，理由：

- 第一版复用框架现成模块、`tp_size=1`、不管 cache，目标侧长什么样，`qwen3.py` **已经用代码给出了答案**；
- 抽目标图的价值变成"自动发现框架里这类节点通常怎么实现"，是**建目录的素材**，不是必经环节；
- 把它放在第一步会让 M1 卡在技术障碍上（`dist.get_rank()`、全局 `Context`、Triton kernel 需要 GPU），
  而这些障碍对第一版要证明的东西没有贡献。

它没有被删掉：放在 M3 建目录时使用，同时也作为"我们对框架的理解是否正确"的交叉验证。

---

## 6. 数据契约

### 6.1 图必须携带的信息：输入顺序与形参名（v1.1 新增）

导师原话："不是只给图啊……图有些信息没有的……比如算子输入的顺序。"

**这是真缺口，现有实现确实丢了。** 两处代码：

```python
# nanovllm_selector/extract/recorder.py:46
def flatten_tensors(obj, out=None):
    # 把 forward 的输入/输出（可能嵌套 list/tuple/dict）拍平成一个 list
    # —— 位置信息在这里丢失
```

```python
# nanovllm_selector/extract/schema.py:52
@dataclass
class TensorRef:
    tensor: str
    role: str | None = None   # 只有 "in" / "out"
```

后果是具体的：

| 节点 | 丢了什么 | 会导致什么 |
| --- | --- | --- |
| `L*.mlp.mul` | 分不出哪路来自 `silu(gate)`、哪路来自 `up` | 换 `SiluAndMul` 时 chunk 顺序错——前 3072 是 gate、后 3072 是 up，数值静默错 |
| `add_rms(x, residual)` | 分不出两个入参谁先谁后 | 写反了数值错，但不报错 |
| `L*.self_attn` | q / k / v 的顺序 | 三者形状可能相同，静默错 |

**修法：三层信息，缺一不可。**

1. **位置**（`path`）：`args[0]`、`args[2][1]`、`kwargs["hidden_states"]`
2. **名称**（`arg_name`）：从 `inspect.signature(module.forward)` 取；合成节点从 aten schema 取
   （`aten.mul.Tensor(Tensor self, Tensor other)` → `self` / `other`）
3. **绑定**（`semantics.bindings`）：公式变量 → 输入，例如

```yaml
op: mul
formula: "y = silu(a) * b"
bindings:
  a: "in[0]"     # 来自 L*.mlp.act_fn 的输出
  b: "in[1]"     # 来自 L*.mlp.up_proj 的输出
```

扩展后的结构：

```python
@dataclass
class TensorRef:
    tensor: str
    role: str | None = None       # in / out
    path: str | None = None       # args[0] / kwargs['hidden_states']
    arg_name: str | None = None   # x / hidden_states / residual / self / other
    index: int | None = None      # 摊平后的序号（用于稳定排序）
```

**验收**：图里每个语义节点的每个输入都有 `path` 与 `arg_name`；
`mlp.mul` 能回答"哪个输入来自 gate 分支、哪个来自 up 分支"；`add_rms` 能回答参数顺序。

> 注意：aten 层本身就保序（`Dispatch.inputs` 是 list），丢序只发生在"语义节点"这一层。
> 所以修复量很小——在 `decompose.py` 建节点时把位置和形参名带上即可。

### 6.2 目标契约（Target Contract）

一个算子（或一组被融合的算子）在目标引擎里的完整要求。YAML 便于人写，运行时转成 dict。

```yaml
# contracts/nanovllm/add_rms.yaml
op: add_rms
family: normalization
engine: nanovllm
impl:
  module: nanovllm.layers.layernorm
  entry: RMSNorm.add_rms_forward
  source_sha256: "<自动填充>"
covers: [rmsnorm, residual_add]      # 吃掉参考图里的哪几种 op
inputs:
  - {name: x,        layout: "[N, hidden]", dtype: activation}
  - {name: residual, layout: "[N, hidden]", dtype: activation}
outputs:
  - {name: x,        layout: "[N, hidden]"}
  - {name: residual, layout: "[N, hidden]"}
state:
  - {name: weight, kind: parameter, shape: "[hidden]", dtype: weight}
context: []                          # 不依赖引擎上下文
constraints:
  contiguous: true
  inplace: false
notes: 双入双出；调用方负责把上一个 add 的结果作为 residual 传入
```

`inputs[].name` 必须与图里节点的 `arg_name` 使用**同一套命名**，否则 6.1 修了也白修。

`context` 字段是引擎差异的落点：

| op | 需要的 context |
| --- | --- |
| `matmul`（列并行） | `tp_size`, `tp_rank`（第一版恒为 1，不实际使用） |
| `attention`（prefill） | `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q/k` |
| `kv_write` | `slot_mapping`, `k_cache`, `v_cache` |
| `attention`（decode） | `context_lens`, `block_tables` |
| `embedding`（词表并行） | `tp_size`, `tp_rank` |

### 6.3 计划（Plan）

```json
{
  "schema_version": "plan.v1",
  "model": "llama-3.2-1b",
  "engine": "nanovllm",
  "reference_graph_sha256": "...",
  "groups": [
    {
      "group_id": "G07",
      "nodes": ["L*.input_layernorm", "L*.residual_add@0"],
      "op": "add_rms",
      "impl": "nanovllm.layers.layernorm.RMSNorm.add_rms_forward",
      "matched_by": "rule:Q3-RMS-1",
      "evidence": ["reference#L*.input_layernorm", "source:layernorm.py:34"]
    }
  ],
  "unbound": [],
  "conflicts": []
}
```

两个字段是硬要求：

- `nodes` 是**列表**，允许一个组覆盖多个参考节点（这是融合的落地方式）
- `matched_by` 必须写清是**规则命中**还是**人工指定**——这是"不能只基于名称"的落地

### 6.4 实现目录（kernels/）

导师的意见：现成算子放到文件夹里，让 agent 按需去查；没有的由 agent 自己写。

```
kernels/
  silu_mul/
    manifest.json        # 检索用：op、签名、形状、dtype、context、来源、sha256
    impl_torch.py        # 参考实现（一定存在，作为兜底）
    impl_triton.py       # 可选
    test_impl.py         # 数值自检
  add_rms/
  qkv_proj/
```

`manifest.json` 的字段与 6.2 的契约对齐，其中 `inputs[].name` 用同一套命名。
agent 的检索方式：按 `op` 与输入形状筛出候选，**只把命中的 manifest 放进上下文**，
`impl_*.py` 按需读——这正是"算子太多，不要全塞进去"的落地。

### 6.5 生成代码的形态

生成物是一个**普通 Python 文件**，和手写的 `qwen3.py` 同构：

```python
# ===== 生成信息（勿手工修改，重跑会覆盖）=====
# generator: modelport 0.1.0
# reference_graph: sha256:...
# plan:            sha256:...
# contracts:       nanovllm@v1 (10 ops)
# generated_at:    2026-11-20T10:00:00+08:00
# =============================================

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()
    ...

class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
    }
```

三条硬要求：

1. **带 provenance 头**，能追溯到哪张图、哪份计划、哪版契约。
2. **第一版只生成骨架与连接**，算子体调用框架已有实现或 kernels/ 里的实现——不内联 kernel 代码。
3. **幂等**：同样的输入生成字节级相同的文件（时间戳可通过开关关闭）。

---

## 7. 阶段计划

### M1：图补全 + 直译版（1–1.5 周）

**做什么**

1. ~~修 6.1：`schema.py` / `recorder.py` / `decompose.py` 补 `path` / `arg_name` / `index` 与 `bindings`。~~
   **已完成（2026-11-10）**：`TensorRef` 增加 `index` / `path` / `arg_name`；
   `recorder.py` 记模块调用的参数位（含 storage 回退）与 aten 算子的参数树位置；
   `semantics.bindings` 输出公式变量→输入；Qwen3 上 28 个模板输入全部有 `path` 与 `arg_name`，
   单测 19 项通过。
2. 写 `port/emit.py` v0：按图**一比一**生成 `llama.py`（不融合），算子体全部调用框架现成模块。
3. 在 nano-vLLM 里注册 `ARCH`，能构造、能 prefill、能 decode。

**为什么先直译**：如果直译都跑不对，后面融合出错就分不清是"图错了"还是"融合规则错了"。
直译是给融合准备一个已知正确的基线。

**注意**：直译版**仍然在引擎里跑**，验收走 V1–V3。它不是"脱离引擎的独立实现"（见 2.3）。

**验收**：V1 节点级全对齐（此时 1:1，可完全对齐）；V2 不崩；V3 8 步 token 一致。

### M2：融合决策（1–1.5 周）

**做什么**

1. 写 `port/fuse.py`：在参考图上按 4.2 的三类模式提出分组，再由确定性检查器验合法性
   （语义等价、形状匹配、不破坏数据流）。
2. 融合后的实现先用 **torch 组合**，例如 `F.silu(x[..., :3072]) * x[..., 3072:]`。
3. 与 M1 的直译版对比：逐节点、逐 token。

**产出**：`plan.json` 里有 `groups`，生成代码里出现融合后的模块。

**验收**：与 M1 输出逐 token 一致；中间张量数量下降（可用图对比证明）。

### M3：契约 + 实现目录（1–1.5 周）

**做什么**

1. 写 10 份契约：`embedding` / `matmul` / `rmsnorm` / `add` / `mul` / `silu` / `rope` /
   `attention` / `kv_write` / `lm_head`。
2. 建 `kernels/` 目录 v1：每种 op 至少 1 个实现（框架自带 / vendored / 自己写），带 manifest。
3. 写**覆盖率检查器**：拿参考图的 22 个模板节点跑一遍，输出未覆盖清单。
4. （辅助）抽目标图 `nanovllm.models.qwen3`，与参考图 diff，校验"我们对框架的理解"是否正确。

**验收**：Qwen3 参考图 22 个模板节点 100% 有候选；diff 结论与人工分析一致。

### M4：agent 驱动（1.5 周）

**做什么**

1. 规划器接 LLM：输入是**图的一部分 + 候选 manifest**（不是全部），输出是 `plan.json` 的增量。
2. 检索：按 op + 形状筛候选，只把 top-k manifest 放进上下文。
3. 全流程串起来：一个命令从 HF 目录到 `llama.py` + `report.md`。

**验收**：Qwen3 与 **Llama-3.2-1B** 两个模型都能过 V1–V3。这是第一个可交付版本。

### M5：新算子生成与性能（开放）

1. 目录里没有合适实现时，agent 自己写（torch 或 triton），写完自动跑 `test_impl.py`。
2. 同一节点 ≥2 个候选时，按硬件与形状选；引入 benchmark 与性能闸门 G3。
3. 旧 demo 的 `geak_packed_swiglu` 在这个阶段作为候选之一回归。

---

## 8. 验收闸门

| 闸门 | 判据 | 不过的后果 |
| --- | --- | --- |
| **G0 接口** | 生成文件 `py_compile` 通过、模型能构造、forward 不抛异常 | 跑不起来 |
| **G1 逐点** | 每个可对齐节点的输出误差 < 1e-4（fp32）/ < 1e-2（bf16） | 静默错误 |
| **G2 端到端** | 8 步 top-1 token 全一致 | 不可用 |
| **G3 性能** | （M5 才启用）不劣于手写实现 | 无意义 |

**闭环定义**：一条命令跑完 G0–G2，失败时报告**第一个不一致的节点**与该节点涉及的契约字段。

---

## 9. 目录与模块

```
nanovllm_selector/
  extract/                # 已有：抽图
    schema.py  source.py  loader.py  recorder.py
    decompose.py  edges.py  template.py  reconcile.py
    render.py  api.py  cli.py
    diff.py               # M3 新增（目标图对比，辅助）
  port/                   # 新增：移植
    contract.py           # M3 契约加载与校验
    catalog.py            # M3 实现目录与覆盖率
    fuse.py               # M2 融合决策
    plan.py               # M2/M3 分组 + 选择
    emit.py               # M1/M4 代码生成
    verify.py             # M1 验证（V1–V3）
    cli.py                # port 子命令
  kernels/                # 新增：可检索实现目录
    silu_mul/  add_rms/  qkv_proj/ ...
  test_extract.py         # 已有
  test_port.py            # 新增
docs/
  model-port-plan.md      # 本文
  graph-extraction-plan.md
```

CLI 设计：

```bash
# 抽参考图
python -m nanovllm_selector.extract.cli --model <hf_dir> --out references/qwen3.json
# 规划（融合 + 选实现）
python -m nanovllm_selector.port.cli plan references/qwen3.json --target nanovllm --out plan.json
# 生成
python -m nanovllm_selector.port.cli emit plan.json --out nanovllm/models/qwen3_generated.py
# 验证（引擎内）
python -m nanovllm_selector.port.cli verify --model <hf_dir> --generated <generated.py>
```

---

## 10. 风险与未决

| 编号 | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R1 | 图的信息不完整（顺序、形参名、副作用） | 生成的代码静默错 | 6.1 的三层信息；`effects` 字段后续补 |
| R2 | 融合判据不严 | 语义不等价却合并 | 融合只"提议"，合法性由确定性检查器兜住；M2 用 M1 做基线 |
| R3 | 目标侧图抽不出来（dist / Context / CUDA） | M3 卡住 | 已降级为辅助项，不阻塞主线 |
| R4 | 生成代码可读性差 | 无法调试 | 只生成骨架；算子体调用现成实现；保留 provenance |
| R5 | Llama 与 Qwen3 的结构差异 | 契约不够用 | 先列差异清单，每条差异对应一条契约字段 |
| R6 | 权重名对不上 | 加载失败或静默错位 | `packed_modules_mapping` 由计划推导，生成时校验 state_dict 键 |
| R7 | TP>1 | 第一版不支持 | 契约预留 `tp_size`，代码里 `assert tp_size == 1` |
| R8 | 版本漂移（torch / transformers / 框架） | 证据失效 | 所有产物记录 sha256 与版本，`env` 已在图里 |
| R9 | 生成代码与手写代码漂移 | 难以回归 | 以 HF 输出为唯一基准，不与他人手写实现比对 |
| R10 | 何时切回运行时装配 | 架构返工 | 见下 |
| R11 | 融合后的实现无人验证 | 错得隐蔽 | 融合版必须与 M1 直译版逐 token 对比 |
| R12 | 目录膨胀后检索不准 | agent 选错实现 | manifest 带形状/dtype/context，先精确过滤再语义检索 |

### R10：为什么先选"生成代码"

| | 生成代码 | 运行时装配 |
| --- | --- | --- |
| 产物 | `llama.py` | 通用 `GraphModel` + 执行计划 |
| 优点 | 可 diff、可调试、与框架原生同构 | 真正自动，加模型不改代码 |
| 缺点 | 每个模型生成一份文件 | 要自研执行器，还要过 CUDA graph |
| 适用 | **正确性验证阶段（现在）** | 模型数量多、需要规模化时 |

**决定：用生成代码。** 理由：它把"生成的实现是否正确"变成一次文件 diff 就能看出的事；
运行时装配会同时引入执行器本身是否正确这个新变量。

**切换条件**（写下来，避免以后反复讨论）：当同一引擎需要支持 ≥5 个模型，
且生成代码的重复率超过 70% 时，再评估运行时装配。

---

## 11. 与旧方案的关系

**保留**

- `extract/` 整个模块（探针、建边、折叠、对账、渲染）
- binder 指纹的思路（config sha / weights sha / dtype / gpu arch / 版本）
- 回退语义（候选不满足就退回原生实现）
- vendored 来源记录（`vendor/manifest.json` 的格式）
- 验收闸门的分级

**改写**

- 目标从"替换一个节点"变成"移植整个模型"
- 从"Target/Candidate/Plan 三张表"升级为"reference graph / target contract / plan"
- 候选库的指标从"候选数量"变成**覆盖率**（参考图节点能否 100% 找到实现）
- 流水线里新增**融合决策**（v1.1）

**废弃**

- 基于 `monkeypatch` + `nn.Module` 包装的安装方式（会改 `state_dict` 键、跨不过 `spawn` 子进程、与 CUDA graph 冲突）
- 只支持单一 `target == "mlp_activation"` 的校验逻辑
- 手工写在 `validate_plan` 里的环境绑定判断（改为由契约字段表达）

---

## 附录 A：术语

| 术语 | 含义 |
| --- | --- |
| 参考图 | 从 HF 模型抽出的计算图，描述"该算什么" |
| 目标图 | 从目标引擎已适配模型抽出的计算图，描述"引擎怎么算"（v1.1 降为辅助） |
| 模板节点 | 28 层折叠后的一层代表 + `instances` 列表 |
| 实例节点 | 展开后的每个具体调用 |
| 融合 | 把参考图的多个节点并成一个实现 |
| 计划（plan） | 分组 + 每组的实现选择 |
| 契约 | 目标引擎对一个算子/实现提出的接口要求 |
| 闸门 | G0–G3 四级验收 |

## 附录 B：v1.1 修订对照

| 导师结论 | v1.0 的状态 | v1.1 的改动 |
| --- | --- | --- |
| 节点 ≠ kernel | 隐含假设一对一 | 新增 1.3 / 4.2；流水线加融合步 |
| 让 agent 去合并 | 无此步骤 | 新增 M2、`port/fuse.py` |
| matmul + silu 会合并 | 只作为例子 | 列为第一优先融合模式 |
| 先不管 kv cache / tp | 已排除 tp，但仍计划自研 | 2.4 明确复用框架 `Attention`；tp 断言 |
| 现成算子放文件夹按需查 | 目录叫 catalog，未定形态 | 6.4 改成 `kernels/` + manifest + 检索 |
| 图缺输入顺序 | **确实缺** | 6.1 三层信息 + M1 第 1 项 |

阶段编号也从 N1–N6 调整为 M1–M5：原 N1（目标图 + diff）降为 M3 的辅助步骤。
