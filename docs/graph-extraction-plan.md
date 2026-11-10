# 从 reference design 抽取完整计算图 —— 工程实现规划

> 状态：v0.1（待评审）
> 适用对象：直接执行本任务的编码 agent
> 关联文档：[kernel-selector-plan.md](kernel-selector-plan.md)（本文件是它「① 参考图重建」这一步的细化实现）
> 本文所有数字分两类：标「实测」的是本机跑出来的；标「推导」的是由 config 算出来、尚未实测的。不要混淆。

---

## 0. 这份文档怎么用

1. 按第 9 章的里程碑顺序做，**不要跳步**。M1 不通过不要开始 M2。
2. 每一步都有【输入】【输出】【验收】三段。验收不通过就停下来修，不要「先跑通再说」。
3. 任何一条无法满足的硬约束，写进第 10 章风险表并回报，不要自己降低标准。
4. 代码只写在 `nanovllm_selector/extract/` 下（新增目录），**不改 `nano-vllm/` 子模块**，不改现有 `agent.py` / `candidates.py` / `run_selector.py`。

---

## 1. 目标与非目标

### 1.1 目标

输入一个 **HF transformers 里的模型**（本地目录），输出它的**完整计算图**：

- 节点 = 一次算子调用（不是模块对象）；
- 边 = 张量级数据依赖；
- 每个张量带 **实测** 的 shape / dtype / stride；
- 每个节点带 **语义标签** 和 **证据链**；
- 28 层折叠成「1 个模板 + instances 28」，人能看懂；
- 输出可序列化 JSON，可 diff、可查询、可单元测试。

### 1.2 非目标（本阶段明确不做）

- 不做「图 → nano-vLLM 实现」的映射（那是下一步「目标图对齐」）。
- 不做 kernel 选择。
- 不追求覆盖 transformers 全部 384 个架构；M2 只追加 Llama 系列验证泛化。
- 不在 GPU 上跑长任务；M1 允许全程 CPU + fp32 + 随机权重（见 §9）。

### 1.3 为什么不用「模块树」代替

模块树只表达「包含」，不表达「谁喂谁」；而且实测同一模型在两个框架里的模块数、参数数都不一样：

| 口径 | 模块数 | 命名参数数 | 来源 |
| --- | --- | --- | --- |
| HF `Qwen3ForCausalLM`（CPU fp32） | **427 实测** | **310 实测**（tied 权重被 dedupe） | 本文实验 |
| nano-vLLM `Qwen3ForCausalLM` | 371 | 227 | 历史实测 |

差 84 个参数来自「q/k/v 分开 vs 打包」以及 tied 权重的计数口径。**按名字对不上，只能按数据流对。**

---

## 2. 抽取策略：三条路线，实测后选 hook

### 2.1 实测结论（本机，torch 2.12.0+rocm10.0.0 / Python 3.12 / transformers 4.57.3）

| 路线 | 结果 | 细节 |
| --- | --- | --- |
| `torch.fx.symbolic_trace(model)` | **失败** | `ValueError: code: co_varnames is too small`（Python 3.12 的 code object 与 fx Proxy 不兼容，属 fx 自身缺陷） |
| `torch.export.export(model, (ids,))` | **失败** | `RuntimeError: Found <class 'transformers.cache_utils.DynamicCache'> in output` |
| `torch.export.export(model_cfg.use_cache=False, ...)` | **成功** | 产出 **2702** 个 aten 级节点 |
| 模块 forward hook | **成功** | 一次 8-token 前向，**426** 次模块调用 |
| `TorchDispatchMode` | **成功** | 一次 8-token 前向，**2664** 个 aten op，38 种 |

### 2.2 结论与理由

**主路线 = forward hook 记录模块边界 + TorchDispatchMode 补齐模块内部。** 理由：

1. **fx 在 Python 3.12 上直接不可用**，等上游修复不现实；而且要 trace 成功还得改 HF 源码（有控制流）。
2. **export 能成功但必须 `use_cache=False`**，改的是模型语义；且输出是 2702 个 aten 节点（`aten.view` 318 个、`_unsafe_view` 254 个），粒度过细，不能直接当「计算图」用。它的价值是**做旁证**：可以拿来交叉验证 matmul 数量。
3. **hook 给出正确的「模块边界」**，这是语义节点最自然的锚点；它的缺陷（漏掉父模块 forward 里的合成算子）恰好被 dispatch 补齐。
4. **dispatch 给出正确的「数据流」**，因为它在 aten 层看到每一个张量，不受视图、in-place 影响。

一句话：**hook 定边界，dispatch 定连接，规则定语义。**

### 2.3 不要做的事

- 不要尝试 `torch.compile(model, backend=...)` 抓图：MI100 上 `attention` 走 math 后端，且编译产物不可导出为可读图。
- 不要用 `torch.jit.trace`：它会在 `past_key_values` 分支上炸，且同样给 aten 级。
- 不要把 export 的 2702 个节点直接当交付物。

---

## 3. 实测数据（写代码前先记住这些数）

一次 `input_ids` 形状 `(1, 8)` 的前向，CPU fp32：

```
模块 hook 调用      426
按对象身份建边      254
再按 storage 匹配   +57  → 311
再按区间重叠        +0
未解析输入          86
未解析的消费者      o_proj×28, post_attention_layernorm×28, down_proj×28,
                    embed_tokens×1, rotary_emb×1
多消费者生产者      84（最大扇出 3）
dispatch op 总数    2664，唯一 38
aten.mm             197   ← 28 层 × 7（q,k,v,o,gate,up,down）+ lm_head = 197 ✓
aten.t              197   ← 每个 Linear 的 weight 转置
aten.view           318
aten._unsafe_view   254
aten.add.Tensor     225
aten.mul.Tensor     368
```

第 0 层的模块调用序列（**这是 hook 的完成序，父模块在子模块之后**）：

```
input_layernorm → q_proj → q_norm → k_proj → k_norm → v_proj → o_proj
→ self_attn → post_attention_layernorm → gate_proj → act_fn → up_proj
→ down_proj → mlp
```

**注意**：`self_attn` 排在 `o_proj` 之后，因为 forward hook 在模块 forward **返回后**才触发。要拿执行起点必须用 forward **pre**-hook。

这 86 条未解析的边，正是本项目最关心的几个算子：

| 未解析消费者 | 条数 | 真正生产者 | 位置 |
| --- | --- | --- | --- |
| `o_proj` | 28 | `scaled_dot_product_attention` | `Qwen3Attention.forward:187` 内部，非模块 |
| `post_attention_layernorm` | 28 | 残差 `residual + hidden_states` | `Qwen3DecoderLayer.forward:245` 内部，非模块 |
| `down_proj` | 28 | `act_fn(gate) * up` 的 `*` | `Qwen3MLP.forward:81` 内部，非模块 |
| `embed_tokens` | 1 | 图输入 `input_ids` | — |
| `rotary_emb` | 1 | 图输入 `position_ids` | — |

**这一条是整份规划的核心论据**：只挂模块 hook，会漏掉 86/397 ≈ 22% 的边，而且漏掉的正好是 swiglu 的 `mul`、残差 `add`、attention 核心——也就是未来要做融合和 kernel 选择的地方。

---

## 4. 数据模型

### 4.1 Python 侧（dataclass，放 `schema.py`）

```python
# nanovllm_selector/extract/schema.py
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "graph.v1"

@dataclass(frozen=True)
class TensorSpec:
    id: str                      # "t0001"，全局唯一
    shape: list[int | str]       # 动态维用 "N"
    dtype: str                   # "torch.bfloat16" / "torch.float32"
    device: str
    stride: list[int]
    contiguous: bool
    storage_id: str              # "s0007"，同一块 storage 的视图共享
    storage_offset: int
    numel: int

@dataclass
class Evidence:
    kind: Literal["observed", "source", "param", "dispatch", "loader", "rule"]
    ref: str                     # "modeling_qwen3.py:81" / "aten.mm" / "config.json:hidden_act"
    detail: str

@dataclass
class Node:
    id: str                      # "L0.mlp.act_fn@0"
    template_id: str             # "L*.mlp.act_fn@0"
    instances: list[int] | None  # None = 非重复节点；[0..27] = 模板节点
    op: str                      # 语义标签："silu" / "matmul" / "rmsnorm" / ...
    op_confidence: Literal["exact", "inferred", "unknown"]
    origin: Literal["module", "synthetic"]   # 来自模块调用 / dispatch 补出来的
    impl: dict                   # {"class": "SiLUActivation", "module_path": "...", "source": {...}}
    semantics: dict              # {"formula": "silu(x)", "reference": {...}}
    inputs: list[dict]           # [{"tensor": "t0001", "role": "x"}]
    outputs: list[dict]          # [{"tensor": "t0002", "role": "out"}]
    state: list[dict]            # 参数/buffer/标量，见 4.4
    attrs: dict                  # {"eps": 1e-06, "bias": false}
    phase: str                   # "prefill" / "decode" / "both"
    evidence: list[Evidence]

@dataclass
class Edge:
    from_node: str
    to_node: str
    tensor: str
    role: str                    # 消费者侧的入参名
    kind: Literal["data", "residual", "cache_write", "cache_read"]

@dataclass
class ParamSpec:
    name: str                    # "model.layers.0.mlp.gate_proj.weight"
    shape: list[int]
    dtype: str
    owner_node: str
    storage_group: str | None    # tied 权重共享时非空

@dataclass
class ExtractedGraph:
    schema_version: str
    model: dict                  # {"path","model_type","config_sha256","weights_sha256","num_layers"}
    env: dict                    # {"torch","transformers","python","device","dtype"}
    nodes: list[Node]
    tensors: dict[str, TensorSpec]
    params: list[ParamSpec]
    edges: list[Edge]
    reconcile: dict              # 对账结果，见 §8
```

### 4.2 JSON 输出

顶层：

```json
{
  "schema_version": "graph.v1",
  "model": {"path": "...", "model_type": "qwen3", "num_layers": 28,
            "config_sha256": "...", "weights_sha256": "..."},
  "env": {"torch": "2.12.0+rocm10.0.0", "transformers": "4.57.3",
          "python": "3.12", "device": "cpu", "dtype": "float32"},
  "phases": ["prefill"],
  "nodes": [ ... ],
  "tensors": { "t0001": { ... } },
  "params": [ ... ],
  "edges": [ ... ],
  "reconcile": { "checks": [ ... ], "ok": true }
}
```

节点样例（Qwen3 第 0 层 MLP 激活，真实字段）：

```json
{
  "id": "L0.mlp.act_fn@0",
  "template_id": "L*.mlp.act_fn@0",
  "instances": [0, 1, "...", 27],
  "op": "silu",
  "op_confidence": "exact",
  "origin": "module",
  "impl": {
    "class": "SiLUActivation",
    "module_path": "model.layers.0.mlp.act_fn",
    "source": {"file": ".../modeling_qwen3.py", "class_line": 71, "sha256": "..."}
  },
  "semantics": {
    "formula": "silu(x) = x * sigmoid(x)",
    "reference": {"symbol": "transformers.Qwen3MLP.forward",
                  "file": ".../modeling_qwen3.py", "line": 81,
                  "code": "down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))"}
  },
  "inputs":  [{"tensor": "t0012", "role": "x"}],
  "outputs": [{"tensor": "t0013", "role": "out"}],
  "state": [],
  "attrs": {},
  "phase": "prefill",
  "evidence": [
    {"kind": "observed", "ref": "hook", "detail": "input (1,8,3072) fp32 contiguous"},
    {"kind": "param",   "ref": "gate_proj.weight", "detail": "shape (3072,1024) → 输入末维 3072"},
    {"kind": "dispatch","ref": "aten.sigmoid + aten.mul", "detail": "module 窗口内 2 个语义 op"}
  ]
}
```

### 4.3 模板化规则

- 节点路径把 `model.layers.<i>.` 归一成 `model.layers.*.`，得到 `template_id`。
- 同一个 `template_id` 的所有节点必须**结构同构**：`op` 相同、`inputs` 的 role 顺序相同、`shape` 去掉层号后相同、`state` 的 key 集合相同。不同构 → 报错，不要静默合并。
- 重复节点的 `instances` 写 `[0..N-1]`；非重复节点 `instances` 为 `null`。
- 跨层边（第 i 层输出 → 第 i+1 层输入）折叠成模板自环边，`kind` 记 `"data"`。
- 首层特例（Qwen3 HF 没有；Llama 有 `first_layer` 分支，nano-vLLM 有 `residual is None` 分支）必须保留成**独立模板**，不能强行合并。

### 4.4 `state` 的四种 kind

沿用 [kernel-selector-plan.md](kernel-selector-plan.md) 的约定：

| kind | 判据 | 例子 |
| --- | --- | --- |
| `parameter` | 模块 `_parameters` 里有 | `q_norm.weight` |
| `buffer` | 模块 `_buffers` 里有 | `rotary_emb.inv_freq` |
| `runtime_tensor` | 普通属性且是 Tensor | `Attention.k_cache`（nano-vLLM） |
| `scalar` | 普通 Python 数值 | `rms_norm_eps = 1e-06` |

`state` 必须记录 `name / kind / shape / dtype / requires_grad`。这是后续 Op Spec 的直接输入。

---

## 5. API 设计

### 5.1 函数级 API

```python
# nanovllm_selector/extract/api.py
def extract_graph(
    model_path: str | Path,
    *,
    dtype: str = "float32",         # M1 用 float32 + CPU，避免 bf16 舍入影响对账
    device: str = "cpu",
    phases: tuple[str, ...] = ("prefill",),
    seq_len: int = 8,               # prefill 的 token 数
    trust_remote_code: bool = False,
    collapse_template: bool = True,
    label_mode: str = "rules",      # "rules" | "rules+llm"
    include_transparent_ops: bool = False,
    reconcile: bool = True,
    output_dir: str | Path | None = None,
) -> ExtractedGraph:
    """从一个 HF 模型目录抽出完整计算图。

    前置条件：
      - model_path 必须是本地目录，含 config.json 与权重（weights 可为随机初始化，
        只要 config 正确；M1 用随机初始化，不加载权重）。
      - 不联网。local_files_only=True 强制。

    副作用：只在 output_dir 下写文件（默认不写）。
    """
```

配套三个小 API：

```python
def extract_graph_from_module(model, config, **kw) -> ExtractedGraph   # 测试用，绕过加载
def load_graph(path) -> ExtractedGraph                                  # 读回 JSON
def diff_graphs(a: ExtractedGraph, b: ExtractedGraph) -> GraphDiff      # M3 用
```

### 5.2 `ExtractedGraph` 的方法

```python
g.to_dict() -> dict            # 可 json.dumps
g.to_json(indent=2) -> str
g.validate() -> None           # 抛 AssertionError，带节点 id
g.query(op=None, phase=None, template=None, has_state=None) -> list[Node]
g.stats() -> dict              # {"nodes": N, "edges": M, "by_op": {...}}
g.template() -> dict           # 只返回模板层（L0 实例化展开）
```

### 5.3 CLI

```bash
# 出图
python -B -m nanovllm_selector.extract.cli \
    --model tmp/models/Qwen3-0.6B \
    --out outputs/graph_qwen3_prefill.json \
    --pretty

# 出图并跑对账（失败返回码 1）
python -B -m nanovllm_selector.extract.cli \
    --model tmp/models/Qwen3-0.6B --check

# 人看的摘要（Markdown 表格，一行一个模板节点）
python -B -m nanovllm_selector.extract.cli \
    --model tmp/models/Qwen3-0.6B --summary
```

CLI 退出码：`0` 成功；`1` 对账失败；`2` 参数/环境错误。`--check` 失败时必须打印**具体哪条对账没过**，不要只打 `AssertionError`。

### 5.4 目录结构

```
nanovllm_selector/
  extract/
    __init__.py          # 导出 extract_graph / load_graph / diff_graphs
    schema.py            # §4 的 dataclass + JSON 校验
    loader.py            # 模型加载（随机权重 / 真实权重两种模式）
    recorder.py          # hook + dispatch 记录（§6.1）
    identity.py          # 张量身份与别名解析（§6.2）
    edges.py             # 建边与补全（§6.3）
    decompose.py         # 模块 → 算子（§6.4）
    template.py          # 折叠（§4.3）
    semantics.py         # 语义规则表（§7）
    source.py            # file:line + sha256（§6.6）
    reconcile.py         # 对账（§8）
    api.py
    cli.py
  tests/
    test_extract_schema.py
    test_extract_qwen3.py
```

新增测试放 `nanovllm_selector/tests/`，用 `python -B -m unittest discover -s nanovllm_selector/tests -v` 跑。

---

## 6. 实现步骤

### S1 加载器（`loader.py`）

【输入】`model_path`、`dtype`、`device`、`weights=False`。
【输出】`(model, config)`，模型 `eval()`、`requires_grad_(False)`。

两种模式：

- `weights=False`（默认，M1 用）：`AutoConfig.from_pretrained(path, local_files_only=True)` 后 `AutoModelForCausalLM.from_config(config)` 随机初始化。**不读权重文件**，秒级完成。
- `weights=True`：`AutoModelForCausalLM.from_pretrained(..., local_files_only=True)`。

【验收】打印 `modules / named_parameters` 数量；Qwen3 随机初始化应为 **427 / 310**（实测）。

### S2 记录器（`recorder.py`）

同时挂三类探针，共享一个单调递增的 `seq` 计数器：

1. **forward pre-hook**：记录 `(start_seq, module_path, class, inputs)`。这是**执行起点**。
2. **forward hook**：记录 `(end_seq, outputs)`。这是**执行终点**。
3. **`TorchDispatchMode`**（包住整个前向）：记录 `(seq, aten_name, input_ids, output_id, output_shape)`。

```python
def record(model, inputs, *, track_dispatch=True) -> Recording:
    ...  # Recording: calls[], dispatches[], input_ids[]
```

注意：

- hook 返回 `None`（不要改输出）。
- 必须用 try/finally 移除所有 handle。
- 记录期间持有所有张量引用（否则 `id()` 会被回收，见风险 R2）。
- `TorchDispatchMode` 在 CPU fp32 上开销可接受（2664 ops，秒级）；GPU 上不要开（会拖慢且可能干扰）。

【验收】对 Qwen3，`len(calls) == 426`、`sum(dispatches) == 2664`（实测值，允许 ±5% 浮动，版本差异时打印实际值并记录）。

### S3 张量身份与建边（`identity.py` + `edges.py`）

见 §6.2 与 §6.3。

【验收】Qwen3 一条 8-token 前向：身份边 254、别名边 +57、补全边 86、**未解析边 0**（除两个图输入）。

### S4 模块 → 算子（`decompose.py`）

见 §6.4。

【验收】节点里 `op == "matmul"` 的数量 == **197**；`op == "rmsnorm"` == **113**。

### S5 模板折叠（`template.py`）

【验收】`L*.` 前缀的模板节点数 == 14 左右（第 0 层实测 14 个模块 + 合成节点）；`instances` 长度都是 28；折叠前后总边数守恒。

### S6 语义标注（`semantics.py`）

【验收】`op_confidence == "unknown"` 的节点数为 0（M1 阶段 Qwen3 必须全命中）；每个节点至少 1 条 `evidence`。

### S7 源码对应（`source.py`）

【验收】每个 `origin=="module"` 的节点都有 `impl.source.file` 与 `impl.source.class_line`；每个文件有 sha256。

### S8 输出与 CLI

【验收】`--out` 生成的 JSON 能被 `json.load` 读回，`load_graph()` 能还原，`g.validate()` 通过。

---

## 6.2 建边算法

### 6.2.1 张量身份的三级匹配

按优先级从高到低，命中即停：

```python
# 级别 1：对象身份（最强）
key1 = id(tensor)                      # 实测命中 254 条

# 级别 2：storage 身份（视图，跨模块边界时形状会变）
key2 = tensor.untyped_storage().data_ptr()   # 实测补 +57 条

# 级别 3：字节区间包含（视图 offset 不同）
span = (tensor.storage_offset(), tensor.storage_offset() + tensor.numel())
# 与某个已知 producer 的 span 有交集 → 认为来自它
# 实测补 +0 条（HF Qwen3 没用到），但 nano-vLLM 的 split 需要
```

**为什么需要级别 2 的实证**：

```
q_proj 输出: shape (1,8,2048)  offset 0  storage 459964224
q_norm 输入: shape (1,8,16,128) offset 0  storage 459964224   ← 同一个 storage！
```

中间隔着 `view(...).transpose(1,2)`，形状和 stride 都变了，`id()` 必然不同。只按形状/storage_offset 匹配也会失败，**必须只看 storage 指针**。

### 6.2.2 误连的防护

只按 storage 指针会误连，典型是 **KV cache**：nano-vLLM 里 `k_cache = kv_cache[0, layer_id]`，28 层共享同一块 storage，每层 offset 不同。规则：

1. **最近生产者优先**：候选 producer 必须 `start_seq(producer) < start_seq(consumer)`，取 seq 最大的那个。
2. **区间包含要求**：级别 3 要求 consumer 的 `[offset, offset+numel)` 落在 producer 的 span 内；级别 2 只允许「offset 相同」或「producer 是 cache 切片且带 `cache_write` 标记」。
3. **一个 tensor 只有一个 producer**：若匹配到多个，报冲突并记录到 `graph.conflicts`，不要静默选第一个。

### 6.2.3 补全未解析输入（关键步骤）

```
对每个 module call c，若其输入张量 t 三级都没匹配上：
    op = dispatches 里「output storage == storage(t) 且 seq < start_seq(c)」的最后一条
    若 op 存在：
        新建 synthetic Node
        op 语义 = 规则表[op.aten_name]（见 §7）
        把 op 的输入张量递归执行 6.2.1（它们通常能匹配到某个 module 输出）
        把这条边记为 data
    否则：
        标记为图输入（input_ids / position_ids / attention_mask / 常量）
```

实测这条规则能补上全部 86 条：`o_proj` 的输入来自 `aten._scaled_dot_product_attention`、`post_attention_layernorm` 的输入来自 `aten.add.Tensor`（残差）、`down_proj` 的输入来自 `aten.mul.Tensor`（swiglu）。

### 6.2.4 in-place 写

用 `tensor._version` 检测：在 pre-hook 记录 `v0`，在 post-hook 记录 `v1`。`v1 > v0` 说明该模块原地改了入参。节点上标 `attrs.inplace = true`，并**把被改的 tensor 的 producer 改成这个节点**（写后值）。nano-vLLM 的残差延迟相加会触发这条；HF 目前不触发（实测 `_version` 字段可用）。

---

## 6.3 节点粒度

### 6.3.1 原则

**节点 = 一次算子调用**，不是模块对象。三个判据：

1. **模块 → 节点**：一个模块调用默认对应 1 个节点（`nn.Linear` → `matmul`，`Qwen3RMSNorm` → `rmsnorm`，`SiLUActivation` → `silu`）。
2. **模块 → 多节点**：当模块 forward 内部有「非平凡算子」时，用 dispatch 记录把它拆开。`Qwen3Attention` 是典型：一次模块调用实际包含 `rope_q`、`rope_k`、`kv_write`、`attention(qk^T, softmax, pv)`、`o_proj` 五段。
3. **多模块 → 一节点**：当两个模块是同一数学式的两半时，合并且保留 `impl` 列表。HF Qwen3 里 `act_fn` 与 `*` 是两个东西，但语义上是一个 `swiglu`；合并与否由 `--fuse-aware` 开关控制，**默认不合**（保持与源码一一对应），另提供 `semantic_view` 字段给出融合候选。

### 6.3.2 算子拆分表（M1 必须实现）

| 模块类 | 拆出的 op | 说明 |
| --- | --- | --- |
| `torch.nn.Linear` | `matmul` | `attrs = {"bias": bool}`，`state = [weight, (bias)]` |
| `Qwen3RMSNorm` | `rmsnorm` | `attrs = {"eps": config.rms_norm_eps}`，`state = [weight]` |
| `SiLUActivation` | `silu` | 无状态 |
| `torch.nn.Embedding` | `embedding` | 输出即层输入 |
| `Qwen3Attention` | `rope_q` / `rope_k` / `attention` / `kv_write` | 见下 |
| `Qwen3DecoderLayer` | `add` × 2（残差） | synthetic，来自未解析边 |
| `Qwen3MLP` | `mul`（swiglu 的乘法） | synthetic，来自未解析边 |

`Qwen3Attention` 的拆分规则（按 dispatch 序列切窗）：

```
svd 之前：rope（cos/sin 乘法）        → rope_q, rope_k
有 past_key_values 时：cache 更新     → kv_write（nano-vLLM 里的 store_kvcache）
attention_interface 调用              → attention（attrs.backend 必须记录，见 §6.5）
o_proj 调用                           → matmul（由子模块 hook 给出，不重复建）
```

**注意**：`attention_interface` 是运行时从 `ALL_ATTENTION_FUNCTIONS` 里选的（CPU 上通常是 `sdpa`，`eager` 是纯 PyTorch 实现）。同一个源码位置可能执行不同实现，**必须把 backend 名字记进 `attrs.backend` 和 evidence**，否则图不可复现。

### 6.3.3 透明算子

以下 aten op **不建节点**，只更新别名表（issue：它们占 dispatch 的大头）：

```
aten.view / _unsafe_view / reshape / transpose / t / permute / squeeze / unsqueeze
/ slice / select / narrow / expand / contiguous / clone / detach / alias
```

实测：2664 个 dispatch op 里，view+_unsafe_view+t 就占 769 个。折叠后剩余语义 op ≈ 1800，再合并同模块内的 view 链，最终节点数应在 **500–900** 区间（28 层 × 约 20–30）。

### 6.3.4 节点 id 规则

```
"L{层号}.{模块相对路径}@{同类调用序号}"         # 模块节点
"L{层号}.{父路径}.synth@{dispatch_seq}"        # 合成节点
根路径用 "R"（如 "R.embed_tokens@0"、"R.lm_head@0"）
```

---

## 6.4 语义标注（规则优先）

### 6.4.1 三级判定

```
1. 模块类名 + config 联合规则（覆盖已知架构）
2. 数据流形状 + 参数形状推断（不依赖名字）
3. 仍未命中 → op="unknown"，op_confidence="unknown"，进 graph.conflicts
```

**第 2 级是「不能只基于名称」的落点**，至少实现两条：

- 输入末维 `d`、权重 `(out, in)`：若节点有参数且 `in` 等于输入末维 → `matmul` 类（再按 `out` 与 config 的 `intermediate_size` / `num_heads*head_dim` 比对区分 gate/up/down/q/k/v/o）。
- 无参数 + 输出与输入同形状 + 对最后一维做归一化 + 有 `weight` 形状等于输入末维 → `rmsnorm`（再看 `eps` 是否存在、是否双输入，区分 `rmsnorm` 与 `add_rmsnorm`）。

### 6.4.2 证据链（每个判断必须带）

| evidence.kind | 来源 |
| --- | --- |
| `source` | `inspect.getsourcefile` + `getsourcelines` → `file:line` |
| `observed` | hook 实测的 shape/dtype/stride |
| `param` | `named_parameters` 里的形状 |
| `dispatch` | aten op 名 + 计数 |
| `loader` | `packed_modules_mapping` 之类 |
| `rule` | 命中的规则编号，如 `R-MATMUL-2` |

**没有证据的标签不许写。** 规则命中不了就写 `unknown`，不要猜。

### 6.4.3 LLM 兜底（可选，默认关）

`label_mode="rules+llm"` 时，对 `op=="unknown"` 的节点，把「源码片段 + 输入输出形状 + 参数形状」发给 LLM 要一个标签。硬约束：

- LLM 只能返回 `{op, reason, evidence_refs}`，`evidence_refs` 必须来自已知证据 id，否则丢弃。
- LLM 结果标 `op_confidence="inferred"`，与规则结果（`exact`）区分。
- 同一节点多次不一致时保留全部候选，进 `conflicts`。

---

## 6.5 源码对应

```python
# nanovllm_selector/extract/source.py
def source_ref(module: nn.Module) -> dict:
    cls = type(module)
    f = inspect.getsourcefile(cls)            # 可能为 None（C 扩展/动态类）
    line = inspect.getsourcelines(cls)[1]
    return {"class": cls.__name__, "qualname": cls.__qualname__,
            "file": f, "class_line": line,
            "sha256": sha256_file(f) if f else None}
```

三种特殊情况必须处理：

1. **`f is None`**：`torch.nn.Linear`、`nn.Embedding` 在 torch 内置但 Python 源码可得；真正 None 的（C 实现）记 `"file": null` 并加 evidence `{"kind":"rule","detail":"builtin"}`。
2. **被 torch 内部吞掉的计算**：`F.scaled_dot_product_attention` 这类在 C++/ATen 里执行，Python 层看不到。处理方式：记 `aten._scaled_dot_product_attention` 的 dispatch 证据 + `attrs.backend`，`op_confidence="inferred"`。
3. **运行时选择的分支**：`attention_interface`、`ACT2FN[config.hidden_act]` 都是查表选出来的。必须把**实际选中的键**（如 `"silu"`、`"sdpa"`）写进 `attrs`，否则同一份源码在不同机器上语义不同。

---

## 7. 语义规则表（M1 必须覆盖）

| op | 判定规则 | 状态来源 |
| --- | --- | --- |
| `embedding` | `isinstance(m, nn.Embedding)` | `weight (vocab, hidden)` |
| `matmul` | `isinstance(m, nn.Linear)` 或无参节点 + 有 2D 权重 | `weight (out, in)` |
| `rmsnorm` | 类名含 `RMSNorm` 且 `hasattr(m,"weight")` 且 `hasattr(m,"eps")` | `weight (hidden,)`、`eps` |
| `silu` | 类名含 `Silu`/`SiLU` 或 `config.hidden_act=="silu"` 对应激活 | 无 |
| `mul` | 无参 + 两输入同形状 + 输出同形状，且生产者是 `silu` 与 `matmul` | 无 |
| `add` | 无参 + 两输入同形状 + 输出同形状（残差） | 无 |
| `attention` | 模块类名含 `Attention` 且内部 dispatch 出现 `_scaled_dot_product_attention` 或 `softmax` | 无（但依赖 cache） |
| `rope` | dispatch 出现 `cos`/`sin` 乘法且输入来自 position_embeddings | 无 |
| `lm_head` | 最后一个 `matmul` 且权重 `(vocab, hidden)` | `weight` |

**规则必须带编号**（`R-EMB-1`、`R-MM-1`…），写进 evidence，方便回归。

---

## 8. 验证方案

### 8.1 Qwen3 对账表（M1 的硬门槛）

| 检查项 | 期望 | 来源 |
| --- | --- | --- |
| 模块数 | 427（HF，CPU 随机初始化） | 实测 |
| 命名参数数 | 310（tied 权重被 dedupe） | 实测 |
| `matmul` 节点数 | 197 = 28×7 + 1 | 实测 dispatch `aten.mm`=197 |
| `rmsnorm` 节点数 | 113 = 28 input + 28 post + 28 q + 28 k + 1 final | 推导，须实测确认 |
| `silu` 节点数 | 28 | 推导 |
| 残差 `add` 节点数 | 56 = 28×2 | 推导 |
| `embedding` | 1 | 推导 |
| `lm_head` | 1 | 推导 |
| 层模板节点数 | 14 个模块节点 + 3 个合成节点（add×2 + mul×1）+ attention 拆分 4 个 | 推导 |
| 未解析边 | 只有 `input_ids`、`position_ids` 等图输入 | 目标 |
| 每条边两端都存在 | 100% | 目标 |
| 每个节点 ≥1 条 evidence | 100% | 目标 |

`--check` 把上表逐条跑一遍，任何一条不符 → 退出码 1，并打印「期望 vs 实际」。

### 8.2 数值对账（可选，M2）

用同一份随机权重跑一次 HF 前向，再把图按拓扑序在 `graph_to_torch.py` 里重放一遍，比较最终 logits 的最大绝对误差 < 1e-5。这一条能抓出「边连错了但形状恰好对得上」的情况，**建议 M2 前完成**。

### 8.3 换模型验证（Llama）

- `--model tmp/models/Llama-3.2-1B`（需另行确认下载）。
- 期望差异（相对 Qwen3）：无 q/k norm（`rmsnorm` 113 → 57）、`attention_bias=false`、`num_key_value_heads` 不同、RoPE 参数不同。
- **验收标准不是「跑通」，而是「差异清单与源码一致」**：把 `diff_graphs(qwen3, llama)` 的结果和 `modeling_llama.py` 的结构差异逐条对上。

---

## 9. 里程碑

### M1 —— Qwen3 出图，人能看懂（预计 3–4 天）

范围：只做 prefill、只做 Qwen3、CPU 随机权重。

交付：
1. `nanovllm_selector/extract/` 全部模块；
2. `outputs/graph_qwen3_prefill.json`；
3. `--summary` 输出一张人可读的 Markdown 表（模板节点 × op × 形状 × 状态 × 证据数）；
4. 单元测试 3 个：schema 校验、对账、模板折叠。

验收命令：

```bash
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --check
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --summary
python -B -m unittest discover -s nanovllm_selector/tests -v
```

通过标准：`--check` 退出码 0；`--summary` 里每行都有 `op`、`instance 数`、`input 形状`、`state`、`evidence 条数`；测试全绿。

### M2 —— 支持 Llama + 数值重放（预计 3–4 天）

- 加载真实权重（bf16, GPU）跑一次 prefill + 一次 decode，两张图（或一张图两个 `phase_shapes`）。
- `graph_to_torch.py`：按图重放，与 HF logits 比对（§8.2）。
- 出 `diff_graphs(qwen3, llama)` 报告。

验收：Llama 的 `--check` 通过（期望值按 §8.3 重算）；重放误差 < 1e-4（bf16）；差异报告与源码一致。

### M3 —— 图 diff 与缺口清单（预计 2–3 天）

- `diff_graphs(reference, target)` 输出 `{added, removed, changed}`，`changed` 细分到 shape/dtype/state。
- 产出一个「框架能力缺口」算子的雏形：给定「nano-vLLM 能执行的算子集合」，输出参考图里哪些节点没有对应实现。

验收：对 Qwen3 自身 diff 为空；对 Llama 能列出「缺少 q/k norm 之外的差异」；缺口清单能被一个手工写的 `capabilities.json` 消费。

### 时间盒

总预算 10 个工作日。任一里程碑超时 50%，**停下汇报**，不要自行扩范围。

---

## 10. 风险与坑

| # | 风险 | 症状 | 规避 |
| --- | --- | --- | --- |
| R1 | forward hook 是**完成序**，父模块晚于子模块 | 图里 `self_attn` 排在 `o_proj` 后面，拓扑序错 | 用 **pre-hook 记起点**、hook 记终点，节点顺序按 `start_seq` |
| R2 | `id()` 被回收后复用 | 随机地把两个张量连起来，跑两次结果不同 | 记录期间**持有所有张量引用**；`id()` 只用于当次录制 |
| R3 | 视图跨模块边界 | `q_proj→q_norm` 连不上（实测） | 加 storage 指针匹配（§6.2.1 级别 2） |
| R4 | KV cache 共享 storage | 28 层互相误连 | 最近生产者 + 区间包含 + 冲突报错（§6.2.2） |
| R5 | in-place 写 | 残差被原地加，producer 指向错 | `_version` 比对（§6.2.4） |
| R6 | 父模块 forward 里的合成算子 | 22% 的边缺失（实测 86 条） | dispatch 补全（§6.2.3） |
| R7 | dispatch 粒度过细 | 2664 个 op，没法看 | 透明算子折叠（§6.3.3） |
| R8 | 共享模块实例 | `RotaryEmbedding` 1 个对象、56 次调用 | 节点按**调用**计，id 带调用序号 |
| R9 | tied embeddings 计数不一致 | HF 310 vs nano-vLLM 227（实测） | 记录 `storage_group`，对账时按 storage 去重 |
| R10 | 混合精度 buffer | `cos_sin_cache` 是 fp32，权重是 bf16 | 每个 tensor 独立记 dtype，不做全局假设 |
| R11 | `torch.export` 需要 `use_cache=False` | 改变模型语义，图里没有 KV cache 写入 | export 只做旁证，不进主流程 |
| R12 | HF 版本漂移 | 行号、类名、默认 backend 全变 | 记录 `transformers.__version__` + 每个源文件 sha256；版本变了先重跑对账 |
| R13 | CPU fp32 与 GPU bf16 形状一致但 stride 不同 | 形状对账通过、性能假设错 | `phase_shapes` 分开记，stride 也入图 |
| R14 | 动态 batch/seq | 图里写死 `(1,8)`，换长度就失效 | 形状里用符号 `"N"` 表示 batch/token 维；每节点记 `symbolic_shape` |
| R15 | `attention_interface` 运行时选后端 | 同一源码在不同机器语义不同 | `attrs.backend` 必须记录（§6.5） |

---

## 附录 A：本文实测数据的复现命令

```bash
# 模块数 / 参数数 / fx / export
docker exec -i nanovllm-selector-rocm-dev python -B - <<'EOF'
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
cfg = AutoConfig.from_pretrained('/workspace/tmp/models/Qwen3-0.6B', local_files_only=True)
m = Qwen3ForCausalLM(cfg).eval()
print(sum(1 for _ in m.modules()), sum(1 for _ in m.named_parameters()))
EOF
```

环境：`torch 2.12.0+rocm10.0.0`、`transformers 4.57.3`、`Python 3.12`、CPU fp32、`input_ids (1,8)`、随机权重。

## 附录 B：与 [kernel-selector-plan.md](kernel-selector-plan.md) 的衔接

本文件替换该文档中：

- 「① 模型解剖 → 计算点清单」→ 改为「① 参考图重建 → 完整计算图」；
- 第 5 章「Qwen3 全覆盖清单」→ 由 `--check` 的对账表（§8.1）自动产出；
- 第 6 章 P1「写三个 Op Spec」→ 改为「先出图，再从图里查 Op Spec」。

**不替换**：三张数据契约的结构（Target / Op Spec / Candidate）、装配方式的 A/B/C、验收闸门 G0–G3。Op Spec 在本文件里以 `Node.state + Node.attrs + Node.semantics` 的形式存在，字段名保持一致。
