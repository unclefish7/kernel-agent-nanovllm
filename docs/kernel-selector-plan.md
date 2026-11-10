# Kernel Selector Agent：学习与实现计划

- 版本：v0.1（草案），2026-10-28
- 读者：项目作者（用来理解系统全貌、安排学习顺序）+ 编码 agent（用来在明确边界内实现）
- 状态：本文是计划与契约草案，不是已完成实现的说明。所有 JSON schema 都允许升级，但升级必须带版本号并保持向后兼容。

## 0. 这份文档怎么用

- 第 1–3 章：**给人读**。回答"这个系统到底是什么样子""为什么现在这版一扩就大改"。尽量少用工程术语。
- 第 4 章：**契约**。给实现者的数据格式与规则，是本项目唯一需要长期稳定的部分。
- 第 5–6 章：**工作量清单 + 阶段计划**。每阶段都有"要学的概念"和"要产出的东西"，学和做交替进行。
- 第 7–8 章：**可扩展性保证 + 给编码 agent 的硬约束**。写代码前必须先读。
- 附录：术语表、旧代码去留对照。

阅读建议：第一次只读第 1、2、3 章和第 6 章；动手前再读第 4 章；每次让 agent 改代码前，把它指向第 7、8 章。

---

## 1. 一页纸：这个系统是什么

完整链路只有五步：

```
① 模型解剖      真实模型跑一遍 + 读 HF 源码
                → 得到"计算点清单"，每个点带一份【算子契约 Op Spec】
                          ↓
② 候选库        收集真实存在的 kernel
                → 每个 kernel 变成一条【候选条目 Candidate】
                          ↓
③ 选择          agent 阅读契约 + 检索出的候选子集
                → 输出一份【方案 Plan】（谁用哪个实现、怎么组合）
                          ↓
④ 装配          确定性执行器把 Plan 装到真实模型上
                → 不训练、不改权重、不碰框架源码
                          ↓
⑤ 验收          逐点数值 → 端到端数值 → 性能
                → 自动判定通过与否，失败定位到 Plan 的具体节点
```

**关键判断：五个部件里，agent 是最小的一个。** 决定这套东西能不能长大的，是 ①②④⑤ 能不能被数据驱动；agent 只是"读数据、填一个 JSON"。

所以本项目的技术主线不是"怎么让模型选得更准"，而是：

> **把「换哪个算子」这件事，从散落在代码里的 if 分支，变成三张可以用程序校验的表。**

三张表：

| 名字 | 回答什么问题 | 谁来写 |
| --- | --- | --- |
| **Target 表** | 框架里哪些位置可以换？ | 模型解剖阶段生成 |
| **Candidate 表** | 有哪些实现可以换上去？ | 人工登记 + 半自动抽取 |
| **Plan** | 这一次换成什么、怎么组合？ | agent 生成 |

外加一个**执行器**：读 Plan，校验、安装、取证。它是纯确定性代码，不含任何模型调用，可以脱离 agent 单独测试。

这三张表就是全部的可扩展性来源。**以后扩展系统 = 往表里加一行，而不是改执行器。**

---

## 2. 诊断：为什么现在这版一扩就大改

上一版 demo（提交 `491b4e7`）能跑，但它把四件本该分开的事混在同一个文件里。具体位置：

| 症状 | 代码位置 | 扩展时会怎样 |
| --- | --- | --- |
| 计算点写死 | `run_selector.prepare()` 直接写 `model.layers.{i}.mlp.act_fn` | 接 RMSNorm 要同时改采集、安装、校验三处 |
| 候选写死 | `candidates.catalog()` 返回手写 list | 加一个候选要改 5 处：catalog、`unsupported_reason`、`validate_plan` 的 ID 集合、`install()` 的分支、prompt 文案 |
| 方案只有一格 | `plan.choice` 只有 `target/candidate_id/reason` | 无法表达"两个 kernel 组合"，格式要推倒 |
| 安装借时序 | 构造期 patch `ModelRunner.warmup_model` | 依赖框架内部调用顺序；多进程不生效；图捕获下计数失效 |
| 包装改模块树 | `SelectedActivation` 用 `self.original = original` | `state_dict` 键名变成 `x.original.weight`；接 RMSNorm（有 weight）直接被硬断言拦下 |
| 检查在热路径 | `unsupported_reason()` 每次 forward 都调 `get_device_properties` | 实测 4.2 µs/次，28 层 decode 每步多出约 118 µs |

这些不是"写得不好"，而是**设计时只考虑了"一个点、一个候选、一次安装"**。所以结论不是打补丁，而是：保留思路，重写结构。

**要保留的四个决策**（它们是上一版真正的价值，重写时原样保留）：

1. **方案指纹绑定**：plan 绑定 checkpoint 哈希、dtype、GPU 架构、Torch/Triton 版本、目标源码哈希、候选表哈希；任何一项变了就拒绝复用。
2. **回退语义明确**：规格不支持 → 回退原生并记录原因；候选自己报错 → 直接失败，不静默回退假装成功。
3. **候选可追溯**：vendored 代码要记录 repo + commit + 路径 + sha256。
4. **prepare / select / run 三步分离**：采集和决策可以慢，正式运行不调 API。

---

## 3. 三个概念（人话版）

### 3.1 计算点（Target）

模型里一个"可以整个换掉"的位置。它需要三个信息：

- **位置**：怎么找到它。例如"所有层的 `mlp.act_fn`"，写成路径模式 `model.layers.*.mlp.act_fn`。
- **语义**：它算的是什么。例如 `swiglu_packed`。注意是**语义 id**，不是类名——实现可以变，语义不能变。
- **装配方式**：怎么换掉它。第一版只有一种：换掉这个模块的 `forward` 函数。

### 3.2 算子契约（Op Spec）

描述"这个计算点接受什么、产出什么"。**它不描述实现，只描述契约。**

类比：它像 C 语言里的函数声明（头文件），而不是函数实现。有了它：

- 候选库能判断"我能不能被装到这里"；
- 组合时能判断"A 的输出能不能接到 B 的输入"；
- 验收时知道该准备什么输入、期望什么输出。

### 3.3 候选（Candidate）

一个真实存在的 kernel + 它凭什么能被换上去。四件套：

1. **能力**：它能提供哪个语义（`provides`）。
2. **适配器**：怎么把框架的输入喂给它、把它的输出还回去（真实 kernel 的签名通常和契约不一致）。
3. **支持检查**：一个**可执行函数**，判断当前输入是否被支持，返回 `(ok, reason)`。
4. **来源与数值特征**：repo/commit/哈希；累加精度、是否原地修改、输出 dtype。

三个东西对齐，才能装配成功：

```
Target 声明的 op  ==  Candidate 声明的 provides
       且        Candidate.support(实际输入) == ok
```

---

## 4. 契约细节（给实现者）

以下 schema 为 v1 草案。字段可以加，含义不能改。所有 schema 带 `schema_version`，执行器必须同时接受当前版本和前一个版本。

### 4.1 Target 表

一个 Target 描述一条（可能覆盖多个模块实例的）替换位置。

```json
{
  "target": "model.layers.*.post_attention_layernorm",
  "op": "add_rms_norm",
  "instances": 28,
  "assemble": "replace_forward",
  "contract": {
    "inputs":  ["x: [N, H] bf16", "residual: [N, H] bf16"],
    "outputs": ["y: [N, H] bf16", "residual_out: [N, H] bf16"],
    "effects": {"mutates_input": false, "allocates": true},
    "state":   ["weight: [H] bf16", "eps: float"]
  },
  "notes": "nano-vLLM 的 Add+RMSNorm 不原地写 residual，而是返回新的 residual；顺序必须是 (归一化结果, 新残差)"
}
```

`assemble` 是装配方式枚举，第一版只实现 `replace_forward`，另两种明确留着（见 4.5）。

**Rule T1**：一个模块可能对应**两个不同**的 Target。例如 Qwen3 的 `input_layernorm`：第 0 层 `residual=None` 走单输入分支，其余 27 层走双输入分支。同一个模块、两种 op。表里必须分两条。

**Rule T2**：Target 的粒度**可以细于模块**。Attention 内部至少有 4 个可独立替换的计算（QK^T、softmax、PV、KV cache 写入），第一版不拆，但要承认"模块级 Target"不是唯一形态，否则以后接 Attention 会被迫重写。

### 4.2 Op Spec

示例（第一个接入点，最简单的一类）：

```json
{
  "op": "swiglu_packed",
  "formula": "y[..., i] = silu(x[..., i]) * x[..., i + I]",
  "reference": {
    "module": "transformers.models.qwen3.modeling_qwen3.Qwen3MLP.forward",
    "source_sha256": "<HF 源码哈希>",
    "weight_packing": {"gate_proj": ["gate_up_proj", 0], "up_proj": ["gate_up_proj", 1]}
  },
  "inputs":  [{"name": "x", "role": "gate_up_packed", "layout": "contiguous",
               "dtype": ["bf16"], "shape": ["N", "2I"]}],
  "outputs": [{"name": "y", "role": "activation", "dtype": ["bf16"], "shape": ["N", "I"]}],
  "params":  [],
  "state":   [],
  "effects": {"mutates_input": false, "allocates": true, "stream": "current"},
  "observed": [{"phase": "prefill", "shape": [128, 6144]},
               {"phase": "decode",  "shape": [1, 6144]}],
  "context": []
}
```

字段说明与**必须现在就设计进去的四个字段**：

| 字段 | 含义 | 为什么现在就要有 |
| --- | --- | --- |
| `role` | 每个输入输出的语义角色 | 同一语义可能有 packed / split 两种布局（HF 是两个张量，nano-vLLM 是打包成一个）。用角色对齐，而不是靠位置对齐。组合时靠它连线 |
| `state` | 不随调用变化的输入：权重、eps | RMSNorm 的 `weight` 属于这里，不属于 `inputs`。混进去会让 plan 把它当数据流 |
| `effects` | 是否原地改、是否新分配、在哪个 stream | nano-vLLM 的 Add+RMSNorm 会改动 residual；不写清楚，正确性判断必错 |
| `context` | 依赖的框架上下文（v1 全为空） | Attention 需要 `block_tables` / `slot_mapping` / `cu_seqlens_q|k` / `is_prefill`。现在留空字段，接 Attention 时不用改 schema |

完整参考源码与打包映射要一并带上，因为 agent 的 reference 就是这些源码（这是项目既定设计：用 Transformers 源码表达"怎么算"，因为源码规定了数学但没规定实现）。

### 4.3 Candidate 表

```json
{
  "id": "geak_packed_swiglu",
  "provides": ["swiglu_packed"],
  "source": {"repo": "AMD-AGI/GEAK-eval", "commit": "453b0218", "path": "geak_eval/data/.../swiglu_fwd.py",
             "sha256": "<vendored 哈希>"},
  "adapter": "adapters/geak_swiglu.py:call",
  "support": "checks/geak_swiglu.py:support",
  "binding": {"backend": "triton", "device": ["gfx908"], "dtypes": ["bf16", "fp16", "fp32"]},
  "numerics": {"accum": "fp32", "out_dtype": "input"},
  "perf": {"measured": false},
  "risks": ["内部 FP32 累加后写回 bf16，舍入与原生分步实现不同"]
}
```

四条规则：

**Rule C1**：`provides` 是能力列表，不是名字。一个融合 kernel 可以同时声明 `provides: ["add", "rms_norm", "add_rms_norm"]`——这就是"组合"的物理基础：不同粒度的候选在同一个池子里竞争。

**Rule C2**：`support` 必须是一个**可执行函数**，返回 `(ok: bool, reason: str)`。禁止用自然语言描述支持范围。理由：自然语言无法被校验，也无法产生可复现的失败原因。

**Rule C3**：适配器必须显式声明它引入的额外开销（例如"需要先把两个张量拼接"）。旧原型里 Qwen3 的拼接成本计入过评测，这个做法保留。

**Rule C4**：候选库 = **索引 + 适配器 + 支持检查 + 来源**，不是"一个放 .py 文件的文件夹"。规模到几百条时，检索是核心问题：agent 不可能看全库，必须按 `(op, dtype, shape 区间, device, backend)` 先过滤。

**扩库路线**（三条，前两条可并行）：

| 阶段 | 数量级 | 做法 | 人工做什么 |
| --- | --- | --- | --- |
| 手工 | 2 → 30 | 从 GEAK-eval / TritonBench / PyTorch eager 里挑同类实现 | 全部手写条目 |
| 半自动 | 30 → 200 | 脚本抽取函数签名与 `autotune` 配置，生成条目草稿 | 只审语义标签和支持范围 |
| 自动 | 200 → 1000+ | 批量抽取 + 用差分测试自动判定支持范围 | 只审失败样本 |

### 4.4 Plan

Plan 是**图**，不是列表。第一版只有一个节点，但格式必须从第一天就支持多节点。

```json
{
  "schema_version": 1,
  "binding": {"checkpoint_sha256": "...", "dtype": "bf16", "gpu_arch": "gfx908",
              "torch": "...", "triton": "...", "catalog_sha256": "..."},
  "targets": [
    {
      "target": "model.layers.*.post_attention_layernorm",
      "op": "add_rms_norm",
      "tensors": {"t_sum": {"dtype": "fp32", "note": "中间量保持 fp32，避免 bf16 二次舍入"}},
      "nodes": [
        {"node": "sum",  "candidate": "torch_add_fp32",
         "in": {"a": "$in[0]", "b": "$in[1]"}, "out": {"s": "$t_sum", "residual": "$out[1]"}},
        {"node": "norm", "candidate": "geak_rmsnorm",
         "in": {"x": "$t_sum", "w": "$state.weight"}, "out": {"y": "$out[0]"}}
      ]
    }
  ]
}
```

命名空间：`$in[i]` = 第 i 个输入，`$out[j]` = 第 j 个输出，`$state.<name>` = 模块状态，普通名字 = 中间张量。

**为什么这个例子重要**：nano-VLLM 原生的 `add_rms_forward` 是先把两个 bf16 输入升到 fp32 相加，再在 fp32 上做归一化，最后才降回 bf16。如果拆成"先 bf16 相加 → 再读 bf16 归一化"，中间多一次舍入，**数值会变**。所以 Plan 必须能表达"中间张量用什么精度承载"，否则组合出来的方案会静默失准。

这条也说明：**组合的难点不是让 agent 输出多个 ID，而是让 Plan 的表达能力和验收能力跟上。**

### 4.5 装配：替换到底怎么做

先列出**禁止**的三种做法（上一版踩过的坑）：

- ❌ 运行时替换框架的类方法（`ModelRunner.warmup_model`）——依赖内部调用顺序，多进程不生效。
- ❌ 把原模块挂成新模块的子模块（`self.original = original`）——`state_dict` 键名变化，有参数的模块直接装不上。
- ❌ 在 `forward` 热路径里做支持检查、计数、日志——开销直接进延迟。

三种**允许**的装配方式，按引入顺序：

| 方式 | 做法 | 适用 | 何时引入 |
| --- | --- | --- | --- |
| A `replace_forward` | 替换模块实例的 `forward` 绑定方法，模块树和权重完全不动 | MLP 激活、RMSNorm、Linear、RoPE | P2 |
| B `replace_call` | 框架 forward 里调用的某个函数符号被替换 | 框架愿意配合改造时 | P5 之后 |
| C `graph_rewrite` | 拦截 `forward`，在计算图层面做子图替换 | 融合、Attention 内部拆分 | P6 |

方式 A 的最小实现要点（写代码前必看）：

1. 用 `module.forward = types.MethodType(fn, module)` 替换，**不要**新建 `nn.Module` 包一层。
2. 原 `forward` 存在闭包或旁路字典里，**不要**存成模块属性。
3. 安装前后必须断言：`[(name, id(p)) for name, p in model.named_parameters()]` 完全相同，且 `list(model.state_dict())` 完全相同。这两条是硬断言，不是建议。
4. 每个实例独立安装、独立失败、可单独恢复。
5. 安装动作只发生在一个明确位置：**`LLM(...)` 构造完成之后、第一次 `generate()` 之前**。理由：此时权重已加载、KV cache 已按原生分配（对比公平），且不依赖框架内部顺序；多进程时每个 rank 各自安装。

代价：候选的首次调用会落在第一次真实请求上，触发 Triton JIT / autotune。所以**性能评测前必须先跑一遍预热请求**，这一条写进验收规范（见 4.6）。

### 4.6 验收：什么叫"真的闭环"

四级闸门，逐级放行：

| 级别 | 判据 | 产物 | 失败时 |
| --- | --- | --- | --- |
| **G0 接口** | 每层被调用次数 > 0；返回值 shape/dtype/device 与契约一致；不崩 | 调用计数、回退原因 | 定位到 target |
| **G1 逐点数值** | 同一输入下，候选输出 vs 原生输出，误差 ≤ 门槛；覆盖全部观测规格 + 边界（空/非连续/奇偶末维/极值） | max / mean / p99 误差，逐规格一条 | 定位到 plan 的**节点** |
| **G2 端到端数值** | 同 token 前缀下完整 logits 对比；固定 greedy 采样逐 token 一致 | logits 误差曲线、token 一致率 | 定位到层 |
| **G3 性能** | 同进程交替、3 轮取中位；TTFT / TPOT / token/s；记录 `num_kvcache_blocks` | 原始样本 + 汇总 | 不阻塞正确性 |

**"闭环"的定义**：一份 Plan 能被一条命令自动跑完 G0–G3，产出可复查报告；任何一级失败都能定位到 Plan 的具体节点或具体规格。只要还需要人去看日志判断"这算不算过"，就不是闭环。

三个必须写死的细节：

1. **门槛从哪来**：不是拍脑袋的常量，而是"原生路径与 Transformers 的偏差量级"。当前基线里 nano-vLLM 对 Transformers 的最大 logit 绝对误差是 `0.6875`（8 步内 top-1 全一致）。候选路径的误差应该与这个量级同阶；明显超过它就是引入新偏差。G1 的逐点门槛同理：先用原生实现跑两遍确认自身可复现，再定阈值。

2. **误差要报分布，不是报布尔**。只报 "pass/fail" 的验收无法定位问题。

3. **性能必须记录 `num_kvcache_blocks`**。nano-vLLM 用 warmup 期间的显存峰值来算 KV cache 块数；候选如果改变了峰值，原生和 Agent 两次运行拿到的块数会不同，对比环境就不一致。

失败信号要能回传：第一版允许只是"拒绝这份 Plan"，但报告格式里必须留出"失败节点 + 失败原因"的字段，为后续"把失败喂回 agent 重选"留接口。

---

## 5. Qwen3 全覆盖工作清单

"完全接入 nano-vLLM 的 Qwen3"具体是多少活，先算清楚。单次前向的调用次数（28 层）：

| Target | op | 单次前向调用次数 | 装配方式 | 难度 |
| --- | --- | --- | --- | --- |
| `model.layers.*.mlp.act_fn` | `swiglu_packed` | 28 | A | 低（已完成） |
| `model.layers.*.input_layernorm`（第 0 层） | `rms_norm` | 1 | A | 低 |
| `model.layers.*.input_layernorm`（其余 27 层） | `add_rms_norm` | 27 | A | 中（双输出 + 残差语义） |
| `model.layers.*.post_attention_layernorm` | `add_rms_norm` | 28 | A | 中 |
| `model.norm`（最终 Norm） | `add_rms_norm` | 1 | A | 中 |
| `model.layers.*.self_attn.q_norm` / `k_norm` | `rms_norm`（3D，最后一维是 head_dim） | 56 | A | 低但形状要注意 |
| `model.layers.*.self_attn.rotary_emb` | `rope` | 28 | A | 中 |
| `model.layers.*.self_attn.attn` | `attention` | 28 | A（粗粒度）/ C（细粒度） | 高 |
| `model.layers.*.self_attn.{qkv_proj,o_proj}` | `matmul` | 56 | A | 中 |
| `model.layers.*.mlp.{gate_up_proj,down_proj}` | `matmul` | 56 | A | 中 |
| `lm_head` | `matmul` | 1 | A | 中 |

两个容易踩的事实：

- **RMSNorm 一共 113 次调用**：57 次单输入单输出（其中 56 次是 3D 的 q/k norm），56 次双输入双输出。同一个"语义"在不同形状下出现，这正是 Op Spec 里 `shape` 要写模板而不是写死数字的原因。
- **Linear 是 4 个模块、113 次调用**，但它们内部有 TP 分片逻辑（`MergedColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear`），替换前必须先读懂分片，否则 TP>1 会错。第一版 TP=1 可以只保证正确性，但 Target 描述里要标明这个限制。

排期上的建议：**先把所有 Norm 打通（P3），再动 Linear 和 Attention**。因为 Norm 覆盖了"有状态 + 多输入多输出 + 延迟残差"这三类难点，而且不涉及分片和 KV cache。

---

## 6. 阶段计划（学习 + 实现交替）

规则：**每个阶段结束时，系统都必须处于可运行、可验证的状态。** 不允许出现"先推倒上层，等下一阶段一起验证"。

### P0 看懂框架（1–2 天，只读不改）

- **学**：PyTorch 的模块树、`state_dict`、`forward` hook、绑定方法。这四个概念是后面所有装配操作的基础。
- **读三个文件**，每个都带着问题读：
  - [model_runner.py](../nano-vllm/nanovllm/engine/model_runner.py)：`run()` → `prepare_prefill`/`prepare_decode` → `set_context` → `model(...)` → `compute_logits`。问题：decode 时 batch 里每一行的 shape 是多少？为什么是 1？
  - [qwen3.py](../nano-vllm/nanovllm/models/qwen3.py)：问题：`hidden_states` 和 `residual` 为什么要成对返回？第 0 层和其余层走的分支差在哪？
  - [context.py](../nano-vllm/nanovllm/utils/context.py)：问题：Attention 需要的 `block_tables` / `slot_mapping` 是从哪来的？
- **产出一个笔记**（可以放在本文末尾或单独文件）：一张 ASCII 调用链图 + 三条你确认过的 shape 事实。
- **完成判据**：能不看代码说出"一个 decode step 从 `run()` 到 logits 经过哪些调用"。
- **小练习**：打印 Qwen3 的 `state_dict().keys()`，数一数有多少个 `weight`，并画出模块树的前三层。

### P1 契约与校验器（纯 CPU，不需要 GPU）

- **学**：JSON Schema 风格的校验思路；单元测试怎么覆盖"应该拒绝的输入"。
- **写**：
  - `contracts/`：Target / OpSpec / Candidate / Plan 的 schema 与校验函数。
  - Qwen3 的 Target 表和三个 Norm 点的 Op Spec（手写，不自动提取）。
  - 校验器单元测试：非法 schema、节点引用不存在、输入角色不匹配、绑定指纹不一致，全部要报错。
- **完成判据**：`python -m unittest` 在没有 GPU 的机器上通过；一份故意写错的 Plan 能被精确定位到出错字段。
- **为什么先做这个**：它把"格式"从"代码"里剥离出来。之后所有扩展都是改数据，改错了会被校验器抓住，而不是等到跑推理才崩。

### P2 最小执行器（一个 op、两个实现，G0 + G1）

- **学**：`module.forward` 替换、如何在不改权重的前提下换实现。
- **写**：
  - `assemble/installer.py`：读 Plan，按 `replace_forward` 安装，硬断言"参数名和 state_dict 键完全不变"。
  - `assemble/instrument.py`：可开关的计数与取证，默认关闭。
  - 只接一个 target（MLP 激活）、两个候选（`native` + `geak_packed_swiglu`）。
  - `verify/numeric.py`：G1，逐规格对比原生输出，报误差分布。
- **完成判据**：一条命令跑完 G0 + G1 并产出报告；把实现换回 native 后结果无变化（证明装配是无损的）。
- **关键**：这一步**不接 agent**。手写 Plan。目标是"脱离 LLM 也能端到端验证"。

### P3 覆盖 Qwen3 的全部 Norm（G2）

- **写**：普通 RMSNorm（2D + 3D 两种形状）、Add+RMSNorm（双入双出 + 延迟残差）；最终 Norm。
- **难点**：Add+RMSNorm 的"新残差"必须与原实现完全一致（包括它返回的是 fp32 相加后降回输入 dtype 的那个值）。这里最容易出静默错误。
- **完成判据**：G2 通过——同 token 前缀的 logits 与原生路径的偏差在原生噪声量级内；固定 greedy 采样逐 token 一致。
- **至此，"完全接入 Qwen3（除 Attention 和 Linear）"达成。** 这是第一个真正的里程碑。

### P4 候选库工程化

- **学**：如何从现有代码里抽取函数签名与常量；差分测试怎么自动判定支持范围。
- **写**：注册表 + 检索（按 op / dtype / shape 区间 / device 过滤）+ 20–50 条真实候选 + 每条一个支持检查函数。
- **完成判据**：同一 op 至少有 3 个候选可切换；用错 dtype 或非连续输入会得到明确拒绝原因，而不是崩溃。
- **为什么放在 agent 之前**：agent 的价值完全取决于候选池的质量。池子只有 2 条时，选谁都是猜。

### P5 接 agent

- **写**：把 Op Spec + 检索后的候选子集喂给 agent；产出 Plan；执行 G0–G2；失败原因写回报告。
- **学**：约束生成（只允许输出登记过的 ID）、失败重试的边界。
- **完成判据**：同一份上下文重复调用，产出结构合法且可执行；非法输出被拒绝而不是被执行；每一级失败能定位到节点。

### P6 组合与融合（进入 Attention 前的地基）

- **写**：多节点 Plan 的执行、中间张量精度声明、C 类装配（图重写的可行性验证）。
- **验证目标**：能表达"两个 op 合成一个融合 kernel"和"一个 op 拆成两步"，且 G1 能分辨两者数值差异。
- **之后**：Linear（TP 分片）、RoPE、Attention（需要 context 字段）、Sampler。

---

## 7. 可扩展性靠什么保证

三条规则，缺一条就会退化成"每次扩展都要大改"：

1. **加东西只加数据，不改执行器。**
   新增一个可替换点 → 往 Target 表加一行；新增一个 kernel → 往 Candidate 表加一条 + 一个 support 函数；新增一个模型 → 加一份 introspect 配置。执行器的代码修改应该是罕见的。

2. **契约只加字段，不改含义；带版本号，执行器同时接受 N 和 N−1。**
   这是应对"表达能力不够"的正规做法。预计会发生的两次升级：P3 加 `state` 的精度声明，P6 加 `context` 和中间张量精度。

3. **每个阶段都有一个能跑通的端到端路径。**
   系统任何时候都能回答"现在哪几个点能被替换、证据是什么"。这是防止"为了扩展先推倒"的唯一办法。

**什么时候算"该改架构"**：当你需要表达的能力超出了 schema（而不是超出了一条数据）时。这时的改动是"加字段 + 升版本"，是可预期的；而"改执行器"应该只在 P6 那种真正换装配方式的时候发生。

---

## 8. 给编码 agent 的硬约束

写任何代码前，先确认不属于下面任何一条。

**禁止**

1. 禁止在运行时替换框架的类或方法（`patch.object(ModelRunner, ...)` 之类）。
2. 禁止把原模块挂成新模块的属性（会造成 `state_dict` 键名变化）。
3. 禁止在 `forward` 热路径做支持检查、计数、日志、`get_device_properties`。
4. 禁止把层号、模型名、候选 id 写进执行器或采集器。
5. 禁止执行 agent 返回的代码；只允许引用登记过的 ID。
6. 禁止在 G1 未通过时宣称"接入成功"或"等价"。
7. 禁止把"调用次数 > 0"当作正确性证据。
8. 禁止改动 `nano-vllm/` 子模块（阶段 P0–P5）。确需改动时，单独提交、单独说明理由。

**必须**

1. 安装前后断言参数名、参数对象 id、`state_dict` 键完全不变。
2. `support` 是函数，返回 `(ok, reason)`；`reason` 要能在报告里定位问题。
3. 每次改动说明：动了哪张表、执行器是否改动、验证到 G0/G1/G2/G3 哪一级。
4. 新增候选必须带来源（repo + commit + path + sha256）和数值特征（累加精度、是否原地、输出 dtype）。
5. 报告要落在 `runs/` 下并带环境指纹；可复现的命令写进 README。
6. 不确定时选择"拒绝"而不是"降级猜测"。

---

## 9. 学习资料与小练习

按需要查阅，不要通读。

- **框架内部**：`nano-vllm/nanovllm/engine/model_runner.py`、`models/qwen3.py`、`layers/layernorm.py`、`layers/activation.py`、`utils/context.py`。这五个文件加起来不到 600 行，是全部接入点的坐标系。
- **工业界的可替换算子长什么样**：[vLLM 的 `custom_op.py`](../references/reps/vllm/vllm/model_executor/custom_op.py)（`CustomOp` / `PluggableLayer` / `op_registry`），以及它的 attention backend 注册机制。重点看它把"平台相关的实现替换"抽象成了什么。
- **论文脉络**：[kernel-agent-papers.md](kernel-agent-papers.md)。当前阶段只需要关注"Astra 如何把 kernel 重新接回推理框架"和"RealisticTritonBench 如何做框架级验证"。

三个动手小练习（都在 P0–P2 期间完成）：

1. **模块树练习**：打印 Qwen3 的模块树和 `state_dict` 键，验证"一个权重一个地址"这个说法。
2. **无损装配练习**：给第 0 层的 `mlp.act_fn` 装一个"完全调用原实现"的替换，跑生成，确认输出与原生逐 token 相同。这证明装配机制本身是安全的。（注意：要用 4.5 的方式 A，不要用 `nn.Module` 包装。）
3. **规格采集练习**：用 forward hook 采集三个 Norm 点和激活点的输入 shape，验证你对 prefill / decode 的理解。当前已知的观测值：prefill `[128,6144] / [41,6144] / [25,6144]`，decode `[1,6144] / [2,6144]`。

---

## 附录 A：术语表

| 术语 | 人话解释 |
| --- | --- |
| 计算点 / Target | 模型里一个可以被整体替换的位置 |
| 算子契约 / Op Spec | 这个位置"接受什么、产出什么"的声明，不含实现 |
| 候选 / Candidate | 一个真实 kernel + 它凭什么能被换上去 |
| 方案 / Plan | agent 给出的、可被程序校验和执行的组合方案 |
| 装配 / Assemble | 把 Plan 变成真实模型上生效的替换 |
| 取证 / Instrument | 运行时记录"谁被调用了多少次"，默认关闭 |
| 闸门 / Gate | 验收的一级，G0 接口、G1 逐点数值、G2 端到端、G3 性能 |
| golden | 原生实现产出的参考结果，是数值验收的标尺 |

## 附录 B：现有代码的去留

| 现有文件 | 处置 | 原因 |
| --- | --- | --- |
| `run_selector.py` 的 prepare/select/run 节奏 | 保留思路，重写 | 三步分离是对的 |
| `run_selector.binding_for()` | 保留，扩展 | 指纹绑定是核心决策 |
| `candidates.catalog()` / `unsupported_reason()` | 重写 | 改为数据 + 可执行 support 函数 |
| `candidates.SelectedActivation` | 删除 | `nn.Module` 包装导致键名变化 |
| `candidates.install()` | 重写 | 改为 `replace_forward` + 硬断言 |
| `agent.select()` 的"只收 ID、不执行代码" | 保留 | 安全边界正确 |
| `agent.SYSTEM_PROMPT` | 重写 | 从"只选一个"改为"检索后出 Plan" |
| `vendor/`（自带来源与哈希） | 保留 | 可追溯性做法正确 |
| `test_selector.py` 中"非法方案必须被拒" | 保留并扩展 | 校验思路正确 |
| `baseline.py` | 保留 | 原生基线仍然有效，G2 的 golden 从它来 |
| `outputs/`、`cache/` | 保持忽略 | 运行产物不入库 |

重写时目录名保留 `nanovllm_selector/`，内部按第 4 章的契约重新组织；`nano-vllm/` 子模块在 P0–P5 期间不修改。
