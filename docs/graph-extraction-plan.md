# 计算图抽取：怎么保证没有东西被悄悄丢掉

> 状态：v2.0（2026-11-15）
> 适用对象：维护 `nanovllm_selector/extract/` 的人
> 前置文档：[model-port-plan.md](model-port-plan.md) 的第 ① 步

---

## 0. 这份文档解决什么

抽取器现在的做法是"跑一次真实前向，边跑边记"。这种做法有一个天生的问题：

**记什么、不记什么，是代码里的 `if` 决定的。写窄了，就是一段计算从图里消失，而图看起来依然完整。**

已经真实发生过一次：`rotary_emb` 同时吐出 cos 和 sin，去重时按"生产者"判重，sin 被当成重复项丢掉。图里 `self_attn` 只有 6 个输入，看起来毫无异常，但按它写出来的 attention 少乘一个 sin，输出全错。

所以本文档只回答一个问题：**怎么让"漏"这件事不可能静默发生。**

---

## 1. 四种漏法

| 漏法 | 具体在哪 | 后果 |
|---|---|---|
| 输入去重写窄 | 按生产者去重，同一模块的多个产出被合掉 | 张量少一个，边少一条 |
| 权重过滤写窄 | 只按对象身份过滤，`weight.t()` 这类视图漏进来 | 图上多出一堆假输入 |
| 透明算子名单写窄 | 遇到没登记的 view/reshape 就 `continue` | 中间张量断链 |
| 算子判定写窄 | `classify_ops` 没命中就标 `unknown` | 节点语义不明，但仍会被当作正常节点 |

共同点：**每一种都是"边扫边判断，不要就扔"，扔的时候没有任何记录。**

---

## 2. 四层防线

| 层 | 查什么 | 靠什么 | 状态 |
|---|---|---|---|
| ① 输入 = 集合差 | 节点的输入集合是否等于"吃进去的 − 自己造的 − 自带的" | 三分桶 + 覆盖等式 | 已完成 |
| ② 三条机械不变量 | 每个算子、每个张量、每个参数位是否都有归属 | 可数的等式 | 已完成 |
| ③ 逐节点重放 | 按图重新算一遍，结果是否和原模型一致 | 独立实现 + 数值比对 | 已完成 |
| ④ decode 与状态 | 跨 step 的 cache 读写在图里是否可见 | 第二次前向 + 合并 trace | 已完成 |

四层查的是**不同**的错误，任何一层都不能替代另一层：

- ① 管"输入集合算错"；
- ② 管"有人被漏掉"；
- ③ 管"语义标签写错"（比如把 `rope_table` 标成 `rope`）；
- ④ 管"只看见了一条执行路径"。

---

## 3. 第一层：输入 = 集合差

### 3.1 为什么不能"扫一遍留下该留的"

节点不是函数。`Qwen3MLP.forward(self, x)` 只有一个参数，但图里的 `L*.mlp.mul` 有两个输入：`act_fn` 的产物和 `up_proj` 的产物，都不是 `x`。

节点的输入不是它"天生"的属性，而是相对节点划分算出来的。划分一旦确定，唯一不依赖主观判断的定义是：

```
输入 = consumed − produced − state
```

- `consumed`：簇内**所有**算子（含透明算子）吃掉的张量；
- `produced`：簇内所有算子产出过的张量（含视图）；
- `state`：模块自己的权重 / buffer / 标量。

### 3.2 三分桶

对每个被消费的张量，判定它属于哪一个桶，只有三种可能：

| 桶 | 判据 |
|---|---|
| internal | 对象在 `produced` 里；或同一存储块上，生产时刻早于消费时刻（视图、切片、原地写） |
| state | 存储块属于该模块自己的参数/buffer |
| input | 其余 |

第二行那个"生产早于消费"是关键：`q` 的切片和 `q` 共享存储块，但 `q` 是在**别的**簇里生产的，所以切片仍然是输入。

### 3.3 断言

```
consumed 的张量总数
  == internal + state + input
```

且三者两两不相交。每个被消费的张量必须**恰好**落进一个桶，没有第四类，也就没有"判断不要就扔掉"的位置。

不满足时记进 `graph.conflicts`，`kind="input_classification"`。

### 3.4 展示层的合并

同一存储块的多个别名只显示成一个输入（比如 `q` 的多种形状），这不影响分类——分类在合并之前完成。

**代码**：`nanovllm_selector/extract/decompose.py` 的 `_Cluster`（存 `consumed`/`produced`）与 `_node_io()`（三分桶）。

---

## 4. 第二层：三条机械不变量

都是可数的等式，不是"看起来对"。

### I1 dispatch 全覆盖

每个 aten 算子必须要么属于某个节点，要么在透明算子白名单里。

```
total == in_nodes + transparent_only + both
```

白名单 `TRANSPARENT` 是显式数据，写在 `decompose.py` 里；加了新算子忘了登记，这里立刻不平。

**实测（Qwen3-0.6B / 8 token）**：total 2412 = 进节点 1359 + 只属透明 628 + 两者都算 425。

### I2 张量有主

每个节点输入必须能连到唯一来源：生产者节点，或一个图输入节点。0 元素的占位张量记在 `attrs.empty_inputs`，不算。

检查方式不是"再算一遍"，而是拿**已经建好的边**去对每个 `_in_srcs` 条目：边必须落在具体张量上。

### I3 入参有槽

每个节点输入的 `index` / `path` / `arg_name` 都不为空。

`path` 是 `args[0]`、`kwargs['position_embeddings'][1]` 这种真实位置，`arg_name` 是模块形参名（合成节点用 aten 的参数名）。少了它，`qkv` 传给谁、`silu` 乘哪一路就全靠猜。

**代码**：`nanovllm_selector/extract/checks.py` 的 `check_invariants()`，由 `api.py` 并入 `graph.reconcile`。

---

## 5. 第三层：逐节点重放

不变量只能保证"没漏"，保证不了"标签对"。`rope_table` 曾经被标成 `rope`——输入输出都在、边也在、槽也对，但语义是错的。

重放要做的：

1. 拿图里每个节点的 `op`、`inputs`、`state`、`attrs`，用一份**独立实现**（解释器）算出结果；
2. 与原模型同一次前向里该节点的实际输出逐元素比对；
3. 全图串起来再比一次 logits。

通过标准：fp32 相对误差 < 1e-5；任何一个节点超差，就定位到具体节点。

这一步同时是"图能不能被别的程序读懂"的检验：如果解释器只能靠偷看源码才能算出结果，说明图里的信息不够。

**已落地**：`nanovllm_selector/extract/replay.py`，每个 op 一份独立实现（matmul / rmsnorm / add / mul / silu / embedding / arange / rope_table / attention / lm_head），只用节点自己的 `op`、`inputs`、`state`、`attrs`。prefill 的 453 个实例节点 + decode 的 452 个，全部对上（fp32 `rtol=1e-5`）。

---

## 6. 第四层：decode 与状态

现在只跑了 prefill。`recorder.py` 里只有一行 `model(input_ids=..., use_cache=True)`，所以 decode 那条分支（`past_key_values` 非空、mask 形状不同、cache 先读再 cat）在图里完全不存在。

要补的不是"再跑一次"，而是：

1. 先跑 prefill 拿到 `DynamicCache`；
2. 用 1 个 token 再跑一次；
3. 两次 trace 合并，decode 独有的节点单独标 `phase="decode"`。

同时补 `state`：KV cache 是节点的**跨 step 状态**，不是普通输入。判据是"这个张量在一次 forward 之外还活着"——它是 `runtime_tensor`，有 `read`/`write` 两种 `effects`。

注意边界：**存储布局（block table、slot_mapping）是引擎的事，不进参考图**；进图的只有"这个节点读写了一份跨 step 的状态"这个事实。

**已落地**：`extract_graph(..., phases=("prefill","decode"))` 会先跑 prefill 拿到 `DynamicCache`，再用 1 个 token 跑第二次，两次 trace 用 `seq_offset` 接在同一根时间轴上。decode 的节点 id 和张量 id 都加 `D.` / `d` 前缀（两个 builder 各自从 `t0000` 编号，不区分会互相覆盖）。

跨 step 状态的判定是机械的：**decode 中某个输入，在 decode 里没有生产者、但在 prefill 的存储块里出现过、且不是参数** —— 它就是 cache。判定后把该输入的 `role` 改成 `state`，并在节点 `state` 里记一条 `kind="runtime_tensor", role="read_write"`。Qwen3 上每层 attention 得到两条（k cache、v cache）。

---

## 7. 不做的事

- **不读源码做静态分析**。名字会骗人（`SiluAndMul` 的实际行为取决于输入是不是打包的），AST 拿不到张量身份。
- **不追求"对所有 transformers 模型可证明正确"**。做不到。目标只是：错了当场炸，炸在具体那一行。
- **不在图里做任何决定**。不合并、不选 kernel、不考虑快不快。图只记录参考实现怎么算。

---

## 8. 当前验收

```
python -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --check
```

20 项全部通过：16 项对账（模块数/参数数/算子数）、3 项不变量、1 项逐节点重放。

单测 26 项：`python -m unittest nanovllm_selector.test_extract`。
