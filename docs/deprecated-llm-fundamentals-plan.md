# 吃透推理代码：LLM 基础知识补全计划

> 目标不是"学过 transformer"，而是**给你一个没见过的推理框架，你能在 2 小时内说出它的执行路径**。
> 教材固定为本仓库的 `nano-vllm/`（1480 行 Python，一个模型，一条链路，没有历史包袱）。
> 训练、多模态、量化部署不在本计划范围内，理由见第 5 节。

**路径约定**：全文 `engine/model_runner.py:112` 这类写法，前缀都是 `nano-vllm/nanovllm/`，即容器内绝对路径 `/workspace/nano-vllm/nanovllm/engine/model_runner.py`（宿主机同一份快照在 `/home/jerryjwc/_reps/kernel-agent-nanovllm/nano-vllm/nanovllm/engine/model_runner.py`，但实验一律以容器内为准）。`example.py`、`bench.py`、`tests/` 相对 `nano-vllm/`。

---

## 0. 怎么用这份文档

### 0.1 三条规则

1. **每个知识点必须有一个你能亲手跑出来的验证**。只读不跑的理解是假的：你能背出"残差是延迟相加"，但改一行代码就崩，说明没懂。
2. **验收标准是可判定的**。每阶段的"验收"是一组具体问题 + 一份要交的产物（脚本或笔记文件）。答不出来就是没过，不要跳。
3. **不许"用术语解释术语"**。你写的每条笔记，如果里面出现一个你没亲手验证过的名词，就把它标红，下一轮先解决它。

### 0.2 环境

- 容器：`nanovllm-fundamentals`（镜像 `jerryjwc/rocm-dev:10.0`），工作目录 `/workspace`
- 卡：AMD MI100（gfx908 / CDNA1），torch `2.12.0+rocm10.0.0`；`torch.cuda.device_count()` 会看到 8 张卡，本计划只跑单卡（`tensor_parallel_size=1`），**不要主动开 TP**
- 模型：`/workspace/tmp/models/Qwen3-0.6B`，已建软链 `~/huggingface/Qwen3-0.6B`（`example.py:8` 写死这个路径）
- 代码：`/workspace/nano-vllm`，已 `pip install -e`，改 `nanovllm/` 下的文件**不需要重装**

**隔离方式：容器里没有任何 bind mount。** 代码、模型、`references/reps/vllm`、`docs/` 都是 `docker cp` 复制进容器层的，容器内怎么改、怎么删都不碰宿主机。旧的 `nanovllm-selector-rocm-dev` 走的是挂载，而且 `nano-vllm/` 还是**只读挂载**（改不动），那是给 selector 项目用的，**这份计划不要用它**。

容器是一次性创建的，命令留在这里备查（镜像可复用，仓库更新后重新 `docker cp` 即可）：

```bash
docker run -d --name nanovllm-fundamentals \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size=16g -w /workspace \
  jerryjwc/rocm-dev:10.0 sh -c 'exec sleep infinity'

docker cp nano-vllm/.                       nanovllm-fundamentals:/workspace/nano-vllm/
docker cp tmp/models/Qwen3-0.6B             nanovllm-fundamentals:/workspace/tmp/models/
docker cp references/reps/vllm              nanovllm-fundamentals:/workspace/references/reps/
docker cp docs/.                            nanovllm-fundamentals:/workspace/docs/
```

`/workspace/nano-vllm` 已经做过一次 `git init` + 基线提交。练习时插的 `print`、改的行，`git diff` 能看到，`git checkout -- <file>` 能还原；改乱了不想收拾，直接 `docker rm -f nanovllm-fundamentals` 重建。

笔记写在容器内 `/workspace/docs/notes/`。容器删了笔记就没了，阶段验收前用 `docker cp nanovllm-fundamentals:/workspace/docs/notes ./notes` 导出一份到宿主机留档。

### 0.3 这块卡上的四个"开关"（先记住，第 8 阶段会回来解释）

| 能力 | 判定处 | MI100 上的实际值 | 后果 |
| --- | --- | --- | --- |
| flash attention | `layers/attention.py:15` | **关**（没装 flash_attn） | 走 `forward_torch` 数学路径 |
| CUDA graph | `engine/model_runner.py:35-37` | **关**（上面的检查把 `enforce_eager` 置 True） | `capture_cudagraph` 是死代码 |
| `torch.compile` | `utils/compilation.py:12` | **开**（capability[0]=9 ≥ 8） | SiluAndMul/RMSNorm/RoPE/Sampler 走编译版 |
| Triton 写 KV cache | `layers/attention.py:54` | **开** | 走 `store_kvcache_kernel` |

**这张表是整份计划里最容易被忽略又最容易踩坑的东西**：网上讲 vLLM 的文章一半在讲 CUDA graph 和 flash attention，而你这块卡上这两条路都不走。你读到 `model_runner.py:206` 那行 `if is_prefill or self.enforce_eager or ...` 时必须知道：在你机器上它**恒真**。

### 0.4 时间预算

总计 **53 小时**，按每天 2–3 小时算是 3–4 周。阶段 0–4 是地基（25h），必须按顺序；阶段 5–8 可以按需调换（见 2.2）。

---

## 1. 一页纸：这个框架长什么样

先建立骨架，细节后面每个阶段填一块。

```
LLM.generate()                            engine/llm_engine.py:60
 └─ while not is_finished():               :73
     └─ step()                            :49
         ├─ scheduler.schedule()          engine/scheduler.py:25    → 选出这一轮跑哪些请求
         ├─ model_runner.call("run", ...) engine/model_runner.py:224
         │   ├─ prepare_prefill/decode    :138 / :181              → 把 Sequence 翻译成 GPU 张量 + 填全局 Context
         │   ├─ run_model                 :205                     → self.model(input_ids, positions)
         │   │   └─ Qwen3ForCausalLM.forward  models/qwen3.py:205  → 28 层
         │   │       └─ compute_logits    :212                     → lm_head
         │   └─ sampler                   layers/sampler.py:10     → logits → token_id
         └─ scheduler.postprocess()       engine/scheduler.py:81   → 追加 token、判断结束、还块
```

四个必须在第一天就分清的抽象：

| 抽象 | 是什么 | 住在哪 |
| --- | --- | --- |
| `Sequence` | **一个请求**的状态机：token 序列、块表、采样参数 | `engine/sequence.py:14` |
| `Scheduler` | **这一轮跑谁**：prefill 队列/decode 队列、token 预算、抢占 | `engine/scheduler.py:25` |
| `BlockManager` | **显存在哪**：块的分配/释放/前缀缓存哈希 | `engine/block_manager.py:26` |
| `Context` | **这一轮的形状参数**：全局单例，模型层从中读 `slot_mapping` 等 | `utils/context.py:6` |

**关键结构洞察**：模型 `forward(input_ids, positions)` 只吃两个参数（`models/qwen3.py:205`），`attention mask`、KV cache 位置、序列边界**全部走全局 `Context`**。这是这份代码最重要的设计选择，也是你以后读任何推理框架都会再见到的模式。

---

## 2. 学习顺序与理由

### 2.1 推荐顺序

```
阶段 0  环境与跑通           2h   ← 建立"能改能跑"的反馈环
阶段 1  nn.Module 工程机制    6h   ← 所有模型代码的语法基础
阶段 2  模型 = 模块树 + 权重流 5h   ← 静态结构：谁是谁、权重从哪来
阶段 3  张量形状与内存布局    5h   ← 静态结构：一个张量到底长什么样
阶段 4  一次 forward 的数据流 7h   ← 动态：端到端跑通一次
阶段 5  KV cache 与分页       8h   ← 动态：为什么 decode 能加速
阶段 6  批处理与调度          6h   ← 动态：多个请求怎么共存
阶段 7  权重加载 + TP 分片    6h   ← 并行：一张卡跑不下怎么办
阶段 8  compile / graph / 性能 8h   ← 外壳：为什么快、哪里慢
```

### 2.2 为什么是这个顺序

- **为什么先 1 再 2**：`qwen3.py` 全是 `nn.Module` 嵌套。你不懂"给属性赋值会自动登记子模块"，读到 `self.act_fn = SiluAndMul()` 只会觉得是普通赋值，后面所有"参数为什么在子模块里"的疑问都解释不了。
- **为什么先 2 再 4**：读 `forward` 时你会不断卡在"`self.weight` 是从哪来的"——这是**权重流**，它和数据流是两条独立的线。先把权重放好，再读数据怎么流。
- **为什么 3 夹在中间**：形状是连接的凭证。`qkv_proj` 输出 4096、`split` 成 2048/1024/1024、`view` 成 `[-1,16,128]`，这三步的形状账算不清，后面 attention 一定读错。
- **为什么 4 在 5 前**：decode 的所有复杂性都建立在"prefill 已经把 K/V 写进 cache"这个前提上。先看单请求一次 forward 的完整链路，再看第二次 forward 怎么复用上一次的结果。
- **为什么 6 在 5 后**：调度器算的是"块够不够"，不先懂块和 `block_table`，`can_append` 那行 `len(seq) % block_size == 1` 看起来就像魔法。
- **为什么 8 最后**：`torch.compile` 和 CUDA graph 是**性能外壳**，不理解语义也能读，但会误判——比如以为 `graph.replay()` 一定在跑（在你这块卡上它没跑）。地基不稳时学这个，只会把"读到的代码"和"实际执行的代码"混为一谈。

### 2.3 哪些可以先跳过

**第一遍读的时候**，下面这些看到就跳过，别停：

- `engine/block_manager.py:36-41`（xxhash 前缀哈希的具体算法）——知道"同一前缀算出同一个 hash"就够了
- `engine/model_runner.py:70-98`（`read_shm`/`write_shm`/`loop`）——TP=1 时全是死代码
- `layers/attention.py:99-136`（`forward_torch` 的数学实现）——第一遍只需要知道"它是没有 flash attention 时的兜底"
- `utils/compilation.py`（16 行）——第一遍只需要知道"它决定用不用编译版"

---

## 3. 阶段计划

### 阶段 0：环境与跑通（2h）

**目标**：能在 30 秒内跑出一个结果，能在任何一行打 `print` 并看到输出。

**要读的代码**
- `example.py:1-39`（39 行，全读）
- `engine/llm_engine.py:17-41`（构造与析构）
- `config.py:6-25`（所有可调参数）

**动手练习**

1. 跑 `python example.py`，记录输出。
2. 用 `python -c "from nanovllm import LLM; ..."` 构造一个 `LLM`，打印 `llm.model_runner.enforce_eager`。**预测值再验证**——预测错了说明第 0.3 节的表没进脑子。
3. 在 `engine/model_runner.py:207` 那行 `return self.model.compute_logits(...)` 上面插一句 `print("run_model", input_ids.shape, is_prefill)`，重跑。数一下 prefill 打了几次、decode 打了几次。
4. 把 `example.py:16` 的 `max_tokens=256` 改成 `8`，观察 `step()` 被调用的次数变化。

**自测问题**（能答出来才算过）
- `LLM(...)` 这一行里，模型权重是什么时候加载的？KV cache 是什么时候分配的？顺序能不能换？
- `example.py:11` 那个 `if capability < 8` 分支在你的卡上走哪边？为什么？
- `llm_engine.py:35` 的 `atexit.register(self.exit)` 和你手动 `llm.exit()` 同时发生会怎样？

**验收**
- 交一份 `notes/00-env.md`，包含：`enforce_eager` 实测值、prefill/decode 打印次数、你对第 3 个问题的实测结果（贴 traceback 也行）。

**预计**：2h（其中 1h 在环境上）

---

### 阶段 1：`nn.Module` 的工程机制（6h）

**目标**：彻底搞清"对象、参数、buffer、普通属性"四者的区别，以及 `state_dict` 的键是怎么拼出来的。这是你目前最薄的一环，也是读一切 PyTorch 代码的地基。

**要读的代码**
- `layers/layernorm.py:7-16`（三种东西同时出现：`self.eps` float、`self.weight` Parameter、无 buffer）
- `layers/activation.py:8-13`（完全无参数的 Module）
- `layers/rotary_embedding.py:19-37`（`register_buffer(..., persistent=False)`）
- `layers/attention.py:63-77`（`self.k_cache` 是普通属性，第 77 行初始化为空 tensor）
- `engine/model_runner.py:124-130`（用 `hasattr` 遍历挂载 kv_cache）

**动手练习**

1. **复现三本账**。手写一个 20 行的 `MiniModule`，只实现 `__setattr__`（拦截 `nn.Parameter`/`nn.Module`/普通值）和 `parameters()`/`named_parameters()`。然后：
   ```python
   class A(nn.Module):
       def __init__(self):
           super().__init__()
           self.w = nn.Parameter(torch.ones(3))
           self.eps = 1e-6
           self.sub = nn.Linear(3, 3)
   a = A()
   print(list(a.named_parameters()))   # 预测：几条？名字是什么？
   ```
2. **验证"改名"事故**。你之前踩过的坑，现在主动复现一次：
   ```python
   class Wrapper(nn.Module):
       def __init__(self, inner):
           super().__init__()
           self.inner = inner        # 注意：这行会把它登记为子模块
   w = Wrapper(nn.Linear(3, 3))
   print(list(w.state_dict().keys()))   # 预测：weight 还是 inner.weight？
   ```
   然后改成 `object.__setattr__` 或 `self._inner = [inner]`，看键怎么变。**这是"包装一个 Module"永远危险的根本原因。**
3. **`state_dict` 键的来源**：对真模型跑
   ```python
   from nanovllm.models.qwen3 import Qwen3ForCausalLM
   from transformers import AutoConfig
   cfg = AutoConfig.from_pretrained("/workspace/tmp/models/Qwen3-0.6B")
   m = Qwen3ForCausalLM(cfg)
   ks = list(m.state_dict().keys())
   print(len(ks), ks[:4], ks[-2:])
   ```
   预测键的数量（提示：227 个参数，但 tied embedding 会不会出现两次？），再验证。第 1 层的 8 个键逐个和 `qwen3.py:127-144` 的属性名对上。
4. **hooks 三件套**。在 `mlp.act_fn` 上挂 `register_forward_pre_hook`、`register_forward_hook`，打印输入/输出形状。再写一个 pre_hook **返回一个 tensor**，观察它是否替换了输入。
5. **`inference_mode` 和 buffer 的坑**：在 `run()` 外面（`model_runner.py:223` 的装饰器）临时把 `torch.inference_mode()` 去掉，看能不能跑；再试着在 `inference_mode` 里给 `self.k_cache` 赋值，观察是否报 "Inplace update to inference tensor"。

**自测问题**
- `Qwen3MLP` 容器自己有几个参数？它子树里有几个？(`qwen3.py:91-117`)
- `RMSNorm.eps` 在 `state_dict` 里吗？为什么？(`layernorm.py:15`)
- `RotaryEmbedding.cos_sin_cache` 为什么不在 `state_dict` 里，但仍然是 buffer？(`rotary_embedding.py:37`)
- `Attention.k_cache` 为什么既不在 `parameters` 也不在 `buffers`？那 `model_runner.py:126` 是怎么找到它的？
- 一个模块对象被 `@lru_cache` 共享后，模块树里出现几次？(`rotary_embedding.py:53`)

**验收**
- 交 `notes/01-module.md`：包含第 2 题的两次 `state_dict` 输出、第 3 题的键数量与"tied embedding 出现几次"的结论、第 4 题 pre_hook 返回 tensor 的结果。
- 能不看代码画出"给 `self.x` 赋值时 `nn.Module.__setattr__` 做判断的流程图"。

**预计**：6h

---

### 阶段 2：模型 = 模块树 + 权重流（5h）

**目标**：能在 10 秒内回答"这个参数属于哪一层、名字叫什么、为什么是这个名字"，并理解权重加载和模型结构是**两件独立的事**。

**要读的代码**
- `models/qwen3.py:186-216`（入口 + `packed_modules_mapping` + tied embedding）
- `models/qwen3.py:120-159`（DecoderLayer）
- `utils/loader.py:12-28`（28 行，全读，这是权重流的全部）
- `layers/linear.py:12-35`（`LinearBase.__init__`，注意第 25-26 行的 `weight_loader` 挂载）
- `layers/embed_head.py:9-32`（vocab 分片 + loader）

**动手练习**

1. **数一遍**。构造模型后跑：
   ```python
   from collections import Counter
   print(len(list(m.modules())), len(list(m.parameters())))
   print(Counter(type(x).__name__ for x in m.modules()).most_common())
   ```
   对照"371 个模块 / 227 个参数"核对，并解释为什么 RMSNorm 是 113 个（提示：`qwen3.py:69-70` 有 2 个，`qwen3.py:143-144` 有 2 个，`qwen3.py:171` 有 1 个）。
2. **权重流追踪**。在 `utils/loader.py:16` 的 `for weight_name in f.keys()` 里加 `print(weight_name)`，只印前 10 个。然后对 `model.gate_proj.weight`（checkpoint 里的名字）手工走一遍第 17-24 行：`k` 匹配到什么？`v, shard_id` 是什么？`param_name` 变成了什么？最后落到哪个参数的哪个切片？
3. **`packed_modules_mapping` 只影响加载，不影响 forward**。把 `qwen3.py:191-192` 两行注释掉（只留 `q/k/v`），重新加载会怎样？报错信息说明什么？
4. **tied embedding**。验证 `qwen3.py:203` 之后 `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`，并解释：既然是同一块存储，为什么 `state_dict` 里可能只有一个键？
5. **画出第 0 层的模块树**（纸或 mermaid），标注每个节点的参数数。验收标准：整棵树加起来正好 8 个参数。

**自测问题**
- `qkv_proj.weight` 的形状是 `(4096, 1024)`，4096 是怎么算出来的？(`linear.py:111` + `qwen3.py:42-48`)
- 为什么 `o_proj` 是 `RowParallelLinear` 而 `qkv_proj` 是 `ColumnParallelLinear`？两者的 `weight` 形状在 TP=2 时差在哪？(`linear.py:63` vs `:140`)
- `model.get_parameter("layers.0.mlp.gate_up_proj.weight")` 能取到吗？`get_parameter` 是怎么靠名字走树的？
- `load_model` 遍历的是 **safetensors 里的键**，不是模型的参数。如果 checkpoint 里有一个键在模型里找不到，会在哪一行崩？

**验收**
- 交 `notes/02-tree.md`：第 0 层模块树图 + 参数计数表 + 第 3 题的报错原文。
- 口头（或笔记）能说清：**数据流**（`forward` 的入参和返回值）和**权重流**（`self.weight` 从 checkpoint 到参数张量）是两条不相交的线。

**预计**：5h

---

### 阶段 3：张量形状与内存布局（5h）

**目标**：看到一个张量，能说出"它有几个维度、每维什么意思、是不是个 view、底层存储多大"。这一阶段直接决定你能不能读 `qwen3.py:77-88` 那 12 行。

**要读的代码**
- `models/qwen3.py:72-88`（12 行，全程形状账）
- `layers/linear.py:152-156`（`F.linear` 不做转置，weight 是 `(out, in)`）
- `layers/rotary_embedding.py:8-16`（`chunk` + `cat`，形状不变）
- `engine/model_runner.py:121-124`（KV cache 的 6 维形状）
- `layers/attention.py:47-60`（`store_kvcache` 的三个 `assert stride`）

**动手练习**

1. **手算一遍前向的形状账**。对 batch=1、序列长 N 的 prefill：
   ```
   hidden_states        [N, 1024]
   qkv_proj 输出        [?]        ← 4096 是什么
   split → q,k,v        [?], [?], [?]
   q.view(-1,16,128)    [?]        ← -1 是多少
   q_norm(q)            [?]        ← RMSNorm 在 3D 上归一化的是哪一维（layernorm.py:25）
   rotary_emb           [?]
   attn 输出            [?]        ← 注意 k 的 head 数是 8 不是 16
   o.flatten(1,-1)      [?]        ← flatten 把哪两维压了
   o_proj 输出          [N, 1024]
   ```
   每一行都要写数字，然后用代码打印验证。
2. **view / chunk / split 是不是共享存储**。对 `qkv` 做 `split`，改 `q[0,0,0]`，看 `qkv` 变不变；对 `cos_sin.chunk(2,-1)` 做同样测试。用 `t.data_ptr()` 和 `t._base` 判断。
3. **stride 检查**。对 `k`（`[N,8,128]`）和 `k_cache`（`[num_blocks,256,8,128]`）打印 `stride()`，对照 `attention.py:50-53` 的三个 assert，说出它们各自在防什么错。
4. **dtype 混用**。`cos_sin_cache` 是 fp32（`rotary_embedding.py:31-36` 用的是 `torch.float`），而 `q` 是 bf16。`apply_rotary_emb` 第 13 行先 `.float()` 再算。如果没有这次转换，误差会差多少？写 5 行代码在 GPU 上量化一下。
5. **KV cache 的 6 维**。`torch.empty(2, layers, blocks, 256, kv_heads, head_dim)`（`model_runner.py:124`）。写出 `k_cache`（`kv_cache[0, i]`）的形状和每个维度的含义，并算出 234 个块时这块显存占多少 GB。

**自测问题**
- `head_dim` 是 128，但 `hidden_size / num_attention_heads = 1024/16 = 64`。这两个数为什么不同？(`qwen3.py:36`)
- `q.view(-1, self.num_heads, self.head_dim)` 里的 `-1` 在 prefill 和 decode 下分别是多少？
- `o.flatten(1,-1)` 和 `o.reshape(-1, 2048)` 在这个上下文里等价吗？有什么前提？
- `qkv.split(...)` 出来的三个 tensor 是 3 份拷贝还是 1 份存储的 3 个视图？

**验收**
- 交 `notes/03-shapes.md`：第 1 题的完整形状表（含实测打印）+ 第 5 题的显存计算。
- 能在白板上 3 分钟讲完 `qkv_proj → attention` 的形状变化。

**预计**：5h

---

### 阶段 4：一次 forward 的完整数据流（7h）

**目标**：从 `input_ids` 一路走到 `logits`，每一跳都知道发生了什么、为什么这么写。重点吃透**残差的延迟相加**和**RMSNorm 的三种形态**——这两处是这份代码最"反直觉"的地方。

**要读的代码**
- `models/qwen3.py:173-183`（`Qwen3Model.forward`，11 行，核心在 residual 的初值）
- `models/qwen3.py:146-159`（DecoderLayer，注意第 152-155 行的两个分支）
- `layers/layernorm.py:18-52`（三种 forward 路径）
- `models/qwen3.py:205-216`（入口 + `compute_logits`）
- `layers/embed_head.py:56-66`（prefill 时只取每个序列的最后一个位置）

**动手练习**

1. **残差对照实验**（本阶段最重要）。用纯 torch 手写一份"标准 pre-norm"参考实现（每层 `x = x + attn(norm(x))`），再跑 nano-vllm 的版本，同一组权重、同一输入，比较输出。**预测：两者应该逐位相等还是只近似相等？** 然后解释为什么 `qwen3.py:153` 是 `hidden_states, residual = self.input_layernorm(hidden_states), hidden_states`——注意它把**未归一化的输入**存进 residual，把归一化结果作为下一层的输入。
2. **残差什么时候加回去**。在 `qwen3.py:182` 之前打印 `hidden_states.abs().mean()` 和 `residual.abs().mean()`，确认最后一步 `self.norm(hidden_states, residual)` 里 `residual is not None`。然后把第 182 行改成 `self.norm(hidden_states)`，看输出差多少。
3. **RMSNorm 的三种形态**。分别在三个位置挂 hook 打印输入形状：
   - 第 0 层 `input_layernorm`（`residual=None`，走 `rms_forward`）
   - 第 1 层 `input_layernorm`（双输入，走 `add_rms_forward`）
   - 第 0 层 `q_norm`（3D 输入，归一化最后一维）
   确认：**同一个类，三种完全不同的语义**。
4. **fp32 累加路径**。`layernorm.py:23-27` 是 `x.float() → 算 → 转回 bf16`。把第 24 行改成不做 `.float()`，比较 logits 最大绝对误差。（这正好呼应你们之前 baseline 对照里 `0.6875` 的误差量级——想想那点误差可能来自哪里。）
5. **prefill 只取最后位置**。`embed_head.py:58-60` 的 `last_indices = cu_seqlens_q[1:] - 1`。给一个 batch=2、长度 [3,5] 的 prefill，算出 `last_indices` 应该是多少，验证，并解释为什么 decode 不需要这一步。

**自测问题**
- 为什么第 0 层不需要做残差相加，而后续层需要？如果统一写成一个分支会怎样？
- `Qwen3Model.forward` 最后一行 `hidden_states, _ = self.norm(...)` 里的 `_` 是什么？为什么扔掉？
- `compute_logits` 在 `ModelRunner` 里是模型外单独调的（`model_runner.py:207`），为什么不像 HF 那样放进 `forward`？
- 28 层里，`hidden_states` 的形状从头到尾变过吗？

**验收**
- 交 `notes/04-forward.md`：手写参考实现的代码 + 与纳米的逐元素对比结果（最大误差、是否 bitwise 相等）+ 第 3 题三个形状。
- 能脱稿讲一遍"从 `input_ids` 到 28 层输出"的完整链路，中间不许看代码。

**预计**：7h

---

### 阶段 5：KV cache 与分页 attention（8h）

**目标**：能解释"为什么 decode 每步只算一个 token 却要知道全部历史"，能从 `Context` 的每个字段反推出这一轮 batch 的真实形状。你已经知道 KV cache 的大意，这一阶段补的是**工程细节**：块怎么分、索引怎么算、写和读怎么分开。

**要读的代码**
- `engine/model_runner.py:112-136`（分配 + 挂载）
- `engine/model_runner.py:138-197`（`prepare_prefill` / `prepare_decode`，逐行）
- `layers/attention.py:24-61`（Triton 写 cache）
- `layers/attention.py:79-136`（三种 forward 路径）
- `utils/context.py:5-27`（27 行，全读）

**动手练习**

1. **打印真值**。在 `prepare_prefill` 和 `prepare_decode` 末尾各加一句打印全部 context 字段，跑 `example.py` 的双 prompt，把每一轮的字段抄下来。然后用这些数字回答：
   - 为什么 prefill 的 `input_ids` 是 120 个数，decode 是 2 个数？
   - `slot_mapping` 的长度和 `input_ids` 的关系是什么？
   - prefill 时 `block_tables` 是 `None`，什么时候不是？
2. **手算 slot**。请求 A 有 100 个 token，被分到块 5；请求 B 有 20 个 token，被分到块 9。写出 prefill 那一轮的 `slot_mapping`（前 5 个 + 后 3 个），和 A 下一次 decode 的 `slot_mapping`。用 `model_runner.py:160-170` 和 `:190` 的公式核对。
3. **写一个 numpy 版 paged attention**。50 行以内：输入 `q[1,16,128]`、`k_cache[blocks,256,8,128]`、`block_tables[1,k]`、`context_lens[1]`，输出 `[1,16,128]`。和 `attention.py:99-136` 的 `forward_torch` 对同样的输入比，误差应该 < 1e-2。
4. **GQA 的 repeat**。`attention.py:120-123` 用 `repeat_interleave` 把 8 个 KV head 扩成 16 个。验证 `num_heads // num_kv_heads == 2` 意味着 query head 0、1 共用 KV head 0。把它改成 `repeat`（不带 interleave）会错成什么样？写出错误的结果形状。
5. **前缀缓存命中**。同一条 prompt 跑两次（两次 `generate` 调用之间不要重建 LLM），观察第二次的 `num_cached_tokens` 和 `cu_seqlens_k > cu_seqlens_q` 的情况（`model_runner.py:171`）。

**自测问题**
- `slot_mapping` 和 `block_tables` 一个是"写地址"一个是"读目录"，具体到张量形状上差在哪？
- 为什么 Triton kernel 里 `slot == -1` 要直接 `return`（`attention.py:37`）？谁会传 -1 进来？
- `can_use_flash_attention` 返回 False 时，`forward` 走哪条路？这条路在 decode 时为什么要把结果 `unsqueeze(1)`（`attention.py:136`）？
- KV cache 是 28 层各一份，还是所有层共享一份大张量？(`model_runner.py:124-130`)
- `context_lens` 是"已经有的长度"还是"含当前 token 的长度"？从 `prepare_decode:189` 反推。

**验收**
- 交 `notes/05-kvcache.md`：两轮 context 全字段抄录 + slot 手算 + numpy 版 paged attention 代码与误差。
- 能在纸上画出"一个 token 的 K/V 从算出来到写进 cache 再到被下一个 token 读出来"的完整路径。

**预计**：8h

---

### 阶段 6：批处理与调度（6h）

**目标**：搞清"多个请求同时来"时，框架怎么决定这一轮跑谁、跑多少、块不够时怎么办。这是推理框架和"跑一个模型"最大的区别。

**要读的代码**
- `engine/scheduler.py:25-73`（`schedule`，prefill 循环 + decode 循环）
- `engine/scheduler.py:81-92`（`postprocess`）
- `engine/block_manager.py:58-108`（分配/追加/回收）
- `engine/sequence.py:55-70`（`num_blocks`、`last_block_num_tokens`）
- `engine/llm_engine.py:49-55`（`step` 的返回值约定）

**动手练习**

1. **让 chunked prefill 真的发生**。把 `max_num_batched_tokens` 设成 128，喂一个 300 token 的 prompt。观察：跑了几轮 prefill？每轮 `num_scheduled_tokens` 是多少？`scheduler.py:42` 那行注释"only allow chunked prefill for the first seq"在你这个场景下生效了吗？
2. **让抢占真的发生**。把 `num_kvcache_blocks` 调到很小（比如 4），同时发 3 个长请求。观察 `preempt`（`scheduler.py:75`）被调用，记录被抢占的 seq 的 `num_cached_tokens` 怎么变。
3. **前缀缓存命中率**。发 5 个共享同一段 500 token 前缀的请求，在 `block_manager.py:75` 的 `allocate` 里打印 `num_cached_blocks`，算命中率。然后把 `hash_blocks`（`:110`）注释掉，再看命中率——**预测会变成多少？**
4. **一次 step 的账**。给 `llm_engine.py:49` 加计时，分别统计 prefill step 和 decode step 的耗时，画出"prompt 长度 vs prefill 时间"和"batch size vs decode 时间"两张表。

**自测问题**
- 一个 step 里能不能既有 prefill 又有 decode？为什么？（看 `scheduler.py:54-55` 的 early return）
- `can_append`（`:104`）里 `len(seq) % block_size == 1` 这个条件在判断什么？为什么是 `== 1` 不是 `== 0`？
- 被抢占的请求重新进来时，为什么能复用之前的 `num_cached_tokens`？
- `postprocess` 里先 `hash_blocks` 再 `append_token`，顺序能换吗？

**验收**
- 交 `notes/06-scheduling.md`：四张实测表（chunked prefill 轮次、抢占记录、命中率前后对比、耗时表）。
- 能解释"为什么这个调度器是单层循环而不是优先级队列"，以及它在什么负载下会退化。

**预计**：6h

---

### 阶段 7：权重加载与 TP 分片（6h）

**目标**：理解"一张卡放不下时怎么切"，以及切完之后**通信点在哪**。这一阶段大部分是读，因为 TP>1 需要多卡。

**要读的代码**
- `layers/linear.py:54-156`（`Column` / `MergedColumn` / `QKV` / `Row` 四种子类）
- `layers/embed_head.py:9-66`（vocab 维切分 + `all_reduce` / `gather`）
- `engine/model_runner.py:19-31`（`dist.init_process_group`）与 `:50-57`（shm）
- `engine/llm_engine.py:22-31`（spawn 子进程）
- `engine/sequence.py:72-83`（`__getstate__` 跨进程只传必要字段）

**动手练习**

1. **TP=1 时 dist 也在跑**。打印 `dist.get_rank()`、`dist.get_world_size()`，确认即使单卡也初始化了 NCCL（`model_runner.py:28`）。解释为什么这份代码不写"if TP>1 才 init"。
2. **手算分片偏移**。TP=2 时，`QKVParallelLinear` 的 `qkv_proj.weight` 形状是多少？第 114-128 行的 `weight_loader` 对 `loaded_shard_id="k"` 算出的 `shard_offset` 和 `shard_size` 分别是多少？用代码验证。
3. **通信点清单**。全文搜 `all_reduce` 和 `gather`，列出所有位置，并解释每一处为什么**必须**通信（提示：`RowParallelLinear` 切的是输入维，所以输出要合并）。
4. **跨进程传什么**。读 `sequence.py:72-83`，说明 prefill 和 decode 分别传了哪些字段，为什么 decode 只传 `last_token` 就够了。
5. （可选，需要 2 张卡）真的起 `tensor_parallel_size=2`，对比生成结果和 TP=1 是否一致。**注意**：`llm_engine.py:24` 用 `spawn`，任何在父进程打的 monkeypatch 都不会传给子进程——这正是你们 demo 里那个安装器在 TP>1 时会静默失效的原因。

**自测问题**
- `ColumnParallelLinear` 切输出维、`RowParallelLinear` 切输入维。为什么是这个组合而不是反过来？
- `MergedColumnParallelLinear.weight_loader` 里 `shard_offset = sum(output_sizes[:id]) // tp_size` 中的整除为什么是安全的？
- `VocabParallelEmbedding.forward` 里那个 `mask`（`embed_head.py:36`）在防什么？
- `lm_head` 的 `gather` 只在 rank0 拿到完整 logits，那 rank1 的 sampler 跑什么？（看 `model_runner.py:226-228`）

**验收**
- 交 `notes/07-tp.md`：分片形状手算表 + 通信点清单 + 第 4 题的字段表。
- 能画出 TP=2 时一次 forward 里"哪里切、哪里通信、哪里拼"的图。

**预计**：6h

---

### 阶段 8：`torch.compile`、CUDA graph 与性能分析（8h）

**目标**：分清"语义代码"和"执行优化"，并能在 profiler 里定位瓶颈。这一阶段要反复回到第 0.3 节那张表——因为这块卡上两个重要机制是关的。

**要读的代码**
- `utils/compilation.py:1-16`（16 行，全读）
- `engine/model_runner.py:205-221`（`run_model` 的三分支）与 `:232-267`（`capture_cudagraph`）
- `layers/attention.py:99-136`（math 兜底为什么不能进 graph）
- `bench.py:1-32`（32 行，全读）

**动手练习**

1. **证明 CUDA graph 在这台机器上是死代码**。构造 `LLM` 后打印 `enforce_eager`，并在 `capture_cudagraph`（`:233`）第一行插 `print("captured")`。跑 example，确认一次都没打印。然后解释 `:206` 那个 `if` 为什么恒真。
2. **`torch.compile` 到底编译了什么**。在 `compilation.py:14` 的 `compiled(*args, **kwargs)` 前后加计时，跑 10 次 forward，打印第一次（含编译）和后面几次的耗时。再看 `TORCH_LOGS="graph"` 环境变量能打印出多少个编译图。
3. **`compile_on_supported_gpu` 的隐藏开销**。第 11-12 行每次调用都要遍历 args 找 tensor、再问一次 `get_device_capability`。写 10 万次循环量一下这个 wrapper 的固定开销。**这正好是你们 demo 里 `unsupported_reason` 每层 4 µs 的同类问题。**
4. **profiler 抓 decode**。用 `torch.profiler` 抓一个 decode step，按 kernel 时间排序，列出前 10 个，并标注每个属于"矩阵乘 / 归一化 / 写 cache / 采样"哪一类。算一下矩阵乘占比——这就是 roofline 分析的入口。
5. **带宽估算**。decode 一步的显存流量 ≈ 权重（0.6B × 2 字节 ≈ 1.2 GB）+ KV cache 读取。用 MI100 的 HBM 带宽（约 1.2 TB/s）算出理论下限，和实测对比，差几倍？

**自测问题**
- CUDA graph 捕获时，Python 代码只执行一次。那么 `get_context()` 在 replay 时返回什么？（这解释了为什么 graph 版本必须靠 `graph_vars` 手动拷贝）
- 为什么 `forward_torch`（`attention.py:99`）里有一堆 `.tolist()` 和 Python 循环，却"正好"适合 eager 而不适合 graph？
- `is_prefill or enforce_eager or bs > 512`（`:206`）这三个条件各自在防什么？
- 如果给你一块有 flash attention 的卡，`bench.py` 的哪几行行为会变？

**验收**
- 交 `notes/08-perf.md`：graph 死代码的证据 + profiler 前 10 kernel 表 + 带宽估算与实测对比。
- 能一句话回答"这份代码在 MI100 上实际执行的是哪条路径"。

**预计**：8h

---

## 4. 你最容易误解的 12 个点

这些全部来自你已经暴露过的困惑，或者代码里最反直觉的地方。**每一条都必须亲手验证一次**，验证方法写在最后一列。

| # | 你以为 | 实际 | 怎么验证 |
| --- | --- | --- | --- |
| 1 | `Qwen3MLP` 是个 MLP，里面有参数 | 容器自己 0 个参数，2 个参数全在 `gate_up_proj`/`down_proj` 子模块里 | `len(list(m.mlp.parameters()))` vs `len(list(m.mlp.named_children()))` |
| 2 | `self.weight` 是 forward 的参数传进来的 | 权重流和 `forward` 的入参是**两条独立的路**，`self.weight` 直接从模块属性读 | 在 forward 里打印 `self.weight.shape`，再看函数签名只有 `x` |
| 3 | `head_dim = hidden_size / num_heads` | 是 128 而不是 64，`head_dim` 来自 config，不能反推 | `qwen3.py:36`；打印 `config.head_dim` |
| 4 | `state_dict` 的键由类名决定 | 由**属性名 + 注册路径**决定，包一层就变 | 阶段 1 练习 2 |
| 5 | `Attention.k_cache` 是 buffer | 是普通属性，靠 `hasattr` 遍历找到（`model_runner.py:127`） | `"k_cache" in dict(m.named_buffers())` → False |
| 6 | `lru_cache` 共享的是"计算结果" | 共享的是**模块对象本身**，28 层指向同一个 `RotaryEmbedding` | `m.layers[0].self_attn.rotary_emb is m.layers[1].self_attn.rotary_emb` → True |
| 7 | prefill 的 `input_ids` 是"每条请求一个 tensor" | 是**所有请求拼接成一个一维张量**，边界靠 `cu_seqlens` | 阶段 5 练习 1 |
| 8 | `slot_mapping` 和 `block_tables` 是一回事 | 前者是写地址（一维），后者是读目录（二维） | 打印两者形状 |
| 9 | `enforce_eager=False` 就会用 CUDA graph | 在这块卡上被强制改成 True，graph 根本不跑 | 阶段 8 练习 1 |
| 10 | `torch.compile` 只在"老卡"上被跳过 | 逻辑反了：`capability < 8` 才跳过，MI100 是 9，所以**在用** | `compilation.py:12`；打印实际走哪支 |
| 11 | `get_context()` 返回一个新对象 | 返回**全局单例**，`set_context` 是整体替换 | `get_context() is get_context()`；读 `context.py:21` |
| 12 | HF 的 `Qwen3MLP` 和这里的 `Qwen3MLP` 一样 | 名字一样，实现、参数打包、调用方式都不同 | 把两份 `forward` 并排贴出来 |

---

## 5. 明确不需要学的

范围爆炸比学得慢更致命。下面这些**这一轮不碰**，理由写清楚，等真需要了再回来。

| 不需要学 | 理由 |
| --- | --- |
| 反向传播、优化器、学习率调度 | 推理不建图，`torch.inference_mode()` 之下没有任何 `.grad`。你要读的 1480 行里，零个反向相关符号 |
| 数据并行（DDP/FSDP）、混合精度训练（amp/GradScaler） | 那是训练侧并行。推理侧的并行只有 TP（本仓库）和 PP（本仓库没有） |
| 预训练与数据 pipeline（tokenizer 训练、数据清洗） | 与本仓库无关。`tokenizer` 只用到 `encode/decode/apply_chat_template` 三个方法 |
| Attention 的数学推导（softmax 为什么稳定、RoPE 的复数证明） | 你只需要知道"结果是什么形状、有什么性质"。推导留给论文 |
| 高级采样（beam search / top-p / repetition penalty） | `sampler.py` 一共 14 行，只有 temperature + 指数采样。`sampling_params.py:11` 明确禁掉了 greedy |
| 自定义 CUDA/HIP 编译工具链 | 你已经在写 HIP/Triton kernel，这一块不是短板 |
| NCCL 调优、通信重叠、多机 | 单卡 MI100，且 `model_runner.py:28` 用的是最朴素的 `tcp://localhost:2333` |
| vLLM 原版的完整调度算法（continuous batching 的各种变体、speculative decoding、chunked prefill 的调参） | 本仓库是简化版（92 行调度器）。先把简化版吃透，再看原版才知道简化了什么 |
| 量化（AWQ/GPTQ/FP8） | 本仓库没有。知道"有这回事"即可 |
| HuggingFace `Trainer` / `PEFT` / `accelerate` | 与本仓库完全无关 |

**判断标准**：如果一个知识点删掉之后，你仍然能完整讲清"`generate()` 到 token 的路径"，它就不在这份计划里。

---

## 6. 综合验收

学完不是"感觉懂了"，而是**能对一份陌生代码做出可验证的判断**。

### 6.1 主任务（限时 2 小时，不许看本计划）

从 `references/reps/vllm/` 里任选一个文件（建议 `vllm/attention/backends/` 下任选一个 backend，或 `vllm/model_executor/layers/linear.py`），产出一份 `notes/final.md`，必须包含：

1. **执行路径图**：从入口函数到最底层 kernel 调用，标出每一跳的文件:行号。
2. **关键数据结构**：至少 3 个（张量形状 + 每个维度的含义）。
3. **三个可验证断言**，每个都要能用 5 行代码或一次 `print` 证明真伪。示例格式：
   > 断言：这个 backend 的 `forward` 在 prefill 时读的是 `block_tables` 而不是连续的 K/V。
   > 验证：打印传入的 `kv_cache` 形状与 `block_table` 的形状，若前者 4 维、后者 2 维则成立。
4. **与 nano-vllm 的三处差异**，每处说明"多解决了什么问题"。

### 6.2 通过标准

- 路径图里**没有一处是猜的**：每一跳你都能指出行号。
- 三个断言里**至少两个被验证为真**，被证伪的那个你能解释为什么猜错。
- 差异分析里**至少一处是 nano-vllm 故意简化的**（比如它有 chunked prefill 的多序列支持，而 `scheduler.py:42` 只允许第一个序列分块）。

### 6.3 反向验收（更狠的一版）

给一个完全没注释的 200 行推理脚本，你要在 30 分钟内回答四个问题：

1. 它有没有 KV cache？如果没有，`forward` 的输入会随步数怎么变？
2. 它的 batch 是怎么组织的——拼接（`cu_seqlens`）还是 padding（`attention_mask`）？
3. 它在哪里同步（`.item()` / `.cpu()` / `torch.cuda.synchronize()`）？这些点是不是每步都发生？
4. 它的 attention 是真 attention 还是被替换成了别的东西？

这四个问题答得出来，说明你读的是**结构**而不是**字面**。

---

## 7. 附录

### 7.1 常用命令

```bash
# 进容器
docker exec -it nanovllm-fundamentals bash

# 跑通（软链已建好）
cd /workspace/nano-vllm && python example.py

# 笔记导出到宿主机（在宿主机执行）
docker cp nanovllm-fundamentals:/workspace/docs/notes ./notes

# 正确性对照（官方 transformers）
python tests/check_qwen3_reference.py

# 单测（当前 3 个文件）
python -m unittest discover -s tests -v

# 吞吐
python bench.py

# 看编译了哪些图
TORCH_LOGS="graph" python example.py 2>&1 | head -50

# 确认这块卡的实际能力
python -c "import torch; print(torch.cuda.get_device_capability(), torch.cuda.get_device_name())"
```

### 7.2 读一段陌生理代码的六个固定动作

每次打开一个 `forward`，按这个顺序做，不要跳：

1. **看签名**：只吃几个参数？返回什么？有没有 `**kwargs`（HF 风格，说明有一堆可选配置）。
2. **看形状**：在纸上写出每个入参的维度含义，标出哪一维是 `batch`、哪一维是 `seq`。
3. **看状态**：函数体里 `self.xxx` 用到了什么？逐个查它是 Parameter、buffer 还是普通属性。
4. **看全局**：有没有 `get_context()` 之类的全局读取？有的话，这个函数就**不是纯函数**，测试时必须先设上下文。
5. **看分支**：`if` 的条件分别对应什么真实场景？把每个分支和"prefill/decode/TP/有无 flash attention"对上。
6. **看副作用**：有没有写共享张量（KV cache）、有没有 `.item()`、有没有 `print`？这些决定它能不能进 CUDA graph。

这套动作做完，一个函数基本就被拆开了。

### 7.3 进度打卡表

| 阶段 | 计划 | 实际 | 验收产物 | 状态 |
| --- | --- | --- | --- | --- |
| 0 环境 | 2h | | `notes/00-env.md` | |
| 1 Module | 6h | | `notes/01-module.md` | |
| 2 模块树 | 5h | | `notes/02-tree.md` | |
| 3 形状 | 5h | | `notes/03-shapes.md` | |
| 4 数据流 | 7h | | `notes/04-forward.md` | |
| 5 KV cache | 8h | | `notes/05-kvcache.md` | |
| 6 调度 | 6h | | `notes/06-scheduling.md` | |
| 7 TP | 6h | | `notes/07-tp.md` | |
| 8 性能 | 8h | | `notes/08-perf.md` | |
| 综合 | 2h | | `notes/final.md` | |

> 笔记放 `docs/notes/` 或你自己的草稿目录都行，但**必须落盘**。不落盘的笔记等于没写——两周后你只会记得"我好像搞懂过"。
