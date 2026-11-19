# ModelPort：给新模型自动适配推理引擎

**一句话**：给一个 HuggingFace 模型目录和一个推理引擎，产出一份该引擎能直接加载运行的模型文件，
并用引擎里的数值证明它算得对。

当前目标引擎是 `nano-vllm/` 子模块，第一个新模型是 Llama-3.2-1B。

---

## 1. 三步主线

```
HF 模型目录 ──① 抽参考图──> graph.json ──┐
                                        │
nano-vLLM 源码 ──② 引擎规约──> spec.yaml ┤──③ 交给 agent ──> nanovllm/models/<model>.py
                                        │         ↑
                        kernels/ 算子库 ─┘   nano-vllm/nanovllm/models/qwen3.py
                          （只读参考）         （同构模板）
```

| 步 | 做什么 | 产物 | 状态 |
| --- | --- | --- | --- |
| ① | 跑一遍 HF 模型，记录它实际做了哪些计算、谁吃谁、每个输入摆在第几个参数位 | `graph.json`（+ 五种人读视图） | 完成 |
| ② | 读引擎源码，写成"引擎对模型的要求"（must / toolbox / context） | `contracts/nanovllm/spec.yaml` | 完成 |
| ③ | 把图、规约、算子库、模板喂给 agent，生成模型文件，并在真实引擎里反复验证到数值对齐 | `outputs/iter_llama4/attempt_4.py` | Llama 单卡通过逐项相等验收 |

引擎侧不需要再改：注册表、契约检查、权重装载核对、KV cache 绑定核对都已补进 `nano-vllm/`（见第 7 节）。

---

## 2. 目录

```
nanovllm_selector/
├── extract/           ① 参考图：loader / recorder / decompose / edges / template / replay / render / cli
├── port/              ③ 生成与验证
│   ├── prompt.py        拼 prompt：任务 + 参考图 + 引擎规约 + 算子索引 + 同构模板
│   ├── generate.py      调模型、剥代码块、静态校验
│   ├── cli.py           单次生成（--dry-run 可只看 prompt）
│   ├── evaluate.py      在真实引擎里跑候选，抓 prefill logits 与 HF 比（子进程）
│   ├── iterate.py       生成 → 评测 → 回灌真实报错 → 重试，轮数封顶
│   └── demo.py          用生成的模型真生成一段文字，并与 HF 对照
├── outputs/           模型、计算图和验收结果
├── fixtures/          测试夹具（Llama-3.2-1B 的 config.json）
└── test_*.py          单测

contracts/nanovllm/spec.yaml   ② 引擎规约
kernels/                       算子库：index.md + 9 类实现（只读、只参考，不调用）
docs/model-port-plan.md        设计文档
```

---

## 3. 快速上手

前提：容器已启动，模型在 `tmp/models/` 下。

```bash
# 0. 进容器（工作目录 /workspace）
docker exec -it -w /workspace nanovllm-selector-rocm-dev zsh

# 1. 重建 Llama 参考图和五种视图
python -B -m nanovllm_selector.extract.cli \
    --model nanovllm_selector/fixtures/llama-3.2-1b --seq-len 5 \
    --outdir nanovllm_selector/outputs/graph_llama32_1b --check

# 2. 只拼 prompt 不发请求，确认材料齐了
python -B -m nanovllm_selector.port.cli --dry-run \
    --graph nanovllm_selector/outputs/graph_llama32_1b/graph.json

# 3. 用已有的生成结果真跑一段文字（最直观）
python -B -m nanovllm_selector.port.demo \
    --model tmp/models/llama-3.2-1B \
    --candidate nanovllm_selector/outputs/iter_llama4/attempt_4.py \
    --prompt "The capital of France is"
```

---

## 4. `demo.py`：最直观的那个入口

它回答一个问题：**agent 写的这份代码，在引擎里到底能不能正常说话？**

一条命令做三件事：

1. **HF 参考**：用 transformers 贪心生成一段文本，作为参照；
2. **引擎自由生成**：把候选类注入 `models.ARCH`，构造 `LLM(...)`，用引擎的采样器生成同样长度的文本；
3. **同前缀逐位置比对**：把 HF 自己生成的前缀逐段喂回引擎，看每一步的 argmax 是否相同。

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | HF 模型目录，**必须含权重**（demo 要真跑） |
| `--candidate` | 必填 | 候选 `.py` 文件，里面要有 `*ForCausalLM` 类 |
| `--arch` | 候选类名 | 注册到 `models.ARCH` 的名字 |
| `--prompt` | `The capital of France is` | 提示词 |
| `--max-tokens` | `40` | 生成多少个 token |
| `--temperature` | `1e-8` | 引擎不允许贪心（`assert temperature > 1e-10`），`1e-8` 在数值上等价于贪心 |
| `--teacher-forced` | `24` | 逐位置比对的位置数，`0` 表示关闭 |
| `--no-hf` | 关 | 跳过 HF 参照，只让引擎生成 |

### 早期候选的实际输出（`iter_llama2/attempt_3.py`）

```
[HF  ] ' Paris. It is the most visited city in the world. It is also the most
        expensive city in the world. The average cost of living in Paris is
        2.5 times higher than the average'

[引擎] ' Paris. It is the most visited city in the world. It is also the most
        expensive city in the world. Paris is a city of art, culture, and
        history. It is also a city'

自由生成：23/40 个 token 相同，从第 23 个开始分叉
同一前缀逐位置 argmax：23/24 相同
```

### 怎么读这个结果

- **分叉不等于算错**。bf16 下很多位置 top-2 只差一点点（这一处 HF 的 top-3 是 `[It, The, Paris]`），
  谁先谁后由几十个 ulp 决定。所以自由生成比 token 相同率意义有限。
- **同一前缀逐位置 argmax 才是判据**：两边看同样的上文，预测的下一个 token 是否相同。
- 想要更严格的整体判断，用 `evaluate.py` 的三档长度 logits 比对（第 6 节），那里有 `max_abs` 和 `cosine`。

### 一个已经踩过的坑

demo 一个进程里先后构造 HF 和引擎**两份**模型。PyTorch 的缓存分配器不会自动把显存退回系统，
第二份在 `allocate_kv_cache` 里可能算出 `num_kvcache_blocks == 0`，直接
`AssertionError`。所以每次用完都显式 `gc.collect() + torch.cuda.empty_cache()`（文件里的 `_free_gpu()`）。
`evaluate.py` 没这个问题，因为它每个候选单开子进程。

---

## 5. 第一步：抽参考图（`extract/`）

给一个模型目录，在 Hugging Face 模型实际运行时记录运算顺序、每个节点用到的数据和模型保存的缓存。

```bash
python -B -m nanovllm_selector.extract.cli \
    --model tmp/models/Qwen3-0.6B \
    --phases prefill,decode --decode-steps 2 --shape-probes \
    --outdir nanovllm_selector/outputs/graph_qwen3 --check
```

| 参数 | 说明 |
| --- | --- |
| `--model` | 本地模型目录；有 `config.json` 就能抽（默认随机初始化，不读权重） |
| `--weights` | 加载真实权重（默认关闭） |
| `--phases` | 默认 `prefill,decode`，只看初始提示可传 `prefill` |
| `--dtype` / `--device` | 默认 `float32` / `cpu` |
| `--seq-len` | 默认 8 个 token |
| `--batch-size` | 主样本并行输入的序列数，默认 1 |
| `--decode-steps` | 连续录制的生成步数，命令行默认 2，可看到“读已有缓存、更新、下一步再读”的全过程 |
| `--shape-probes` | 命令行默认开启。用批大小和提示长度 (1,5)、(1,17)、(2,5) 推断各轴，再用独立输入 (2,11) 检验。快速单样本检查可用 `--no-shape-probes` |
| `--no-collapse` | 不做层模板折叠，输出 28 层全部实例 |
| `--text` / `--summary` / `--mermaid` / `--json` | 打到 stdout 的人读视图 |
| `--out` / `--outdir` | 只写 JSON / 一次写六种视图 |
| `--check` | 跑对账，未通过退出码 1 |

六种产物：`graph.json`（机器读）、`graph.md`、`graph.mmd`、`graph.dot`、`graph.svg`、`graph.html`（推荐人看）。

**图是什么**：一次真实前向的流水账，只记录事实，不做决定。
节点 = 一次模块调用（`nn.Linear` 内部的 `t + mm + view` 算一个 matmul 节点）；
两次模块调用之间不合并——所以 nano-vLLM 里的四处融合（`add_rms`、`qkv_proj`、`gate_up_proj`、`SiluAndMul`）
在图里分别是 2 / 3 / 2 / 2 个节点，一处都没有提前合并。

**输入是算出来的**：节点输入 = `consumed − produced − state`（簇内所有算子吃掉的张量，
减去自己产出的，再减去权重/buffer），并做覆盖等式断言。这样"去重时把 sin 丢掉"这类静默错误会当场炸。

**四层防线**：

| 层 | 管什么 | 结果 |
| --- | --- | --- |
| ① 集合差 + 断言 | 输入不再靠人工判断 | 覆盖等式不成立就进 `conflicts` |
| ② 三条机械不变量 | aten 全覆盖 / 张量有主 / 入参有槽 | Qwen3：`2412 = 1359 进节点 + 628 只透明 + 425 两者都算` |
| ③ 逐节点重放 | 用录制的输入独立复算，每个对外输出都比较尺寸、类型和数值 | 跳过节点、丢输出、交换同形输出都会使检查失败 |
| ④ decode 与状态 | 记录本层实际保存的键和值，标出下一步读的是前一步哪个输出 | 每次读写都保留两端的槽位，找不到张量编号就报错 |

各阶段单独对账。`--shape-probes` 另外验证各维度的关系，图和 Agent 的输入里都保留每个槽位的实测尺寸。B 表示序列数，S 表示当前输入长度，P 表示已有缓存的位置数。与配置及缓存读写身份能对应上的轴会标明字段名；固定数字只表示几次实测取值相同，未确定的关系记为 `?`。

---

## 6. 第三步：生成与验证（`port/`）

### 6.1 只看 prompt

```bash
python -B -m nanovllm_selector.port.cli --dry-run
```

打印任务、参考图、引擎规约、算子索引和模型模板的字符数。抽取的阶段、尺寸样本和引擎数值示例都会影响总长度。逐项相等模式会附上 `port/exact_reference.md` 中单卡实测过的运算顺序、数据布局和舍入位置。

### 6.2 单次生成

```bash
python -B -m nanovllm_selector.port.cli \
    --graph nanovllm_selector/outputs/graph_llama32_1b/graph.json \
    --out nanovllm_selector/outputs/llama_gen.py
```

省略 `--out` 时写入 `nanovllm_selector/outputs/generated.py`。从环境变量取 `DEEPSEEK_API_KEY` / `DEEPSEEK_API_BASE` / `DEEPSEEK_MODEL`；
`.env` 里的值需要先 `export`，`make_llm()` 不读 `.env`。

### 6.3 迭代生成（推荐）

```bash
python -B -m nanovllm_selector.port.iterate \
    --model tmp/models/llama-3.2-1B \
    --graph nanovllm_selector/outputs/graph_llama32_1b/graph.json \
    --spec contracts/nanovllm/spec.yaml \
    --kernels kernels/index.md \
    --template nano-vllm/nanovllm/models/qwen3.py \
    --out-dir nanovllm_selector/outputs/iter_llama5 \
    --max-iters 3
```

上例使用宿主仓库的引擎规约和算子索引。在现有容器内，两份输入材料分别位于 `tmp/port-deps/contracts/nanovllm/spec.yaml` 和 `tmp/port-deps/kernels/index.md`。本次已验收的四个候选按生成顺序保存在 `outputs/iter_llama4/`，其中第四个根据第三个的评测反馈续跑生成。

每轮写 `attempt_N.py`，并在**子进程**里跑真实引擎：

- 把候选类注入 `models.ARCH`，构造 `LLM(...)`；
- 三档长度 128 / 256 / 768 各跑一次 prefill，抓那一份 logits（构造时的 warmup 要跳过）；
- 再对最长的那条跑一次生成，抓 prefill 之后的 8 步 logits——这几步是真的在读 KV cache；
- Hugging Face 从原始提示独立建立缓存，随后逐个输入引擎实际生成的词元。引擎命中历史提示前缀时，报告的 `cached_prefix_tokens` 记录复用长度，HF 从自己先前计算的缓存中取相同前缀，以相同的输入长度执行。每步比较整个词表，包括连续生成的第一份分数。
- 单卡的默认判据要求所有分数均为有限数且逐项相等，报告记录 `exact`、最大绝对差、余弦相似度及首选词。两种精度模式都要求抓全指定步数。`evaluate` 和 `iterate` 均可用 `--no-exact` 复现既有的近似验收标准（首选词相同、最大绝对差 ≤ 2、余弦相似度 ≥ 0.999）。下文历史实验的“通过”按当时的近似标准统计。
- 严格验收在两端同时开启 PyTorch 确定性算法，结束后恢复调用前的设置。这样相同硬件、相同输入的参考计算可重复；报告里可核对 `deterministic_algorithms`。在本机，255 个词元后第一次生成时，未开启此选项的 HF 自身重复运行也出现过 0.125 的最大差。

失败时把**真实报错**和每档数字回灌给模型，而不是用自然语言说"再检查一下"。

`--max-iters` 用完就停，留下每一轮的文件和报告。

### 6.4 单独评测一个文件

```bash
HIP_VISIBLE_DEVICES=0 python -B -m nanovllm_selector.port.evaluate \
    --model tmp/models/llama-3.2-1B \
    --candidate nanovllm_selector/outputs/iter_llama4/attempt_4.py \
    --tp 1 --lengths 128,256,768 --decode-steps 8 --exact \
    --report nanovllm_selector/outputs/iter_llama4/verified.json \
    --logits-out nanovllm_selector/outputs/iter_llama4/logits.pt
```

退出码 0 通过。失败阶段会写明是 `load`（装载）、`class`（找不到类）、`run`（引擎里跑挂）还是 `compare`（数值不符）。
`--decode-steps` 控制逐 token 走 KV cache 的步数，设 0 就退回"只验 prefill"。

报告里 `comparisons` 是 prefill（每个长度一条），`decode.items` 是逐步的 KV cache 验证，每条都带 `max_abs_diff` 和 `cosine`。

---

## 7. 引擎侧基底（`nano-vllm/` 里补的四件事）

引擎启动时依次做四个动作，每个都有明确的失败提示：

| 动作 | 位置 | 行为 |
| --- | --- | --- |
| 找到它 | `nanovllm/models/__init__.py` | `ARCH` 按 `config.architectures` 查表；查不到时列出"config 要的"和"已注册的" |
| 验它 | 同上 `check_model_contract()` | 构造前检查 `forward(input_ids, positions)`、`compute_logits`、`packed_modules_mapping` |
| 装它 | `nanovllm/utils/loader.py` | 装完核对：checkpoint 有模型没有 → 报键名；模型参数没被填 → 报参数名；支持 `ignored_weight_keys` |
| 绑它 | `nanovllm/engine/model_runner.py` | KV cache 切片数量必须等于 `num_hidden_layers`，多了少了都报 |

`model_runner.py` 里原本写死 `Qwen3ForCausalLM` 的两行，改为 `resolve_model_class(hf_config)` + `self.model_cls(hf_config)`。

---

## 8. 实测结果

### Llama 单卡严格验收

当前 Llama 图含 60 个节点模板、85 条边和 685 次实际计算，覆盖提示阶段与两次连续生成。长度 128、256、768 的三个提示、连续生成首步及后续八步，共比较 12 对完整词表分数，每对 128,256 项，最大绝对差为 0。长度 255 的独立提示跨缓存块生成八步也全部逐项相等。各步数据及产物路径见第 12 节。

### 早期参考图

| 模型 | 模板节点 | 边 | 实例节点（prefill / +decode） | 对账 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 22 | 29 | 453 / 905 | 20/20 |
| Llama-3.2-1B | 20 | 27 | 229 / 457 | 16/16 |

Llama 每层 14 个节点（没有 q/k norm），16 × 14 + 5 = 229；重放全部通过。

### 迭代生成（Llama-3.2-1B）

| 轮 | 结果 | 引擎或比对给出的原因 |
| --- | --- | --- |
| 1 | 装载失败 | `lm_head.weight` 没被填充：`lm_head.weight.data = embed.weight.data` 是两个独立参数 |
| 2 | 数值不过 | 直接用了 `get_rope`（只认 `rope_theta`），llama3 缩放没实现；768 长度余弦 0.9975 |
| 3 | 通过 | 自己写了 `LlamaRotaryEmbedding` + `_llama3_inv_freq`；三档余弦 0.999846 / 0.999905 / 0.999895 |

一个必须记住的教训：第一轮只比 5 个 token 时第 2 轮就"通过"了——**假通过**。
把长度拉到 768 后，同一份错实现的误差随长度单调增长（0.30 → 0.58 → 1.17），
才和真正的实现错误区分开。短序列看不出 rope 的问题。

---

## 9. 环境

- 容器 `nanovllm-selector-rocm-dev`，仓库挂载在 `/workspace`（`nano-vllm/` 只读）。
- Python 3.12.3，PyTorch `2.12.0+rocm10.0.0`，transformers `4.57.3`，Triton `3.8.0`。
- 模型：`tmp/models/Qwen3-0.6B/`（有权重）、`tmp/models/llama-3.2-1B/`（有权重，2.48GB，来自 modelscope）。
- API key 放 `nanovllm_selector/.env`，运行前 export。
- `nano-vllm/` 是 git submodule，改动要先在子模块提交，再在父仓库更新引用。

---

## 10. 测试

```bash
# 容器内，工作目录 /workspace
python -B -m unittest nanovllm_selector.test_extract          # 31 项，约 80 秒
python -B -m unittest -v nanovllm_selector.test_graph_contracts
python -B -m unittest nanovllm_selector.port.test_port        # 29 项
python -B -m unittest nanovllm_selector.test_port_iterate     # 13 项
python -B -m unittest discover -s nano-vllm/tests -v          # 引擎 16 项
```

`test_extract` 一组需要真实模型目录，缺目录时整组跳过，可用 `EXTRACT_MODEL` 覆盖。

---

## 11. 遗留：旧的 MLP 选择 demo

`run_selector.py` / `agent.py` / `candidates.py` / `baseline.py` 是第一阶段的东西：
在**已经适配好的** Qwen3 上，让 agent 从两个候选里选一个 MLP 激活实现（原生 vs GEAK swiglu），
验证它能装进真实推理。它证明了"选择 + 安装 + 回退"这条链路，但只覆盖一个节点、候选只有两个。

现在的主线不是它——主线是"给一个没适配的模型，把它适配出来"。旧 demo 保留可跑，
不再是验收对象。

---

## 12. 产物与清理

生成的模型 Python 文件、计算图、需要保留的测试日志、逐步评测报告和完整分数统一放在 `nanovllm_selector/outputs/`。独立评测省略 `--report` 时，会将 `<候选名>.eval.json` 写在候选所在的 `outputs/` 子目录；对于项目输出目录之外的候选，默认报告放在 `outputs/` 根目录。

| 路径 | 内容 |
| --- | --- |
| `nanovllm_selector/outputs/graph_qwen3/` | Qwen3 的六种图产物 |
| `nanovllm_selector/outputs/graph_llama32_1b/` | Llama 的六种图产物 |
| `nanovllm_selector/outputs/iter_llama4/` | 当前 Llama 单卡结果。`attempt_4.py` 为最终模型，`trace.json` 记录四次尝试，各轮另有 `attempt_N.report.json` |
| 同目录的 `verified.json`、`logits.pt`、`cold_255.json` | 单卡逐步报告、双方完整词表分数和跨缓存块报告 |
| 同目录的 `faults.json`、`without_history.py`、`without_history.json`、`regression.log` | 计算图破坏检查、故意跳过历史的反例及其报告、回归测试记录 |
| `nanovllm_selector/outputs/iter_llama4/experiments/` | 前期试验的原始文件与当时的记录，需要清理中间产物时可单独移除此目录 |
| `nanovllm_selector/outputs/iter_llama2/` | 三轮生成的 `attempt_N.py` + `trace.json` |
| `nanovllm_selector/outputs/iter_qwen3/` | Qwen3 冒烟 |
| `tmp/models/llama-3.2-1B/` | 下载的权重，2.48GB |

仓库收录了 `outputs/iter_llama4/` 中通过单卡验收的 `attempt_4.py`、`verified.json` 和 `cold_255.json`。运行生成的其他候选、日志、完整分数及本机缓存默认由 Git 忽略。完整分数文件可用第 6 节的命令重新生成。
清理示例：`rm -rf nanovllm_selector/outputs/iter_llama2`、`rm -rf tmp/models/llama-3.2-1B`（删了要重新下载）。
