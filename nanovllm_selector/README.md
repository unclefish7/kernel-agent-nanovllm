# nano-vLLM 真实 Qwen3 推理的算子选择 Agent

## 1. 当前状态与阶段目标

**原生基线和第一版 MLP 激活 Agent 接入均已完成冒烟验证。** 本轮按用户要求优先打通“Agent 决策能安装并实际执行”，完整数值验收和性能调优后续再做。当前只接 MLP 激活，尚未接 RMSNorm/Add+RMSNorm。`nano-vllm/` 源码未修改；不将接入通过表述为严格数值等价或性能优化。

用户已将模型下载到宿主机 `tmp/models/Qwen3-0.6B/`，容器内对应 `/workspace/tmp/models/Qwen3-0.6B/`。权重 SHA256 与下载元数据一致，311 个张量及配置、分词器均能离线读取。当前没有自动下载任务；本轮只在选择阶段调用一次 DeepSeek，正式运行不调用 API。

另有一条独立能力：从 HuggingFace 模型目录直接还原整张计算图（不依赖类名，靠 hook + aten 算子实测 + 对账），见第 10 节。这是"给 nano-vLLM 跑一个它没适配的模型"的前置步骤。

### 已完成的 Agent 接入：只选 MLP 激活

本轮入口为 [run_selector.py](run_selector.py)，分三步，所有实验在已有的新容器 `nanovllm-selector-rocm-dev` 内运行：

| 阶段 | 实际做什么 | 主要产物 |
| --- | --- | --- |
| prepare | 用真实权重执行原生模型，通过 hook 采集全部 28 层激活的实际输入规格；读取 Transformers 和 nano-vLLM 源码 | `context.json`：源码、接口、5 种输入规格、硬件、候选和版本绑定 |
| select | DeepSeek 阅读 context，只返回候选 ID 和理由；检查 ID、替换位置及方案结构 | `request.json`、`response.json`、`plan.json` |
| run | 离线读取 plan，在真实权重加载后、原 warmup 前安装激活模块，再执行原 `generate()` | `integration.json`：选择、逐层调用、回退、生成和恢复结果 |

候选只有 `native` 和 `geak_packed_swiglu`。后者是现有 GEAK-eval 的 `_swiglu_fwd`，直接接收 packed gate/up；独立保存在 [vendor/swiglu_fwd.py](vendor/swiglu_fwd.py)，来源版本、哈希及移除导入测试的说明见 [manifest.json](vendor/manifest.json)，保留 Apache-2.0 许可证。不导入旧原型，不从零写 kernel，也不执行 Agent 返回的代码。

Agent 本次真实选择了 `geak_packed_swiglu`，理由是 `[N, 6144]` 的连续 BF16 输入、gate/up 两半布局以及输出规格与候选一致；同时承认舍入和性能尚未验证。没有人工改写决定。全部 28 层安装成功，真实生成中累计调用候选 504 次：prefill 112 次、decode 392 次；warmup 另计 28 次，回退为 0。双请求和 281-token 提示词均生成了指定的 8 个 token；这是有限长度的接入测试，不是文本质量评测。

安装器保留参数对象，结束后可恢复原激活。通过局部替换 `ModelRunner.warmup_model` 在原构造流程的正确时点安装；构造完成后这个类方法补丁立即撤销。该方式仅用于本轮固定单进程、TP=1、eager 试验，不声称已支持多进程或图捕获。

当前只做方案结构、支持范围、返回规格及基本接口测试。非连续输入等不支持规格会走原生回退并记录原因；候选编译/执行异常直接报错，不会静默回退后伪称成功。完整模型的 logits 正确性、生成一致性与加速比尚未验证。上游 autotune 原样保留，配置质量未评估。

从宿主机仓库根目录复现：

```bash
# 1. 原生模型采集真实输入，不调用 API。
docker exec nanovllm-selector-rocm-dev python -B run_selector.py prepare

# 2. 仅此步骤需要 API。先参照 .env.example 创建本目录的 .env 并填写密钥。
docker exec --env-file nanovllm_selector/.env \
  nanovllm-selector-rocm-dev python -B run_selector.py select

# 3. 直接重放已有方案，无需 API 密钥，不重新选择。
docker exec nanovllm-selector-rocm-dev python -B run_selector.py run
```

默认产物在宿主机 `nanovllm_selector/outputs/mlp_agent/`。需要保留不同试跑时，三条命令都传入同一个新的 `--run-dir outputs/<试跑名>`，不要覆盖已有记录；prepare/select 失败时不要继续 run，以免误用旧方案。方案绑定 checkpoint、配置、dtype、GPU 架构、Torch/Triton、原生目标源码和候选版本；不匹配时拒绝安装。

本次首跑通过 `docker exec --env-file operator_selector/.env` 只注入旧位置中已有的 DeepSeek 配置，没有复制密钥、复用旧容器或导入旧模块。后续可将配置独立放在本目录 `.env`；它已忽略。请求记录只有正文，不记录 Authorization。正式运行只读方案，不需要 `.env`。目前 API 请求只发送公开模块源码和规格，不发送权重数值。

新增逻辑分工：

- [agent.py](agent.py)：固定任务边界的 prompt、一次 API 请求、解析与保存选择。
- [candidates.py](candidates.py)：候选目录、支持检查、安装/恢复、按 warmup/prefill/decode 统计调用。
- [run_selector.py](run_selector.py)：采集、决策、完整生成三个命令的编排。
- [test_selector.py](test_selector.py)：5 项接口测试，包括非法方案拒绝、权重/恢复、回退、模拟 API 解析和 MI100 当前 stream 调用；不代表完整数值验收。

检查命令：`docker exec nanovllm-selector-rocm-dev python -B -m unittest -v test_selector`。原生模式仍使用下面的 `baseline.py`，不会导入安装器或请求 Agent。

### 已完成的原生基线

入口为 [baseline.py](baseline.py)，不依赖旧 `operator_selector/`，不安装替换算子。以下命令在宿主机仓库根目录执行，实际推理发生在新容器内：

```bash
docker exec nanovllm-selector-rocm-dev python -B baseline.py
```

容器已将工作目录设为 `/workspace/nanovllm_selector`。默认报告保存为宿主机 `nanovllm_selector/outputs/baseline.json`；也可在容器 shell 中执行 `python -B baseline.py`。

本次运行环境为 PyTorch `2.12.0+rocm10.0.0`、HIP `7.15.26333`、Transformers `4.57.3`，nano-vLLM 子模块版本为 `04ea53a899650326fd1a3f90350d9fc83b8130e5`。配置为 BF16、单卡、eager、最大上下文 512、每次最多调度 128 个 token、最多 2 个请求、显存预算 25%。未安装 FlashAttention，Attention 使用仓库已有的 PyTorch math 兼容路径；eager 指禁用引擎的图捕获，原模块的 `torch.compile` 装饰器仍保持不变。

实际验证结果：

- 依赖检查和已有 4 项单元测试通过。
- 原生随机采样完成中英文双请求生成；包括中文自我介绍和 `2 + 3 = 5`。
- 固定输入的 1 次 prefill 和 7 步 decode，与同一 checkpoint、同一 BF16 dtype 的 Transformers 比较；8 步最高分 token 全部一致。
- 281-token 提示词首次经过 3 次 prefill 和 7 步 decode；重复请求的首步复用 256 个缓存 token，只计算剩余 25 个。两次生成 token 完全一致。

**数值边界：** 上述 Transformers 对照是在相同 token 前缀上进行，最大 logit 绝对误差为 `0.6875`，没有通过或宣称严格 `assert_close`。当前结果证明这些请求的原生完整推理链路跑通，不证明所有输入下与 Transformers 数值等价。后续替换算子需要另行制定数值门槛，并与同配置的 nano-vLLM 原生路径逐步对照；不能把本次 token 一致当作全面正确性保证。本次也不是正式性能 benchmark，不使用包含编译、调试和对照开销的耗时宣称推理性能。

阶段目标：在 AMD MI100 上加载真实 Qwen3 checkpoint，通过已有 nano-vLLM 完整生成文本；让 Agent 在指定的计算位置选择已有实现，验证这些实现确实进入了真实推理，并且结果正确。

本阶段优先解决“能够正确接入”，不要求一定加速。不开发新的 kernel，不实现推理引擎，不尝试一次支持任意新模型。第一版以仓库示例的 Qwen3-0.6B 为目标，实际使用前先确认本地权重、配置和运行环境。

旧的 [operator_selector](../operator_selector/README.md) 已归档。它证明了独立计算片段可以由 Agent 组装和评测，但不代表完整推理已接入。新目录不导入旧目录的 Python 模块，不依赖其 JSON 格式、候选 ID、实验输出或容器；其中的测试和计时经验可以参考，候选实现需要追溯到原开源来源并重新验证。

## 2. 先理解我们要接入什么

几个必要名词：

| 名词 | 在这里的意思 |
| --- | --- |
| checkpoint | 训练好的模型权重以及配套配置、分词器文件，不是随机初始化的小模型 |
| forward | 模型收到输入后，实际执行的计算过程 |
| 算子 / kernel | 完成某项计算的接口及其具体实现，例如 RMSNorm 或 SiLU 与乘法融合 |
| reference | 判断结果是否正确的标准；用真实 Transformers 模型及 nano-vLLM 原实现做对照 |
| prefill | 一次处理提示词中的多个 token，并建立后续使用的缓存 |
| decode | 使用已有缓存继续生成新 token，会多次重复调用模型 |
| KV cache | 保存 Attention 的历史 Key/Value，避免每步重新计算全部历史 |
| Agent | 阅读计算和候选说明、返回选择方案的模型；不参与每一个 token 的实时生成 |

nano-vLLM 原有链路大致为：

```text
用户提示词 → 分词 → LLM.generate()
                      ↓
                 引擎与调度器
                      ↓
                  ModelRunner
                      ↓
               Qwen3 模型的 forward
                      ↓
       Attention、Norm、MLP 等真实计算
                      ↓
                logits → 采样 → 新 token
                      ↓
              重复 decode，直到结束
```

我们保留这条链路，只替换其中经过确认的计算实现。生成接口、调度、Attention、KV cache 和采样仍由 nano-vLLM 负责。

现有源码入口：

- [Qwen3 模型与 Decoder/MLP](../nano-vllm/nanovllm/models/qwen3.py)。
- [ModelRunner 的模型加载、预热和执行](../nano-vllm/nanovllm/engine/model_runner.py)。
- [原激活实现](../nano-vllm/nanovllm/layers/activation.py)、[原 Norm 实现](../nano-vllm/nanovllm/layers/layernorm.py)。
- [权重加载与打包映射](../nano-vllm/nanovllm/utils/loader.py)、[已有模型对照测试](../nano-vllm/tests/check_qwen3_reference.py)。

## 3. 为什么不能把旧片段直接装进去

Transformers 的 Qwen3 MLP 分别计算 gate_proj 和 up_proj，而本仓库 nano-vLLM 已经把它们合并为 gate_up_proj，加载权重时也有对应映射。因此，激活候选接收的是一个打包张量，不应再要求运行两次投影。

另一个重要区别是残差。Transformers 通常直接返回一次相加后的结果；nano-vLLM 的 Decoder 返回 `(hidden_states, residual)`，把部分相加推迟到后续 Norm 中执行。不能将这个二元组改成单个 Tensor，也不能提前相加后让下一层再加一次。

旧实验固定的 256/1024 维度、FP32、随机输入和片段性能，也不能直接用于真实 checkpoint。新方案必须读取实际模型规格，保留权重、dtype、布局、残差精度及输入输出约定。

## 4. 本阶段替换哪些位置

先固定三个接口，再由 Agent 决定各接口使用哪个实现。接入位置相对于 ModelRunner 持有的 Qwen3 模型描述如下：

| 接入点 | 实际位置与输入 | 必须保持的输出 |
| --- | --- | --- |
| MLP 激活 | model.layers.<层号>.mlp.act_fn；输入为打包的 gate/up | SiLU(gate) * up，最后一维宽度减半 |
| 普通 RMSNorm | Decoder 的 input_layernorm/post_attention_layernorm 中 residual 为 None 的分支 | 与原实现相同规格的归一化 Tensor |
| Add+RMSNorm | 上述 Norm 中 residual 不为 None 的分支 | `(归一化结果, 新残差)`，顺序、dtype 和输入修改约定与原接口一致 |

第一轮不替换 Attention 内的 Q/K Norm、最终输出 Norm、Linear、RoPE、Attention、KV cache、调度器或采样器。不能按类名遍历后把所有 Norm 都无区别替换。

每个位置保留原生实现作为候选和安全回退。某个优化 kernel 不支持实际输入时，不得强行选用，也不得为了让它通过而悄悄改变模型计算。

## 5. 完整流程分成“准备”和“运行”

### 准备阶段：允许调用 Agent

```text
真实 checkpoint + 未修改的 nano-vLLM
    ↓ 原生推理，采集目标位置的实际规格
参考源码 + 接口约定 + 实测规格 + 硬件 + 已验证的候选
    ↓ Agent 选择
结构化方案：每个接入点使用哪个候选
    ↓ 静态及支持范围检查（当前已实现）；完整数值检查后续补充
可安装的接入试验方案（尚非完整正确性验收通过的方案）
```

采集和检查可以额外执行参考计算，但这些开销不算入正式推理性能。优先保存规格和误差统计，不默认保存大量激活张量。

### 正式运行：不请求 Agent API

```text
构建 nano-vLLM Qwen3 模型
    ↓
按原映射加载真实权重
    ↓
读取匹配当前环境的方案，安装选中的实现
    ↓
warmup，随后由原框架管理缓存和执行
    ↓
照常调用 LLM.generate()，完成 prefill 和多步 decode
```

安装点应位于权重加载后、模型预热前。真实规格在前一轮原生基线中采集，不能等每个 token 到来时再调用 Agent。第一版使用 eager 模式，不同时处理 graph capture 或复杂编译接入。

## 6. 需要准备的模块、输入和产出

以下是完整阶段的职责划分。目前已实现上文所述的 MLP 接入子集，候选完整数值验证、多接口覆盖和性能评测仍是规划，不复制旧原型的交叉调用结构。

| 职责 | 输入 | 做什么 | 产出 |
| --- | --- | --- | --- |
| 基线与规格采集 | checkpoint 路径、原生 nano-vLLM、测试提示词 | 完整生成，记录三个接入点在 prefill/decode 的 shape、dtype、stride、残差规格 | 原生基线与运行规格 |
| 候选登记和验证 | 框架已有实现或固定版本开源 kernel、原模块参数、运行规格 | 登记数学含义及输入输出；以代码检查限制，验证数值与输入修改行为 | 可调用候选、说明及验证结果 |
| Agent 选择 | 参考源码、实际接口、规格、硬件、候选说明 | 请求模型并校验返回的 ID、接入点和适用条件 | 选择方案与原始请求/响应记录 |
| 安装与回退 | 加载完权重的模型、已验证方案 | 只替换指定位置，保留原接口和权重；不支持时回退并说明原因 | 可通过 generate 使用的模型 |
| 检查与评测入口 | 原生/Agent 两种模式、同一组请求和配置 | 验证调用路径、正确性、完整生成与性能 | 可复查报告和原始测量记录 |

候选信息分开维护：名称、签名、实际张量规格等尽量自动读取；数学语义及特殊限制需要确认，不能只信一段自然语言描述。应提供可执行的支持检查，覆盖 dtype、维度、stride、设备等，并用测试验证“返回两个结果”“不改写输入”等约定。

Agent 获得原模型 forward、MLP/Norm 源码、nano-vLLM 接口说明、实际运行规格和候选目录；不需要看到权重数值，不提供人工最快答案。输出只引用已登记候选，不允许返回任意 Python/kernel 代码直接执行。

方案与模型版本/规格、dtype、硬件和候选版本关联。条件不匹配时不能直接复用。相同规格的多个层可共享选择结论，但不能共享本来不同的 Norm 权重。

适配代码只绑定原模块需要的参数或子层，不持有整个父 Decoder；避免循环模块引用、重复参数注册，以及破坏 state_dict 和设备迁移。不复制或重新随机初始化模型权重。kernel 使用 PyTorch 当前 stream，不能擅自切到另一条未同步的执行队列。

## 7. 实施顺序与每步完成标准

### 第一步：真实原生基线

- 检查本地镜像、Torch/ROCm、模型文件与 nano-vLLM 依赖。现有针对旧 NVIDIA 的能力判断需要在 MI100 上核对，不能直接视为 ROCm 兼容结论；torch.cuda 命名本身不表示用了 NVIDIA。
- 单卡、小批量、eager 模式，以一种明确确认可运行的 dtype 加载真实 Qwen3-0.6B；原生和替换版必须使用同样的 dtype。
- 运行已有单元测试，完成提示词 prefill 和连续多步生成，并与 Transformers 做必要对照。

完成标准：关闭 selector 时，真实 nano-vLLM 推理可靠运行，并记录版本、模型、dtype、上下文和批量设置。当前已有对照脚本有固定路径及测试范围，需要核对后使用，不能宣称它已经覆盖所有 decode 情况。

### 第二步：只接入 MLP 激活

当前“Agent 返回方案并在真实生成中执行”的接入子目标已通过；下列完整数值完成标准尚未达到，不因调用成功而打勾。

- 采集真实 packed 输入规格，准备原生实现与至少一个兼容的开源优化候选。
- 先人工指定候选用于诊断，验证接口及同一输入上的输出，再让 Agent 选择；两种结果必须区分记录。
- 先在单层试装，通过后扩展到同规格层，确认完整 generate 走到了所选实现。

完成标准：MLP 接入前后输出满足约定误差，多步生成正常；能够报告实际候选调用，不是只报告“安装成功”。

### 第三步：接入普通 RMSNorm 与 Add+RMSNorm

- 分开验证 residual=None 和带残差分支，读取真实权重、eps、输入/残差 dtype。
- 保留 nano-vLLM 的延迟残差加法语义，不替换整个 Decoder 尾部，不改权重打包映射。
- 验证归一化输出、新残差、输入修改约定及组合后的完整 logits。

完成标准：三个接入接口均覆盖；单层和完整模型的数值、残差状态与原生路径一致到规定容差。

### 第四步：可切换的完整推理验证与评测

- 提供明确的原生/Agent 方案开关和恢复方式；正式推理只加载方案，不调用 API。
- 用短/较长提示词、单请求/小批量、多步 decode 和缓存复用检查完整链路。
- 在相同输入 token 上比较中间结果和 logits；固定采样进行生成检查，不能仅凭文本“看起来正常”判断正确，也不把随机采样差异直接判成 kernel 错误。
- 正确性通过后再测完整生成延迟、输出 token/s、显存；逐步增加 prefill/decode 分段计时。记录输入/输出长度和批量，并排除 API、模型加载、编译、调优及诊断计算时间；正式计时关闭逐调用日志。

完成标准：原生与 Agent 方案都能稳定完成同一组请求，候选实际生效、回退透明，测试和性能结果可复查。不要求第一版一定加速。

## 8. 阶段验收清单与边界

- [x] 真实 checkpoint 的原生 nano-vLLM 基线冒烟通过，不再用随机小模型代替；严格数值等价不在本次通过范围内。
- [x] Agent 使用真实运行规格选择已登记的 MLP 激活实现，保存方案和原始 API 记录。
- [ ] 三个接口保持权重、残差、dtype、shape、device/stream 与输入修改约定。
- [ ] 至少一个接入点在真实推理中实际调用非原生候选，并通过正确性验证；目前真实调用已证明，完整正确性验证仍待完成。如果 Agent 全部选原生，只证明回退和流程，不替代优化候选接入验收，也不能人工改选后伪称 Agent 的决定。
- [ ] prefill、多步 decode、小批量和缓存复用均经过验证，实际调用路径有证据。
- [ ] 原生/Agent 开关可用，不支持的条件可以安全回退并记录原因。
- [ ] 有相同配置下的完整推理测量；不能把片段加速比当作整个模型的加速比。

本阶段允许人工固定 Qwen3 接入位置和候选集合，Agent 负责选择实现；不声称它已能自动发现任意新模型的替换边界。先跑稳这条真实链路，再考虑不同 MLP 数学结构、更多候选、性能反馈，以及是否接入更大的框架。暂不做多卡 TP/PP、新 Attention 适配或 kernel 生成。

## 9. 环境、目录与操作约束

已复用本地镜像 `jerryjwc/rocm-dev:10.0`，通过本目录的 [compose.yaml](compose.yaml) 建立独立容器 `nanovllm-selector-rocm-dev`。源码和 `nano-vllm/` 通过 bind mount 共享，不 COPY 进镜像；其中 `nano-vllm/` 在容器内只读挂载。旧 `operator-selector-rocm-dev` 容器已移除，其宿主机文件和镜像保留。

从仓库根目录管理容器：

```bash
docker compose -f nanovllm_selector/compose.yaml up -d --pull never --no-build
docker exec -it nanovllm-selector-rocm-dev /usr/bin/zsh
# 用完暂时停止；需要删除容器时再用 down，不会删除宿主机挂载文件。
docker compose -f nanovllm_selector/compose.yaml stop
```

依赖清单见 [requirements.txt](requirements.txt)。当前容器已具备清单中的依赖，Torch/ROCm 沿用镜像；新增的 requests 也已在容器中，无需再次下载。重建容器后需要重新安装这些容器内依赖，不需要重新下载宿主机已有模型。以下检查在容器内执行：

```bash
python -B -m pip check
python -B -m unittest discover -s /workspace/nano-vllm/tests -v
```

已有检查结果为依赖检查通过，4 项单元测试通过，分别覆盖 CPU/GPU Attention 兼容路径、预热分块边界和 FP32 残差。容器只暴露 GPU 0；完整模型的原生冒烟结果及 BF16 数值对照边界见本文开头。

运行前先确认本地模型位置。缺少权重时，说明下载来源、保存路径、缓存路径和预计大小，获得确认后再下载；遵守仓库关于 `tmp/` 和目录外写入的要求。不在检查环境时隐式拉取镜像、更新依赖或下载模型。

运行产物放在本目录的 `outputs/`；缓存通过 Compose 环境变量显式设置在本目录 `cache/`。原生产物包括安装日志 `outputs/install.log`、依赖检查 `outputs/pip_check.log`、单元测试 `outputs/unit_tests.log`、早期失败下载日志 `outputs/model_download.log`、原生生成日志 `outputs/native_smoke.log`、初次数值诊断 `outputs/reference_probe.log`，以及 `outputs/baseline.log` 和 `outputs/baseline.json`。Agent 接入产物在 `outputs/mlp_agent/`；本次另有 `outputs/mlp_prepare.log`、`outputs/mlp_select.log`、`outputs/mlp_run.log`、`outputs/selector_tests.log`、`outputs/selector_regression.log` 和 `outputs/selector_pip_check.log`。`outputs/`、`cache/`、本地 `.env` 及仓库根目录 `tmp/` 已加入忽略规则，不提交模型、密钥或运行产物。

上述报告及日志如不再需要，可逐个用 `rm nanovllm_selector/outputs/<文件名>` 清理；缓存可在停止相关任务后清理 `nanovllm_selector/cache/` 中对应子目录，下次运行会重新生成或重新编译。这里仅说明清理方法，没有执行清理；`docker compose down` 不会清理这些宿主机文件，也不会删除模型。

Agent 子目录内的记录可逐个用 `rm nanovllm_selector/outputs/mlp_agent/<文件名>` 清理；删除 `plan.json` 后将不能直接重放，需重新 select。单元测试的临时文件位于 `cache/tmp/`，由 `TemporaryDirectory` 在测试结束时自动删除，其余编译缓存保留。当前未清理真实运行记录。

后续若需要修改 nano-vLLM，应保持改动集中，先在子模块单独提交，再在父仓库提交引用更新；按仓库规范处理 fork 上的对应提交。当前原生基线和 MLP 接入均未修改子模块，尚无完整数值验收或正式性能对比数据。

## 10. 计算图抽取（extract）

`extract/` 解决的是另一类问题：**给一个 HuggingFace 模型目录，不读它的类名、只看它实际跑了什么，把整张计算图还原出来**。这是"跑一个 nano-vLLM 没适配的模型"的前置步骤——Qwen3 有现成实现所以不需要图，新模型没有实现，才需要把 HF 的规格翻译成图。

当前是 M1：只做参考侧（HF）的图、只做 prefill、CPU/fp32、随机初始化权重（只读 `config.json`，不需要 safetensors）。参考图与 nano-vLLM 可执行图的差集在 M3。

### 一条命令

```bash
# 容器内，工作目录 /workspace
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --check
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --text      # 终端树
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --summary   # Markdown 表
python -B -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --outdir nanovllm_selector/outputs/graph_qwen3
```

退出码：0 对账通过，1 对账未通过，2 出错。`--out` 只写 JSON，`--outdir` 一次写六种文件。

### 输出（六种，JSON 给机器，其余给人）

| 文件 | 用途 |
| --- | --- |
| `graph.json` | 机器可读的完整图：节点、边、张量形状、状态、证据 |
| `graph.md` | 概览 + 逐节点表 + 内嵌 Mermaid |
| `graph.mmd` | Mermaid 源码，可粘到任何支持 Mermaid 的地方 |
| `graph.dot` | Graphviz 源码，需要的话自己 `dot -Tpng` |
| `graph.svg` | 自带的离线分层布局，不依赖外部工具 |
| `graph.html` | **推荐的入口**：内嵌 SVG + 算子覆盖 + 对账表 + 逐节点明细，双击即可看 |

模板折叠：28 层结构相同，图上只画 1 个模板节点并标 `×28层`（JSON 里是 `instances: [0..27]`）。Qwen3-0.6B 折叠后 **22 个节点 / 28 条边**，展开是 453 个实例节点。层与层之间的两条边（残差直接喂下一层）单独标 `kind=layer`，不画在图上，避免出现假环。

### 数据契约

- `Node`：`id / op / inputs / outputs / state / attrs / impl / semantics / evidence / origin / template_id / instances`
- `TensorRef`（每个输入/输出）：`tensor / role / index / path / arg_name`
- `Edge`：`from_node / to_node / tensor / role / kind(data|state|layer)`
- `evidence.kind`：`observed`（hook 实测）/ `dispatch`（aten 算子）/ `param`（参数形状）/ `source`（源码 file:line + sha256）/ `rule`（规则编号）

**没有证据的标签不写。** `op` 的判定来自实测 aten 算子集合，不是模块类名：例如 `rsqrt + mean` 判 `rmsnorm`，`mm + 该模块 out_features == vocab_size` 判 `lm_head`。

### 输入位置：图里不只有"谁连谁"

节点只记"吃了哪几个张量"是不够的。`SiluAndMul` 要求打包张量的前一半是 gate、后一半是 up
（`x, y = x.chunk(2, -1)`）；`store_kvcache(key, value, ...)` 的 key/value 同形同源，传反了不报错、结果全错。
所以每个输入还带三个字段：

| 字段 | 含义 | 例子 |
| --- | --- | --- |
| `index` | 在节点 `inputs` 里的下标 | `0` |
| `path` | 原始调用里的位置表达式 | `args[0]`、`kwargs['position_embeddings'][0]`、`aten.cat` 里的 `args[0][1]` |
| `arg_name` | 该位置对应的形参名 | `hidden_states`、`input`、`self` |

节点的 `semantics.bindings` 再把公式变量绑到位置：`L*.mlp.mul` 是 `{"a": "in[0]", "b": "in[1]"}`，
配合边就能读出 `in0 ← act_fn`、`in1 ← up_proj`。终端视图直接显示：

```
[17] L*.mlp.mul ×28层            mul     1×8×3072 → 1×8×3072   —
       ↑ in0 self ← L*.mlp.act_fn  [args[0]]
       ↑ in1 other ← L*.mlp.up_proj  [args[1]]
```

`arg_name` 优先取 `inspect.signature(module.forward)` 的形参名；模块内部先 view/cast
（`F.linear` 会把 3D 输入摊成 2D）导致对象和 storage 都变了时，退回 aten schema 的参数名
（`self` / `other` / `tensors`），`path` 仍是真实参数树位置，不会把列表下标误当成参数下标。

### 对账（`--check`，16 项）

模块 427、命名参数 310、matmul 197、rmsnorm 113、silu 28、残差 add 56、embedding 1、lm_head 1，另查边端点存在、每节点至少 1 条证据、unknown 算子为 0。这些数字同时与 HF 模块树和计划 §8.1 的常量比对，两边都过才算通过。Qwen3-0.6B 当前 16/16。

### 三个实现上的坑（都已修）

1. **节点不是模块**：一个 `nn.Linear` 内部有 `mm/t/view` 四个 aten 算子，语义上只有一个矩阵乘；一个 `Qwen3DecoderLayer` 只做两次残差 add。所以节点 = "同一次模块调用里连续执行的一串非透明算子"，透明算子（view/transpose/reshape/…）只传播别名、不建节点。
2. **权重不能当图输入**：权重常以 `weight.t()` 视图进入算子，张量对象变了但 storage 没变。只按对象身份过滤会多出 215 个"权重输入节点"，必须按 storage 过滤。
3. **折叠会造假环**：第 0 层的残差输入来自 embedding，折叠后看起来像 `L*.residual_add@1 → L*.input_layernorm` 的回边。按原始层号判断跨层边并单独标记后，拓扑分层才正常。

### 单测

```bash
python -B -m unittest -v nanovllm_selector.test_extract   # 19 项，约 48 秒
```

模型目录不存在时整组跳过，可用 `EXTRACT_MODEL` 指向别处。
