# nano-vLLM 真实 Qwen3 推理的算子选择 Agent

## 1. 当前状态与阶段目标

**第一步原生推理基线已完成冒烟验证，尚未实现 Agent 接入。** 已在独立 Compose 容器内加载真实 Qwen3-0.6B，使用 MI100 单卡、BF16、小批量和 eager 模式完成文本生成、分块 prefill、多步 decode 和缓存复用。`nano-vllm/` 源码未修改；不将冒烟通过表述为严格数值等价或性能优化。

用户已将模型下载到宿主机 `tmp/models/Qwen3-0.6B/`，容器内对应 `/workspace/tmp/models/Qwen3-0.6B/`。权重 SHA256 与下载元数据一致，311 个张量及配置、分词器均能离线读取。当前没有自动下载任务，也没有调用 Agent API。

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
    ↓ 静态检查、真实输入正确性检查
可安装的已验证方案
```

采集和检查可以额外执行参考计算，但这些开销不算入正式推理性能。优先保存规格和误差统计，不默认保存大量激活张量。

### 正式运行：不请求 Agent API

```text
构建 nano-vLLM Qwen3 模型
    ↓
按原映射加载真实权重
    ↓
读取已验证方案，安装选中的实现
    ↓
warmup，随后由原框架管理缓存和执行
    ↓
照常调用 LLM.generate()，完成 prefill 和多步 decode
```

安装点应位于权重加载后、模型预热前。真实规格在前一轮原生基线中采集，不能等每个 token 到来时再调用 Agent。第一版使用 eager 模式，不同时处理 graph capture 或复杂编译接入。

## 6. 需要准备的模块、输入和产出

以下是职责规划，不代表这些文件已经存在；先保持一个入口和少量清楚的模块，不复制旧原型的交叉调用结构。

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
- [ ] Agent 使用真实运行规格选择已登记实现，保存方案和原始 API 记录。
- [ ] 三个接口保持权重、残差、dtype、shape、device/stream 与输入修改约定。
- [ ] 至少一个接入点在真实推理中实际调用非原生候选，并通过正确性验证；如果 Agent 全部选原生，只证明回退和流程，不替代优化候选接入验收，也不能人工改选后伪称 Agent 的决定。
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

依赖清单见 [requirements.txt](requirements.txt)。当前容器已安装该清单，Torch/ROCm 沿用镜像；没有调用 Agent API。重建容器后需要重新安装这些容器内依赖，不需要重新下载宿主机已有模型。以下检查在容器内执行：

```bash
python -B -m pip check
python -B -m unittest discover -s /workspace/nano-vllm/tests -v
```

已有检查结果为依赖检查通过，4 项单元测试通过，分别覆盖 CPU/GPU Attention 兼容路径、预热分块边界和 FP32 残差。容器只暴露 GPU 0；完整模型的原生冒烟结果及 BF16 数值对照边界见本文开头。

运行前先确认本地模型位置。缺少权重时，说明下载来源、保存路径、缓存路径和预计大小，获得确认后再下载；遵守仓库关于 `tmp/` 和目录外写入的要求。不在检查环境时隐式拉取镜像、更新依赖或下载模型。

运行产物放在本目录的 `outputs/`；缓存通过 Compose 环境变量显式设置在本目录 `cache/`。当前产物包括安装日志 `outputs/install.log`、依赖检查 `outputs/pip_check.log`、单元测试 `outputs/unit_tests.log`、早期失败下载日志 `outputs/model_download.log`、原生生成日志 `outputs/native_smoke.log`、初次数值诊断 `outputs/reference_probe.log`，以及可复现入口生成的 `outputs/baseline.log` 和 `outputs/baseline.json`。`outputs/`、`cache/`、本地 `.env` 及仓库根目录 `tmp/` 已加入忽略规则，不提交模型、密钥或运行产物。

上述报告及日志如不再需要，可逐个用 `rm nanovllm_selector/outputs/<文件名>` 清理；缓存可在停止相关任务后清理 `nanovllm_selector/cache/` 中对应子目录，下次运行会重新生成或重新编译。这里仅说明清理方法，没有执行清理；`docker compose down` 不会清理这些宿主机文件，也不会删除模型。

后续若需要修改 nano-vLLM，应保持改动集中，先在子模块单独提交，再在父仓库提交引用更新；按仓库规范处理 fork 上的对应提交。当前原生基线未修改子模块，尚无 Agent 接入结果或正式性能对比数据。
