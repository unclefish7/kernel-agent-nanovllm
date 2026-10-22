# Norm + MLP + 残差连接实验（历史原型，已归档）

本实验已停止扩展，不作为 nano-vLLM 真实推理接入实现；状态及不兼容原因见 [目录说明](README.md)。本页记录独立计算片段的历史方法和结果，不代表完整模型生成已经通过验证。后续方向转到根目录 `nanovllm_selector/`，不再按文中的旧“下一步”推进。

## 任务与参考来源

不生成 kernel，让 Agent 读取 Transformers 4.57.3 的真实 Qwen3DecoderLayer/Phi3DecoderLayer 源码，为 Attention 之后的连续计算片段选择已有算子：

```text
x（Attention 输出）+ residual（Attention 前的输入）
        ↓
保存相加结果 summed → RMSNorm → MLP → 加回 summed
```

`tail_task.py` 从原 forward 的第一次残差相加开始提取语句，保留实际 post_attention_layernorm、MLP、Phi3 dropout 调用以及原权重。Phi3 在 eval 模式下 dropout 为恒等操作。它不是任意模型提取器，不运行 Attention，也不验证 KV cache 或整个模型推理。

每个案例另运行真实 Decoder forward，只将 self_attn 临时替换为返回指定 x 的模块；验证提取片段的输出与其一致。不要把人工组合方案作为 reference。

设置：MI100 GPU 0、FP32、无梯度、hidden_size=256、intermediate_size=1024，随机共享权重；Norm 权重随机设为 0.5～1.5，不仅测试全 1 权重。每个案例使用 3 行和目标行数两组正负随机输入，目标行数为 1、128、512。不下载 checkpoint。

## 模块与候选

| 文件 | 职责 |
| --- | --- |
| tail_task.py | 初始化真实 Decoder，提取源码，构造两个边界输入，检查原 forward |
| tail_candidates.py | 候选目录、参数绑定和适配；构造 12 种人工组合；按当前负载调优 |
| agent.py | 通用 prompt、真实 API 请求、候选及张量引用检查 |
| tail_demo.py | 原始对照、独立候选检查、正确性、共同计时及结果保存 |
| tail_suite.py | 顺序运行两种模型 × 三种行数，汇总结果 |
| test_tail.py | 源文件哈希、双输出及任意张量名、权重共享、非法方案检查 |

候选不只有不同包装：有两个不同的 RMSNorm kernel、两个不同的 SwiGLU kernel，以及独立的融合 Add+RMSNorm kernel。另有 PyTorch 基础操作、原 Norm 和原 Linear。Qwen3 目录有 12 个调用条目，Phi3 也有 12 个：Phi3 多一个切分条目、少一个 Linear；不能把条目数说成独立 kernel 数量。

| 类别 | 候选及来源 | 适配说明 |
| --- | --- | --- |
| RMSNorm | rms_reference | 原 Transformers 模块和权重 |
| RMSNorm | rms_row：GEAK-eval rmsnorm_fused.py | 原 TritonLlamaRMSNorm |
| RMSNorm | rms_loop：GEAK-eval rmsnorm_implementation.py | 增减一个 batch 维的 view，调用原 wrapper |
| Add + RMSNorm | add_rms：GEAK-eval layer_norm_ops.py | 原 _layer_norm_fwd，设置 RMS 模式、无 bias；返回顺序为 normalized、summed，不原地修改输入 |
| SwiGLU | swiglu_tiles：GEAK-eval swiglu_fwd.py | Qwen3 直接传两个 Tensor 和 stride 启动原 kernel，不再拼接；Phi3 直接使用原 packed wrapper |
| SwiGLU | swiglu_row：GEAK-eval swiglu_triton.py | 原 wrapper 假设输入输出行 stride 相同；先连续化 gate/up，再取第三个返回值，复制开销计入计时 |
| 基础操作 | torch_add、torch_silu、torch_mul、Phi3 的 split_halves | 切分返回 view，不保证连续 |
| Linear | 原 mlp.* Linear | 共享原始对象、权重；每个恰好调用一次，不拆分或合并 |

新增上游文件的前向定义存于 `vendor/`，保留原版权头，附 LICENSE；`manifest.json` 记录 GEAK-eval 提交、原路径、原文件 SHA256 和本地 SHA256。仅移除测试及未使用的反向定义，不修改保留的函数或 kernel 正文。已有 swiglu_fwd.py 仍使用 Compose 的只读挂载。

## Agent 看到和输出什么

输入包括真实 Decoder、提取片段、MLP、RMSNorm、激活源码，以及输入布局、执行模式、硬件和候选目录。权重及 eps 已由执行器绑定。外部张量只有 x、residual，中间张量名称自行指定，双输出用数组表达。

不提供人工方案、测得的性能、最快组合名称或应该在哪里融合的答案。每个案例只做一次成功选择，不加入性能反馈或自动修复循环。静态检查先验证 ID、输入数量、张量依赖和返回数量，随后用真实 reference 检查完整片段输出。

## 逐步运行

复用已有 Compose 镜像与容器，不需要新增依赖。下列 Python 命令在容器 `/workspace/operator_selector` 内执行：

```bash
python -B tail_demo.py reference --model qwen3 --tokens 128
python -B tail_demo.py candidates --model qwen3 --tokens 128
python -B tail_demo.py manual --model qwen3 --tokens 128
python -B tail_demo.py agent --model qwen3 --tokens 128
python -B tail_demo.py compare --model qwen3 --tokens 128
python -B -m unittest test_tail -v
```

只有 agent 模式请求 API，compare 复用保存的方案。每次在同一模式运行会覆盖该目录的结果。模型可以换为 phi3，tokens 可以为 1、128、512。

运行全部六个案例（允许为缺失方案请求 API）：

```bash
python -B tail_suite.py --generate
```

不加 --generate 时不调用 API；缺少方案会停止。已有方案直接复用，但原始对照、候选和人工组合会重新运行，随后共同评测。任一子流程失败即停止，没有忽略失败后继续报成功。

注意：`docker compose start` 不刷新旧容器中的密钥。如果当前 `.env` 已更新，可从宿主机仓库根目录显式注入，再运行全套流程，无需重建容器：

```bash
docker exec --env-file operator_selector/.env operator-selector-rocm-dev python -B tail_suite.py --generate
```

不要将密钥写入源码或聊天。当前 `.env` 与缓存、输出均被 Git 忽略。

## 评测与结果解释

- 每个候选先独立与 PyTorch/原模块对照，融合 Add+RMSNorm 的两个输出都检查。
- 再检查每个完整组合与真实片段一致，rtol=1e-4、atol=1e-5，并检查输入没有被改写。
- 人工对照为 4 种归一化路径（原模块、row、loop、融合加法）× 3 种激活路径（PyTorch 分开、tiles、row），共 12 种合法组合。
- 计时继续复用 KernelBench timing.py。每项预热 10 次、保留 100 个原始样本，沿用丢弃首个样本及冷 L2 策略，正序/反序/正序测三轮。
- 每个规模先重新调优 tiles 和 Add+RMSNorm，清除前一规模的 autotuner 配置缓存，再固定该次配置；所有人工与 Agent 方案共用它。固定配置的 RMSNorm/SwiGLU 同样记录上游来源。
- 汇总耗时为三轮中位数的中位数；加速比是本轮 reference 汇总耗时除以方案汇总耗时。计时不包含 API 请求、编译、调优及正确性检查，包含布局转换和 Python 执行器开销。
- `agent_slowdown_vs_best` 是 Agent 耗时 / 12 种人工组合中最快者耗时，越接近 1 越好。不是理论上所有合法计算图的全局最优；微小差异可能只是波动。
- 验证通过只能说明当前模型、FP32、布局和规格下成立，不能推出任意未知模型或低精度均成立，更不能证明 Agent 比规则或穷举更好。

## 实际完成结果

已实际运行两种模型 × 三种行数的六个案例，所有案例均依次完成原 Decoder 对照、独立候选检查、12 种人工方案、真实 Agent 请求和共同评测。六次成功 API 调用合计使用 28,538 tokens；没有根据测得结果修改 Agent 的方案或追加性能反馈。

各案例的 12 个人工方案和 Agent 方案均通过两种行数的完整片段正确性及输入未改写检查。共保存 25,200 个共同计时样本（6 案例 × 14 个实现含 reference × 3 轮 × 100 次）。以下为三轮中位数的中位数，耗时单位为微秒：

| 模型 | 行数 | reference | Agent | 12 种人工组合中最快 | Agent 加速比 | Agent / 最快耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 | 1 | 72.16 | 69.28 | 48.88 | 1.04× | 1.42 |
| Qwen3 | 128 | 122.87 | 94.24 | 94.24 | 1.30× | 1.00 |
| Qwen3 | 512 | 116.31 | 83.52 | 83.52 | 1.39× | 1.00 |
| Phi3 | 1 | 60.32 | 79.60 | 41.92 | 0.76× | 1.90 |
| Phi3 | 128 | 126.24 | 91.84 | 92.00 | 1.37× | 1.00 |
| Phi3 | 512 | 116.16 | 77.60 | 74.08 | 1.50× | 1.05 |

六个案例的 Agent 均选用 add_rms 和 swiglu_tiles，中间保留各模型的原 Linear，最后加回 add_rms 的第二个返回值。Qwen3 为六步方案，Phi3 为五步方案，后者直接读取打包投影。它能理解完整依赖并生成可执行方案，但没有随输入规模选择不同组合。

在本轮共同对照中，Qwen3/1 最快是 row__tiles，Phi3/1 最快是 row__row；不能将小规模退化简单归因于某一个融合 kernel，也不能认为保留 PyTorch 激活一定最快。人工独立运行与共同对照之间的微小排名会变化，文档表格以 suite 的共同对照为准。

相同方案也有计时波动。例如 Phi3/1 的人工 add_fused__tiles 三轮为 79.92、72.96、72.40 us，Agent 三轮为 79.60、72.72、85.04 us；它们共用候选与执行器，不能把这段差异归因于 API 或 Agent 内部推理。两者都明显慢于本轮约 42 us 的最快人工组合。未对单 kernel 做 profiling，不在这里断言退化根因。Phi3/128 的 Agent 略快于同方案人工实现也视为波动，不代表超越了候选组合。

容器内 `python -B -m unittest test_tail -v` 的 5 项测试全部通过：本地上游文件哈希、与原 Decoder 对照及任意中间名/双输出、错误方案拒绝、实际返回数量检查，以及旧 Qwen3/Phi3 MLP 的分开/融合回归。另核对了保留定义与上游 AST 一致，14 个 Python 文件语法检查通过。没有运行 Attention、完整模型生成或 nano-vLLM 推理。

首次 Qwen3/128 API 调用曾返回 401：旧容器保存的密钥与当前 `.env` 不同。显式注入当前 `.env` 后成功。失败的上下文和日志保存在 `outputs/tail/qwen3/128/agent_auth_failed/` 与 `agent_auth_failed.log`；未修改密钥或手工修正模型方案。认证失败请求不计入上面的六次成功调用统计。

本轮结论是：**复杂片段的识别、算子组合和评测流程已跑通；正确选择计算不等于正确预测性能，小规模案例已经暴露出静态选择的不足。** 下一步可以围绕这些失败案例做 profiling，再比较静态规则或有限性能反馈，而不是先扩大到 Attention。

## 产物与清理

每个案例保存在 `outputs/tail/<model>/<tokens>/`：各阶段的 task、reference_check、catalog、candidate_check、tuning、调用方案、results 及日志；agent 目录还保存 prompt、原始响应和 token 用量，不保存密钥。compare 的 results 保留每个方案的所有原始计时样本。

`outputs/tail/suite.json` 汇总六个案例，`outputs/tail/suite.log` 是本次批次日志，`outputs/tail/tests.log` 是补充测试日志。缓存位于项目 `cache/`，不改变原来的缓存路径。

如不再保留本轮结果，可在容器中执行下面的命令；只删除本轮产物，不删除源码、旧 MLP 结果、密钥或缓存。未提交结果不能从 Git 恢复。本次没有执行清理：

```bash
rm -r /workspace/operator_selector/outputs/tail
```
