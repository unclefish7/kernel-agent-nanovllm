# 最小算子选择 Agent

## 目标与总体安排

以真实 Hugging Face Transformers 的 Qwen3MLP、Phi3MLP 为 reference，让 Agent 阅读其初始化和 forward 源码，划分计算片段，并选择已有算子。Agent 输出调用方案，不编写 kernel。

分两步完成：

1. **验证候选和评测器**：人工分别指定分开执行与融合执行方案，验证完整 MLP 输出和替换片段的正确性，再计时。
2. **接入 Agent**：将真实模块源码、模型配置、输入规格、硬件信息和候选目录交给 DeepSeek Flash。它返回 JSON 方案，执行相同的组装与验证流程。

当前处理这两种 MLP 的激活、乘法及打包张量切分部分，所有 Linear 保留原实现和原权重。没有集成 nano-vLLM，没有支持任意未知模型，也没有实现自动搜索或修复。

## 模块与输入输出

| 文件 | 输入 | 功能 | 产出 |
| --- | --- | --- | --- |
| task.py | 模型名称 qwen3 或 phi3 | 初始化真实 reference，提取源码和 Linear 规格，准备输入 | reference、两组输入、模型上下文 |
| candidates.py | 原模块、已有 GEAK-eval 算子、调用方案 | 登记候选、按计时输入调优并固定共享配置、执行方案 | 可调用的替换 MLP、融合配置 |
| agent.py | 模型上下文、候选目录 | 请求 DeepSeek 识别计算、划分并选择 | JSON 方案与理由 |
| evaluation.py | reference、方案字典、同一组输入 | 正确性检查，交替测三轮，复用 KernelBench 计时 | 配置、测试顺序、原始样本、统计与加速比 |
| demo.py | manual、agent 或 compare 模式 | 串联模块；compare 复用已保存的 Agent 方案 | outputs/ 下的任务、方案和结果 |

方案中的每一步包含 op、inputs、output，op 必须是候选目录里的 ID。执行器通过已登记函数调用，不执行 Agent 生成的 Python 源码。单输出的 output 为字符串，切分操作的 output 为两个名字的数组；张量按名字存放在执行器的字典中。

候选目录中的 parameters 只表示接口形参，方案 inputs 才表示实际张量引用；唯一外部张量名为 x。请求返回后先检查候选 ID、参数数量和张量是否已定义，再执行方案。两种模型共用同一套 prompt 规则，不在 prompt 中预填某个模型的选择答案。

## Reference 与候选来源

- Reference：transformers==4.57.3 的 `transformers.models.qwen3.modeling_qwen3.Qwen3MLP`、`transformers.models.phi3.modeling_phi3.Phi3MLP`。直接实例化，不手写一个类似模型代替。
- 配置：hidden_size=256、intermediate_size=1024、hidden_act=silu，FP32、MI100 GPU 0。
- 权重：随机初始化；reference 与替换方案共享同一组 Linear 权重，不下载 checkpoint。
- 正确性输入：[8, 256] 和 [128, 256]，随机输入乘以 2，包含正负值。
- 计时输入：[128, 256]；gate/up 的 shape 为 [128, 1024]，Phi3 的打包投影为 [128, 2048]。
- 分开执行：PyTorch SiLU 和 Multiply。
- 融合候选：本地 GEAK-eval 的 `_swiglu_fwd`，源文件为 `geak_eval/data/TritonBench/data/TritonBench_G_v1/swiglu_fwd.py`。它是现成 Triton 实现，不由 Agent 或本项目生成。
- GEAK-eval 文件底部有导入即运行的测试，适配器只加载测试分隔线之前的原始定义；源文件保持不变，Triton JIT 仍可读取其真实路径。
- Qwen3 的 triton_swiglu 适配器需要将 gate/up 拼接后调用；拼接成本包含在片段和完整 MLP 的计时中。Phi3 的 triton_swiglu_packed 直接传入原本已打包的投影结果，不做拼接。两个接口复用同一个已有 kernel。

| Reference | 原始投影 | 原始非 Linear 部分 | 融合候选接口 |
| --- | --- | --- | --- |
| Qwen3MLP | gate_proj(x)、up_proj(x)，各输出 [T, 1024] | silu(gate) * up | 两个输入，适配器先拼接 |
| Phi3MLP | gate_up_proj(x)，输出 [T, 2048] | 沿最后一维切成 gate/up，再 up * silu(gate) | 一个打包输入，直接调用 |

原 Linear 按真实模块动态登记。Phi3 的人工分开方案使用 split_halves 产生两个 Tensor，再调用 SiLU 和 Multiply；融合方案直接覆盖这段计算，不拆分原 Linear 权重。评测器的片段输入准备目前显式支持上述两种格式；完整 MLP 的正确性始终与真实 reference.forward 比较，不能将它称作任意模型的自动提取器。

## 环境与运行

复用本地镜像 `jerryjwc/rocm-dev:10.0`，新建独立 Compose 容器 `operator-selector-rocm-dev`，不 build、不拉取镜像。

```bash
cd operator_selector
docker compose up -d
docker exec -it operator-selector-rocm-dev zsh
```

容器内工作目录为 `/workspace/operator_selector`。源码通过 bind mount 共享；GEAK-eval 算子文件和 KernelBench timing.py 单独只读挂载。不需要安装完整 vLLM、GEAK-eval 或 KernelBench。

新建或重建容器后，安装 Transformers 和 HTTP 客户端依赖（不替换镜像已有的 Torch/Triton）：

```bash
python -m pip install --no-cache-dir --retries 0 --timeout 15 -r requirements.txt
```

先运行人工方案：

```bash
python demo.py manual --model phi3
```

再运行 Agent（会产生 API 费用）：

```bash
python demo.py agent --model phi3
```

已有 Agent 方案后，用同一进程公平比较三份方案，不再调用 API：

```bash
python demo.py compare --model phi3
```

将 --model 改为 qwen3 即可运行原模型；不指定时默认 qwen3。当前输出按模型隔离：compare 读取 `outputs/<model>/agent/selected_plan.json`，将人工分开、人工融合和保存的 Agent 方案共同评测，结果写入 `outputs/<model>/compare/`。同模型、同模式再次运行会覆盖对应结果。旧版 outputs/manual/、outputs/agent/、outputs/compare/ 保留为 Qwen3 历史记录，新版不会自动读取这些旧目录；运行新版 Qwen3 compare 前需先运行该模型的 agent 模式生成新路径下的方案。

Compose 默认读取 `../references/reps/KernelBench/.env`，复用已有 DeepSeek 密钥。也可在本目录创建 `.env` 覆盖，字段见 `.env.example`；不要提交密钥，不需要在聊天中提供。默认模型为 deepseek-flash。

结束后在宿主机使用 docker compose stop；不再需要容器时使用 docker compose down。安装依赖在容器 `/opt/pytorch-venv/`，stop/start 保留，down 后重建需要重新安装。

## 评测说明

- 比较两组输入上的激活片段和完整 MLP 输出，rtol=1e-4、atol=1e-5。
- 随机权重、输入保持相同；关闭梯度；不训练，也不声称支持反向传播。
- 直接复用 KernelBench timing.py 的 Event 计时函数，不加载其整个包或生成流程。
- 先按计时输入 [128, 256] 得到 gate/up [128, 1024] 或打包投影 [128, 2048]，用此规模调优融合实现，再进行 [8, 256] 和 [128, 256] 的正确性测试。避免先按 8 行选配置再拿 128 行计时。
- 一次评测中所有方案共享同一个加载的融合实现，调优后只保留选中的配置；记录 BLOCK_N、num_warps、num_stages、num_ctas 和调优 shape。不是修改上游 kernel 源码。
- 每项 warmup 10 次、保留 100 个样本，并丢弃第一个计时样本，使用原函数的冷 L2 策略。每个 scope 测三轮，第二轮将 reference/方案的顺序反转。
- `results.json` 的 fusion 字段记录配置；rounds 数组记录每轮 order、activation 和 mlp；各实现的 samples_ms 保存全部 100 个原始样本。
- 报告均值、中位数、波动和 reference/selected 加速比，单位为 ms。保留拼接及 Python 方案执行产生的实际开销。
- manual 与 agent 是独立进程，各自的结果不能保证相同调优和运行状态。人工/Agent 的直接对照应使用 compare；模型 API 请求不在计时范围内。
- 单次 demo 只证明当前环境和规格能执行，不代表选到了候选库最优实现，也不能证明 Agent 优于规则或穷举。

## 当前工作进度

### Phi3 适配与实际验证

2026-10-14 已运行人工方案、真实 Agent 调用和共同对照；另运行 Qwen3 人工回归，两种模型的分开与融合方案均通过两种输入规模的片段和完整 MLP 检查。五个 Python 文件均通过语法检查。容器内补充测试还验证了四种组合（两种模型 × 分开/融合）的任意中间张量命名、原 Linear 对象共享，以及方案检查能拒绝未定义张量；HTTP 使用模拟，没有新增 API 费用或测试文件。

实际 Agent 方案为三步：gate_up_proj(x) → triton_swiglu_packed → down_proj。中间张量名由 Agent 自行给出，执行器不依赖固定的 gate/up/hidden 名称。它识别了原始打包投影与 chunk 的语义，并选择无需额外拼接的已有接口；没有生成新 kernel。

共同对照使用 MI100 GPU 0、FP32、输入 [128, 256]，各方案共享按计时规模调优的 BLOCK_N=256、num_warps=4、num_stages=3、num_ctas=1。下表耗时为各轮中位数的中位数，单位为微秒；加速比为该汇总 reference 耗时除以方案耗时。

| 方案 | 非 Linear 片段 us | 片段加速比 | 完整 MLP us | 完整 MLP 加速比 |
| --- | --- | --- | --- | --- |
| Transformers reference | 24.16 | 1.00× | 96.16 | 1.00× |
| 人工分开 | 24.16 | 1.00× | 95.84 | 1.00× |
| 人工融合 | 13.76 | 1.76× | 85.60 | 1.12× |
| Agent 方案 | 13.76 | 1.76× | 85.60 | 1.12× |

本次完整 MLP 有约 1.12× 加速，人工融合与 Agent 结果基本一致。Phi3 不需要 Qwen3 适配器中的额外拼接；同时两种模型的 Linear 数量和实际张量布局也不同，不能把跨模型的耗时差异全部归因于某一个因素。这仍不是 Agent 优于穷举或规则的证据。

首次 API 返回正确的三步计算，但将外部张量写为 input 而非 x，执行失败。随后将候选形参与实际张量名明确区分，增加执行前引用检查，再调用 API 成功；没有手动改写 Agent 答案，也没有实现自动修复循环。首次和第二次调用分别使用 2375、3654 tokens，共两次计费请求，均 finish_reason=stop。

- `outputs/phi3/manual/`：人工方案与结果；`outputs/phi3/manual.log` 为日志。
- `outputs/phi3/agent/`：成功调用的 task、prompt、response、selected_plan 和 results；`outputs/phi3/agent.log` 为日志。
- `outputs/phi3/agent_first_attempt/`、`outputs/phi3/agent_first_attempt.log`：首次失败的上下文、响应、方案与日志。
- `outputs/phi3/compare/results.json`：共同对照的正确性、配置与 2400 个原始计时样本；`outputs/phi3/compare.log` 为日志。
- `outputs/qwen3/manual/`、`outputs/qwen3/manual.log`：Qwen3 人工回归结果及日志，没有为回归重新调用 API。

### Qwen3 历史验证

- 两步均已完成实际运行验证，日期为 2026-10-14。
- 独立容器正常运行；Torch 2.12.0+rocm10.0.0、Triton 3.8.0、Transformers 4.57.3，pip check 通过。
- 人工分开方案和融合方案均通过两组输入的激活片段、完整 MLP 正确性检查。
- DeepSeek Flash 实际返回了四步方案：gate_proj、up_proj、triton_swiglu、down_proj。JSON 按登记接口执行，不生成或修改 kernel。Agent 方案也通过了相同正确性检查。
- 本次一次 API 调用使用 902 个输入 tokens、1411 个输出 tokens（包含 1193 个思考 tokens），finish_reason=stop。完整元数据保存在 outputs/agent/response.json，未保存密钥。
- 正确性容差保持 rtol=1e-4、atol=1e-5，未运行不相关测试或完整模型推理。

最初独立运行的单轮中位数计时如下，单位为微秒。仅作为历史记录，不用它判定 Agent 执行器的额外开销。

| 方案 | 片段 reference / selected | 片段加速比 | 完整 MLP reference / selected | 完整 MLP 加速比 |
| --- | --- | --- | --- | --- |
| 人工分开 | 17.12 / 17.12 | 1.000 | 89.44 / 90.24 | 0.991 |
| 人工融合 | 17.12 / 28.32 | 0.605 | 89.60 / 100.63 | 0.890 |
| Agent 选择融合 | 17.44 / 70.72 | 0.247 | 90.40 / 144.63 | 0.625 |

这些结果证明闭环正确执行，而不是性能优化成功。当前小尺寸下，融合方案的拼接、Triton 调用等开销可能抵消收益；具体瓶颈尚未 profiling。Agent 与人工融合计划完全相同，且共用执行器；两次独立运行计时有明显差异，不能将差异直接归因于 Agent。当前 Agent 没有接收人工 benchmark 结果，选择不是经过性能搜索的最优结论。

评测可比性修复后，已实际运行 `python demo.py compare`，没有调用 API。三份方案均通过两组输入的片段和完整 MLP 正确性检查；共用按 [128, 1024] 调优的 BLOCK_N=256、num_warps=4、num_stages=3、num_ctas=1。

| 方案 | 三轮片段中位耗时 us | 三轮完整 MLP 中位耗时 us | 片段中位数加速比 | 完整 MLP 中位数加速比 |
| --- | --- | --- | --- | --- |
| 人工分开 | 16.96、17.12、16.96 | 90.40、90.40、90.40 | 0.991–1.000 | 0.998–1.000 |
| 人工融合 | 22.88、22.88、22.88 | 95.52、95.52、95.68 | 0.741 | 0.943–0.946 |
| 保存的 Agent 方案 | 22.88、22.88、22.88 | 95.44、95.68、95.68 | 0.741 | 0.943–0.947 |

本轮 reference 片段约 16.96 us，完整 MLP 约 90.2–90.4 us。人工融合与 Agent 差距基本消失，但不能据此确定旧结果异常的唯一原因。当前融合仍没有加速。2400 个原始样本（4 个实现含 reference × 2 个 scope × 3 轮 × 100 次）完整保存在 outputs/compare/results.json。

outputs/manual/ 保存两个人工计划及 results.json；outputs/agent/ 保存真实上下文、prompt.json、response.json、selected_plan.json 和 results.json；outputs/manual.log、outputs/agent.log 为运行日志。outputs/candidate_check.log 记录已有 GEAK-eval 融合候选的独立正确性检查。

所有功能通过 demo 命令验证，不另建测试框架。原 GEAK-eval、KernelBench、nano-vLLM 仓库未修改。

## 后续安排

1. 当前已增加 Phi3 打包投影写法；继续增加真实 MLP 的语义变化，检查识别和适配是否正确。
2. 增加输入规模与已有候选；给 Agent 测量反馈，再比较选择与候选库穷举最优的差距。
3. 对照 vLLM GLUFuser 的规则识别，找出规则匹配遗漏而 Agent 能正确处理的案例。
4. 再扩展 RMSNorm、Q/K Norm 与 RoPE，最后考虑 Attention 及 nano-vLLM 接入。

## 产物与清理

outputs/ 保存任务上下文、方案、API 响应和评测结果；cache/ 保存安装临时文件、Triton/ROCm/Hugging Face 缓存。当前还存在 cache/tmp/torchinductor_root/，为运行时创建的缓存目录；pip 临时下载目录已自动清理。两者及 .env 均被忽略。本实验不下载模型权重或数据集。

如不再保留产物，可在容器内删除这两个明确目录（会丢失结果，未提交的结果不能从 Git 恢复）：

```bash
rm -r /workspace/operator_selector/outputs /workspace/operator_selector/cache
```

删除 cache 后重新运行前执行 `mkdir -p cache/tmp`；无需删除源码或密钥。目录中已存在一个空的 cache/home（早期配置留下），与实际 HOME 无关，可用 `rmdir cache/home` 单独删除。
