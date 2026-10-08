# Kernel Agent 相关论文阅读清单

本文整理与 Kernel/Operator Generator Agent 方向相关的代表性论文，重点关注以下流程：

```text
PyTorch 或真实模型实现
→ 算子提取与分析
→ CUDA、HIP 或 Triton kernel 生成
→ correctness 验证
→ benchmark 与 profiling
→ 自动修复和性能优化
→ 推理框架集成
```

整理时间：2026-09-23。该方向发展较快，论文状态和开源实现可能继续变化。

## 推荐阅读顺序

第一轮建议只阅读四篇，先建立完整的研究脉络：

1. KernelBench：理解任务、数据集和评价指标如何定义。
2. Astra：理解如何优化来自真实 LLM 推理框架的 kernel。
3. STARK：理解多 Agent、反馈和搜索机制。
4. KernelEvolve：理解异构硬件和生产环境中的 Agentic kernel 开发。

第二轮再阅读 TritonBench、KernelBench-X、KernelGenBench 等评测工作，重点分析现有评价体系的不足。最后阅读 RealisticTritonBench 和 Atrex-Bench，理解研究如何从孤立算子转向真实框架和端到端推理负载。

## 第一轮：核心论文

### 1. KernelBench: Can LLMs Write Efficient GPU Kernels?

- 论文：[arXiv:2502.10517](https://arxiv.org/abs/2502.10517)
- 代码：[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- 本地参考仓库：`references/reps/KernelBench/`
- 发表：ICML 2025

KernelBench 将任务定义为：给定 PyTorch reference implementation，让模型生成正确且高效的 GPU kernel。它提供 250 个不同复杂度的 PyTorch workload，并使用 `fast_p` 衡量生成结果是否正确，以及是否达到指定的性能提升阈值。

阅读重点：

- reference、测试输入和生成代码之间的接口；
- correctness 与性能测试的执行顺序；
- Level 1、Level 2、Level 3 的任务差异；
- `fast_p` 指标如何同时表达正确性和性能；
- execution feedback 为什么能改善结果，以及它的局限性。

### 2. Astra: A Multi-Agent System for GPU Kernel Performance Optimization

- 论文：[arXiv:2509.07506](https://arxiv.org/abs/2509.07506)

Astra 不从 PyTorch reference 开始生成新 kernel，而是从 SGLang 中提取已有 CUDA kernel，再通过编码、测试、profiling 和规划等专门 Agent 进行迭代优化。生成结果可以重新接入原推理框架。

阅读重点：

- 为什么优化已有 kernel 与从 PyTorch 生成 kernel 是两类不同任务；
- 多 Agent 之间如何划分 planning、coding、testing 和 profiling；
- profiler 信息如何转化为下一轮修改建议；
- 单 kernel speedup 如何与真实推理框架联系起来。

这篇论文和本项目的“面向真实 LLM inference framework”目标非常接近，但主要基于 CUDA 和 SGLang。

### 3. STARK: Strategic Team of Agents for Refining Kernels

- 论文：[arXiv:2510.16996](https://arxiv.org/abs/2510.16996)
- 发表：ICLR 2026

STARK 将 kernel 优化视为一个需要持续搜索的问题，使用多个 Agent 分别承担规划、编码和反思等职责，并通过动态上下文和搜索策略在已有候选之间平衡探索与利用。

阅读重点：

- 多 Agent 相比单 Agent 的具体作用；
- grounded instruction 如何把优化策略转换成代码修改；
- 动态上下文如何控制历史尝试和反馈信息；
- 候选搜索、选择与终止条件如何设计。

### 4. KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

- 论文：[arXiv:2512.23236](https://arxiv.org/abs/2512.23236)
- 发表：ISCA 2026

KernelEvolve 面向 NVIDIA GPU、AMD GPU 和 Meta 自研加速器等异构硬件，将 kernel 优化建模为图搜索问题，并支持 Triton、CuTe DSL 等不同编程抽象。它强调从 kernel specification 出发，在不同模型、算子和硬件之间自动生成与优化实现。

阅读重点：

- kernel specification 包含哪些信息；
- Agent 如何选择编程语言和优化策略；
- retrieval 与历史优化经验如何参与 prompt 构造；
- 同一算子在不同硬件上的正确性和性能如何评价。

它是当前清单中与 AMD 和异构硬件适配最直接相关的论文。

## 第二轮：Triton 与评测体系

### 5. TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators

- 论文：[arXiv:2502.14752](https://arxiv.org/abs/2502.14752)
- 代码：[thunlp/TritonBench](https://github.com/thunlp/TritonBench)

TritonBench 专门评估 LLM 生成 Triton operator 的能力，包含来自开源项目的真实算子，以及与 PyTorch 接口对应的生成任务。它同时关注 functional correctness 和 GPU efficiency。

阅读重点：

- TritonBench-G 与 TritonBench-T 的数据来源；
- 自然语言描述、PyTorch 接口和 Triton 实现之间的关系；
- 如何衡量生成 operator 的 GPU 利用效率；
- Triton 是否真的降低了模型生成高性能 kernel 的难度。

### 6. KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels

- 论文：[arXiv:2605.04956](https://arxiv.org/abs/2605.04956)
- 代码：[BonnieW05/KernelBenchX](https://github.com/BonnieW05/KernelBenchX)

KernelBench-X 按任务类型分析 kernel generation 的失败边界，并研究 correctness、性能与硬件之间的关系。论文特别指出，生成结果通过正确性测试并不意味着它比 PyTorch baseline 更快，跨硬件的 speedup 也可能发生明显变化。

阅读重点：

- 哪些算子类别最难生成；
- 编译成功、数值正确和性能有效三者为何不能等同；
- 不同硬件上的性能方差；
- 迭代修复为什么可能提高正确率却降低平均性能。

### 7. KernelGenBench: A Multi-Source and Multi-Chip Benchmark for LLM-based Kernel Generation

- 论文：[arXiv:2607.27231](https://arxiv.org/abs/2607.27231)

KernelGenBench 从多个算子来源构建任务，并在多种芯片平台上评价 LLM 和 Agent 生成的 Triton kernel，重点研究 hardware portability 和 Agent 成本。

阅读重点：

- benchmark 如何避免只围绕 PyTorch 人工小算子；
- 同一个生成方法如何跨芯片测试；
- hardware-specific optimization 与可移植性之间的矛盾；
- 每个成功算子的推理轮次和 token 成本。

这篇论文与 MI100 环境密切相关，因为它提醒我们不能把某张 GPU 上的成功结果直接推广到 gfx908。

## 第三轮：真实框架与生产负载

### 8. RealisticTritonBench: A Benchmark for Triton-Kernel Generation in Real-World AI Frameworks

- 论文：[arXiv:2608.12004](https://arxiv.org/abs/2608.12004)

RealisticTritonBench 从真实 AI 框架中修改 Triton kernel 的 pull request 构造任务，要求生成实现能够回到原框架并通过端到端测试。它不再只关注孤立 kernel 的运行时间。

阅读重点：

- 如何从真实 pull request 提取任务描述和工程上下文；
- 如何构造可复现的框架级测试环境；
- kernel-level benchmark 和 end-to-end benchmark 的差异；
- 如何防止模型通过 fallback 或测试漏洞获得虚假正确性。

它和本项目“生成 operator 后接入 nano-vLLM”的验证方式非常接近。

### 9. Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent

- 论文：[arXiv:2607.14541](https://arxiv.org/abs/2607.14541)

该工作提出 Atrex-Bench 和 Atrex-Kernel-Agent。任务和 shape 来自真实推理 trace，并根据算子占用的 GPU 时间赋予权重。Agent 使用 profiling、迭代修改和分层优化知识库生成候选。

阅读重点：

- 如何从生产 trace 选择真正值得优化的 kernel；
- Amdahl 定律如何影响优化优先级；
- 为什么只报告 correctness 或单算子 speedup 可能产生误导；
- 如何识别生成代码中的 PyTorch fallback。

### 10. CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

- 论文：[arXiv:2602.24286](https://arxiv.org/abs/2602.24286)

CUDA Agent 研究如何通过数据合成、可执行工具环境、自动验证、profiling 和强化学习，让模型本身学习 CUDA kernel 开发能力。它与调用通用 LLM 进行多轮提示的路线不同。

阅读重点：

- kernel 训练数据如何构造；
- 编译、正确性和性能信号如何形成 reward；
- Agent 工具使用能力如何参与强化学习；
- 训练专用模型与使用通用代码 Agent 的成本差异。

该工作主要针对 NVIDIA CUDA。当前阶段可以了解其方法，但不适合作为 MI100 上的首个复现实验。

## 与本地参考仓库的对应关系

当前 `references/reps/` 中的项目可以和论文配合阅读：

| 本地目录 | 用途 | 建议关注内容 |
| --- | --- | --- |
| `references/reps/KernelBench/` | 基础 benchmark | task 定义、correctness、timing、`fast_p` |
| `references/reps/KernelAgent/` | PyTorch 到 Triton 的 Agent workflow | 子图提取、并行候选、验证、优化产物 |
| `references/reps/GEAK-eval/` | ROCm/Triton 评测 | ROCm 数据集、`torch.allclose`、性能评价 |
| `references/reps/GEAK/` | AMD kernel 与端到端优化 Agent | Triton/HIP workflow、profiling、vLLM/SGLang |
| `references/reps/Apex/` | LLM serving kernel benchmark 与 Agent 环境 | 推理 workload、热点算子、端到端评分 |

KernelAgent、GEAK 和 Apex 的开源代码更新速度可能快于正式论文。阅读时应同时记录代码提交版本和论文版本，避免把当前仓库行为直接当成论文实验设置。

## 与当前项目的关系

现有工作主要覆盖两条路线：

```text
PyTorch 小型 reference → CUDA/Triton kernel
```

```text
已有 CUDA/Triton kernel → Agent 自动优化
```

当前项目设想的链条更强调模型上下文和推理框架适配：

```text
HuggingFace 或模型官方实现
→ 理解 Transformer 模块
→ 提取 operator、shape、dtype 和语义约束
→ 生成 PyTorch/Triton/HIP 实现
→ correctness 与 benchmark
→ 接入 nano-vLLM 进行端到端验证
```

现阶段不需要立即判断创新点。学习这些论文时，应重点观察它们如何回答以下问题：

1. Agent 获得什么形式的输入和上下文？
2. reference 的语义和测试输入如何构造？
3. correctness 是否覆盖不同 shape、dtype、边界值和数值稳定性？
4. benchmark 是否只测孤立 kernel，还是测真实推理吞吐？
5. profiling 信息如何反馈给 Agent？
6. 生成实现如何回接原框架？
7. 同一实现换到 AMD MI100 等不同硬件后是否仍然正确、有效？
