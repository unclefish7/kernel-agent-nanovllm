# 三种注意力模型的单卡适配计划

目标是让 agent 从 Transformers 原生模型的运行证据生成可由 nano-vLLM 加载的 Python 模型。沿用[项目设计](../../model-port-plan.md)的抽图、引擎规约、生成与数值验收流程。按下列顺序执行，每个模型独立记录结果。

## 共用验收条件

- [ ] 记录模型版本、权重校验和、Transformers 版本和实际加载的类。参照模型必须完整装载同一份权重。
- [ ] 在 `nanovllm_selector/extract/` 的常规入口录制 prefill（整段提示）、至少两步 decode（读取历史逐词计算）、多种长度与批大小；核对输出、状态来源与随输入变化的路径。
- [ ] 对照 `contracts/nanovllm/spec.yaml` 和当前引擎实现。若需扩充引擎，先写能复现缺口的测试，行为用张量、状态或能力契约表达，已有模型运行原有回归。
- [ ] 用 `nanovllm_selector/port/` 将通过检查的图、规约、算子索引、模板交给 agent。保留原始候选及失败反馈。在单卡上比较独立参考与引擎在相同提示及后续每个位置的完整词表分数，用独立输入、跨缓存块、批次换序复验。逐项相等和近似结果分别报告。
- [ ] 权重放仓库 `tmp/models/`；生成的 Python、图、报告放 `nanovllm_selector/outputs/<模型>/`。在项目 README 记录实测范围、复现命令与剩余缺口。

## 1. DeepSeek-V3 tiny-random（本轮开始）

- [x] 用项目现有 Transformers 4.57.3 原生类核对两层模型的权重、路由和单步参考。`yujiepan/deepseek-v3-tiny-random@cc35c93ec8328de7650c3d86a2a316d1d58cde02` 有 256 个非数路由偏置，保留作诊断；用公开的有限权重 `katuni4ka/tiny-random-deepseek-v3@9e56bebd7c1fda4920e0ed9ae9f090e13d02c363` 验证内容驱动的专家选择。[实测报告](../../../nanovllm_selector/outputs/deepseek_v3_tiny/README.md)
- [ ] 检查 MLA（多头潜在注意力）的张量来源：共享内容向量产生每个头的内容键和值，位置键单独旋转后在头间共享；参考模型缓存展开的 K 和 V，每头宽度分别为 32 和 16。
- [ ] 检查 MoE（按输入选择专家网络）的路由分数、每词的专家编号、专家权重与结果累加；以长度相同而内容不同的输入验证编号来自计算。一次执行只覆盖选中的专家。
- [ ] 在当前引擎验证展开缓存的正确性，按共用条件验收生成候选。压缩态 KV 缓存的显存优化另行验证。产物目录为 `outputs/deepseek_v3_tiny/`。

## 2. Qwen3.5-0.8B-Base（待执行）

- [ ] 独立准备支持 Qwen3.5 的 Transformers 环境，记录版本。先用 `hf-tiny-v2/tiny-random-Qwen3_5ForCausalLM` 覆盖两类注意力，再用 `Qwen/Qwen3.5-0.8B-Base` 的官方权重复验 24 层文本路径。
- [ ] 对照 Gated DeltaNet 的卷积历史、矩阵衰减、误差写回、更新后读取，以及完整注意力的输出门控。真实文本配置中每请求每个线性注意力层有 16 个头，每头矩阵为 `[128,128]`。
- [ ] 验证同一输入整段与分块处理、两个请求换序和释放后复用。比较中间状态及完整分数，分别记录逐步算法和分块算法的舍入差异。产物目录为 `outputs/qwen35_08b/`。

## 3. GLM-5 架构 tiny-random（待执行）

- [ ] 在兼容原生类的 Transformers 环境使用 `hyper-accel/tiny-random-glm-moe-dsa`。保留该权重自带的 `index_topk=64`。
- [ ] 检查 DSA（按查询动态筛选历史位置的稀疏注意力）中，从索引器分数、因果约束、位置编号到最终注意力的依赖关系。
- [ ] 测试 63、64、65、100 个可见位置，并用长度相同、内容不同的提示核对入选位置及完整词表分数。超过 64 个位置时验证确有候选被排除。产物目录为 `outputs/glm5_dsa_tiny/`。
