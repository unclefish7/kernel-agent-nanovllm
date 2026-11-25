# DeepSeek-V3 tiny-random 单卡适配进度

有限 FP32 检查点已在单张 AMD Instinct MI100 上完成整模型适配。Codex 使用项目的四个探索接口，从本地配置、引擎规约和 Python 模板开始，实现了两层网络的 MLA（多头潜在注意力）、稠密前馈网络、按词元选择的专家和共享专家。发布文件与首次通过验收的候选逐字节一致。[完整模型](model_deepseek_v3.py) · [移植记录](codex_explore/process.md)

## 主动探索与端到端结果

编码 agent 先读源码，再用四个提示词元、两个后续词元运行 Transformers。它编写了一个独立的 PyTorch 探针，重算归一化、带历史的注意力和专家加权输出。9 次完整张量对照都逐项相等，随后生成候选。[工具证据摘要](codex_explore/evidence.json) · [原样探针](codex_explore/probe.py)

| 冻结候选的用例 | 比较位置 | 词表与精度 | 最大绝对差 |
| --- | ---: | --- | ---: |
| 常规文本，提示长度 128、256、768，含前缀复用及随后 8 步生成 | 12/12 | 每位置 102,400 个 FP32 分数 | 0 |
| 独立文本，255 词元起步，随后 8 步生成并跨过 256 词元的 KV 页边界 | 10/10 | 每位置 102,400 个 FP32 分数 | 0 |

在物理 GPU2 的两次验收中，候选 SHA-256 均为 `195ac1ad4f8baba576476f37054a58e79e4c99ff1ef15662c34c5ce61ef6774d`。引擎与 Transformers 分别在独立进程中计算，参照沿用引擎实际生成的词元历史。两个输入组合计比较 2,252,800 个分数。[常规报告](codex_explore/attempt_1.report.json) · [独立报告](codex_explore/attempt_1.holdout.json)

同一份发布文件还经过两次补充验证。物理 GPU1 通过正式 CLI 重验，常规 12/12、独立 10/10。物理 GPU2 将两条长度不同的请求同时送入引擎，生成期间调换请求所在的行，再让较长的请求单独继续。交换初始提交顺序重做一遍，共有 16 个位置的完整 FP32 分数与各自历史的 Transformers 参照逐项相等。[CLI 复验](verified/trace.json) · [批次换序](batch_order.json)

已配置的 DeepSeek 生成后端在物理 GPU0 上也独立尝试了移植。它最初把展平的引擎输入当成三维张量。修订后能完整生成，公开输入通过 5/12。利用逐层诊断，它改写了查询低秩投影后的 RMSNorm（均方根归一化），提升到 7/12。之后首次可对齐的数值差异出现在第一层注意力输出投影的输入。它多次尝试在原生模型中查询候选自己定义的 `attn` 属性，未取得该探针的数值。到本次轮次上限，严格公开验收仍为 7/12，未进入独立输入验收。[生成记录摘要](deepseek_agent.json) · [7/12 的逐层诊断](active_repair/attempt_2.diagnosis.json)

引擎侧补了一个通用缺口。PyTorch 的持久缓冲区是随模型权重保存的张量，DeepSeek 的路由器用它存放专家选择的校正值。装载器现在从 checkpoint 填充参数和持久缓冲区，校验目标形状，并跳过由模型计算得到的非持久缓冲区。模型侧的路由公式和注意力仍由 agent 在候选 Python 中实现。引擎也按配置声明的架构名匹配本地模型类并核对接口。[nano-vLLM 装载器](../../../nano-vllm/nanovllm/utils/loader.py) · [vLLM 的权重装载实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/utils/#vllm.model_executor.models.utils.AutoWeightsLoader)

## 选择数值参考

| 项目 | 最初选择的 BF16 权重 | 动态路由验证用的 FP32 权重 |
| --- | --- | --- |
| 来源 | [yujiepan 固定版本](https://huggingface.co/yujiepan/deepseek-v3-tiny-random/tree/cc35c93ec8328de7650c3d86a2a316d1d58cde02) | [katuni4ka 固定版本](https://huggingface.co/katuni4ka/tiny-random-deepseek-v3/tree/9e56bebd7c1fda4920e0ed9ae9f090e13d02c363) |
| 每个头的 K / V 宽度 | 32 / 16 | 4 / 2 |
| 每个 token 的路由 | 从 256 个专家中选 8 个 | 从 160 个专家中选 6 个 |
| 磁盘路由修正偏置 | 256 个 NaN | 160 个零，全部有限 |
| 原生数值、路由编号与权重 SHA-256 | [reference.json](reference.json) | [finite_reference.json](finite_reference.json) |

在 Transformers 4.57.3 CPU 环境使用本地原生类逐项核对 checkpoint，并分别以 FP32、BF16 比较缓存解码和从头计算。第一份权重的最终分数有限，但路由器用带 NaN 的修正分数选择专家；实际混合权重来自修正前的分数，所以这个错误能躲过仅检查模型最终输出是否有限的测试。

第二份权重在两个长度为 5 的提示中，仅末尾词元编号从 53 换到 97，末位的 6 个专家就从 `[90,112,68,98,101,75]` 变为 `[19,70,150,76,74,1]`。这个样本可以检验 agent 有没有将一次抽图观测到的专家编号写死。[复现脚本](reference.py)

## 计算图与候选生成

首次用有限权重抽图时通过 24/31 项。模块总数包含未调用的专家，旧检查却要求每步访问所有线性层；抽取器也缺少动态取索引、按索引累加和数学注意力等计算的独立复算。[初次抽图日志](finite_extract.log)

修改后，检查使用本次调用的模块路径与执行区间核对投影。路由保留 top-k（选取得分最高的若干项）的值和编号，后续取数和原位累加都逐条复算。参考提示及两步 decode 的 370 个计算节点全部通过，底层算子均有重放规则。用另外三种输入规模运行时，三个样本的节点或连线都发生变化，抽取器记录每次差异，停止将这些样本合并成同一条路径。有限权重最终通过 33/34 项。原版 268 个节点重放通过，所测三阶段均只读了 256 个专家中的 8 个。[有限权重最终日志](finite_extract_final.log) · [原版最终日志](extract_final.log)

图驱动入口在组装上下文时拦下了有限权重的图。跨输入合并的图仍缺少一个可执行的专家选择循环。主动探索入口允许 agent 查询原生实现、实际发生的分支及其数值，直接走完整模型验收。这次适配沿用了现有计算图格式。[图驱动入口记录](agent_gate.log)

## 分页注意力

使用 CPU 参考核对权重后，在 GPU 上截取原生模型实际算出的 Q/K/V。在两层注意力中分别调用 nano-vLLM 的 `Attention.forward`，给每头 V 补零到 K 的宽度，输出按原 V 宽度裁回。提示长 255，随后读写第 256、257 个位置，两个物理页按反序分配。每一步都核对过去所有位置的 K/V 内容、填零列和注意力输出。层间继续传递原生输出，因此误差按层独立测量。此实验覆盖实测输入下的展开式 KV，整模型验收还须运行候选的权重装载、MLA 投影和专家路由。

| 检查点 | 测到的层和阶段 | FP32 最大绝对差 | BF16 最大绝对差 |
| --- | --- | ---: | ---: |
| 原版 32/16 | 2 层 × 3 阶段，两种精度的缓存内容均逐项相等 | 1.49e-8 | 0.0009765625 |
| 有限版 4/2 | 2 层 × 3 阶段，两种精度的缓存内容均逐项相等 | 3.73e-9 | 0.000244140625 |

早期分页注意力对照分别使用 2e-6 和 2e-3 的绝对容差。整模型候选从分页缓存读出 V 后恢复每头的实际宽度，再调用原生参照使用的注意力运算。上方端到端报告记录了该候选在有限 FP32 权重上的严格结果。[GPU 脚本](paged_mla.py) · [原版结果](paged_mla_original.json) · [有限版结果](paged_mla.json)

在当前引擎 `ba57abd` 上，Qwen3 的严格版候选、Llama、MiniCPM 和 LFM 各通过 12 个常规位置和 10 个独立位置，最大绝对差均为零。项目原有套件通过 159 项，引擎通过 33 项。早期使用另一份 Qwen3 候选的容差记录保留用于追溯。[当前四模型回归](../explore_regression/engine_ba57abd/summary.json) · [早期 Qwen3 记录](qwen_regression.json)

## 复现与产物

在仓库根目录，用已构建的执行镜像复验发布文件。选择新的输出目录保存这次报告。`--from-candidate` 会先验收输入的 Python，若两组通过就直接结束，本次 GPU1 复验的模型接口调用数为零。

```sh
/usr/bin/python3 -B -u -m nanovllm_selector.port.iterate --explore \
  --model tmp/models/deepseek-v3-tiny-katuni4ka --device 1 \
  --from-candidate nanovllm_selector/outputs/deepseek_v3_tiny/model_deepseek_v3.py \
  --out-dir nanovllm_selector/outputs/deepseek_v3_tiny/recheck --max-iters 2 --max-turns 1
```

从仓库根目录下载有限权重及分词器，校验结果应为 `OK`。

```sh
model=tmp/models/deepseek-v3-tiny-katuni4ka
revision=9e56bebd7c1fda4920e0ed9ae9f090e13d02c363
mkdir -p "$model"
for name in config.json generation_config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json; do
    curl -fL "https://huggingface.co/katuni4ka/tiny-random-deepseek-v3/resolve/$revision/$name" -o "$model/$name" || break
done
printf '%s  %s\n' cfbdbe3b14fae3948138aae5b254c9981826f84bd4069e5d688d8079ce437fc3 "$model/model.safetensors" | sha256sum -c -
```

在项目容器的 `/workspace` 执行，`HIP_VISIBLE_DEVICES=2` 在本机容器映射到 PCI `0000:03:00.0`。在其他机器先核实映射。

```sh
out=nanovllm_selector/outputs/deepseek_v3_tiny
model=tmp/models/deepseek-v3-tiny-katuni4ka
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -B "$out/reference.py" --model "$model" --out "$out/finite_reference.json"
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -B -m nanovllm_selector.extract.cli \
    --model "$model" --weights --seq-len 5 --decode-steps 2 --shape-probes --check --out "$out/finite_graph_final.json"
python -B -m nanovllm_selector.port.cli --dry-run --graph "$out/finite_graph_final.json"
HIP_VISIBLE_DEVICES=2 python -B "$out/paged_mla.py" --model "$model" --out "$out/paged_mla.json"
```

下载的原版与有限版可分别用 `rm -r tmp/models/deepseek-v3-tiny-random` 和 `rm -r tmp/models/deepseek-v3-tiny-katuni4ka` 清理。计算图及本地运行日志位于本报告目录，可按需删除对应的 `*graph*.json` 和 `*.log` 后复跑。运行期间使用项目已有的 `nanovllm_selector/cache/`，删除缓存后下次运行会重新生成。
