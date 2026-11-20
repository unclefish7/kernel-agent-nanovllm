# MiniCPM5-1B 单卡适配验收

DeepSeek 首轮生成的 Python 文件直接通过了 nano-vLLM 数值验收。使用 OpenBMB 发布的 MiniCPM5-1B 权重，在一张 MI100 上以 BF16（16 位浮点格式）运行。标准输入和独立输入共核对 22 组下一词分数，每组包含词表全部 130,560 项，最大绝对误差为 0。[生成的模型文件](agent/attempt_1.py) · [标准输入](verified.json) · [独立输入](holdout.json)

## 计算图能表达什么

这份权重在 Transformers 中使用 `LlamaForCausalLM`。它有 24 层，隐藏向量宽度为 1536，有 16 个查询头，每头宽度为 128。查询投影需要输出 2048 个数，存入历史缓存的键和值各有 2 个头、各占 256 个数。模型的输入嵌入和最终词表打分各有自己的权重。已将图中 219 个参数的名称和形状与真实权重文件逐一核对。[图的交互视图](graph/graph.html) · [参数与缓存核对](graph_audit.json)

抽图覆盖完整输入的首次计算，以及随后两个逐词生成步骤。图中分别标出了每层的键和值从首次计算传给下一步、再传给第三步的连接。抽取器用三组批量和长度推断张量各维度的变化，再用保留的批量 2、长度 11 检验。记录的 5,252 次底层操作全部有归属，1,021 个节点的输出全部通过检查。[计算图 JSON](graph/graph.json)

实际交给 DeepSeek 的图来自配置初始化的模型，在 CPU 上用 FP32 抽取。另一次加载发布权重、按相同条件抽图，两次均通过 27 项检查。除权重加载标记和内存地址外，两份图所有字段相同。目前抽取器对这种带分片编号的权重文件会把图内的 `weights_sha256` 留空。本次单独计算权重文件的 SHA-256，并与固定模型版本的发布清单核对。[两次抽图对照](weights_audit.json) · [模型版本和文件校验值](snapshot.json)

## 引擎与生成代码各自通过了哪些检查

| 检查 | 输入与判据 | 实测 |
| --- | --- | --- |
| 引擎计算能力 | 先用已有的 Llama 实现装入 MiniCPM 权重，与 Transformers 比较完整词表 | 12/12 组逐项相等。[对照结果](control.json) |
| 引擎停止行为 | 在实际前向计算后控制采样结果，分别送入两个结束词、普通词和忽略结束词的请求 | 修复前在第二个结束词 `130073` 后多输出了 `42`。修复后两个结束词均立即停止，允许忽略结束词的请求到长度上限才停止。[修复前](stop_before.json) · [修复后](stop_control.json) |
| Agent 独立生成 | 提供计算图摘要、引擎规约、可用算子索引、Qwen3 模板及通用数值对齐说明。Llama 对照代码和独立输入均未放入消息 | 最多三轮，第一轮即通过。保留原始候选和生成记录。[首轮消息](initial_messages.json) · [轮次记录](agent/trace.json) |
| 生成代码复验 | 在新进程装入同一候选。先处理 128、256、768 个词元，再用最长输入连续生成 8 步。Transformers 从自己的权重和输入建立历史缓存 | 12/12 组相等。四次请求实际复用的前缀长度是 0、0、256、512。[逐步报告](verified.json) |
| 独立文本与缓存换块 | 取一段中英混合文本的前 255 个词元。两次请求的可复用前缀均为 0，接着生成 8 步，越过 256 词元缓存块边界 | 10/10 组相等。包含单独运行的首次计算、连续生成的首次计算和 8 次读取历史的计算。[输入文本](holdout.txt) · [逐步报告](holdout.json) |
| 普通生成与结束词 | 用最终候选从“中国的首都是”生成 8 个词元，再对同一候选控制采样测试结束条件 | 普通生成的 8 个词元编号与 Transformers 一致，4 种停止行为全部符合预期。[普通生成](generation.json) · [停止测试](stop_final.json) |

验证覆盖物理 GPU 0、每次一条请求、标准输入最长 768 个词元。本模型可以检验工程中对注意力宽度、前缀复用、分页缓存和多结束词的处理。多卡、并发批处理、配置声明的 131,072 词元上下文长度，以及其他架构的计算方式仍需单独验证。

## 复现

模型固定版本为 `ac83f813ba7da263c51a9f3ec37c2072597e54af`。权重分片从 OpenBMB 的 ModelScope 镜像取得，与 Hugging Face 发布的 SHA-256 一致。模型文件在仓库 `tmp/models/MiniCPM5-1B/`。本次使用 Python 3.12.3、PyTorch 2.12.0+rocm10.0.0 和 Transformers 4.57.3。[发布清单](https://huggingface.co/openbmb/MiniCPM5-1B/tree/ac83f813ba7da263c51a9f3ec37c2072597e54af)

以下命令在项目容器的 `/workspace` 执行。此机器上 ROCm 的设备编号 2 对应物理 GPU 0，PCI 地址为 `0000:03:00.0`。在其他机器应按实际 PCI 地址选择设备。

```sh
export HIP_VISIBLE_DEVICES=2
model=tmp/models/MiniCPM5-1B
out=nanovllm_selector/outputs/minicpm5_1b
candidate="$out/agent/attempt_1.py"
rerun="$out/rerun"
mkdir -p "$rerun"

HIP_VISIBLE_DEVICES= python -B -m nanovllm_selector.extract.cli \
  --model "$model" --seq-len 5 --decode-steps 2 --shape-probes \
  --outdir "$rerun/graph" --check
HIP_VISIBLE_DEVICES= python -B -m nanovllm_selector.extract.cli \
  --model "$model" --weights --seq-len 5 --decode-steps 2 \
  --shape-probes --out "$rerun/weights.json" --check

python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$candidate" \
  --exact --report "$rerun/verified.json" --logits-out "$rerun/logits.pt"
python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$candidate" \
  --exact --lengths 255 --decode-steps 8 --prompt-file "$out/holdout.txt" \
  --report "$rerun/holdout.json" --logits-out "$rerun/holdout_logits.pt"
python -B -m nanovllm_selector.outputs.minicpm5_1b.check_stop --model "$model" \
  --candidate "$candidate" --report "$rerun/stop.json"
```

如要重新调用已配置的 DeepSeek 生成候选，可在相同容器中运行下列命令。每一轮会使用接口额度。

```sh
python -B -m nanovllm_selector.port.iterate --model "$model" --graph "$out/graph/graph.json" \
  --spec tmp/port-deps/contracts/nanovllm/spec.yaml --kernels tmp/port-deps/kernels/index.md \
  --template nano-vllm/nanovllm/models/qwen3.py --exact --max-iters 3 --timeout 300 \
  --out-dir "$rerun/agent"
```

## 本地产物

本目录的 `logits.pt` 和 `holdout_logits.pt` 保存了引擎和 Transformers 的完整原始分数，均已从磁盘重新读取并逐项验证。`graph/weights.json` 是真实权重抽出的图。`*.log` 记录了各步骤的终端输出，包括抽取及适配流程 82 项、选择器 5 项、引擎 18 项回归测试，全部通过。

在仓库根目录可用 `rm -f nanovllm_selector/outputs/minicpm5_1b/*.pt nanovllm_selector/outputs/minicpm5_1b/*.log` 清理本次原始分数与日志。模型下载占用约 2.1 GiB，可单独清理 `tmp/models/MiniCPM5-1B/`。容器共用缓存位于 `nanovllm_selector/cache/`，其他实验可能继续使用它。清理本次复现的输出可删除 `nanovllm_selector/outputs/minicpm5_1b/rerun/`。
