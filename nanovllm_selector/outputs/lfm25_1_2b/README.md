# LFM2.5-1.2B-Base 在 nano-vLLM 上的适配

## 真实权重单卡验收

DeepSeek 生成的 Python 文件在物理 GPU 0（MI100，PCI 地址 `0000:03:00.0`）上以 BF16 执行。每一步都用 Transformers 原生模型独立算出 65,536 个词的分数。两套提示中所有保存的分数均有限，逐值相等。[通过验收的模型文件](agent_repair/attempt_1.py) · [完整分数文件的复核记录](scores_audit.json)

| 用例 | 实测结果 |
| --- | --- |
| 128、256、768 词的提示 | 三次提示末位、独立生成的提示末位和后续八步，共 12/12 组分数相等。[报告](verified.json) |
| 生成代码时未提供的 255 词提示 | 两次提示末位和八步续写，共 10/10 组分数相等。[提示](holdout.txt) · [报告](holdout.json) |
| 两个请求同时运行 | 分别以 A、B 和 B、A 的顺序加入，中途交换行号，A 结束后 B 继续。两轮共 16/16 个请求步的分数与各自的独立参考一致。[报告](batch_order.json) |
| 自然语言生成 | 两端都从同一句中文提示生成“需要测量水位的平均值”，8 个词元编号完全一致。[原生模型](native_reference.json) · [引擎生成](generation.json) |
| 已有模型 | Llama、MiniCPM 各 12/12 组分数相等，Qwen3 按原标准通过（最大绝对误差 0.34375）。三者均观测到 256 和 512 词的前缀命中；MiniCPM 的四个停止条件用例通过。[Llama](regression_llama.json) · [MiniCPM](regression_minicpm.json) · [Qwen3](regression_qwen.json) · [停止条件](regression_minicpm_stop.json) |

首轮有三个候选。第 2 版在第一层归一化就产生差异；第 3 版修正归一化与旋转位置编码后，128 词的分数精确相等，256 和 768 词仍有误差。在第 3 版上只更换注意力调用方式，单请求与换序并发对照都达到零误差。把这条实验结果连同第 3 版反馈给 DeepSeek 后，它另行生成了上表中的最终文件。最终验收直接加载此文件。[首轮记录](agent/trace.json) · [首处分歧](diagnosis_2.json) · [注意力对照](control_attention.json) · [修复消息](repair_messages.json) · [修复轮记录](agent_repair/trace.json)

**分块计算有一项可复现的数值边界。** 同一条 24 词提示整段处理时，引擎与 Transformers 的完整分数相等。改为 16+8 两段再续写两步，两端生成的三个词元仍然相同，但分块引擎与整段 Transformers 的分数最大相差 0.2099609375。第一处差异出现在第一个注意力层：相同输入按不同分组计算时，24 个位置的 49,152 个 BF16 元素有 19 个不同，最大差约 0.000031。40 次窗口检查以及 24 次键、值和因果遮罩检查均通过。只在独立的 Transformers 对照中按相同方式拆分注意力，三步全词表便与分块引擎逐项相等。这个结果限定在所测样本、设备和精度。[原始分块报告](chunked.json) · [独立对照](chunk_control.json) · [复现脚本](chunk_control.py)

## 复验

在项目容器的 `/workspace` 中执行。下列相对路径均从仓库根目录计算。本机容器内设备编号 2 对应上列物理 GPU 0。

```sh
export HIP_VISIBLE_DEVICES=2
out=nanovllm_selector/outputs/lfm25_1_2b
model=tmp/models/LFM2.5-1.2B-Base
candidate=$out/agent_repair/attempt_1.py
NANOVLLM_MODEL_FILE="$candidate" python -B -c '
import sys
from nanovllm import LLM, SamplingParams
llm = LLM(sys.argv[1], max_num_seqs=1, max_model_len=256, gpu_memory_utilization=0.25, enforce_eager=True)
print(llm.generate(["为了记录河流水位的每日变化，观测员首先"], SamplingParams(temperature=1e-8, max_tokens=8)))
' "$model"
python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$candidate" --exact \
  --report "$out/verified.json" --logits-out "$out/logits.pt"
python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$candidate" --exact \
  --lengths 255 --prompt-file "$out/holdout.txt" --report "$out/holdout.json" --logits-out "$out/holdout_logits.pt"
python -B -m nanovllm_selector.outputs.lfm25_1_2b.check_batch_order \
  --model "$model" --candidate "$candidate" --exact --report "$out/batch_order.json"
```

两份 `.pt` 文件为本机完整分数，约 11 MiB。可用 `rm -f "$out/logits.pt" "$out/holdout_logits.pt"` 清理；上述命令可以重建。官方权重位于 `tmp/models/LFM2.5-1.2B-Base/`（单个权重文件约 2.18 GiB），下载清单位于 `tmp/lfm25-1_2b-base-download-7453bca9/`，本轮容器规约副本位于 `tmp/lfm25-port/`。结束复验后可用 `rm -r` 分别删除这些目录。本地运行日志位于 `$out/*.log`，框架缓存位于 `nanovllm_selector/cache/`；清理缓存后，下次测试会重新建立。

## 计算图与引擎

使用官方权重在 CPU 上执行完整的 16 层模型，先输入提示，再连续生成两步。三个阶段分别记录到 1,048、1,088、1,088 次底层操作。图中有 686 个节点与 1,040 条边。681 个计算节点通过数值检查，31 个高层含义尚未命名的节点保留了通过检查的底层操作顺序、参数和写入位置。[模型版本与校验和](snapshot.json) · [生成修正候选时使用的图](graph_final.json)

再次抽图时，对每个输入规模分别保留了实际执行的操作。两个逐词阶段共有 60 个节点出现两条路径；120 条配方均有逐步数值验证。例如同一卷积层的输出投影，在单请求时做批量矩阵乘（`bmm`），在两个请求时做普通矩阵乘（`mm`）。加入这项检查后，真实权重图通过 31/31 项检查。最新上下文已在本地构建；上表的候选来自此前记录的修复轮。[最新图](graph_variants.json) · [分支审计](variants_audit.json) · [最新上下文](variant_messages.json)

每个阶段都有 22 份实际状态。6 层注意力各保存一份键和一份值，10 层卷积各保存一个固定长度的窗口。每生成一个词，窗口保留旧的两列并写入一列。模型对象还分配了 6 个未参与计算的窗口，图中单独记录。最终抽图在两种批大小和三种提示长度下识别了卷积产生的额外两个位置；首轮生成使用的早期图仍把相关的 10 个轴标为未知。实际发送的消息保存在本目录。[首轮生成上下文](initial_messages.json)

引擎为每个请求分配独立槽位来保存固定大小的状态。长提示分块期间沿用原槽位，请求抢占后从原始词元重建状态，结束后允许槽位复用。混合模型在不同请求之间从头计算相同前缀；纯键值缓存模型保留原有的前缀命中。相关用例包括换序、抢占、完成后复用及显存预算。[引擎测试](../../../nano-vllm/tests/test_request_state.py)

子仓库 `main` 和 `feat/request-state-storage` 已推送至 `fe704c8`，其中也包含 `fix/minicpm5-generation-eos`。在此版本上复跑了 27 项引擎测试、5 项 GPU 选择器测试和三个已有模型。MI100 使用 PyTorch 直接执行；设备不支持的图捕获路径由模拟捕获与重放的用例验证。[最终版本的回归结果](final_regressions.json)

父仓库 137 项 CPU 回归覆盖计算图、尺寸、原语分支、数值参照、并发请求归属和生成失败后的反馈。其中故意损坏计算图的测试会删除状态读写来源、更改操作或移除分支，检验生成前能否发现问题。[回归日志](project_final.log)

## 修改前的诊断记录

以下结果来自父仓库 `6f8deed` 与子模块 `285e898`，其中命令和路径按当时的状态保留。

当前流程在抽取首次输入的历史缓存时退出，未产出计算图。随后调用 agent 的上下文组装入口也因缺图退出。引擎和验收程序的独立检查还分别发现了混合层缓存分配和前缀复用问题。[抽图原始日志](extract.log) · [上下文组装日志](agent_dry_run.log)

## 验证范围

- 父仓库版本 `6f8deed`，nano-vLLM 子模块版本 `285e898`。容器中使用 Transformers 4.57.3、PyTorch 2.12.0+rocm10.0.0。
- 采用官方配置的完整 16 层与原始宽度，在 CPU 上随机初始化权重。10 层使用短卷积（组合最近三个位置），6 层使用注意力（读取过往位置的键和值）。配置文件的 SHA-256 为 `15d6157fb6df3f8272e2fe90e18f57727ccf02a125c94469198b0f3281510185`。[官方配置](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base/blob/main/config.json)
- 原生实现先处理 5 个词元，再连续输入 2 个词元。三个位置的词表分数均为有限值，最高分词元均与从头计算相同，最大绝对差小于 0.000015。本次数值用于区分历史状态错误与计算舍入差异。发布权重在 GPU 上的推理精度、吞吐量及生成代码的表现均未实测。[诊断脚本](probe.py) · [完整结果](probe.json)

## 已复现的问题

1. **抽取器读取不了模型返回的两类历史。** 标准命令以退出码 2 停在 `cache_tensors`。该接口从 `cache.layers` 逐层读取键和值，原生模型返回的 `Lfm2HybridConvCache` 把注意力键、注意力值和卷积窗口分别保存。实测 6 个注意力层的历史长度依次为 5、6、7，10 个卷积层各自的窗口始终为 `[1, 2048, 3]`。每前进一步，这 10 个窗口都保留旧窗口的最后两列，再写入新列。只清零第一个卷积层的历史，再输入同一词元，就改变了 65,536 项下一词分数中的 65,535 项。[缓存读取代码](../../extract/trace_io.py)
2. **底层操作被记录，但缺少卷积语义与数值检查。** 对同一个原生模型独立调用现有记录器与节点分类器。首次计算中 10 个一维卷积均有底层卷积记录，但都标为 `unknown`。逐词路径中第一层的窗口移动、写入、乘法、求和也被归在未知节点，输出投影使用的批量矩阵乘同样被标为未知。三个阶段各有 20 个未知节点。现有节点数值检查也没有卷积实现。因此就算单独解决缓存读取，也不能据此宣称卷积计算已通过检查。[节点分类](../../extract/decompose.py) · [数值检查规则](../../extract/replay.py)
3. **引擎按总层数要求注意力缓存。** 在 CPU 上调用原始 `ModelRunner.allocate_kv_cache`，按官方层表提供 6 个带键值缓存接口的模块和 10 个带卷积窗口接口的模块，并模拟显存容量。方法给 6 个注意力模块分配了缓存，给卷积模块分配了 0 个窗口，随后因总数不等于 16 报错。这项实验复现的是引擎接口约束，没有装入生成候选或启动 GPU 引擎。[分配逻辑](../../../nano-vllm/nanovllm/engine/model_runner.py)
4. **验收程序复用前缀时，参考答案可能出错。** 验收程序会把已保存的缓存截短，再一次送入剩余提示。按当前 Transformers 的原生行为，把长度 5 的缓存截回 3 后，注意力历史长度变成了 3，但 10 个卷积窗口全都停在位置 5。再送入同一词元，与从头处理前 3 词所得参考分数相比，最大绝对差为 5.006。此外，即使用正确的长度 3 缓存，一次送入后续两个词元与逐个送入的末步分数仍相差 5.818。逐个送入与整段从头计算仅差约 0.000013。这两项是当前容器 CPU 参照路径的独立实验；尚未运行 GPU 参照路径。[验收程序中的前缀处理](../../port/evaluate.py)
5. **激活函数计数会误报。** 把首次前向得到的节点交给现有对账函数，它得到“期望 0、实际 16”。LFM2 用函数直接执行 SiLU（一种逐元素计算），每层一次。检查程序只按指定名称的模块对象计数，所以把正确出现的 16 次计算当成多余。该结果来自独立运行对账函数，正式抽图在更早的缓存步骤就退出了。[对账逻辑](../../extract/reconcile.py)

## 复现与本地文件

在项目容器的 `/workspace` 中执行以下命令。第一条抽图命令预期报错，第二条独立诊断命令会打印 JSON。

```sh
export HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -B -m nanovllm_selector.extract.cli --model tmp/models/LFM2.5-1.2B-Base \
  --seq-len 5 --decode-steps 2 --shape-probes --check \
  --out nanovllm_selector/outputs/lfm25_1_2b/graph.json
python -B -m nanovllm_selector.outputs.lfm25_1_2b.probe
```

测试脚本、JSON、日志和本报告都在 `nanovllm_selector/outputs/lfm25_1_2b/`。下载目录 `tmp/models/LFM2.5-1.2B-Base/` 中只有官方配置（1,224 字节）。两处目录均可在不再需要时单独删除。容器临时缓存沿用已有的 `nanovllm_selector/cache/`。
