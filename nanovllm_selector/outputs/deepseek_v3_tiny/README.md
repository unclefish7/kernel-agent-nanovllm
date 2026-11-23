# DeepSeek-V3 tiny-random 单卡适配进度

已完成两个公开检查点的原生参考，并在物理 GPU 0（MI100，PCI `0000:03:00.0`）验证了 nano-vLLM 的分页注意力。当前整模型的生成候选和逐步完整词表验收仍待完成。

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

正式生成入口在组装上下文时拦下了有限权重的图。要进入整模型验收，图还需表达可由 agent 实现的同一个循环：路由器算出编号和权重，按编号调用对应专家、取该专家的参数，再按 token 位置累加。可以用已记录的多条轨迹验证这个循环；每个编号的来源及每个专家的权重绑定均须可核对。当前的分支差异清单只能指出哪里变了。[入口实际结果](agent_gate.log)

## 分页注意力

使用 CPU 参考核对权重后，在 GPU 上截取原生模型实际算出的 Q/K/V。在两层注意力中分别调用 nano-vLLM 的 `Attention.forward`，给每头 V 补零到 K 的宽度，输出按原 V 宽度裁回。提示长 255，随后读写第 256、257 个位置，两个物理页按反序分配。每一步都核对过去所有位置的 K/V 内容、填零列和注意力输出。层间继续传递原生输出，因此误差按层独立测量。此实验覆盖实测输入下的展开式 KV，整模型验收还须运行候选的权重装载、MLA 投影和专家路由。

| 检查点 | 测到的层和阶段 | FP32 最大绝对差 | BF16 最大绝对差 |
| --- | --- | ---: | ---: |
| 原版 32/16 | 2 层 × 3 阶段，两种精度的缓存内容均逐项相等 | 1.49e-8 | 0.0009765625 |
| 有限版 4/2 | 2 层 × 3 阶段，两种精度的缓存内容均逐项相等 | 3.73e-9 | 0.000244140625 |

FP32 和 BF16 对照的绝对容差分别设为 2e-6 和 2e-3。两个注意力后端的舍入结果有差异；项目端到端逐项相等的标准仍需独立检验。[GPU 脚本](paged_mla.py) · [原版结果](paged_mla_original.json) · [有限版结果](paged_mla.json)

引擎现在按注意力模块声明的本地头数和物理头宽分配缓存。项目 156 项测试、引擎 30 项测试通过。单卡对已有模型各比较 12 组完整词表分数，覆盖 128、256、768 词提示和后续 8 步。Llama、MiniCPM、LFM 均逐项相等，Qwen3 达到原有近似验收标准，最大绝对差为 0.3447265625。[项目测试](project_tests_final.log) · [引擎测试](engine_tests.log) · [Llama](llama_regression.json) · [MiniCPM](minicpm_regression.json) · [LFM](lfm_regression.json) · [Qwen3](qwen_regression.json)

## 复现与产物

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
