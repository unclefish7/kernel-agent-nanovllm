# Qwen3-0.6B 单卡数值对照

本地修订的 [model_qwen3.py](model_qwen3.py) 在 nano-vLLM 中使用 Qwen3-0.6B 的真实权重推理。参照为容器内的 Transformers。在同一张 MI100 上，两端均使用 BF16（16 位浮点数），开启确定性算法。以下“相等”要求一个位置上全部 151,936 个词的分数逐项相等。

| 候选和输入 | 完全相等的位置数 | 最大绝对差 |
| --- | ---: | ---: |
| 旧候选，128、256、768 词提示及生成 | 0/12 | 0.3125 |
| 本地修订，同一组输入 | 12/12 | 0 |
| 本地修订，独立文本的 255 词提示及生成 | 10/10 | 0 |

三个长度各比较一次提示末尾的预测，随后比较生成时提示末尾的预测和后续八步。常规测试命中了 256 和 512 词缓存前缀。独立文本的生成跨过 256 词的缓存页边界。[严格基线](baseline.json) · [修订结果](strict.json) · [独立复验](holdout.json)

旧候选直接使用引擎的归一化、位置编码、合并投影和激活函数，执行逻辑与引擎自带 Qwen3 模板相同。早期生成记录的最大绝对差已达 0.4375，当时按近似标准接受。后来的 DeepSeek 回归继续用这一文件和旧标准，12 个位置首选词均相同，最大绝对差 0.3447265625。上表的基线使用严格模式，启用了确定性计算。[历史候选](../iter_qwen3/attempt_1.py) · [早期记录](../iter_qwen3/trace.json) · [历史回归](../deepseek_v3_tiny/qwen_regression.json)

## 第一个差异

128 个词经过嵌入层后，每个词有 1,024 个特征值。第 0 层输入归一化收到的 131,072 个数完全相同，输出已有 34,962 个不同。把第一个词的特征从 0 开始编号，第 3 个特征在旧候选中是 0.486328125，在参考中是 0.484375。这一步发生在第一次注意力计算之前。[逐层诊断](diagnosis.json)

设 `u` 是用 FP32 算出的归一化值，`w` 是该特征的权重，`B(x)` 表示把 x 舍入到 BF16。在同一份真实输入上，逐步执行 `B(B(u) * w)` 与 Transformers 全部相等。手动计算 `B(u * w)` 与旧候选的编译结果全部相等。仅改变中间舍入就复现了这一层的全部差异。[脚本](norm_probe.py) · [实验结果](norm_probe.json)

## 全模型逐组对照

每行保留上一行的修改，在同一份旧候选上增加一项，原始权重、输入、设备和比较标准一致。最后一行仍使用旧候选的模型类。

| 本行新增的改动 | 完全相等的位置数 | 最大绝对差 |
| --- | ---: | ---: |
| [归一化及残差加法按参考顺序舍入](control_norm.json) | 0/12 | 0.375 |
| [位置旋转 RoPE 按参考精度生成并应用正弦、余弦](control_norm_rope.json) | 0/12 | 0.3125 |
| [注意力按参考的调用形状和遮罩计算，仍使用引擎的分页缓存](control_norm_rope_attention.json) | 0/12 | 0.3125 |
| [前馈层逐步计算 SiLU 激活及乘法](control_norm_rope_attention_activation.json) | 0/12 | 0.3125 |
| [把合并的矩阵乘拆成参考使用的各次投影](control_all.json) | 12/12 | 0 |

最后两行只改变投影的执行方式。以注意力为例，原先把查询、键和值的权重拼成一个大矩阵，一次乘完再拆出三个结果；对照组对同一输入分别做三次矩阵乘。门控和上投影也按相同方式拆开。两种写法在实数算术中等价，但改变矩阵形状可以改变底层乘加与舍入顺序。实验确定这组切换能消除前一行的剩余差异；两类投影各自的贡献没有单独测量。中间行的最大差取自所有位置、所有词，不能用它的增减给单个算子分摊误差。

## 复现

在项目容器的 `/workspace` 运行，使用已核实空闲的一张卡。本次为物理 GPU 0，PCI 地址 `0000:03:00.0`，对应本机容器 `HIP_VISIBLE_DEVICES=2`。以下命令假设单卡可见。

```sh
out=nanovllm_selector/outputs/qwen3_exact
model=tmp/models/Qwen3-0.6B
old=nanovllm_selector/outputs/iter_qwen3/attempt_1.py
python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$old" --exact --report "$out/baseline.json"
python -B -m nanovllm_selector.port.diagnose --model "$model" --candidate "$old" --length 128 --report "$out/diagnosis.json"
python -B -m nanovllm_selector.outputs.qwen3_exact.norm_probe > "$out/norm_probe.json"
for group in norm norm_rope norm_rope_attention norm_rope_attention_activation all; do
    python -B -m nanovllm_selector.outputs.qwen3_exact.controls "$group" || break
done
python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$out/model_qwen3.py" --exact --report "$out/strict.json"
python -B -m nanovllm_selector.port.evaluate --model "$model" --candidate "$out/model_qwen3.py" --exact --lengths 255 --prompt-file nanovllm_selector/outputs/lfm25_1_2b/holdout.txt --report "$out/holdout.json"
```

第一条评测命令预期返回 1，报告记录了旧候选的严格比较失败。实验的 JSON 和本地运行日志位于本目录，日志可用 `rm -f nanovllm_selector/outputs/qwen3_exact/*.log` 清理。运行复用 `nanovllm_selector/cache/` 下的编译和运行时缓存。
