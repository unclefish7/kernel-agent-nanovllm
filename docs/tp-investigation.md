# 多卡（TP>1）排查记录

2026-11-19 在无桥接的物理 GPU 0–3 上用 Llama-3.2-1B 复测，双卡约 7.76 秒、四卡约 30.69 秒一个续写词元。同一进程配置增加 `NCCL_P2P_DISABLE=1` 后，通信改走主机共享内存，分别降至 36 毫秒和 34 毫秒。逐步计时和数值对照见 [Llama 双卡与四卡实测报告](../nanovllm_selector/outputs/tp_llama_0_3/README.md)。以下为 11 月 17 日的 Qwen3 实验记录。

- 日期：2026-11-17
- 环境：容器 `nanovllm-selector-rocm-dev`，8×MI100（容器内可见 GPU0–3），RCCL 2.30.4，PyTorch 2.12.0+rocm10.0.0
- 结论一句话：**TP=2 数值正确，但每个 forward 稳定 13 秒；退化点在"跑一次真实模型前向"这件事本身，不在集合通信、不在 ModelPort 生成的代码。**

## 1. 现象

| 配置 | 生成 8 token | 每 token |
|---|---|---|
| tp=1 | 0.39 s（稳态） | 约 49 ms |
| tp=2 | 108 s | 约 13 s |

数值上 tp=2 是对的：同一输入 tp=1 vs tp=2 的 prefill logits
`max_abs=0.375`、`mean_abs=0.0558`、`cosine=0.999832`、argmax 相同、top4 相同，
与 tp=1 vs HF 的固有差（`max_abs=0.4375`、`cos 0.9998`）同一量级。

顺带确认：**`temperature=1e-8` 不等于确定性**。同一条 tp=1 连跑两次，第 6 个 token
分别是 9625（France）和 15344（Italy）。所以多卡一致性只能用 logits 判定，不能用 token。

## 2. 事实链

1. **CUDA event 掐时间**：一次 decode forward `cpu≈53 ms`、`gpu≈12990 ms`。
   CPU 只用 53 ms 把活排完，GPU 真的占着 13 秒。
2. **profiler**：58 次集合通信，CUDA 时间 `12.949 s`（99.93%），单个
   `ncclDevKernel_Generic_2` 平均 **223 ms**；真正的计算
   `aten::scaled_dot_product_attention` 只有 2.78 ms、`aten::matmul` 2.65 ms。
3. **chrome trace**：rank0 的 58 个 nccl kernel 首尾相接、每个约 223 ms，GPU 全程停在
   集合通信内核里，没有空隙。
4. **对照实验（全新进程、同样两张卡）**：单次 all_reduce 0.29 ms；连发 57 次每次 0.25 ms；
   前面夹一次 `torch.mm`、`inference_mode`/`no_grad`、先分配 6 GB、每次新建张量
   ——全部仍是 0.25 ms。**RCCL 本身没问题。**

## 3. 二分定位

在 `ModelRunner.__init__` 的各个阶段插"手动 all_reduce + 计时"：

| 位置 | 耗时 |
|---|---|
| init_process_group 之后 | 0.29 ms |
| 模型构造 + 权重装载之后 | 0.91 ms |
| warmup 之前 | 0.37 ms |
| **warmup 之后** | **205 ms** |
| allocate_kv_cache 之后 | 206 ms |

退化发生在**第一次前向（warmup）期间**，模型构造、权重装载、KV cache 分配都不是触发点。

把 `warmup_model` 整个换成"57 次裸 all_reduce"，再跑 generate：

```
[rank0] naked batch0: 322.246 ms/op     ← 首次建链
[rank0] naked batch1:   0.254 ms/op
...
generate 30.12s                          ← 之后跑真实模型，仍然慢
```

即：**引擎进程本身、通信组本身都是好的；一旦真的跑一次模型前向，集合通信就变成 223 ms/次，
并且此后一直如此。**

## 4. 排除表

| 假设 | 实测 | 结论 |
|---|---|---|
| 集合通信实现慢 | 裸测 0.25 ms/次；引擎内 513 次 p50 0.1 ms | 排除 |
| torch.compile | `TORCHDYNAMO_DISABLE=1` 后仍 12.99 s/forward | 排除 |
| `forward_torch` 每层 `.tolist()` | 改成主机侧列表传下来，仍 12.99 s | 排除 |
| CPU 饥饿 | 128 核、无 cgroup quota、load 5.86 | 排除 |
| 显存不足/换页 | `gpu_memory_utilization` 0.2 → 0.05 仍 13.9 s | 排除 |
| 队列深度（58 次连发） | 每次 all_reduce 后加 `cuda.synchronize()` 仍 27.4 s | 排除 |
| RCCL 通道/协议 | `NCCL_MAX_NCHANNELS=1`、`NCCL_PROTO=Simple`、`NCCL_ALGO=Ring` 均无变化 | 排除 |
| 设备序 | `set_device` 移到 `init_process_group` 之前，无变化 | 排除 |
| 端口/残留进程 | 端口已随机化；容器内只剩 defunct 僵尸 | 排除 |
| 显存分配器 | 每次新建张量的 all_reduce 仍 0.25 ms | 排除 |
| 传输路径 | 引擎与裸测的 RCCL 日志一致：4 通道、`via P2P/IPC`、同一对 PCI 设备 | 排除 |

## 5. 顺带修掉的真 bug

这三个都会伪装成"多卡很慢"，值得记下来：

1. **僵尸进程占端口**：一次失败脚本被 `timeout` 杀掉后，容器里的 python 进程仍活着并监听
   2333，后续每次运行都 `EADDRINUSE`，子进程里还出现 `FileNotFoundError`。
   `timeout` 只杀 `docker exec` 客户端，不杀容器内进程。
   修：`config.master_port`（0 = 自动选空闲端口），`llm_engine` 里 `free_port()`。
2. **子进程崩溃、父进程空等 601 秒**：rank1 在入组前就因"模型类没登记"崩了，rank0 在
   `init_process_group` 里等到 601 秒超时。修：spawn 之前先 `resolve_model_class` +
   `check_model_contract` 预检；`init_process_group` 加 120 秒超时。
3. **候选类跨进程不可见**：`models.ARCH[arch] = cls` 只改父进程内存，spawn 出来的 rank1
   重新 import 后看不见。修：`register_candidate_file()` + `NANOVLLM_MODEL_FILE` 环境变量。

## 6. 尚未定位

最可疑的方向是 **RCCL 在这对卡 + 这个容器里的慢路径**（模型前向里的 P2P 传输退化成高延迟），
但还没有证据。下一步可做：

1. 换到 XGMI 那组卡（GPU4–7，空闲时）对比：如果变快，说明是 PCIe 组的问题。
2. tp=2 跑一个 2 层的小模型：如果仍 223 ms/次，说明与模型规模无关。
3. 在容器里跑 `rccl-tests` 的 `all_reduce_perf -b 8 -e 8M`，与引擎内逐次对比。

## 7. 对 ModelPort 的影响

- **不阻塞主线**：生成、评测、迭代环都在 tp=1 下完成并验证。
- `evaluate --tp 2` 现在能跑，但一次三档长度 + decode 的评测约需 10 分钟，暂不作默认。
- 真正属于 ModelPort 的适配点只有第 5 节第 3 条（候选类跨进程登记），已修。
- 结论：TP 视为**引擎侧能力**，不列入 ModelPort 的验收项；本条记录留作环境待查。
