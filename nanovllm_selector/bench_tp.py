"""用单条请求测量 TP 的首词和续写延迟，在运行引擎前核对物理卡。"""

import argparse
import atexit
import json
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch


def check_devices(buses):
    actual = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        actual.append(f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0")
    if actual != buses:
        raise RuntimeError(f"可见 GPU 的 PCI 地址为 {actual}，期望 {buses}")


def save_report(path, report):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        os.fchmod(stream.fileno(), 0o644)
    Path(stream.name).replace(path)


def measure(args):
    os.environ["NANOVLLM_MODEL_FILE"] = str(args.candidate.resolve())
    from nanovllm import LLM, SamplingParams

    buses = args.buses.split(",")
    check_devices(buses)
    report = {"pci_devices": buses, "torch": str(torch.__version__),
              "model": str(args.model), "candidate": str(args.candidate),
              "nccl_p2p_disable": os.getenv("NCCL_P2P_DISABLE"),
              "requested_runs": args.requests, "requests": [], "complete": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_report(args.report, report)
    start = perf_counter()
    llm = LLM(str(args.model), tensor_parallel_size=len(buses), enforce_eager=True,
              max_model_len=256, max_num_batched_tokens=256, max_num_seqs=1,
              gpu_memory_utilization=0.25)
    try:
        report["init_s"] = perf_counter() - start
        print(f"引擎启动 {report['init_s']:.3f} 秒", flush=True)
        torch.manual_seed(42)
        for run in range(args.requests):
            # 128 个词元，小于一个缓存块。每次请求使用不同的首词。
            prompt = [1000 + run] + list(range(1001, 1128))
            llm.add_request(prompt, SamplingParams(temperature=1e-8, max_tokens=4, ignore_eos=True))
            steps = []
            while not llm.is_finished():
                start = perf_counter()
                outputs, scheduled = llm.step()
                elapsed = (perf_counter() - start) * 1000
                steps.append(elapsed)
                print(f"请求 {run + 1} 步 {len(steps)} 调度量 {scheduled} 耗时 {elapsed:.2f} ms",
                      flush=True)
            if len(steps) != 4 or len(outputs) != 1 or len(outputs[0][1]) != 4:
                raise RuntimeError("引擎返回的生成步数或词元数量与请求不同")
            report["requests"].append({"prefill_ms": steps[0], "decode_ms": steps[1:],
                                       "token_ids": outputs[0][1]})
            report["median_decode_ms"] = median(t for r in report["requests"] for t in r["decode_ms"])
            save_report(args.report, report)
    finally:
        atexit.unregister(llm.exit)
        llm.exit()
    report["complete"] = True
    save_report(args.report, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--buses", required=True, help="按进程内顺序列出可用的 PCI 地址，逗号分隔")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--requests", type=int, choices=(1, 2), default=2,
                        help="测一条或两条请求，每条生成四个词元")
    args = parser.parse_args()
    measure(args)
