"""在一次独立数值验收的第二个续写步记录主卡等待跨卡通信的时间。"""

import os
from statistics import median
from time import perf_counter
from unittest.mock import patch

import torch
import torch.distributed as dist


def main():
    from nanovllm_selector.bench_tp import check_devices
    from nanovllm_selector.port.evaluate import main as evaluate
    from nanovllm.engine.model_runner import ModelRunner

    check_devices(os.environ["TP_EXPECT_BUSES"].split(","))
    run = ModelRunner.run
    decode_steps = 0

    def wrapped(self, seqs, is_prefill):
        nonlocal decode_steps
        if not is_prefill:
            decode_steps += 1
        if is_prefill or decode_steps != 2:
            return run(self, seqs, is_prefill)
        spans = []

        def record(fn):
            def call(tensor, *args, **kwargs):
                before = torch.cuda.Event(enable_timing=True)
                after = torch.cuda.Event(enable_timing=True)
                before.record()
                out = fn(tensor, *args, **kwargs)
                after.record()
                spans.append((fn.__name__, tensor.numel() * tensor.element_size(), before, after))
                return out
            return call

        start = perf_counter()
        with patch.object(dist, "all_reduce", record(dist.all_reduce)), patch.object(dist, "gather", record(dist.gather)):
            tokens = run(self, seqs, is_prefill)
        torch.cuda.synchronize()
        total = (perf_counter() - start) * 1000
        print(f"decode_wall_ms={total:.3f}")
        for name in ("all_reduce", "gather"):
            ops = [s for s in spans if s[0] == name]
            if ops:
                times = [a.elapsed_time(b) for _, _, a, b in ops]
                print(f"{name} count={len(ops)} bytes_per_rank={sorted({s[1] for s in ops})} "
                      f"stream_wait_total_ms={sum(times):.3f} median_ms={median(times):.3f}")
        return tokens

    with patch.object(ModelRunner, "run", wrapped):
        return evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
