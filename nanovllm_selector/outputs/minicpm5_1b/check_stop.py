"""在真实模型前向后控制采样结果，检查调度器对每一种结束词的处理。"""

import argparse
import atexit
import hashlib
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    os.environ["NANOVLLM_MODEL_FILE"] = str(args.candidate.resolve())
    from nanovllm import LLM, SamplingParams

    llm = LLM(args.model, enforce_eager=True, max_model_len=256,
              max_num_seqs=1, gpu_memory_utilization=0.25)
    try:
        model_file = Path(inspect.getfile(type(llm.model_runner.model).forward)).resolve()
        assert model_file == args.candidate.resolve()
        cases = []
        for name, sequence, ignore, expected in [
            ("结束词一", [1, 42], False, [1]),
            ("结束词二", [130073, 42], False, [130073]),
            ("普通词后结束", [42, 1], False, [42, 1]),
            ("忽略结束词并达到长度上限", [130073, 1, 42], True, [130073, 1, 42]),
        ]:
            stream = iter(sequence)

            def sample(logits, temperatures):
                return logits.new_tensor([next(stream)], dtype=torch.long)

            with patch.object(llm.model_runner.sampler, "forward", sample):
                result, = llm.generate([[0, 42]], SamplingParams(
                    temperature=0.1, max_tokens=len(sequence), ignore_eos=ignore), use_tqdm=False)
            actual = result["token_ids"]
            cases.append(dict(name=name, actual=actual, expected=expected, ok=actual == expected))
        report = dict(ok=all(x["ok"] for x in cases), candidate=str(model_file),
                      sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(), cases=cases)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return int(not report["ok"])
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    raise SystemExit(main())
