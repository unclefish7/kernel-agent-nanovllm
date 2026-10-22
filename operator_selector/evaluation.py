"""复用 KernelBench 计时，对共享权重的完整 MLP 和替换片段分别评测。"""

import importlib.util
import os
from pathlib import Path
import statistics

import torch

from candidates import SelectedMLP, prepare_swiglu
from task import project, original_activation

ROOT = Path(__file__).resolve().parent
TIMING_SOURCE = Path(os.getenv("KERNELBENCH_TIMING_SOURCE", str(
    ROOT.parent / "references/reps/KernelBench/src/kernelbench/timing.py"
)))
spec = importlib.util.spec_from_file_location("selector_kernelbench_timing", TIMING_SOURCE)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def measure(operation, inputs):
    times = timing.time_execution_with_cuda_event(
        operation, inputs, num_warmup=10, num_trials=100, verbose=False, device=0,
    )
    result = timing.get_timing_stats(times)
    result["median"] = statistics.median(times)
    result["samples_ms"] = times
    return result


def evaluate(reference, inputs, plans):
    models = {name: SelectedMLP(reference, plan) for name, plan in plans.items()}
    result = {"correctness": {}, "tested_shapes": [list(x.shape) for x in inputs],
              "benchmark_shape": list(inputs[-1].shape), "timing_unit": "ms",
              "fusion": None, "rounds": []}
    with torch.no_grad():
        x = inputs[-1]
        projected = project(reference, x)
        if any(step["op"].startswith("triton_swiglu") for plan in plans.values() for step in plan["steps"]):
            result["fusion"] = prepare_swiglu(projected)

        for name, selected in models.items():
            for test_input in inputs:
                test_projected = project(reference, test_input)
                torch.testing.assert_close(selected.activation(test_projected),
                                           original_activation(reference, test_projected), rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(selected(test_input), reference(test_input), rtol=1e-4, atol=1e-5)
            result["correctness"][name] = "通过：两个输入规模，分别比较激活片段及完整 MLP"

        activation_reference = lambda values: original_activation(reference, values)
        for round_id in range(3):
            order = ["reference", *models]
            if round_id % 2:
                order.reverse()
            measured = {"round": round_id + 1, "order": order}
            for scope, args in (("activation", [projected]), ("mlp", [x])):
                operations = {"reference": activation_reference if scope == "activation" else reference}
                operations.update({name: model.activation if scope == "activation" else model
                                   for name, model in models.items()})
                stats = {name: measure(operations[name], args) for name in order}
                for name in models:
                    stats[name]["speedup_mean"] = stats["reference"]["mean"] / stats[name]["mean"]
                    stats[name]["speedup_median"] = stats["reference"]["median"] / stats[name]["median"]
                measured[scope] = stats
            result["rounds"].append(measured)
    return result
