import sys
from pathlib import Path
from typing import Callable

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR / "06_triton_tiled_softmax"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "07_pytorch_hip_tiled_softmax"))

import hip_tiled_softmax  # noqa: E402
from tiled_softmax import tiled_softmax  # noqa: E402


def measure_latency_us(
    operation: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int = 20,
    repeats: int = 200,
) -> float:
    for _ in range(warmup):
        operation(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        operation(x)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def main() -> None:
    implementations = {
        "PyTorch": lambda x: torch.softmax(x, dim=-1),
        "Triton": tiled_softmax,
        "HIP": hip_tiled_softmax.tiled_softmax,
    }
    shapes = ((3, 17, 200), (16, 128, 200), (32, 256, 256))

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'shape':>18} {'PyTorch(us)':>14} {'Triton(us)':>14} {'HIP(us)':>14}")

    with torch.inference_mode():
        for shape in shapes:
            x = torch.randn(shape, device="cuda", dtype=torch.float32)
            expected = implementations["PyTorch"](x)
            torch.testing.assert_close(
                implementations["Triton"](x), expected, rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(
                implementations["HIP"](x), expected, rtol=1e-4, atol=1e-5
            )

            latencies = {
                name: measure_latency_us(operation, x)
                for name, operation in implementations.items()
            }
            print(
                f"{str(shape):>18} "
                f"{latencies['PyTorch']:>14.2f} "
                f"{latencies['Triton']:>14.2f} "
                f"{latencies['HIP']:>14.2f}"
            )


if __name__ == "__main__":
    main()
