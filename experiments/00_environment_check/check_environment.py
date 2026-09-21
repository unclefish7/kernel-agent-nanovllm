from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Sequence


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def check_device_files() -> bool:
    print_section("GPU 设备文件")
    kfd_exists = os.path.exists("/dev/kfd")
    render_nodes = sorted(glob.glob("/dev/dri/render*"))
    print(f"/dev/kfd: {'存在' if kfd_exists else '缺失'}")
    print(f"render 节点: {', '.join(render_nodes) if render_nodes else '未找到'}")
    return kfd_exists and bool(render_nodes)


def run_command(name: str, args: Sequence[str], max_lines: int = 12) -> bool:
    executable = shutil.which(args[0])
    if executable is None:
        print(f"{name}: 未找到命令 {args[0]}")
        return False

    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (result.stdout + result.stderr).strip().splitlines()
    print(f"{name}: 退出码 {result.returncode}")
    for line in output[:max_lines]:
        print(f"  {line}")
    if len(output) > max_lines:
        print(f"  ... 省略 {len(output) - max_lines} 行")
    return result.returncode == 0


def check_rocm_tools() -> bool:
    print_section("ROCm 工具")
    checks = [
        run_command("rocminfo", ["rocminfo"], max_lines=16),
        run_command("rocm-smi", ["rocm-smi", "--showproductname"], max_lines=16),
        run_command("hipcc", ["hipcc", "--version"], max_lines=8),
    ]
    return all(checks)


def check_pytorch() -> bool:
    print_section("PyTorch ROCm")
    if importlib.util.find_spec("torch") is None:
        print("PyTorch 未安装。当前镜像可以检查 ROCm 工具，但不能运行算子实验。")
        return False

    import torch
    import torch.nn.functional as F

    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm/HIP: {torch.version.hip}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.version.hip is None:
        print("当前 PyTorch 不是 ROCm 构建。")
        return False
    if not torch.cuda.is_available():
        print("PyTorch 无法访问 AMD GPU。")
        return False

    device_count = torch.cuda.device_count()
    print(f"GPU 数量: {device_count}")
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        architecture = getattr(properties, "gcnArchName", "未知")
        print(
            f"GPU {index}: {properties.name}; 架构={architecture}; "
            f"显存={properties.total_memory / 1024**3:.1f} GiB"
        )

    device = torch.device("cuda:0")

    left = torch.randn((128, 128), device=device, dtype=torch.float32)
    right = torch.randn((128, 128), device=device, dtype=torch.float32)
    product = left @ right
    torch.cuda.synchronize(device)
    if not torch.isfinite(product).all().item():
        print("矩阵乘法产生了非有限值。")
        return False
    print("GPU 矩阵乘法: 通过")

    gate = torch.randn((256, 1024), device=device, dtype=torch.float32)
    up = torch.randn_like(gate)
    reference = F.silu(gate) * up
    equivalent = gate * torch.sigmoid(gate) * up
    torch.cuda.synchronize(device)
    torch.testing.assert_close(reference, equivalent, rtol=1e-5, atol=1e-6)
    print("SiLUAndMul 等价性检查: 通过")
    return True


def main() -> int:
    print("AMD ROCm 算子实验环境检查")
    print(f"Python: {sys.version.split()[0]}")

    results = {
        "GPU 设备文件": check_device_files(),
        "ROCm 工具": check_rocm_tools(),
        "PyTorch ROCm": check_pytorch(),
    }

    print_section("检查结果")
    for name, passed in results.items():
        print(f"{name}: {'通过' if passed else '失败'}")

    if all(results.values()):
        print("\n环境检查全部通过，可以开始下一个算子实验。")
        return 0

    print("\n环境检查未全部通过，请根据上面的输出定位缺失项。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
