import torch
import triton
import triton.language as tl


@triton.jit
def block_sum_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets, mask=offsets < n, other=0.0)
    block_sum = tl.sum(values, axis=0)
    tl.store(output_ptr + program_id, block_sum)


def block_sum(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    blocks = triton.cdiv(x.numel(), block_size)
    partial_sums = torch.empty(blocks, device=x.device, dtype=torch.float32)
    block_sum_kernel[(blocks,)](
        x, partial_sums, x.numel(), BLOCK_SIZE=block_size
    )
    return partial_sums


def main() -> None:
    n = (1 << 20) + 13
    block_size = 256
    x = torch.ones(n, device="cuda", dtype=torch.float32)

    partial_sums = block_sum(x, block_size)
    expected = torch.full_like(partial_sums, float(block_size))
    expected[-1] = n - (partial_sums.numel() - 1) * block_size
    torch.testing.assert_close(partial_sums, expected)

    print(f"元素数量: {n}")
    print(f"Program 数量: {partial_sums.numel()}")
    print(f"每个 Tile 的元素数: {block_size}")
    print(f"最后一个 Tile 的和: {partial_sums[-1].item()}")
    print("检查通过")


if __name__ == "__main__":
    main()
