import torch
import triton
import triton.language as tl


@triton.jit
def tiled_softmax_kernel(
    input_ptr,
    output_ptr,
    row_count,
    col_count,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    row_tile_id = tl.program_id(1)

    row_offsets = row_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    offsets = (
        batch_id * row_count * col_count
        + row_offsets[:, None] * col_count
        + col_offsets[None, :]
    )
    mask = (row_offsets[:, None] < row_count) & (
        col_offsets[None, :] < col_count
    )

    values = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(values, axis=1, keep_dims=True)
    numerators = tl.exp(values - row_max)
    row_sum = tl.sum(numerators, axis=1, keep_dims=True)
    output = numerators / row_sum
    tl.store(output_ptr + offsets, output, mask=mask)


def tiled_softmax(x: torch.Tensor, block_m: int = 4) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    assert x.ndim == 3

    batch_size, row_count, col_count = x.shape
    block_n = triton.next_power_of_2(col_count)
    output = torch.empty_like(x)
    grid = (batch_size, triton.cdiv(row_count, block_m))
    tiled_softmax_kernel[grid](
        x,
        output,
        row_count,
        col_count,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return output


def main() -> None:
    torch.manual_seed(0)
    shape = (3, 17, 200)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    expected = torch.softmax(x, dim=-1)
    actual = tiled_softmax(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    print(f"输入形状: {shape}")
    print("Grid: (3, 5)")
    print("Tile: (4, 256)")
    print("Reduction: max(axis=1) + sum(axis=1)")
    print("检查通过")


if __name__ == "__main__":
    main()
