import torch

import hip_tiled_softmax


def main() -> None:
    torch.manual_seed(0)
    shape = (3, 17, 200)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    expected = torch.softmax(x, dim=-1)
    actual = hip_tiled_softmax.tiled_softmax(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    print(f"输入形状: {shape}")
    print("Grid: (3, 5)")
    print("Block: 256 threads")
    print("逻辑 Tile: (4, 256)")
    print("检查通过")


if __name__ == "__main__":
    main()
