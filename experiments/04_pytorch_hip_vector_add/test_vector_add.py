import torch

import hip_vector_add


def main() -> None:
    torch.manual_seed(0)

    for size in (1, 257, 1 << 20):
        a = torch.randn(size, device="cuda", dtype=torch.float32)
        b = torch.randn(size, device="cuda", dtype=torch.float32)

        expected = a + b
        actual = hip_vector_add.vector_add(a, b)
        torch.testing.assert_close(actual, expected)
        print(f"size={size}: 通过")


if __name__ == "__main__":
    main()
