import torch

from models.cbam import CBAM


def main():
    x = torch.randn(2, 512, 7, 7)

    cbam = CBAM(
        channels=512,
        reduction=16,
        spatial_kernel=7,
    )

    y = cbam(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)

    assert y.shape == x.shape

    print("CBAM shape test passed.")


if __name__ == "__main__":
    main()
