import torch.nn as nn


__all__ = [
    "ResNet18UpSampleSequentialBackbone",]


class ResidualAdd(nn.Module):
    def __init__(self, fn, upsample):
        super().__init__()
        self.fn = fn
        self.upsample = upsample

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += self.upsample(res)
        return x


class ResidualSequentialTinyBlock(nn.Sequential):
    """
    Example:
        >>> import torch
        >>> block = ResidualSequentialTinyBlock(64, 32, 2, "ReLU")
        >>> x = torch.randn(1, 64, 4, 4)
        >>> y = block(x)
        >>> y.shape
        torch.Size([1, 32, 8, 8])
        >>> block = ResidualSequentialTinyBlock(64, 64, 1, "ReLU")
        >>> x = torch.randn(1, 64, 4, 4)
        >>> y = block(x)
        >>> y.shape
        torch.Size([1, 64, 4, 4])
    """
    def __init__(self, in_channels, out_channels, stride, activation="ReLU", bias=False, ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(  # main
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=bias)
                    if in_channels != out_channels else
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.BatchNorm2d(out_channels),
                    getattr(nn, activation)(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.BatchNorm2d(out_channels),
                ),
                nn.Sequential(  # upsample
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=bias),
                    nn.BatchNorm2d(out_channels),
                ) if in_channels != out_channels else nn.Identity(),
            ),
            getattr(nn, activation)(),
        )


class ResidualSequentialLayer(nn.Sequential):
    def __init__(self, layer_nums, in_channels, out_channels, stride, activation="ReLU", bias=False):
        super().__init__(
            ResidualSequentialTinyBlock(in_channels, out_channels, stride, activation, bias),
            *[ResidualSequentialTinyBlock(out_channels, out_channels, 1, activation, bias)
              for _ in range(layer_nums - 1)]
        )


class ResNet18UpSampleSequentialBackbone(nn.Sequential):
    """
    Example:
        >>> import torch
        >>> model = ResNet18UpSampleSequentialBackbone()
        >>> x = torch.randn(1, 512, 4, 4)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 3, 64, 64])
    """
    def __init__(self, activation="ReLU", in_channels=512, out_channels=3):
        super().__init__(
            ResidualSequentialLayer(2, in_channels, 256, 2, activation),
            ResidualSequentialLayer(2, 256, 128, 2, activation),
            ResidualSequentialLayer(2, 128, 64, 2, activation),
            ResidualSequentialLayer(2, 64, 32, 2, activation),
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, ),
                nn.BatchNorm2d(out_channels),
                getattr(nn, activation)(),  # ReLU, etc.
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            ),
        )
