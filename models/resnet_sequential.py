import torch.nn as nn

__all__ = [
    "ResidualSequentialLargeBlock",
    "ResidualSequentialLayer",
    "ResNet18SequentialBackbone",
    "ResNet34SequentialBackbone",
    "MiniResNetSequentialBackbone",
    "ResNetSequentialClassifierHead",
    "ResNet18Sequential",
    "ResNet34Sequential",
    "MiniResNetSequential",
]


class ResidualAdd(nn.Module):
    def __init__(self, fn, downsample):
        super().__init__()
        self.fn = fn
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += self.downsample(res)
        return x


class ResidualSequentialTinyBlock(nn.Sequential):
    """
    Example:
        >>> import torch
        >>> block = ResidualSequentialTinyBlock(3, 64, 1, "ReLU")
        >>> x = torch.randn(1, 3, 224, 224)
        >>> y = block(x)
        >>> y.shape
        torch.Size([1, 64, 224, 224])
    """
    def __init__(self, in_channels, out_channels, stride, activation="ReLU", bias=False, ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(  # main
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
                    nn.BatchNorm2d(out_channels),
                    getattr(nn, activation)(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.BatchNorm2d(out_channels),
                ),
                nn.Sequential(  # downsample
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                    nn.BatchNorm2d(out_channels),
                ) if in_channels != out_channels else nn.Identity(),
            ),
            getattr(nn, activation)(),
        )


class ResidualSequentialLargeBlock(nn.Sequential):
    pass


class ResidualSequentialLayer(nn.Sequential):
    def __init__(self, layer_nums, in_channels, out_channels, stride, activation="ReLU", bias=False):
        super().__init__(
            ResidualSequentialTinyBlock(in_channels, out_channels, stride, activation, bias),
            *[ResidualSequentialTinyBlock(out_channels, out_channels, 1, activation, bias)
              for _ in range(layer_nums - 1)]
        )


class ResNet18SequentialBackbone(nn.Sequential):
    def __init__(self, activation="ReLU", in_channels=3):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, ),
                nn.BatchNorm2d(64),
                getattr(nn, activation)(),  # ReLU, etc.
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            ResidualSequentialLayer(2, 64, 64, 1, activation),
            ResidualSequentialLayer(2, 64, 128, 2, activation),
            ResidualSequentialLayer(2, 128, 256, 2, activation),
            ResidualSequentialLayer(2, 256, 512, 2, activation),
        )


class ResNet34SequentialBackbone(nn.Sequential):
    def __init__(self, activation="ReLU", in_channels=3):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, ),
                nn.BatchNorm2d(64),
                getattr(nn, activation)(),  # ReLU, etc.
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            ResidualSequentialLayer(3, 64, 64, 1, activation),
            ResidualSequentialLayer(4, 64, 128, 2, activation),
            ResidualSequentialLayer(6, 128, 256, 2, activation),
            ResidualSequentialLayer(3, 256, 512, 2, activation),
        )


class MiniResNetSequentialBackbone(nn.Sequential):
    def __init__(self, activation="ReLU", in_channels=3):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, ),
                nn.BatchNorm2d(16),
                getattr(nn, activation)(),  # ReLU, etc.
            ),
            ResidualSequentialLayer(2, 16, 16, 1, activation),
            ResidualSequentialLayer(2, 16, 32, 2, activation),
            ResidualSequentialLayer(2, 32, 64, 2, activation),
        )


class ResNetSequentialClassifierHead(nn.Sequential):
    def __init__(self, num_classes, in_channels=512):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )


class ResNet18Sequential(nn.Sequential):
    def __init__(self, num_classes, activation="ReLU", in_channels=3):
        super().__init__(
            ResNet18SequentialBackbone(activation, in_channels),
            ResNetSequentialClassifierHead(num_classes, 512),
        )


class ResNet34Sequential(nn.Sequential):
    def __init__(self, num_classes, activation="ReLU", in_channels=3):
        super().__init__(
            ResNet34SequentialBackbone(activation, in_channels),
            ResNetSequentialClassifierHead(num_classes, 512),
        )


class MiniResNetSequential(nn.Sequential):
    def __init__(self, num_classes, activation="ReLU", in_channels=3):
        super().__init__(
            MiniResNetSequentialBackbone(activation, in_channels),
            ResNetSequentialClassifierHead(num_classes, 64),
        )
