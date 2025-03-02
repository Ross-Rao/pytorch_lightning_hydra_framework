# python import
# package import
import torch.nn as nn
# local import

__all__ = ['SimpleConvDecoder']


class SimpleConvDecoder(nn.Module):
    """
    Example:
        >>> import torch
        >>> decoder = SimpleConvDecoder(512, 3, 5)
        >>> x = torch.randn(1, 512, 4, 4)
        >>> y = decoder(x)
        >>> y.shape
        torch.Size([1, 3, 128, 128])
        >>> y.sum().backward()
    """
    def __init__(self, in_channels, out_channels, layers, activation='ReLU', mode='bicubic', align_corners=False):
        super().__init__()
        assert layers > 0 and in_channels > out_channels * 2 ** layers

        layers_lt = []
        for i in range(layers):
            conv_in_channels = in_channels // 2 ** i
            conv_out_channels = in_channels // 2 ** (i + 1)
            if i != layers - 1:
                kernel_size, padding = 3, 1  # inline
            else:
                kernel_size, padding = 5, 2
            layers_lt.append(nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=kernel_size, padding=padding)),
            layers_lt.append(getattr(nn, activation)()),  # ReLU, etc.
            layers_lt.append(nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners))

        penultimate_channels = in_channels // 2 ** layers
        if penultimate_channels != out_channels:
            layers_lt.append(nn.Conv2d(penultimate_channels, out_channels, kernel_size=5, padding=2))

        layers_lt.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers_lt)

    def forward(self, x):
        return self.layers(x)
