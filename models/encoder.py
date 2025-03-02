# python import
# package import
import torch.nn as nn
from torchvision.models.resnet import resnet34, resnet50, resnet18
# local import

__all__ = ['ResNetEncoder']


class ResNetEncoder(nn.Module):
    """
    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 3, 224, 224)
        >>> model = ResNetEncoder(backbone='resnet34', pretrained=True, freeze_all=False)
        >>> output = model(tensor)
        >>> print(output.shape)  # torch.Size([1, 512, 7, 7]) but torch will turn 4d tensor into 2d tensor
        torch.Size([1, 25088])
    """
    def __init__(self, backbone, in_channels=3, pretrained=True, freeze_all=False):
        super(ResNetEncoder, self).__init__()
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError

        # if in_channels is not 3, change the first layer of the resnet
        original_in_channels = self.resnet.conv1.in_channels  # 3
        if original_in_channels != in_channels:
            assert in_channels > 0, 'in_channels must be greater than 0'
            conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            data = self.resnet.conv1.weight.data
            # copy the data from the original weight to the new weight
            self.resnet.conv1 = conv
            self.resnet.conv1.weight.data[:, :min(in_channels, original_in_channels), :, :]\
                = data[:, :min(in_channels, original_in_channels), :, :]

        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()

        if freeze_all:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
