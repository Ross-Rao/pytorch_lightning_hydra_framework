# python import
# package import
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34, resnet50, resnet18
# local import
from models.attention import Attention
from utils.util import get_multi_attr

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


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TransformerEncoderLayer(nn.Module):
    """
    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 32, 512)
        >>> model = TransformerEncoderLayer(d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1)
        >>> output = model(tensor, tensor)
        >>> print(output.shape)
        torch.Size([1, 32, 512])
        >>> loss = output.sum()
        >>> loss.backward()
    """
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, activation='GELU',
                 attention_dropout=0.1, drop_path_rate=0.1, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = d_model

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Dropout(dropout),
        )

        self.attention = Attention(dim=d_model, num_heads=num_heads,
                                   attention_dropout=attention_dropout,
                                   projection_dropout=dropout, )

        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, q, kv):
        # attention op include dropout
        x = self.drop_path(self.attention(self.q_norm(q), self.kv_norm(kv)))
        x = self.norm(x + q)
        result = self.drop_path(self.feed_forward(x))
        result = self.norm(result + x)
        return result
