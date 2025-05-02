# python import
# package import
import torch
import torch.nn as nn
# local import
from models.attention import Attention


__all__ = ['FeedForwardSequentialBlock',
           'TransformerEncoderSequentialLayer',
           'TransformerSequentialEncoder']


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc. networks, however,
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
    Obtained from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardSequentialBlock(nn.Sequential):
    def __init__(self, dim, hidden_dim, output_dim=None, activation='GELU', dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim if output_dim is None else output_dim),
        )


class TransformerEncoderSequentialLayer(nn.Sequential):
    """
    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 32, 512)
        >>> model = TransformerEncoderSequentialLayer(d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1)
        >>> output = model(tensor)
        >>> print(output.shape)
        torch.Size([1, 32, 512])
        >>> loss = output.sum()
        >>> loss.backward()
    """
    def __init__(self, d_model,
                 num_heads,
                 output_dim=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='GELU',
                 attention_dropout=0.1,
                 drop_path_rate=0.1,
                 ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(d_model),
                Attention(d_model, num_heads, attention_dropout, dropout),
                DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(d_model),
                FeedForwardSequentialBlock(d_model, dim_feedforward, output_dim, activation, dropout),
                DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
            )),
        )


class TransformerSequentialEncoder(nn.Sequential):
    """
    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 32, 512)
        >>> model = TransformerSequentialEncoder(d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1)
        >>> output = model(tensor)
        >>> print(output.shape)
        torch.Size([1, 32, 512])
    """
    def __init__(self, n_layers: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderSequentialLayer(**kwargs) for _ in range(n_layers)])


class TransformerSequentialClassificationHead(nn.Sequential):
    def __init__(self, d_model, num_classes):
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
