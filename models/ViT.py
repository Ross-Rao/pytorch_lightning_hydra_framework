# python import
# package import
import torch
import torch.nn as nn
# local import
from models.transformer_sequential import TransformerSequentialEncoder

__all__ = ['PatchEmbedding', 'SequentialVisionTransformer', 'ViTModule', 'ViTSequentialClassificationHead']


class PatchEmbedding(nn.Module):
    """
    Example:
        >>> import torch
        >>> patch_size = 64
        >>> emb_size = 256
        >>> in_channels = 3
        >>> img_size = 64
        >>> x = torch.randn(1, in_channels, 64, 64)
        >>> model = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        >>> model(x).shape
        torch.Size([1, 197, 256])
    """
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positions
        return x


class ViTSequentialClassificationHead(nn.Sequential):
    def __init__(self, d_model, n_classes):
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )


class SequentialVisionTransformer(nn.Sequential):
    """
    Example:
        >>> import torch
        >>> in_channels = 3
        >>> patch_size = 16
        >>> emb_size = 512
        >>> img_size = 224
        >>> d_model = emb_size
        >>> num_heads = 8
        >>> dim_feedforward = 2048
        >>> dropout = 0.1
        >>> n_classes = 1000
        >>> x = torch.randn(1, in_channels, 224, 224)
        >>> model = SequentialVisionTransformer(in_channels, n_classes, patch_size,
        ...                           emb_size, img_size, d_model, num_heads, dim_feedforward, dropout)
        >>> model(x)[:, 0].shape
        torch.Size([1, 1000])
    """
    def __init__(self, in_channels,
                 n_classes,
                 patch_size,
                 emb_size,
                 img_size,
                 d_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 n_layers=12):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerSequentialEncoder(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                                         dropout=dropout, n_layers=n_layers),
            ViTSequentialClassificationHead(d_model, n_classes)
        )


class ViTModule(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 patch_size,
                 emb_size,
                 img_size,
                 d_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 n_layers=12):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoder = TransformerSequentialEncoder(d_model=d_model, num_heads=num_heads,
                                                                dim_feedforward=dim_feedforward, dropout=dropout,
                                                                n_layers=n_layers)
        self.classification_head = ViTSequentialClassificationHead(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x)
        return x[:, 0]
