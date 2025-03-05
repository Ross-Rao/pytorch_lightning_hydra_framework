import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SingleHeadSelfAttention', 'MultiHeadSelfAttention', 'Attention']


def scaled_dot_product_attention(q, k, v, attention_dropout=None):
    """
    softmax((q @ k^T) / sqrt(d_k)) @ v
    """
    dim_key = k.size(-1)
    attention_score = (q @ k.transpose(-2, -1)) / np.sqrt(dim_key)

    if attention_dropout is not None:
        attention_score = attention_dropout(attention_score)

    attention_weight = F.softmax(attention_score, dim=-1)
    result = attention_weight @ v
    return result


class SingleHeadSelfAttention(nn.Module):
    """
    Example:
        >>> x = torch.randn(2, 10, 64)
        >>> attention = SingleHeadSelfAttention(64, 8)
        >>> result = attention(x)
        >>> result.size()
        torch.Size([2, 10, 8])
    """
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.wq = nn.Linear(embed_dim, head_dim, bias=False)
        self.wk = nn.Linear(embed_dim, head_dim, bias=False)
        self.wv = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x):
        q = self.wq(x)  # y = xA
        k = self.wk(x)
        v = self.wv(x)

        result = scaled_dot_product_attention(q, k, v)
        return result


class MultiHeadSelfAttention(nn.Module):
    """
    Example:
        >>> x = torch.randn(2, 10, 64)
        >>> attention = MultiHeadSelfAttention(64, 8, 8)
        >>> result = attention(x)
        >>> result.size()
        torch.Size([2, 10, 64])
    """
    def __init__(self, embed_dim, head_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadSelfAttention(embed_dim, head_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        result = self.fc(head_outputs)
        return result


class Attention(nn.Module):
    """
    Example:
        >>> self_attn = Attention(dim=512, num_heads=8, attention_dropout=0.1, projection_dropout=0.1)
        >>> x = torch.rand(16, 32, 512)
        >>> self_attn(x, x).shape
        torch.Size([16, 64, 512])
    """
    def __init__(self, dim, num_heads, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, q, kv):
        (b, n, c), h = q.shape, self.num_heads

        # (b, n, c) -> (b, h, n, e)
        q = self.w_q(q).reshape(b, n, h, -1).transpose(1, 2)
        kv = self.w_kv(kv).reshape(b, n, 2, h, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        x = scaled_dot_product_attention(q, k, v, self.attention_dropout)
        x = x.transpose(1, 2).reshape(b, n, c)  # (b, h, n, e) -> (b, n, h, e) -> (b, n, c)

        x = self.proj(x)
        x = self.projection_dropout(x)
        return x
