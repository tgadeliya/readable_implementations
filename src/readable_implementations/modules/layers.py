from collections import OrderedDict

import torch
from torch.nn import Module
from ..modules.attention import MultiHeadAttention
from ..modules.utils import TransformerFeedForward, LayerNorm


class TransformerEncoder(Module):
    def __init__(self, num_blocks=6, num_heads=8):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            OrderedDict(
                [
                    (f"enc{idx}", TransformerEncoderBlock(num_heads=num_heads))
                    for idx in range(num_blocks)
                ]
            )
        )

    def forward(self, inp):
        return self.blocks(inp)


class TransformerEncoderBlock(Module):
    def __init__(self, hidden_size=768, num_heads=8, ff_hidden_size=1024):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(
            d_model=hidden_size, n_heads=num_heads
        )
        self.LayerNorm1 = LayerNorm(d_model=hidden_size)
        self.FeedForward = TransformerFeedForward(
            in_dim=hidden_size,
            out_dim=hidden_size,
            hidden=ff_hidden_size,
        )
        self.LayerNorm2 = LayerNorm(d_model=ff_hidden_size)

    def forward(self, x):
        x_mha = self.MultiHeadAttention(x)
        x = self.LayerNorm1(x + x_mha)
        x_ff = self.FeedForward(x)
        x = self.LayerNorm2(x + x_ff)
        return x


class TransformerDecoderBlock(Module):
    def __init__(self):
        self.MaskedMultiHeadAttention = MultiHeadAttention(is_masked=True)
        self.MultiHeadAttention = MultiHeadAttention()
        self.LayerNorm1 = LayerNorm()
        self.FeedForward = TransformerFeedForward()
        self.LayerNorm2 = LayerNorm()

    def forward(self, x):
        x_mmha = self.MaskedMultiHeadAttention(x)
        x = self.LayerNorm1(x + x_mmha)
        x_mha = self.MultiHeadAttention(x)
        x = self.LayerNorm1(x + x_mha)
        x_ff = self.FeedForward(x)
        x = self.LayerNorm1(x + x_ff)
        return x
