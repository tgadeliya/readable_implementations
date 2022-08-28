from collections import OrderedDict

import torch
from torch.nn import Module
from ..modules.attention import MultiHeadAttention
from ..modules.utils import TransformerFeedForward, LayerNorm


class TransformerEncoder(Module):
    def __init__(self, num_blocks=6, num_heads=8, d_model: int = 768, d_ff: int = 2048):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        f"enc{idx}",
                        TransformerEncoderBlock(
                            d_model=d_model,
                            num_heads=num_heads,
                            ff_hidden_size=d_ff,
                        ),
                    )
                    for idx in range(num_blocks)
                ]
            )
        )

    def forward(self, inp):
        return self.blocks(inp)


class TransformerDecoder(Module):
    def __init__(
        self, num_blocks: int = 6, num_heads: int = 8, d_model=768, d_ff=2048
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        f"dec{idx}",
                        TransformerDecoderBlock(
                            num_heads=num_heads,
                            d_model=d_model,
                            d_ff=d_ff,
                        ),
                    )
                    for idx in range(num_blocks)
                ]
            )
        )

    def forward(self, x, encoder_output):
        for block in self.blocks:
            out = block(x, encoder_output)
        return out


class TransformerEncoderBlock(Module):
    def __init__(self, d_model=768, num_heads=8, ff_hidden_size=1024):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(d_model=d_model, n_heads=num_heads)
        self.LayerNorm1 = LayerNorm(d_model=d_model)
        self.FeedForward = TransformerFeedForward(
            in_dim=d_model,
            out_dim=d_model,
            hidden=ff_hidden_size,
        )
        self.LayerNorm2 = LayerNorm(d_model=ff_hidden_size)

    def forward(self, x):
        x_mha = self.MultiHeadAttention(v=x, k=x, q=x)
        x = self.LayerNorm1(x + x_mha)
        x_ff = self.FeedForward(x)
        x = self.LayerNorm2(x + x_ff)
        return x


class TransformerDecoderBlock(Module):
    def __init__(self, num_heads: int, d_ff: int = 2048, d_model: int = 768) -> None:
        super().__init__()
        self.MaskedMultiHeadAttention = MultiHeadAttention(
            d_model=d_model, n_heads=num_heads, is_masked=True
        )
        self.MultiHeadAttention = MultiHeadAttention(
            d_model=d_model, n_heads=num_heads, is_masked=True
        )
        self.LayerNorm1 = LayerNorm(d_model=d_model)
        self.FeedForward = TransformerFeedForward(
            in_dim=d_model, out_dim=d_model, hidden=d_ff
        )
        self.LayerNorm2 = LayerNorm(d_model=d_model)
        self.LayerNorm3 = LayerNorm(d_model=d_ff)

    def forward(self, x, encoder_output):
        x_mmha = self.MaskedMultiHeadAttention(v=x, k=x, q=x)
        x = self.LayerNorm1(x + x_mmha)
        x_mha = self.MultiHeadAttention(v=encoder_output, k=encoder_output, q=x)
        x = self.LayerNorm2(x + x_mha)
        x_ff = self.FeedForward(x)
        x = self.LayerNorm3(x + x_ff)
        return x
