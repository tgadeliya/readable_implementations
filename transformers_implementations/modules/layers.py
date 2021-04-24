import torch
from torch.nn import Module
from ..modules.attention import MultiHeadAttention
from ..modules.attention import MultiHeadAttention
from ..modules.utils import TransformerFeedForward, LayerNorm


class TransformerEncoder(Module):
    def __init__(self):

        self.MultiHeadAttention = MultiHeadAttention()
        self.LayerNorm1 = LayerNorm()
        self.FeedForward = TransformerFeedForward()
        self.LayerNorm2 = LayerNorm()

    def forward(self, x):
        x_mha = self.MultiHeadAttention(x)
        x = self.LayerNorm1(x + x_mha)
        x_ff = self.FeedForward(x)
        x = self.LayerNorm2(x + x_ff)
        return x


class TransformerDecoder(Module):
    def __init__(self):
        self.MaskedMultiHeadAttention = MultiHeadAttention()
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
