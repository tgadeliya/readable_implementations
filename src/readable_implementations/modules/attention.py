import torch
from torch.nn import Module
from readable_implementations.modules.utils import Linear, Softmax


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_key: int = 64,
        d_value: int = 64,
        d_model: int = 512,
        max_len: int = 512,
        n_heads: int = 8,
        is_masked: bool = False,
    ):
        super().__init__()

        self.d_k = d_key
        self.d_q = self.d_k  # dims should equal to calculate attention
        self.d_v = d_value

        self.d_model = d_model
        self.n_heads = n_heads
        self.is_masked = is_masked

        # We want to perform Attention simultaneously
        # for every head, so we have to extend Parameter
        # matrices
        self.K_proj = Linear(d_model, self.d_k * self.n_heads)
        self.Q_proj = Linear(d_model, self.d_q * self.n_heads)
        self.V_proj = Linear(d_model, self.d_v * self.n_heads)

        self.scaling_factor = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        self.softmax = Softmax(dim=-1)
        self.final_linear = Linear(self.d_v * self.n_heads, d_model)

        # TODO: Why we can't precalculate mask on every step instead of max_len
        attention_mask = torch.triu(torch.ones(max_len,max_len), diagonal=1)
        # We don't want to optimize mask
        self.register_buffer("mask", attention_mask)

    def forward(self, x):

        bs, max_len, d_model = x.size()
        assert d_model == self.d_model, "Input emb and attention emb aren't compatible"

        Q = (
            self.Q_proj(x).view(bs, max_len, self.n_heads, self.d_k).transpose(1, 2)
        )
        K = (
            self.K_proj(x).view(bs, max_len, self.n_heads, self.d_k).transpose(1, 2)
        )
        V = (
            self.V_proj(x).view(bs, max_len, self.n_heads, self.d_k).transpose(1, 2)
        )

        Q_K = torch.matmul(
            Q, torch.transpose(K, -1, -2)
        )
        # scale
        Q_K = Q_K / self.scaling_factor
        # add masked attention
        if self.is_masked:
            # self.mask will be broadcasted for scores
            Q_K.masked_fill_(self.mask, float("-inf"))

        # softmax
        scores = self.softmax(Q_K)
        # multiply V by scores
        attention_by_heads = torch.matmul(scores, V)
        concatenated_heads = attention_by_heads.view(bs, max_len, -1)
        output_attention = self.final_linear(concatenated_heads)
        return output_attention
