import typing as T

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from .utils import init_positional_embeddings


class Embeddings(Module):
    def __init__(self, d_emb, vocab_size, max_model_len) -> None:
        super().__init__()
        self.d_emb = d_emb
        self.max_model_len = max_model_len
        # TODO: Add normal distribution initialization
        self.emb_matrix = Parameter(torch.zeros(size=(vocab_size, d_emb)))

        # precompute all positional encodings up to maximal possible len
        # TODO: split up positional encoding and Embeddings
        self.pos_enc_max = Parameter(
            init_positional_embeddings(max_model_len, d_emb), requires_grad=False
        )

    def forward(self, x: T.List[T.List[int]]) -> torch.Tensor:
        lens = [len(l) for l in x]
        # TODO: Understand whether we pad up to maximal model len or batch len
        max_len = min(self.max_model_len, max(lens))
        bs = len(x)

        x_emb = torch.zeros(size=(bs, max_len, self.d_emb))

        for i in range(bs):
            # TODO: Get rid of transposition
            x_emb_i = self.emb_matrix[x[i], :] + self.get_pe(lens[i]).T
            x_emb[i, : lens[i], :] = x_emb_i

        x_emb /= torch.sqrt(torch.Tensor([self.d_emb]))
        return x_emb

    def get_pe(self, inp_len: int) -> torch.Tensor:
        return self.pos_enc_max[:, :inp_len]
