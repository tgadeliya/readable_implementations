import typing as T
from torchtyping import TensorType  # type: ignore

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter


class Embeddings(Module):
    def __init__(self, d_emb, vocab_size, max_model_len) -> None:
        super(Embeddings, self).__init__()
        self.d_emb = d_emb
        self.max_model_len = max_model_len
        # TODO: Add normal distribution initialization
        self.emb_matrix = Parameter(torch.zeros(size=(vocab_size, d_emb)))
        # precompute all positional encodings up to maximal possible len
        self.pos_enc_max = Parameter(
            self.precompute_pe(max_model_len, d_emb), requires_grad=False
        )

    def forward(self, x: T.List[T.List[int]]) -> torch.Tensor:
        lens = [len(l) for l in x]
        max_len = max(lens)
        bs = len(x)

        print(lens, max_len, bs)

        # TensorType("batch", "max_len", "d_model")
        x_emb = torch.zeros(
            size=(bs, max_len, self.d_emb)
        )
        print("X_EMB", x_emb.size())

        for i in range(bs):
            # TODO: Get rid of transposition
            print(i, self.emb_matrix[x[i], :].size(), self.get_pe(lens[i]).T.size())
            x_emb_i = self.emb_matrix[x[i], :] + self.get_pe(lens[i]).T
            x_emb[i, : lens[i], :] = x_emb_i

        x_emb /= torch.sqrt(torch.Tensor([self.d_emb]))
        return x_emb

    def get_pe(self, inp_len: int) -> torch.Tensor:
        return self.pos_enc_max[:, :inp_len]

    @staticmethod
    def precompute_pe(
        max_model_len: int, d_emb: int
    ):
        # TensorType("d_emb", "max_model_len") 
        pos_arg = torch.arange(0, max_model_len)
        dim_arg = 10000 ** (torch.div(torch.arange(0, d_emb), 2.0) / d_emb).reshape(
            -1, 1
        )

        # perform broadcasting to obtain
        # matrix (d_emb, max_model_len),
        # where each pe[:,i] is a positional embedding
        # for position i
        pe = pos_arg / dim_arg
        # apply sin, cos
        pe[::2] = torch.sin(pe[::2])
        pe[1::2] = torch.cos(pe[1::2])
        return pe
