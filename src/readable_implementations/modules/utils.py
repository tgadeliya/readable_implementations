import math

import torch
from torch import Tensor, tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class TransformerFeedForward(Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512, hidden: int = 2048):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.lin1 = Linear(in_dim, hidden)
        self.lin2 = Linear(hidden, out_dim)
        self.ReLU = ReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.ReLU(x)
        x = self.lin2(x)
        return x


class Linear(Module):
    def __init__(self, in_dim, out_dim, initialization="kaiming"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = Parameter(torch.Tensor(out_dim))

        if initialization == "kaiming":
            init_kaiming(self.weight)
            init_kaiming(self.bias)

    def forward(self, x):
        assert (
            x.size()[-1] == self.in_dim
        ), f"Input dimensions {x.size()} don't correspond to in_dim {self.in_dim}"
        x = torch.matmul(x, self.weight)
        x += self.bias
        return x


# TODO: Add GELU


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0)


class LayerNorm(Module):
    def __init__(self, d_model: int = 512, eps=1e-5):
        super().__init__()
        self.H = d_model
        self.eps = eps

    def forward(self, x):
        # x (bs, max_len, d_model)
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        mean_x2 = (x ** 2).mean(dim=(1, 2), keepdims=True)
        var = mean_x2 - mean ** 2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm


class Softmax(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = dim

    def forward(self, x: Tensor) -> Tensor:
        x = torch.exp(x)
        sum = x.sum(dim=self.dim)
        x /= sum.unsqueeze(self.dim)
        return x


def init_kaiming(t, mode="fan_in"):
    """
    Initilize input tensor with two assumptions:
    - non_linearity=ReLU
    - input tensor has <=2 dimensions
    """
    fan_in = t.size()[0]
    fan_out = 1 if len(t.size()) == 1 else t.size()[1]
    assert mode in ["fan_in", "fan_out"], f"Mode {mode} is a wrong type!"
    fan = fan_in if mode == "fan_in" else fan_out
    gain = math.sqrt(2.0)  # ReLU case
    with torch.no_grad():
        return t.normal_(0, gain / math.sqrt(fan))
