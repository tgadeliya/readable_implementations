import pytest
import torch

from readable_implementations.modules.utils import Linear, TransformerFeedForward, LayerNorm, ReLU, Softmax


class TestLinear:
    def test_happy_path(self):
        in_dim , out_dim = 10, 5
        lin = Linear(in_dim, out_dim)
        inp = torch.rand(4, 5, in_dim, requires_grad=True)
        out = lin(inp)

        true = torch.rand_like(out)
        loss = ((out - true) ** 2).mean()
        loss.backward()


class TestFeedForward:
    def test_happy_path(self):
        in_dim , out_dim = 10, 5
        ff = TransformerFeedForward(in_dim, out_dim, hidden=200)
        inp = torch.rand(4, 5, in_dim, requires_grad=True)
        out = ff(inp)
        loss = out.mean()
        loss.backward()


class TestLayerNorm:
    def test_happy_path(self):
        bs , max_len, d_model = 3, 128, 768

        ln = LayerNorm(d_model)
        inp = torch.rand(bs, max_len, d_model, requires_grad=True)

        out = ln(inp)
        loss = out.mean()
        loss.backward()

class TestReLU:
    def test_happy_path(self):
        bs , max_len, d_model = 3, 128, 768

        relu = ReLU()
        inp = torch.rand(bs, max_len, d_model, requires_grad=True)

        out = relu(inp)
        loss = out.mean()
        loss.backward()

class TestSoftmax:
    def test_happy_path(self):
        bs , max_len, d_model = 3, 128, 768

        softmax = Softmax(dim=-1)
        inp = torch.rand(bs, max_len, d_model, requires_grad=True)
        out = softmax(inp)
        loss = out.mean()
        loss.backward()