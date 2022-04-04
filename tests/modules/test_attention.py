import pytest
import torch

from readable_implementations.modules import MultiHeadAttention

class TestMHA:
    def test_happy_path(self):
        torch.autograd.set_detect_anomaly(True)
        bs, max_len, d_model = 3, 512, 768
        inp = torch.rand(bs, max_len, d_model, requires_grad=True)

        mha = MultiHeadAttention(d_model=d_model)
        out = mha(inp)

        loss = out.mean()
        loss.backward()
