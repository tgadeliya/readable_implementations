import torch
from torch import rand

import pytest
from readable_implementations.modules import TransformerEncoderBlock, TransformerEncoder


@pytest.fixture()
def batch_size():
    return 5


@pytest.fixture()
def num_blocks():
    return 6


@pytest.fixture()
def hidden_size():
    return 768


@pytest.fixture()
def num_heads():
    return 8


@pytest.fixture()
def max_seq_len():
    return 512


@pytest.fixture()
def inp(batch_size, max_seq_len, hidden_size):
    return rand(batch_size, max_seq_len, hidden_size)


@pytest.fixture()
def encoder_block(hidden_size, num_heads):
    return TransformerEncoderBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )


@pytest.fixture()
def encoder(num_heads, num_blocks):
    return TransformerEncoder(num_blocks=6, num_heads=num_heads)


class TestTransformerEncoderBlock:
    def test_happy_path(self, inp, encoder_block):
        out = encoder_block(inp)
        true = torch.rand_like(out)
        loss = (true - out).mean()
        loss.backward()

    def test_output(self, inp, encoder_block):
        out = encoder_block(inp)
        assert out.size() == inp.size()



class TestTransformerEncoder:
    def test_happy_path(self, inp, encoder):
        out = encoder(inp)
        assert out.size() == inp.size()
