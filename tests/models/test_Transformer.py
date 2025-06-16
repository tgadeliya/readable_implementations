from readable_implementations.models import Transformer
import pytest


@pytest.fixture()
def inp():
    return [[13, 123, 23, 5, 64, 8]]


@pytest.fixture()
def model():
    return Transformer()


class TestTransformer:
    def test_happy_path(self, model, inp):
        out = model(inp)
