import pytest
import torch

from readable_implementations import VisionTransformer


@pytest.fixture()
def patch_size():
    return 16


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture()
def hidden_size():
    return 768


@pytest.fixture()
def num_patches():
    return 4

@pytest.fixture()
def num_labels():
    return 10


@pytest.fixture()
def image_batch_patched(batch_size, patch_size, num_patches, hidden_size):
    return torch.rand(batch_size, 3, 32, 32)


@pytest.fixture()
def model(patch_size, num_patches, hidden_size, num_labels):
    return VisionTransformer( num_labels= num_labels,patch_size=patch_size,hidden_size=hidden_size, num_patches=num_patches)


class TestVisualTransformer:
    def test_happy_path(self, model, image_batch_patched):
        out = model(image_batch_patched)
        loss = out
        loss.backward()

    def test_output(self, model, image_batch_patched, batch_size, num_labels):
        out = model(image_batch_patched)

        assert out.size() == (batch_size, num_labels)
        assert not torch.isnan(out).any()
