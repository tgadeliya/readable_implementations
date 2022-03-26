import pytest
import torch

from readable_implementations.modules.embeddings import EmbeddingPatches


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
def image_batch_patched(batch_size, patch_size, num_patches, hidden_size):
    return torch.rand(batch_size, num_patches, patch_size * patch_size * 3)


@pytest.fixture()
def embedding(patch_size, num_patches, hidden_size):
    return EmbeddingPatches(
        patch_size=patch_size, hidden_size=hidden_size, num_patches=num_patches
    )


class TestEmbeddingPatches:
    def test_happy_path(self, embedding, image_batch_patched):
        embedding(image_batch_patched)

    def test_output_dim(
        self, embedding, image_batch_patched, batch_size, num_patches, hidden_size
    ):
        output = embedding(image_batch_patched)
        assert len(output.size()) == 3
        assert output.size() == (batch_size, num_patches, hidden_size)
