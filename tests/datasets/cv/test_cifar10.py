import pytest
import torch
from torch.utils.data import DataLoader

from readable_implementations.datasets import CIFAR10


# Dataset output:
#   - Tensor
#   - Normalize
#   - dims: C x H x W

# Dataloader:
# B x


@pytest.fixture()
def path_to_dir():
    return "/Users/tsimur.hadeliya/code/repos/readable_implementations/res/data/cifar-10-batches-py"


@pytest.fixture()
def dataset(path_to_dir):
    return CIFAR10(path_to_dir=path_to_dir)


class TestCIFAR10:
    def test_happy_path(self):
        dataset = CIFAR10(
            path_to_dir="/Users/tsimur.hadeliya/code/repos/readable_implementations/res/data/cifar-10-batches-py"
        )
        dataset.__getitem__(1)

    def test_getitem(self, dataset):
        example = dataset.__getitem__(0)

        assert len(dataset) == 50000
        assert example.keys() == set(["input", "target"])

    def test_image_output(self, dataset):
        example = dataset.__getitem__(0)

        assert isinstance(example["input"], torch.Tensor)
        assert example["input"].size() == (3, 32, 32)

    def test_dataloading(self, dataset):
        dataloader = DataLoader(dataset, batch_size=5)
        batch = next(iter(dataloader))

        assert batch["input"].size() == (5, 3, 32, 32)
