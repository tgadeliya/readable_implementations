import pytest

from readable_implementations.datasets import CIFAR10


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
        assert example.keys() == set(["image", "label"])
