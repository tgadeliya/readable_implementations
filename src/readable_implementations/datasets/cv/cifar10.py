from pathlib import Path
import pickle
from itertools import chain
from typing import Union, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

Example = Dict[str, Union[int, np.ndarray]]


class CIFAR10(Dataset):
    """
    Original CIFAR10 Dataset

    Output Tensor

    """

    INTERNAL_PATH_TO_FILES = "../../../../res/data/cifar-10-batches-py"

    def __init__(
        self, path_to_dir: Optional[Path] = INTERNAL_PATH_TO_FILES, mode: str = "train"
    ) -> None:
        # TODO: Add normalization option
        self.mode = mode
        path_to_dir = Path(path_to_dir)
        cifar10_dict = self.read_cifar_data(path_to_dir)
        self.label_names = cifar10_dict["label_names"]
        self.images = cifar10_dict["images"]
        self.labels = cifar10_dict["labels"]
        assert len(self.images) == (len(self.labels))

    def __getitem__(self, idx: int) -> Example:
        """Return normalized(!) image tensor with label index.
        Use getitem_unnormalized to get raw image tensor by idx.
        Output:
            image: C x H x W
            labels: int
        """
        return {"input": self.images[idx], "target": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def read_cifar_data(self, dir_path: Path) -> Dict:
        labels = []
        data_files = []

        file_to_read = "data_batch_*" if self.mode == "train" else "test_batch"
        for data_path in dir_path.glob(file_to_read):
            data = self.unpickle(data_path)
            data_files.append(data[b"data"])
            labels.append(data[b"labels"])

        # flatten labels
        labels = list(chain.from_iterable(labels))
        # stack and reshape into image, where dims are N x (C x H x W)
        data_files = np.vstack(data_files).reshape(-1, 3, 32, 32)
        data_files = torch.Tensor(data_files)

        meta_file_path = dir_path.joinpath("batches.meta")
        label_names = [
            label.decode("utf-8")
            for label in self.unpickle(meta_file_path)[b"label_names"]
        ]

        return {"images": data_files, "labels": labels, "label_names": label_names}

    @staticmethod
    def unpickle(file: str) -> Dict:
        with open(file, "rb") as fo:
            file_dict = pickle.load(fo, encoding="bytes")
        return file_dict

    def getitem_unnormalized(self, idx: int):
        NotImplementedError("It is not implemented!")
