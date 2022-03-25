import requests
import os
import pathlib
import tarfile

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR_DEFAULT = os.path.join(CURRENT_DIR, "../../../res/data")


def cifar10_downloader():
    CIFAR10_TAR_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    dir_name = OUTPUT_DIR_DEFAULT
    output_file_path = os.path.join(dir_name, 'cifar-10-python.tar.gz')

    pathlib.Path(dir_name).mkdir(exist_ok=True)

    response = requests.get(CIFAR10_TAR_URL, stream=True)
    if response.status_code == 200:
        with open(output_file_path, "wb") as f:
            f.write(response.raw.read())

    # unpack downloaded tar file
    with tarfile.open(output_file_path) as f:
        f.extractall(dir_name)

    # delete file
    pathlib.Path(output_file_path).unlink()

    print("LOL")

AVAILABLE_DATASETS_DOWNLOADER = {
    "cifar10": cifar10_downloader
}


def download_dataset(dataset_name: str) -> None:
    try:
        dowloader = AVAILABLE_DATASETS_DOWNLOADER[dataset_name.lower()]
        dowloader()
    except KeyError:
        KeyError(f"Downloader for {dataset_name} is not implemented")


if __name__ == '__main__':
    cifar10_downloader()