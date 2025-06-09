import os
import pickle
import tarfile
import numpy as np
from PIL import Image
import urllib.request
import mlx.core as mx
from pathlib import Path

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CACHE_DIR = Path("data/resized")
RAW_DIR = Path("data/raw")

# Download and extract dataset
def download_cifar100():
    if RAW_DIR.exists():
        return 
    RAW_DIR.mkdir(parents=True)
    tar_path = RAW_DIR / "cifar-100-python.tar.gz"

    print("Downloading CIFAR-100...")
    urllib.request.urlretrieve(CIFAR_URL, tar_path)

    with tarfile.open(tar_path) as tar: 
        print("Extracting...")
        tar.extractall(path=RAW_DIR)

    print("Done.")


# Load raw dataset
def load_raw(split):
    file = RAW_DIR / "cifar-100-python" / f"{split}"

    with open(file, "rb") as fo: 
        data = pickle.load(fo, encoding="bytes")

    print(f"All of the keys in the dictionary: {data.keys()}")
    print(data[b'data'])

    #images = data["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    #labels = np.array(data["fine_labels"], dtype=np.int32)

    #return images, labels


if __name__ == "__main__":
    print("Downloading...")
    download_cifar100()

    print("Loading raw data...")
    load_raw("train")