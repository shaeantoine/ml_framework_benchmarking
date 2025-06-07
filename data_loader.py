import os
import pickle
import urllib.request
import tarfile
import numpy as np
from pathlib import Path
import mlx.core as mx

# CIFAR-100 URL and file path
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
DATA_DIR = Path("./data/cifar100")
EXTRACTED_DIR = DATA_DIR / "cifar-100-python"

# Download and extract dataset
def download_cifar100():
    if not EXTRACTED_DIR.exists():
        os.makedirs(DATA_DIR, exist_ok=True)
        tar_path = DATA_DIR / "cifar-100-python.tar.gz"
        print("Downloading CIFAR-100...")
        urllib.request.urlretrieve(CIFAR_URL, tar_path)
        print("Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=DATA_DIR)
        print("Done.")

# Load and preprocess dataset
def load_cifar100(split="train"):
    assert split in ["train", "test"]
    with open(EXTRACTED_DIR / f"{split}", "rb") as f:
        data_dict = pickle.load(f, encoding="latin1")
    
    # Data: shape (N, 3, 32, 32), Labels: shape (N,)
    data = data_dict["data"].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(data_dict["fine_labels"], dtype=np.int32)

    # Normalize to [0, 1] then apply ImageNet normalization
    data /= 255.0
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    data = (data - mean) / std

    return data, labels

# Simple batching function
def make_batches(data, labels, batch_size, shuffle=True):
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, len(data), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_x = mx.array(data[batch_idx])  # shape [B, 3, 32, 32]
        batch_y = mx.array(labels[batch_idx])  # shape [B]
        yield batch_x, batch_y
