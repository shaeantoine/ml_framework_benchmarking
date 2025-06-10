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

    images = data[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(data[b"coarse_labels"], dtype=np.int32)

    return images, labels

# Splitting dataset into batches
def batching(images, labels, batch_size=128, shuffle=True): 
    indicies = np.arange(len(images))
    if shuffle: 
        np.random.shuffle(indicies)

    for start_idx in range(0, len(images), batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indicies[start_idx:end_idx]
        yield images[batch_idx], labels[batch_idx]
       

# Resizing dataset
def resize(images, labels, split, size=(224,224)):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_img_path = CACHE_DIR / f"{split}_images_{size[0]}x{size[1]}.npy"
    cache_lbl_path = CACHE_DIR / f"{split}_labels.npy"

    if cache_img_path.exists() and cache_lbl_path.exists():
        print(f"🔄 Using cached {split} data...")
        return np.load(cache_img_path), np.load(cache_lbl_path)

    print(f"Resizing {split} images to {size}...")
    resized = []
    for i, img in enumerate(images):
        img = np.transpose(img, (1, 2, 0))  # (32, 32, 3)
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize(size, Image.BICUBIC)
        resized_img = np.array(pil_img).astype(np.float32) / 255.0
        resized_img = np.transpose(resized_img, (2, 0, 1))  # (3, H, W)
        resized.append(resized_img)

        if (i + 1) % 5000 == 0:
            print(f"{i + 1}/{len(images)} done")

    resized = np.stack(resized)
    np.save(cache_img_path, resized)
    np.save(cache_lbl_path, labels)
    print(f"✅ Saved {split} set to {cache_img_path}")
    return resized, labels


if __name__ == "__main__":
    print("Downloading...")
    download_cifar100()

    print("Loading raw data...")
    images, labels = load_raw("train")

    print("Resizing raw data...")
    #resize(images, labels, "train")
