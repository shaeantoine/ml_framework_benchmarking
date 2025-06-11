import os
import numpy as np
from glob import glob

# Optional imports
import mlx.core as mx
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

class DiskCachedDataset:
    def __init__(self, split="train", size=(224, 224), cache_dir="cache", shuffle=True):
        self.cache_dir = cache_dir
        pattern = os.path.join(cache_dir, f"{split}_batch_*_{size[0]}x{size[1]}.npz")
        self.batch_files = sorted(glob(pattern))
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.batch_files)

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        data = np.load(self.batch_files[idx])
        return data["images"], data["labels"]

    def __iter__(self):
        for path in self.batch_files:
            data = np.load(path)
            yield data["images"], data["labels"]

    # --- MLX Generator ---
    def as_mlx(self):
        def generator():
            for x, y in self:
                yield mx.array(x), mx.array(y)
        return generator

    # --- PyTorch Dataset ---
    class TorchDiskDataset(Dataset):
        def __init__(self, batch_files):
            self.batch_files = batch_files

        def __len__(self):
            return len(self.batch_files)

        def __getitem__(self, idx):
            data = np.load(self.batch_files[idx])
            x = torch.from_numpy(data["images"]).float()
            y = torch.from_numpy(data["labels"]).long()
            return x, y

    def as_torch(self, batch_size=128, num_workers=0, shuffle=True):
        dataset = self.TorchDiskDataset(self.batch_files)
        return DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=num_workers)

    # --- TensorFlow Dataset ---
    def as_tf(self):
        def generator():
            for x, y in self:
                yield x.astype(np.float32), y.astype(np.int32)

        sample = np.load(self.batch_files[0])
        x_shape = sample["images"].shape
        y_shape = sample["labels"].shape

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=tf.float32),
                tf.TensorSpec(shape=y_shape, dtype=tf.int32)
            )
        )
