from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_TAR = "CIFAR-10-C.tar"
CIFAR10C_DIR = "CIFAR-10-C"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def make_cifar10_loaders(
    root: str | Path,
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    root = Path(root)
    train_set = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=train_transform(),
    )
    test_set = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=eval_transform(),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def download_cifar10c(root: str | Path) -> Path:
    """Download and extract CIFAR-10-C if missing.

    The official Zenodo archive is about 2.9 GB. In Colab, put `root` on
    Google Drive to avoid downloading it every session.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / CIFAR10C_DIR
    if target_dir.exists() and (target_dir / "labels.npy").exists():
        return target_dir

    tar_path = root / CIFAR10C_TAR
    if not tar_path.exists():
        print(f"Downloading CIFAR-10-C from {CIFAR10C_URL}")
        urllib.request.urlretrieve(CIFAR10C_URL, tar_path)

    print(f"Extracting {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(root)
    return target_dir


class CIFAR10CDataset(Dataset):
    """Single corruption/severity view of CIFAR-10-C.

    Official CIFAR-10-C files store all five severities in a single `.npy`:
    rows 0:10000 are severity 1, rows 40000:50000 are severity 5.
    """

    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int,
        transform: Callable | None = None,
    ) -> None:
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in [1, 5]")
        self.root = Path(root)
        self.base_dir = self.root / CIFAR10C_DIR
        self.corruption = corruption
        self.severity = severity
        self.transform = transform or eval_transform()

        image_path = self.base_dir / f"{corruption}.npy"
        label_path = self.base_dir / "labels.npy"
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10-C files under {self.base_dir}. "
                "Run download_cifar10c(root) first."
            )

        start = (severity - 1) * 10_000
        end = severity * 10_000
        images = np.load(image_path, mmap_mode="r")
        labels = np.load(label_path)
        self.images = images[start:end]
        self.labels = labels[start:end]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(np.asarray(self.images[idx]).astype(np.uint8))
        label = int(self.labels[idx])
        return self.transform(image), label


def make_cifar10c_loader(
    root: str | Path,
    corruption: str,
    severity: int,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    dataset = CIFAR10CDataset(root, corruption, severity, transform=eval_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
