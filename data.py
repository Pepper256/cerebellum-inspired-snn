"""MNIST loading helpers for this implementation.

Hard requirement: MNIST only, no other dataset. Data lives in <repo>/data/MNIST,
pre-downloaded by <repo>/download_mnist.py, so this module can run offline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def _find_data_root() -> Path:
    """Locate the MNIST directory by walking UP from this file.

    A fixed `parents[1]` breaks as soon as this module is copied into a
    subdirectory (e.g. a frozen baseline snapshot): it then points at
    <impl>/data and silently re-downloads a whole second copy of MNIST.  Search
    upward for an existing MNIST directory instead, and only fall back to
    <this dir>/data if none is found.

    NOTE (archive copy): the fallback is `here.parent / "data"` rather than
    `here.parents[1] / "data"`.  In a fresh standalone clone this file sits at
    the repository root, so `parents[1]` would point OUTSIDE the repo and
    download MNIST into the parent directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "MNIST").exists():
            return parent / "data"
    return here.parent / "data"


DATA_ROOT = _find_data_root()
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _transform(normalize: bool):
    ops = [transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    return transforms.Compose(ops)


def get_loaders(
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 0,
    seed: int = 0,
    normalize: bool = False,
    train_limit: int | None = None,
) -> dict[str, DataLoader | None]:
    """Return {'train','val','test'} DataLoaders over MNIST.

    The 60k training split is divided into (train, val) so the reported accuracy
    is never tuned on the test set.
    """
    tf = _transform(normalize)
    train = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tf)

    val_set = None
    if val_size and val_size > 0:
        gen = torch.Generator().manual_seed(seed)
        n_val = min(val_size, len(train))
        train, val_set = random_split(train, [len(train) - n_val, n_val], generator=gen)

    if train_limit is not None:
        train = Subset(train, range(min(train_limit, len(train))))

    def loader(ds, shuffle: bool, bs: int = batch_size):
        if ds is None:
            return None
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return {
        "train": loader(train, True),
        "val": loader(val_set, False),
        "test": loader(test, False),
    }


def num_classes() -> int:
    return 10
