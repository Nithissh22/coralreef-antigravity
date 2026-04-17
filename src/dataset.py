"""
dataset.py — PyTorch Dataset & DataLoader helpers for coral reef images.

Provides:
    CoralReefDataset        ImageFolder-style dataset with optional augmentation
    get_transforms()        Train / val / test transform pipelines
    split_dataset()         Stratified train / val / test split
    get_dataloaders()       One-call helper returning three DataLoaders
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.utils import load_config, set_seed, setup_logger

logger = setup_logger("dataset")


# ================================================================
#  Transforms
# ================================================================

def get_transforms(
    image_size: int = 224,
    augmentation_cfg: dict | None = None,
    mode: Literal["train", "val", "test"] = "train",
) -> transforms.Compose:
    """Return a ``torchvision.transforms.Compose`` pipeline.

    Training mode adds random augmentations; val / test only do resize +
    normalize.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225],
    )

    if mode == "train" and augmentation_cfg and augmentation_cfg.get("enabled", False):
        aug_transforms = [transforms.Resize((image_size, image_size))]

        if augmentation_cfg.get("horizontal_flip", False):
            aug_transforms.append(transforms.RandomHorizontalFlip())
        if augmentation_cfg.get("vertical_flip", False):
            aug_transforms.append(transforms.RandomVerticalFlip())

        rotation = augmentation_cfg.get("random_rotation", 0)
        if rotation:
            aug_transforms.append(transforms.RandomRotation(rotation))

        cj = augmentation_cfg.get("color_jitter", {})
        if cj:
            aug_transforms.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0),
                    contrast=cj.get("contrast", 0),
                    saturation=cj.get("saturation", 0),
                    hue=cj.get("hue", 0),
                )
            )

        aug_transforms += [transforms.ToTensor(), normalize]
        return transforms.Compose(aug_transforms)

    # Val / Test
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


# ================================================================
#  Dataset
# ================================================================

class CoralReefDataset(Dataset):
    """ImageFolder-style dataset that reads images from class sub-directories.

    Expected layout::
        root/
            healthy/
                img001.jpg
                ...
            bleached/
                img100.jpg
                ...
            diseased/
                img200.jpg
                ...
    """

    def __init__(
        self,
        root: str | Path,
        class_names: list[str] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        if class_names is None:
            class_names = sorted(
                d.name for d in self.root.iterdir() if d.is_dir()
            )
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                logger.warning("Class directory not found: %s", cls_dir)
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        logger.info(
            "Loaded %d images across %d classes from %s",
            len(self.samples), len(self.class_names), self.root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ================================================================
#  Splitting
# ================================================================

def split_dataset(
    dataset: CoralReefDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Stratified train / val / test split.

    ``test_ratio = 1 - train_ratio - val_ratio``.
    """
    labels = [label for _, label in dataset.samples]
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=labels, random_state=seed,
    )
    temp_labels = [labels[i] for i in temp_idx]
    relative_val = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=relative_val, stratify=temp_labels, random_state=seed,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


# ================================================================
#  DataLoader factory
# ================================================================

def get_dataloaders(
    cfg: dict | None = None,
    data_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """High-level helper: returns ``(train_loader, val_loader, test_loader)``."""
    if cfg is None:
        cfg = load_config()

    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})
    image_size = ds_cfg.get("image_size", 224)
    batch_size = cfg["training"].get("batch_size", 32)
    seed = ds_cfg.get("random_seed", 42)
    set_seed(seed)

    root = Path(data_dir) if data_dir else Path(cfg["paths"]["processed_data"])
    class_names = ds_cfg.get("classes")

    # Build dataset with training transforms (split handles train/val/test later)
    full_dataset = CoralReefDataset(root, class_names=class_names, transform=None)

    train_sub, val_sub, test_sub = split_dataset(
        full_dataset,
        train_ratio=ds_cfg["split_ratios"]["train"],
        val_ratio=ds_cfg["split_ratios"]["val"],
        seed=seed,
    )

    # Assign transforms per split
    train_tf = get_transforms(image_size, aug_cfg, mode="train")
    eval_tf = get_transforms(image_size, mode="val")

    train_sub.dataset = _TransformWrapper(full_dataset, train_tf)
    val_sub.dataset = _TransformWrapper(full_dataset, eval_tf)
    test_sub.dataset = _TransformWrapper(full_dataset, eval_tf)

    num_workers = 0

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ================================================================
#  Internal helpers
# ================================================================

class _TransformWrapper(Dataset):
    """Wraps an existing dataset to override its transform at split level."""

    def __init__(self, base_dataset: CoralReefDataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_path, label = self.base.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
