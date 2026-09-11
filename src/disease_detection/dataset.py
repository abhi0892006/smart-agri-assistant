"""
Dataset loading and splitting for the Disease Detection module.

Loads images from data/disease_images/<ClassName>/*.jpg using
torchvision's ImageFolder convention, then splits into stratified
train/val/test sets (same pattern as the tabular modules, adapted for
images).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


@dataclass
class DiseaseDataSplits:
    train_dataset: Subset
    val_dataset: Subset
    test_dataset: Subset
    class_names: list[str]


def _build_transforms(train: bool) -> transforms.Compose:
    size = config.DISEASE_IMAGE_SIZE
    if train:
        # Light augmentation only (small dataset — aggressive augmentation
        # risks distorting disease-specific visual cues like leaf spots).
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_datasets(data_dir=None):
    """Load the full image folder twice (once per transform pipeline) so
    train gets augmentation and val/test don't, while sharing the same
    underlying file list and stratified split indices.
    """
    data_dir = data_dir or config.DISEASE_IMAGES_DIR
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(
            f"No images found under {data_dir}. Expected subfolders per "
            f"class, e.g. {data_dir}/Apple___healthy/*.jpg. "
            "See data/disease_images/README.md."
        )

    train_tf = _build_transforms(train=True)
    eval_tf = _build_transforms(train=False)

    full_train = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    full_eval = datasets.ImageFolder(root=str(data_dir), transform=eval_tf)

    class_names = full_train.classes
    missing = set(config.DISEASE_CLASSES) - set(class_names)
    if missing:
        raise ValueError(
            f"Expected classes not found in {data_dir}: {missing}. "
            f"Found: {class_names}"
        )

    return full_train, full_eval, class_names


def prepare_splits(cfg: dict | None = None) -> DiseaseDataSplits:
    cfg = cfg or config.DISEASE_TRAIN_CONFIG
    full_train, full_eval, class_names = load_datasets()

    targets = np.array(full_train.targets)
    indices = np.arange(len(targets))

    test_size = cfg["test_split"]
    val_size = cfg["val_split"]

    idx_train_val, idx_test = train_test_split(
        indices, test_size=test_size, random_state=cfg["random_seed"], stratify=targets
    )
    relative_val_size = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val_size,
        random_state=cfg["random_seed"],
        stratify=targets[idx_train_val],
    )

    train_dataset = Subset(full_train, idx_train)
    val_dataset = Subset(full_eval, idx_val)   # no augmentation for val/test
    test_dataset = Subset(full_eval, idx_test)

    return DiseaseDataSplits(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        class_names=class_names,
    )
