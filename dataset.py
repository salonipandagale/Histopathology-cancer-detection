import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Paths
# ============================================================

IMAGE_FILE = "camelyonpatch_level_2_split_valid_x.h5"
LABEL_FILE = "camelyonpatch_level_2_split_valid_y.h5"


# ============================================================
# Transformations
# ============================================================

def get_train_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_val_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


# ============================================================
# PCam Dataset
# ============================================================

class PCamDataset(Dataset):

    def __init__(
        self,
        data_dir,
        indices,
        transform=None
    ):
        self.data_dir = data_dir
        self.indices = np.asarray(indices)
        self.transform = transform

        image_path = os.path.join(data_dir, IMAGE_FILE)
        label_path = os.path.join(data_dir, LABEL_FILE)

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found:\n{image_path}"
            )

        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Label file not found:\n{label_path}"
            )

        # Read labels once because they are very small
        with h5py.File(label_path, "r") as f:
            self.labels = np.asarray(f["y"]).reshape(-1)

        self.image_path = image_path
        self._h5 = None

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.image_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = int(self.indices[idx])

        h5_file = self._open_h5()

        image = h5_file["x"][real_idx]
        label = float(self.labels[real_idx])

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, torch.tensor(label, dtype=torch.float32)


# ============================================================
# DataLoaders
# ============================================================

def get_dataloaders(
    data_dir,
    img_size=224,
    batch_size=8,
    num_workers=0,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    **kwargs
):

    print("Creating training dataset...")

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    label_path = os.path.join(data_dir, LABEL_FILE)

    with h5py.File(label_path, "r") as f:
        labels = np.asarray(f["y"]).reshape(-1)

    indices = np.arange(len(labels))

    # Small subset for CPU development/testing
    rng = np.random.default_rng(seed)

    subset_size = min(2000, len(indices))

    indices = rng.choice(
        indices,
        size=subset_size,
        replace=False
    )

    # Keep labels aligned with the selected images
    labels = labels[indices]

    # Re-index selected samples from 0 to subset_size-1
    indices = np.arange(len(labels))

    print(f"Using subset: {len(indices)} images")
    print(f"Class 0: {np.sum(labels == 0)}")
    print(f"Class 1: {np.sum(labels == 1)}")

    # --------------------------------------------------------
    # Train / temp split
    # --------------------------------------------------------

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=seed
    )

    # --------------------------------------------------------
    # Validation / test split
    # --------------------------------------------------------

    temp_labels = labels[temp_idx]

    relative_test_size = test_split / (val_split + test_split)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=seed
    )

    print("\nDataset split:")
    print(f"Train      : {len(train_idx)}")
    print(f"Validation : {len(val_idx)}")
    print(f"Test       : {len(test_idx)}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_ds = PCamDataset(
        data_dir=data_dir,
        indices=train_idx,
        transform=get_train_transform(img_size)
    )

    val_ds = PCamDataset(
        data_dir=data_dir,
        indices=val_idx,
        transform=get_val_transform(img_size)
    )

    test_ds = PCamDataset(
        data_dir=data_dir,
        indices=test_idx,
        transform=get_val_transform(img_size)
    )

    # --------------------------------------------------------
    # Weighted sampler
    # --------------------------------------------------------

    train_labels = labels[train_idx]

    class_counts = np.bincount(train_labels.astype(int))

    class_weights = 1.0 / class_counts

    sample_weights = class_weights[
        train_labels.astype(int)
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Dataset test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing PCam dataset")
    print("=" * 60)

    data_dir = "./data/pcamv1"

    with h5py.File(
        os.path.join(data_dir, LABEL_FILE), "r"
    ) as f:
        labels = np.asarray(f["y"]).reshape(-1)

    indices = np.arange(len(labels))

    dataset = PCamDataset(
        data_dir=data_dir,
        indices=indices[:10],
        transform=get_train_transform(224)
    )

    image, label = dataset[0]

    print("Image shape :", image.shape)
    print("Image dtype :", image.dtype)
    print("Label       :", label.item())

    print("\nDataset test passed.")