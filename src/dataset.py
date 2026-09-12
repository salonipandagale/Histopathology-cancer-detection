"""
src/dataset.py
--------------
PCam dataset loading, preprocessing, augmentation, and DataLoaders.

Development mode:
    Uses a small subset of PCam for CPU experimentation.

Final mode:
    Supports the official PCam train/validation/test split files.
"""

import os

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# PCam file names
# ============================================================

TRAIN_IMAGE_FILE = "camelyonpatch_level_2_split_train_x.h5"
TRAIN_LABEL_FILE = "camelyonpatch_level_2_split_train_y.h5"

VALID_IMAGE_FILE = "camelyonpatch_level_2_split_valid_x.h5"
VALID_LABEL_FILE = "camelyonpatch_level_2_split_valid_y.h5"

TEST_IMAGE_FILE = "camelyonpatch_level_2_split_test_x.h5"
TEST_LABEL_FILE = "camelyonpatch_level_2_split_test_y.h5"


# ============================================================
# Image transformations
# ============================================================

def get_train_transform(img_size=224):
    """
    Training augmentation pipeline.

    Histopathology images can contain variation in orientation,
    scale, brightness, contrast, and color. These augmentations
    help the model learn robust visual features.
    """

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
    """
    Validation/test preprocessing.

    No random augmentation is applied during validation or testing.
    """

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
    """
    PyTorch Dataset for PCam HDF5 files.

    Images are loaded lazily from the HDF5 file so that the
    complete dataset does not need to fit into RAM.
    """

    def __init__(
        self,
        image_path,
        label_path,
        indices=None,
        transform=None
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found:\n{image_path}"
            )

        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Label file not found:\n{label_path}"
            )

        # ----------------------------------------------------
        # Read labels once
        # ----------------------------------------------------

        with h5py.File(label_path, "r") as f:
            self.labels = np.asarray(f["y"]).reshape(-1)

        # ----------------------------------------------------
        # If indices are not supplied, use the entire dataset
        # ----------------------------------------------------

        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = np.asarray(indices)

        # HDF5 file is opened lazily inside each worker/process
        self._h5 = None

    # --------------------------------------------------------
    # Open HDF5 file lazily
    # --------------------------------------------------------

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.image_path, "r")

        return self._h5

    # --------------------------------------------------------
    # Dataset length
    # --------------------------------------------------------

    def __len__(self):
        return len(self.indices)

    # --------------------------------------------------------
    # Get one sample
    # --------------------------------------------------------

    def __getitem__(self, idx):

        # Original PCam index
        real_idx = int(self.indices[idx])

        h5_file = self._open_h5()

        # Read image using original index
        image = h5_file["x"][real_idx]

        # Read corresponding label using SAME original index
        label = float(self.labels[real_idx])

        # Apply transformations
        if self.transform is not None:

            transformed = self.transform(
                image=image
            )

            image = transformed["image"]

        return image, torch.tensor(
            label,
            dtype=torch.float32
        )


# ============================================================
# Helper: load labels
# ============================================================

def load_labels(label_path):
    """
    Load labels from a PCam HDF5 label file.
    """

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Label file not found:\n{label_path}"
        )

    with h5py.File(label_path, "r") as f:
        labels = np.asarray(f["y"]).reshape(-1)

    return labels


# ============================================================
# Helper: create weighted sampler
# ============================================================

def create_weighted_sampler(labels):
    """
    Create a WeightedRandomSampler to reduce the effect of
    class imbalance during training.
    """

    labels = np.asarray(labels).astype(int)

    class_counts = np.bincount(labels)

    # Prevent division by zero
    class_counts = np.maximum(class_counts, 1)

    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


# ============================================================
# Development DataLoaders
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
    subset_size=2000,
    use_sampler=True,
    **kwargs
):
    """
    Create train, validation, and test DataLoaders.

    CURRENT DEVELOPMENT MODE
    -------------------------
    Uses the PCam validation split and selects a small subset
    for CPU experimentation.

    IMPORTANT:
    The selected indices always remain the ORIGINAL PCam
    indices, so images and labels stay correctly aligned.

    Later, the final pipeline will use the official PCam
    train/validation/test files directly.
    """

    # ========================================================
    # Paths
    # ========================================================

    image_path = os.path.join(
        data_dir,
        VALID_IMAGE_FILE
    )

    label_path = os.path.join(
        data_dir,
        VALID_LABEL_FILE
    )

    # ========================================================
    # Load labels
    # ========================================================

    print("=" * 60)
    print("Loading PCam dataset")
    print("=" * 60)

    labels = load_labels(label_path)

    all_indices = np.arange(len(labels))

    print(f"Available images: {len(all_indices)}")

    # ========================================================
    # Select subset for CPU development
    # ========================================================

    if subset_size is not None:

        subset_size = min(
            subset_size,
            len(all_indices)
        )

        rng = np.random.default_rng(seed)

        # IMPORTANT:
        # Keep these as ORIGINAL PCam indices.
        selected_indices = rng.choice(
            all_indices,
            size=subset_size,
            replace=False
        )

    else:

        selected_indices = all_indices

    # Labels corresponding to the selected ORIGINAL indices
    selected_labels = labels[selected_indices]

    print(
        f"Using subset: {len(selected_indices)} images"
    )

    print(
        f"Class 0: {np.sum(selected_labels == 0)}"
    )

    print(
        f"Class 1: {np.sum(selected_labels == 1)}"
    )

    # ========================================================
    # Split selected samples
    # ========================================================

    # Positions inside selected_indices
    subset_positions = np.arange(
        len(selected_indices)
    )

    # --------------------------------------------------------
    # Train vs temporary set
    # --------------------------------------------------------

    train_pos, temp_pos = train_test_split(
        subset_positions,
        test_size=(val_split + test_split),
        stratify=selected_labels,
        random_state=seed
    )

    # Labels for temporary set
    temp_labels = selected_labels[temp_pos]

    # --------------------------------------------------------
    # Validation vs test
    # --------------------------------------------------------

    relative_test_size = (
        test_split /
        (val_split + test_split)
    )

    val_pos, test_pos = train_test_split(
        temp_pos,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=seed
    )

    # ========================================================
    # Convert positions back to ORIGINAL PCam indices
    # ========================================================

    train_idx = selected_indices[train_pos]
    val_idx = selected_indices[val_pos]
    test_idx = selected_indices[test_pos]

    # ========================================================
    # Display split information
    # ========================================================

    print("\nDataset split:")

    print(
        f"Train      : {len(train_idx)}"
    )

    print(
        f"Validation : {len(val_idx)}"
    )

    print(
        f"Test       : {len(test_idx)}"
    )

    # ========================================================
    # Dataset objects
    # ========================================================

    train_ds = PCamDataset(
        image_path=image_path,
        label_path=label_path,
        indices=train_idx,
        transform=get_train_transform(img_size)
    )

    val_ds = PCamDataset(
        image_path=image_path,
        label_path=label_path,
        indices=val_idx,
        transform=get_val_transform(img_size)
    )

    test_ds = PCamDataset(
        image_path=image_path,
        label_path=label_path,
        indices=test_idx,
        transform=get_val_transform(img_size)
    )

    # ========================================================
    # Weighted sampler
    # ========================================================

    if use_sampler:

        train_labels = labels[train_idx]

        sampler = create_weighted_sampler(
            train_labels
        )

    else:

        sampler = None

    # ========================================================
    # DataLoaders
    # ========================================================

    pin_memory = torch.cuda.is_available()

    # --------------------------------------------------------
    # Training loader
    # --------------------------------------------------------

    if sampler is not None:

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    else:

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # --------------------------------------------------------
    # Validation loader
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # --------------------------------------------------------
    # Test loader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# Official PCam DataLoaders
# ============================================================

def get_official_dataloaders(
    data_dir,
    img_size=224,
    batch_size=8,
    num_workers=0,
    use_sampler=True
):
    """
    Create DataLoaders using the official PCam
    train / validation / test splits.

    This function will be used for the final experiment.
    """

    # ========================================================
    # File paths
    # ========================================================

    train_image_path = os.path.join(
        data_dir,
        TRAIN_IMAGE_FILE
    )

    train_label_path = os.path.join(
        data_dir,
        TRAIN_LABEL_FILE
    )

    valid_image_path = os.path.join(
        data_dir,
        VALID_IMAGE_FILE
    )

    valid_label_path = os.path.join(
        data_dir,
        VALID_LABEL_FILE
    )

    test_image_path = os.path.join(
        data_dir,
        TEST_IMAGE_FILE
    )

    test_label_path = os.path.join(
        data_dir,
        TEST_LABEL_FILE
    )

    # ========================================================
    # Load labels
    # ========================================================

    train_labels = load_labels(train_label_path)
    valid_labels = load_labels(valid_label_path)
    test_labels = load_labels(test_label_path)

    # ========================================================
    # Create datasets
    # ========================================================

    train_ds = PCamDataset(
        image_path=train_image_path,
        label_path=train_label_path,
        transform=get_train_transform(img_size)
    )

    val_ds = PCamDataset(
        image_path=valid_image_path,
        label_path=valid_label_path,
        transform=get_val_transform(img_size)
    )

    test_ds = PCamDataset(
        image_path=test_image_path,
        label_path=test_label_path,
        transform=get_val_transform(img_size)
    )

    # ========================================================
    # Training sampler
    # ========================================================

    if use_sampler:

        sampler = create_weighted_sampler(
            train_labels
        )

    else:

        sampler = None

    pin_memory = torch.cuda.is_available()

    # ========================================================
    # DataLoaders
    # ========================================================

    if sampler is not None:

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    else:

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # ========================================================
    # Print dataset information
    # ========================================================

    print("=" * 60)
    print("Official PCam dataset")
    print("=" * 60)

    print(f"Train      : {len(train_ds)}")
    print(f"Validation : {len(val_ds)}")
    print(f"Test       : {len(test_ds)}")

    print(
        f"Train class 0: {np.sum(train_labels == 0)}"
    )

    print(
        f"Train class 1: {np.sum(train_labels == 1)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# Quick dataset test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing PCam Dataset")
    print("=" * 60)

    data_dir = "./data/pcamv1"

    label_path = os.path.join(
        data_dir,
        VALID_LABEL_FILE
    )

    image_path = os.path.join(
        data_dir,
        VALID_IMAGE_FILE
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(image_path):
        print(
            f"\nImage file not found:\n{image_path}"
        )
        print(
            "\nPCam dataset has not been downloaded yet."
        )
        raise SystemExit

    if not os.path.exists(label_path):
        print(
            f"\nLabel file not found:\n{label_path}"
        )
        print(
            "\nPCam dataset has not been downloaded yet."
        )
        raise SystemExit

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels = load_labels(label_path)

    print(
        f"\nTotal images: {len(labels)}"
    )

    print(
        f"Class 0: {np.sum(labels == 0)}"
    )

    print(
        f"Class 1: {np.sum(labels == 1)}"
    )

    # --------------------------------------------------------
    # Test first few samples
    # --------------------------------------------------------

    dataset = PCamDataset(
        image_path=image_path,
        label_path=label_path,
        indices=np.arange(
            min(10, len(labels))
        ),
        transform=get_train_transform(128)
    )

    image, label = dataset[0]

    print("\nSample test:")
    print(
        f"Image shape : {image.shape}"
    )

    print(
        f"Image dtype : {image.dtype}"
    )

    print(
        f"Label       : {label.item()}"
    )

    print("\nDataset test passed successfully.")