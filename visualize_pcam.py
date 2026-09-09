import gzip
import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = r"data\pcamv1"

IMAGE_FILE = os.path.join(
    DATA_DIR,
    "camelyonpatch_level_2_split_valid_x.h5.gz"
)

LABEL_FILE = os.path.join(
    DATA_DIR,
    "camelyonpatch_level_2_split_valid_y.h5.gz"
)


def load_h5_gz(path, key):
    with gzip.open(path, "rb") as gz:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(gz.read())
            temp_path = tmp.name

    try:
        with h5py.File(temp_path, "r") as f:
            return f[key][:]
    finally:
        os.remove(temp_path)


print("Loading PCam validation images...")
images = load_h5_gz(IMAGE_FILE, "x")

print("Loading PCam labels...")
labels = load_h5_gz(LABEL_FILE, "y").reshape(-1)

print("Images:", images.shape)
print("Labels:", labels.shape)
print("Class distribution:")
print("  Non-cancer (0):", np.sum(labels == 0))
print("  Cancer (1):   ", np.sum(labels == 1))


# Find examples from both classes
negative_indices = np.where(labels == 0)[0][:4]
positive_indices = np.where(labels == 1)[0][:4]

indices = np.concatenate([negative_indices, positive_indices])


plt.figure(figsize=(12, 6))

for i, idx in enumerate(indices):
    plt.subplot(2, 4, i + 1)

    plt.imshow(images[idx])

    label = "Cancer" if labels[idx] == 1 else "Non-cancer"
    plt.title(label)

    plt.axis("off")

plt.tight_layout()

output_path = "pcam_samples.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nSaved visualization to: {output_path}")