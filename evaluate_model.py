import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

from dataset import get_dataloaders
from model import HistoClassifier


# ============================================================
# Configuration
# ============================================================

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Histopathology Model Evaluation")
print("=" * 60)
print(f"Device: {device}")


# ============================================================
# Data
# ============================================================

print("\nLoading test data...")

_, _, test_loader = get_dataloaders(
    data_dir=config["data"]["data_dir"],
    img_size=config["data"]["img_size"],
    batch_size=config["training"]["batch_size"],
    num_workers=config["data"]["num_workers"],
    use_sampler=False
)


# ============================================================
# Model
# ============================================================

backbone = config["model"]["backbone"]

model = HistoClassifier(
    backbone=backbone,
    pretrained=False,
    dropout=config["model"]["dropout"],
    hidden_dim=config["model"]["hidden_dim"]
)

checkpoint_path = (
    "./outputs/checkpoints/resnet18_pcam_best.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")


# ============================================================
# Predictions
# ============================================================

all_labels = []
all_probabilities = []

print("\nRunning inference...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        logits = model(images).view(-1)

        probabilities = torch.sigmoid(logits)

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy().reshape(-1)
        )


y_true = np.array(all_labels).astype(int)
y_prob = np.array(all_probabilities)

y_pred = (y_prob >= 0.5).astype(int)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# ============================================================
# Print Results
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Sensitivity : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Cancer"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Save Results
# ============================================================

results_dir = config["paths"]["results_dir"]

os.makedirs(
    results_dir,
    exist_ok=True
)


# ============================================================
# Confusion Matrix Plot
# ============================================================

plt.figure(figsize=(6, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Cancer"]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Histopathology Cancer Classification")
plt.tight_layout()

cm_path = os.path.join(
    results_dir,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ============================================================
# ROC Curve
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve — Histopathology Cancer Classification"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    results_dir,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()


print("\nSaved:")
print(f"Confusion Matrix → {cm_path}")
print(f"ROC Curve        → {roc_path}")

print("\nEvaluation completed successfully.")