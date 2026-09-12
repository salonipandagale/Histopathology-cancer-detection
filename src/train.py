import os
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

from dataset import get_official_dataloaders
from model import HistoClassifier


DATA_DIR = "./data/pcamv1"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 3

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

MODEL_NAME = "resnet18"
PRETRAINED = True
DROPOUT = 0.4
HIDDEN_DIM = 128

CHECKPOINT_DIR = "./outputs/checkpoints"
RESULTS_DIR = "./outputs/results"
LOG_DIR = "./outputs/logs"

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "resnet18_pcam_best.pth"
)

FINAL_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "resnet18_pcam_final.pth"
)

HISTORY_FILE = os.path.join(
    RESULTS_DIR, "resnet18_training_history.csv"
)

LOG_FILE = os.path.join(
    LOG_DIR, "resnet18_training.log"
)

SEED = 42
GRAD_CLIP = 1.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )


def calculate_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, zero_division=0
    )
    recall = recall_score(
        labels, predictions, zero_division=0
    )
    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            labels, probabilities
        )
    except ValueError:
        roc_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        probabilities = (
            torch.sigmoid(logits)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        batch_labels = (
            labels
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_probabilities.extend(
            probabilities
        )
        all_labels.extend(batch_labels)

        total_loss += (
            loss.item() * images.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        total_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    return epoch_loss, metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for images, labels in progress:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        ).float().view(-1, 1)

        logits = model(images)
        loss = criterion(logits, labels)

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .reshape(-1)
        )

        batch_labels = (
            labels
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_probabilities.extend(
            probabilities
        )
        all_labels.extend(batch_labels)

        total_loss += (
            loss.item() * images.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        total_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    return epoch_loss, metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "model_name": MODEL_NAME,
            "img_size": IMG_SIZE,
        },
        path,
    )


def log_metrics(title, loss, metrics):
    logging.info(title)
    logging.info(f"Loss        : {loss:.4f}")
    logging.info(
        f"Accuracy    : {metrics['accuracy']:.4f}"
    )
    logging.info(
        f"Precision   : {metrics['precision']:.4f}"
    )
    logging.info(
        f"Recall      : {metrics['recall']:.4f}"
    )
    logging.info(
        f"F1          : {metrics['f1']:.4f}"
    )
    logging.info(
        f"ROC-AUC     : {metrics['roc_auc']:.4f}"
    )
    logging.info(
        f"Sensitivity : {metrics['sensitivity']:.4f}"
    )
    logging.info(
        f"Specificity : {metrics['specificity']:.4f}"
    )


def main():
    set_seed(SEED)
    setup_logging()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.info("=" * 70)
    logging.info("PCam Histopathology Cancer Classification")
    logging.info("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logging.info(f"Device: {device}")

    if device.type == "cuda":
        logging.info(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        logging.info(
            "CUDA not available. Training on CPU."
        )

    logging.info("")
    logging.info("Loading official PCam dataset...")

    train_loader, val_loader, test_loader = (
        get_official_dataloaders(
            data_dir=DATA_DIR,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            use_sampler=False,
        )
    )

    logging.info(
        f"Train samples      : {len(train_loader.dataset)}"
    )
    logging.info(
        f"Validation samples : {len(val_loader.dataset)}"
    )
    logging.info(
        f"Test samples       : {len(test_loader.dataset)}"
    )

    logging.info("")
    logging.info(f"Creating model: {MODEL_NAME}")

    model = HistoClassifier(
        backbone=MODEL_NAME,
        pretrained=PRETRAINED,
        dropout=DROPOUT,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    logging.info(
        f"Total parameters    : "
        f"{total_params / 1e6:.2f}M"
    )
    logging.info(
        f"Trainable parameters: "
        f"{trainable_params / 1e6:.2f}M"
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_val_auc = -float("inf")
    history = []

    logging.info("")
    logging.info("=" * 70)
    logging.info("Starting training")
    logging.info("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        logging.info("")
        logging.info(
            f"Epoch {epoch}/{EPOCHS}"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Learning rate: {current_lr:.6e}"
        )

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        logging.info("")
        log_metrics(
            "Training metrics:",
            train_loss,
            train_metrics,
        )

        logging.info("")
        log_metrics(
            "Validation metrics:",
            val_loss,
            val_metrics,
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
            }
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
            )

            logging.info("")
            logging.info(
                f"Best checkpoint saved "
                f"(Val ROC-AUC: {best_val_auc:.4f})"
            )

    save_checkpoint(
        FINAL_CHECKPOINT,
        model,
        optimizer,
        scheduler,
        EPOCHS,
        val_metrics,
    )

    pd.DataFrame(history).to_csv(
        HISTORY_FILE,
        index=False,
    )

    logging.info("")
    logging.info("=" * 70)
    logging.info("Training completed")
    logging.info("=" * 70)

    logging.info(
        f"Best validation ROC-AUC: "
        f"{best_val_auc:.4f}"
    )
    logging.info(
        f"Best checkpoint: {BEST_CHECKPOINT}"
    )
    logging.info(
        f"Final checkpoint: {FINAL_CHECKPOINT}"
    )
    logging.info(
        f"Training history: {HISTORY_FILE}"
    )
    logging.info("")
    logging.info(
        "Test set was not used during training."
    )


if __name__ == "__main__":
    main()