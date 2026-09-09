"""
train.py
--------
Training pipeline for histopathology patch classification.
"""

import os
import random

import numpy as np
import torch
import yaml

from dataset import get_dataloaders
from model import HistoClassifier, get_loss_fn, get_optimizer


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.float().to(device).view(-1)

        optimizer.zero_grad()

        logits = model(images).view(-1)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.float().to(device).view(-1)

        logits = model(images).view(-1)

        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    seed = config["training"]["seed"]

    set_seed(seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Histopathology Cancer Classification")
    print("=" * 60)
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------

    data_dir = config["data"]["data_dir"]
    img_size = config["data"]["img_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    print()
    print("Dataset configuration:")
    print(f"Data directory : {data_dir}")
    print(f"Image size     : {img_size}")
    print(f"Batch size     : {batch_size}")
    print(f"Workers        : {num_workers}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print()
    print("Creating DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_sampler=True
    )

    print()
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    backbone = config["model"]["backbone"]
    pretrained = config["model"]["pretrained"]
    dropout = config["model"]["dropout"]
    hidden_dim = config["model"]["hidden_dim"]

    print()
    print(f"Creating model: {backbone}")

    model = HistoClassifier(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        hidden_dim=hidden_dim
    )

    model = model.to(device)

    print("Model created successfully.")

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    pos_weight = config["training"]["pos_weight"]

    criterion = get_loss_fn(
        pos_weight=pos_weight,
        device=str(device)
    )

    print("Loss function: BCEWithLogitsLoss")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    optimizer_name = config["training"]["optimizer"]

    optimizer = get_optimizer(
        model=model,
        lr=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer_name
    )

    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {learning_rate}")

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    epochs = config["training"]["epochs"]

    print()
    print("=" * 60)
    print(f"Starting training for {epochs} epoch(s)")
    print("=" * 60)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(epochs):

        print()
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 60)

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_accuracy = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_accuracy:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_accuracy:.4f}")

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_dir = config["paths"]["checkpoint_dir"]

            os.makedirs(
                checkpoint_dir,
                exist_ok=True
            )

            checkpoint_path = os.path.join(
                checkpoint_dir,
                "resnet18_pcam_best.pth"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "img_size": img_size,
                    "hidden_dim": hidden_dim,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch + 1
                },
                checkpoint_path
            )

            print()
            print(
                f"Best model saved to: {checkpoint_path}"
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Training completed successfully.")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()