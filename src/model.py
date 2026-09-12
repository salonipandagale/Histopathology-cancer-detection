"""
src/model.py
------------
Patch-level classifier for histopathology images.

Supported backbones:
    - EfficientNet-B4
    - ResNet-50
    - ResNet-34
    - ResNet-18

The models use ImageNet-pretrained feature extractors followed
by a custom binary classification head.

Binary classification task:
    0 -> Non-tumor
    1 -> Tumor
"""

import logging

import timm
import torch
import torch.nn as nn
from torchvision import models


logger = logging.getLogger(__name__)


# ============================================================
# Classification Head
# ============================================================

class ClassificationHead(nn.Module):
    """
    Custom binary classification head.

    Architecture:
        Features
           ↓
        Dropout
           ↓
        Linear
           ↓
        ReLU
           ↓
        Dropout
           ↓
        Linear
           ↓
        Single logit

    The output is a raw logit. Sigmoid is applied only when
    converting the logit into a probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.4
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),

            nn.Linear(
                in_features,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=dropout / 2
            ),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.

        Returns:
            Tensor of shape (batch_size, 1)
        """

        return self.head(x)


# ============================================================
# Main Histopathology Classifier
# ============================================================

class HistoClassifier(nn.Module):
    """
    Patch-level tumor classifier for PCam.

    Supported backbones:
        efficientnet_b4
        resnet50
        resnet34
        resnet18

    Args:
        backbone:
            Feature extractor architecture.

        pretrained:
            Whether to use ImageNet pretrained weights.

        hidden_dim:
            Number of neurons in the hidden classification layer.

        dropout:
            Dropout probability.

        freeze_bn:
            Whether BatchNorm layers should remain frozen
            during training.

    Output:
        Raw binary classification logits of shape (B, 1).
    """

    SUPPORTED_BACKBONES = [
        "efficientnet_b4",
        "resnet50",
        "resnet34",
        "resnet18"
    ]

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        freeze_bn: bool = False
    ):
        super().__init__()

        # ----------------------------------------------------
        # Validate backbone
        # ----------------------------------------------------

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone

        # IMPORTANT:
        # Store this flag so the overridden train() method
        # can keep BatchNorm layers frozen.
        self._freeze_bn_flag = freeze_bn

        # ----------------------------------------------------
        # EfficientNet-B4
        # ----------------------------------------------------

        if backbone == "efficientnet_b4":

            self.encoder = timm.create_model(
                "efficientnet_b4",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg"
            )

            in_features = self.encoder.num_features

        # ----------------------------------------------------
        # ResNet family
        # ----------------------------------------------------

        else:

            # torchvision pretrained weights
            if pretrained:
                weights = models.get_model_weights(
                    backbone
                ).DEFAULT
            else:
                weights = None

            base_model = getattr(
                models,
                backbone
            )(
                weights=weights
            )

            in_features = base_model.fc.in_features

            # Remove original ImageNet classification layer.
            #
            # ResNet output:
            #     (B, 512, 1, 1) for ResNet-18/34
            #     (B, 2048, 1, 1) for ResNet-50
            #
            # Flatten to:
            #     (B, feature_dimension)
            self.encoder = nn.Sequential(
                *list(base_model.children())[:-1],
                nn.Flatten()
            )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = ClassificationHead(
            in_features=in_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Freeze BatchNorm if requested
        # ----------------------------------------------------

        if self._freeze_bn_flag:
            self._freeze_batchnorm()

        # ----------------------------------------------------
        # Model statistics
        # ----------------------------------------------------

        total_params = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        logger.info(
            f"Model: {backbone} | "
            f"Total params: {total_params / 1e6:.2f}M | "
            f"Trainable: {trainable_params / 1e6:.2f}M"
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Input image tensor of shape:
                (B, 3, H, W)

        Returns:
            Raw logits of shape:
            (B, 1)
        """

        features = self.encoder(x)

        logits = self.classifier(
            features
        )

        return logits

    # ========================================================
    # Probability prediction
    # ========================================================

    def predict_proba(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model logits into tumor probabilities.

        Returns:
            Probability in range [0, 1].
        """

        logits = self.forward(x)

        return torch.sigmoid(logits)

    # ========================================================
    # Feature extraction
    # ========================================================

    def get_features(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract feature embeddings before classification head.

        Useful for:
            - Feature visualization
            - t-SNE / PCA
            - Downstream analysis
            - Interpretability
        """

        return self.encoder(x)

    # ========================================================
    # Freeze encoder
    # ========================================================

    def freeze_encoder(self):
        """
        Freeze the entire feature extractor.

        Only the classification head will be trainable.
        """

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        logger.info(
            "Encoder frozen. "
            "Only classification head will be trained."
        )

    # ========================================================
    # Unfreeze encoder
    # ========================================================

    def unfreeze_encoder(self):
        """
        Unfreeze the entire feature extractor.
        """

        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        logger.info(
            "Encoder unfrozen. "
            "Full model will be trained."
        )

    # ========================================================
    # Freeze BatchNorm
    # ========================================================

    def _freeze_batchnorm(self):
        """
        Freeze all BatchNorm layers.

        This:
            1. Sets BatchNorm layers to evaluation mode.
            2. Prevents their parameters from being updated.
        """

        for module in self.modules():

            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d
                )
            ):

                module.eval()

                for parameter in module.parameters():
                    parameter.requires_grad = False

    # ========================================================
    # Override train()
    # ========================================================

    def train(
        self,
        mode: bool = True
    ):
        """
        Override PyTorch train() so frozen BatchNorm layers
        remain in evaluation mode.
        """

        super().train(mode)

        if self._freeze_bn_flag:
            self._freeze_batchnorm()

        return self


# ============================================================
# Loss Function
# ============================================================

def get_loss_fn(
    pos_weight: float = 1.0,
    device: str = "cpu"
) -> nn.Module:
    """
    Binary Cross Entropy with Logits.

    BCEWithLogitsLoss combines:
        Sigmoid + Binary Cross Entropy

    in a numerically stable implementation.

    Args:
        pos_weight:
            Weight applied to positive examples.

            > 1:
                Penalizes false negatives more strongly.

            = 1:
                No additional weighting.

        device:
            Device on which the loss weight tensor is stored.

    Returns:
        BCEWithLogitsLoss instance.
    """

    pos_weight_tensor = torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=device
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )


# ============================================================
# Optimizer
# ============================================================

def get_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer: str = "adamw"
):
    """
    Create optimizer using differential learning rates.

    Encoder:
        lr * 0.1

    Classification head:
        lr

    This is useful for transfer learning because the
    pretrained encoder usually requires smaller updates than
    the newly initialized classification head.
    """

    # --------------------------------------------------------
    # Classification head parameters
    # --------------------------------------------------------

    head_params = list(
        model.classifier.parameters()
    )

    head_ids = {
        id(parameter)
        for parameter in head_params
    }

    # --------------------------------------------------------
    # Encoder parameters
    # --------------------------------------------------------

    encoder_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_ids
    ]

    # --------------------------------------------------------
    # Differential learning-rate groups
    # --------------------------------------------------------

    parameter_groups = [
        {
            "params": encoder_params,
            "lr": lr * 0.1
        },
        {
            "params": head_params,
            "lr": lr
        }
    ]

    # --------------------------------------------------------
    # Optimizer selection
    # --------------------------------------------------------

    optimizer = optimizer.lower()

    if optimizer == "adamw":

        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=weight_decay
        )

    elif optimizer == "adam":

        return torch.optim.Adam(
            parameter_groups,
            weight_decay=weight_decay
        )

    elif optimizer == "sgd":

        return torch.optim.SGD(
            parameter_groups,
            momentum=0.9,
            weight_decay=weight_decay
        )

    else:

        raise ValueError(
            f"Unknown optimizer: {optimizer}. "
            f"Choose from: adamw, adam, sgd."
        )


# ============================================================
# Learning-rate scheduler
# ============================================================

def get_scheduler(
    optimizer,
    scheduler: str = "cosine",
    epochs: int = 30,
    warmup_epochs: int = 0
):
    """
    Create a learning-rate scheduler.

    Supported:
        cosine
        step
        plateau

    Note:
        Warm-up is currently handled separately by the training
        pipeline. This function only creates the main scheduler.
    """

    scheduler = scheduler.lower()

    # --------------------------------------------------------
    # Cosine annealing
    # --------------------------------------------------------

    if scheduler == "cosine":

        scheduler_epochs = max(
            1,
            epochs - warmup_epochs
        )

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_epochs,
            eta_min=1e-6
        )

    # --------------------------------------------------------
    # Step decay
    # --------------------------------------------------------

    elif scheduler == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

    # --------------------------------------------------------
    # Reduce on validation metric
    # --------------------------------------------------------

    elif scheduler == "plateau":

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3
        )

    else:

        raise ValueError(
            f"Unknown scheduler: {scheduler}. "
            f"Choose from: cosine, step, plateau."
        )


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    val_auc: float,
    path: str
):
    """
    Save a training checkpoint.

    Stores:
        - epoch
        - model weights
        - optimizer state
        - validation ROC-AUC
        - backbone name
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_auc":
            float(val_auc),

        "backbone":
            model.backbone_name
    }

    # Create parent directory if necessary
    checkpoint_dir = os.path.dirname(path)

    if checkpoint_dir:
        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

    torch.save(
        checkpoint,
        path
    )

    logger.info(
        f"Checkpoint saved → {path} "
        f"(epoch={epoch}, AUC={val_auc:.4f})"
    )


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer=None,
    device: str = "cpu"
) -> dict:
    """
    Load model checkpoint.

    Args:
        model:
            Model instance with the same architecture.

        path:
            Checkpoint path.

        optimizer:
            Optional optimizer to restore.

        device:
            Device used for loading.

    Returns:
        Checkpoint dictionary.
    """

    if not torch.jit.is_scripting():

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False
        )

    else:

        checkpoint = torch.load(
            path,
            map_location=device
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint.get(
        "epoch",
        "unknown"
    )

    val_auc = checkpoint.get(
        "val_auc",
        None
    )

    if isinstance(val_auc, (float, int)):
        auc_text = f"{val_auc:.4f}"
    else:
        auc_text = str(val_auc)

    logger.info(
        f"Checkpoint loaded ← {path} "
        f"(epoch={epoch}, AUC={auc_text})"
    )

    return checkpoint


# ============================================================
# Model summary helper
# ============================================================

def count_parameters(
    model: nn.Module
):
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Quick Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing Histopathology Models")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Test both models that will be compared
    # --------------------------------------------------------

    for backbone in [
        "resnet18",
        "efficientnet_b4"
    ]:

        print("\n" + "-" * 60)
        print(f"Testing: {backbone}")
        print("-" * 60)

        try:

            model = HistoClassifier(
                backbone=backbone,
                pretrained=False,
                hidden_dim=128,
                dropout=0.4
            ).to(device)

            # Smaller input keeps the CPU smoke test manageable.
            dummy = torch.randn(
                2,
                3,
                128,
                128,
                device=device
            )

            # Forward pass
            logits = model(dummy)

            # Probability prediction
            probabilities = model.predict_proba(
                dummy
            )

            # Feature extraction
            features = model.get_features(
                dummy
            )

            total_params, trainable_params = (
                count_parameters(model)
            )

            print(
                f"Input shape       : {dummy.shape}"
            )

            print(
                f"Feature shape      : {features.shape}"
            )

            print(
                f"Logit shape        : {logits.shape}"
            )

            print(
                f"Probability shape  : {probabilities.shape}"
            )

            print(
                f"Total parameters   : "
                f"{total_params / 1e6:.2f}M"
            )

            print(
                f"Trainable params   : "
                f"{trainable_params / 1e6:.2f}M"
            )

            print(
                f"Probability range  : "
                f"{probabilities.min().item():.4f} - "
                f"{probabilities.max().item():.4f}"
            )

            print(
                f"✓ {backbone} test passed"
            )

        except Exception as error:

            print(
                f"✗ {backbone} test failed:"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

    print("\n" + "=" * 60)
    print("Model tests completed.")
    print("=" * 60)