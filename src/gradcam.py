"""
gradcam.py
----------
Grad-CAM explainability for histopathology patch cancer classification.

Designed for the current HistoClassifier implementation used in this project.
Supports ResNet-style Sequential backbones and automatically finds the
last suitable convolutional layer.

Output:
    Original Patch | Grad-CAM Heatmap | Grad-CAM Overlay
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer=None):

        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # ----------------------------------------------------
        # Automatically find a suitable final Conv2d layer
        # ----------------------------------------------------
        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer

        print(
            "Grad-CAM target layer:",
            self._get_module_name(target_layer)
        )

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    # --------------------------------------------------------
    # Find last convolutional layer
    # --------------------------------------------------------

    def _find_target_layer(self):

        last_conv = None
        last_name = None

        for name, module in self.model.named_modules():

            if isinstance(module, nn.Conv2d):

                last_conv = module
                last_name = name

        if last_conv is None:
            raise RuntimeError(
                "Could not find a Conv2d layer in the model."
            )

        print(
            f"Automatically selected Grad-CAM layer: {last_name}"
        )

        return last_conv

    # --------------------------------------------------------
    # Get module name
    # --------------------------------------------------------

    def _get_module_name(self, target):

        for name, module in self.model.named_modules():

            if module is target:
                return name

        return "Unknown"

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def _save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # ========================================================
    # Generate Grad-CAM
    # ========================================================

    def generate(self, image_tensor):

        self.model.zero_grad()

        # Make sure gradients are enabled
        with torch.enable_grad():

            output = self.model(image_tensor)

            # Binary classification:
            # model outputs one logit
            score = output[0, 0]

            score.backward()

        # Check hooks
        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured. "
                "Check the selected target layer."
            )

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured. "
                "Check the selected target layer."
            )

        gradients = self.gradients
        activations = self.activations

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep only positive influence
        cam = F.relu(cam)

        # Remove batch/channel dimensions
        cam = cam.squeeze()

        # Convert to NumPy
        cam = cam.detach().cpu().numpy()

        # ----------------------------------------------------
        # Normalize to [0, 1]
        # ----------------------------------------------------

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        return cam.astype(np.float32)

    # ========================================================
    # Create overlay
    # ========================================================

    def overlay(
        self,
        original_image,
        heatmap,
        alpha=0.45
    ):

        h, w = original_image.shape[:2]

        # Resize heatmap to original image
        heatmap = cv2.resize(
            heatmap,
            (w, h)
        )

        # Convert to uint8
        heatmap_uint8 = np.uint8(
            heatmap * 255
        )

        # Apply color map
        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # OpenCV BGR → RGB
        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # Blend
        overlay = (
            alpha * heatmap_color
            +
            (1 - alpha) * original_image
        )

        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(np.uint8)

        return overlay

    # ========================================================
    # Visualization
    # ========================================================

    def visualize(
        self,
        original_image,
        image_tensor,
        probability=None,
        actual_label=None,
        prediction=None,
        save_path=None,
        patch_id=None
    ):

        # Generate heatmap
        heatmap = self.generate(
            image_tensor
        )

        # Generate overlay
        overlay_image = self.overlay(
            original_image,
            heatmap
        )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14, 4)
        )

        # Original
        axes[0].imshow(
            original_image
        )

        axes[0].set_title(
            "Original Patch",
            fontsize=11,
            fontweight="bold"
        )

        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(
            heatmap,
            cmap="jet"
        )

        axes[1].set_title(
            "Grad-CAM Heatmap",
            fontsize=11,
            fontweight="bold"
        )

        axes[1].axis("off")

        # Overlay
        axes[2].imshow(
            overlay_image
        )

        axes[2].set_title(
            "Grad-CAM Overlay",
            fontsize=11,
            fontweight="bold"
        )

        axes[2].axis("off")

        # ----------------------------------------------------
        # Title information
        # ----------------------------------------------------

        title_parts = []

        if patch_id is not None:
            title_parts.append(
                f"Patch: {patch_id}"
            )

        if actual_label is not None:

            actual_text = (
                "Cancer"
                if actual_label == 1
                else "Normal"
            )

            title_parts.append(
                f"Actual: {actual_text}"
            )

        if prediction is not None:
            title_parts.append(
                f"Predicted: {prediction}"
            )

        if probability is not None:
            title_parts.append(
                f"Tumor Probability: {probability:.3f}"
            )

        if title_parts:

            fig.suptitle(
                " | ".join(title_parts),
                fontsize=12,
                fontweight="bold"
            )

        plt.tight_layout()

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        if save_path:

            save_dir = os.path.dirname(
                save_path
            )

            if save_dir:
                os.makedirs(
                    save_dir,
                    exist_ok=True
                )

            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight"
            )

            print(
                f"Grad-CAM saved to: {save_path}"
            )

        plt.show()

        plt.close(fig)

    # ========================================================
    # Remove hooks
    # ========================================================

    def remove_hooks(self):

        if self.forward_hook is not None:
            self.forward_hook.remove()

        if self.backward_hook is not None:
            self.backward_hook.remove()


# ============================================================
# Main Demo
# ============================================================

def main():

    print("=" * 60)
    print("Grad-CAM Histopathology Demo")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Import model
    # --------------------------------------------------------

    from src.model import HistoClassifier

    # --------------------------------------------------------
    # Create model
    #
    # IMPORTANT:
    # The checkpoint was trained with hidden_dim=128.
    # --------------------------------------------------------

    model = HistoClassifier(
        backbone="resnet18",
        pretrained=False,
        hidden_dim=128
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        "./outputs/checkpoints/"
        "resnet18_pcam_best.pth"
    )

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Load PCam validation image
    # --------------------------------------------------------

    image_file = (
        "./data/pcamv1/"
        "camelyonpatch_level_2_split_valid_x.h5"
    )

    label_file = (
        "./data/pcamv1/"
        "camelyonpatch_level_2_split_valid_y.h5"
    )

    if not os.path.exists(image_file):

        raise FileNotFoundError(
            f"Image dataset not found:\n"
            f"{image_file}"
        )

    if not os.path.exists(label_file):

        raise FileNotFoundError(
            f"Label dataset not found:\n"
            f"{label_file}"
        )

    # --------------------------------------------------------
    # Select patch
    # --------------------------------------------------------

    patch_index = 0

    with h5py.File(
        image_file,
        "r"
    ) as f:

        original_image = f["x"][patch_index]

    with h5py.File(
        label_file,
        "r"
    ) as f:

        actual_label = int(
            f["y"][patch_index].squeeze()
        )

    print(
        "Patch index:",
        patch_index
    )

    print(
        "Actual label:",
        "Cancer"
        if actual_label == 1
        else "Normal"
    )

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    transform = A.Compose([

        A.Resize(
            128,
            128
        ),

        A.Normalize(
            mean=(
                0.485,
                0.456,
                0.406
            ),
            std=(
                0.229,
                0.224,
                0.225
            )
        ),

        ToTensorV2()

    ])

    transformed = transform(
        image=original_image
    )

    image_tensor = transformed["image"]

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        ).item()

    prediction = (
        "Cancer"
        if probability >= 0.5
        else "Normal"
    )

    print(
        "Predicted:",
        prediction
    )

    print(
        f"Tumor probability: "
        f"{probability:.4f}"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gcam = GradCAM(
        model
    )

    output_path = (
        "./outputs/gradcam/"
        "gradcam_sample.png"
    )

    try:

        gcam.visualize(
            original_image=original_image,
            image_tensor=image_tensor,
            probability=probability,
            actual_label=actual_label,
            prediction=prediction,
            save_path=output_path,
            patch_id=f"PCam_{patch_index}"
        )

    finally:

        gcam.remove_hooks()

    print("=" * 60)
    print("Grad-CAM completed successfully!")
    print("=" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()