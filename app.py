import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms

from model import HistoClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Histopathology Cancer Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .prediction {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
    }

    .probability {
        font-size: 1.4rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #ffffff;
        text-align: center;
    }

    .metric-title {
        font-size: 0.9rem;
        color: #666666;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
    }

    .disclaimer {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff8e6;
        border: 1px solid #ead9a6;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

CHECKPOINT_PATH = "outputs/checkpoints/resnet18_pcam_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 128
HIDDEN_DIM = 128


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    backbone = checkpoint.get(
        "backbone",
        "resnet18"
    )

    hidden_dim = checkpoint.get(
        "hidden_dim",
        HIDDEN_DIM
    )

    model = HistoClassifier(
    backbone=backbone,
    pretrained=False,
    hidden_dim=hidden_dim
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    last_conv = None
    last_name = None

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name

    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found in the model."
        )

    return last_conv, last_name


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, image_tensor):

    target_layer, layer_name = find_last_conv_layer(model)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        model.zero_grad()

        output = model(image_tensor)

        probability = torch.sigmoid(output)

        output.backward(
            gradient=torch.ones_like(output)
        )

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(
                image_tensor.shape[2],
                image_tensor.shape[3]
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return probability.item(), cam, layer_name

    finally:

        forward_handle.remove()
        backward_handle.remove()


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(original_image, cam):

    original = np.array(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(original)

    ax.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    ax.axis("off")

    plt.tight_layout(
        pad=0
    )

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    buffer.seek(0)

    return Image.open(buffer)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Architecture:** ResNet18"
    )

    st.write(
        "**Task:** Binary classification"
    )

    st.write(
        "**Dataset:** PCam"
    )

    st.write(
        "**Input:** Histopathology image patch"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.divider()

    st.subheader("Development Metrics")

    st.metric(
        "Accuracy",
        "78.50%"
    )

    st.metric(
        "Sensitivity",
        "94.64%"
    )

    st.metric(
        "Specificity",
        "57.95%"
    )

    st.metric(
        "F1 Score",
        "83.14%"
    )

    st.metric(
        "ROC-AUC",
        "90.12%"
    )

    st.divider()

    st.caption(
        "These metrics were obtained using a "
        "2,000-image development subset and "
        "one training epoch."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">Histopathology Cancer Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based tissue patch classification with '
    'Grad-CAM explainability'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        f"Unable to load the trained model: {e}"
    )

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Histopathology Image")

uploaded_file = st.file_uploader(
    "Upload a histopathology tissue patch",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN INFERENCE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.divider()

        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.subheader(
                "Input Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # PREDICTION + GRAD-CAM
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing image..."
        ):

            probability, cam, layer_name = generate_gradcam(
                model,
                image_tensor
            )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if probability >= 0.5:

            prediction = "CANCER"
            confidence = probability

        else:

            prediction = "NORMAL"
            confidence = 1 - probability

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "Prediction"
            )

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="prediction">
                        {prediction}
                    </div>

                    <div class="probability">
                        Confidence: {confidence * 100:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                f"Probability of cancerous tissue: "
                f"**{probability * 100:.2f}%**"
            )

            st.progress(
                float(probability)
            )

            if prediction == "CANCER":

                st.warning(
                    "The model classified this patch "
                    "as cancerous."
                )

            else:

                st.success(
                    "The model classified this patch "
                    "as normal."
                )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Grad-CAM Explainability"
        )

        st.write(
            "The Grad-CAM visualization highlights "
            "regions of the image that contributed "
            "to the model's prediction."
        )

        gradcam_image = create_gradcam_overlay(
            image,
            cam
        )

        col3, col4 = st.columns(
            [1, 1]
        )

        with col3:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col4:

            st.image(
                gradcam_image,
                caption="Grad-CAM Visualization",
                use_container_width=True
            )

        st.caption(
            f"Grad-CAM target layer: {layer_name}"
        )

        # ----------------------------------------------------
        # MODEL DETAILS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Analysis Details"
        )

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:

            st.markdown(
                """
                <div class="metric-card">

                <div class="metric-title">
                Model
                </div>

                <div class="metric-value">
                ResNet18
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with detail_col2:

            st.markdown(
                """
                <div class="metric-card">

                <div class="metric-title">
                Input Size
                </div>

                <div class="metric-value">
                128 × 128
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with detail_col3:

            st.markdown(
                """
                <div class="metric-card">

                <div class="metric-title">
                Device
                </div>

                <div class="metric-value">
                """
                + str(DEVICE)
                + """
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="disclaimer">

            <strong>Research Disclaimer</strong><br><br>

            This application is intended for educational
            and research purposes only. The model has been
            developed using a limited PCam development subset
            and has not undergone clinical validation.

            The predictions should not be used for medical
            diagnosis or treatment decisions.

            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"Error while processing the image: {e}"
        )

else:

    st.info(
        "Upload a histopathology image patch to begin analysis."
    )

    st.markdown(
        """
        ### How it works

        1. Upload a histopathology image.
        2. The image is resized and normalized.
        3. ResNet18 extracts visual features.
        4. The model predicts the probability of cancer.
        5. Grad-CAM generates a visual explanation.
        """
    )