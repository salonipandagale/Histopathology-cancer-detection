# Histopathology Cancer Detection & Explainability

A deep learning-based histopathology image classification project for detecting cancerous tissue patches using **ResNet18**, transfer learning, image augmentation, and **Grad-CAM explainability**.

This project was adapted and extended from an existing histopathology deep learning repository, with modifications to the dataset pipeline, training workflow, evaluation pipeline, checkpointing, and Grad-CAM visualization.

The project is designed as a practical exploration of **AI for Digital Pathology**, particularly patch-level cancer classification and model interpretability.

---

## Project Overview

Digital pathology involves analyzing extremely large Whole-Slide Images (WSIs) of tissue samples using computational methods. A single WSI can contain millions of pixels, making direct processing computationally expensive.

A common approach is to divide a WSI into smaller image patches and use deep learning models to classify individual patches.

This project focuses on:

1. Processing histopathology image patches.
2. Classifying patches as normal or cancerous.
3. Training a CNN-based image classification model.
4. Evaluating classification performance using multiple metrics.
5. Generating Grad-CAM visualizations to understand which regions influence the model's prediction.

The overall workflow is:

```text
Histopathology Images
        |
        v
Image Preprocessing
        |
        v
Data Augmentation
        |
        v
ResNet18 Feature Extraction
        |
        v
Classification Head
        |
        v
Cancer Probability
        |
        v
Model Evaluation
        |
        v
Grad-CAM Explainability
```

---

## Objective

The primary objective is to develop a deep learning pipeline capable of identifying cancerous tissue patterns from histopathology image patches while also providing visual explanations for the model's predictions.

The project combines:

- Computer Vision
- Deep Learning
- Transfer Learning
- Histopathology Image Classification
- Data Augmentation
- Model Evaluation
- Explainable AI
- Grad-CAM

---

## Key Features

- Histopathology patch classification
- ResNet18-based CNN architecture
- Transfer learning using pretrained ImageNet weights
- Binary cancer classification
- Data augmentation using Albumentations
- Stratified dataset splitting
- Weighted sampling for class balancing
- BCEWithLogitsLoss for binary classification
- AdamW optimizer
- Cosine learning-rate scheduling
- Gradient clipping
- Model checkpointing
- Accuracy, precision, recall, specificity and F1 evaluation
- ROC-AUC analysis
- Confusion matrix visualization
- Grad-CAM explainability
- Automatic selection of the final convolutional layer for Grad-CAM
- CPU-compatible training and inference pipeline

---

# Dataset

## PCam Dataset

The project uses the **PatchCamelyon (PCam)** dataset.

PCam is a histopathology image classification dataset derived from lymph node sections. It contains small RGB tissue patches with binary labels indicating the presence or absence of metastatic tissue.

Each image patch has dimensions:

```text
96 × 96 × 3
```

The dataset contains:

```text
327,680 total image patches

Training:   262,144
Validation: 32,768
Test:       32,768
```

The classes are approximately balanced:

```text
0 → Normal / Non-cancer
1 → Cancer
```

PCam is particularly useful for demonstrating how deep learning can be applied to histopathology image classification.

Dataset source:

```text
https://github.com/basveeling/pcam
```

---

## Working Dataset

The complete PCam training image file is several gigabytes in size and was not available locally during development.

Therefore, the available **PCam validation split containing 32,768 labelled images** was used as the working dataset.

A reproducible subset of **2,000 images** was sampled from this dataset for model development and experimentation.

The 2,000 images were split into:

```text
Training:    1,600
Validation:    200
Test:          200
```

The sampled dataset maintained an approximately balanced class distribution:

```text
Normal : 1008
Cancer :  992
```

Important:

The 200-image evaluation split used in this project is derived from the PCam validation split and should therefore be considered a **development/evaluation subset**, not the official PCam test benchmark.

The reported metrics are intended to demonstrate the functionality and behavior of the implemented pipeline rather than represent a definitive benchmark on the complete PCam dataset.

---

# Model Architecture

## ResNet18

The project uses **ResNet18** as the primary image classification backbone.

ResNet18 is a convolutional neural network based on residual learning. Residual connections help deeper networks learn effectively by allowing information to flow through shortcut connections.

The architecture used in this project can be represented as:

```text
Input Image
    |
    v
ResNet18 Encoder
    |
    v
Feature Representation
    |
    v
Dropout
    |
    v
Fully Connected Layer
    |
    v
ReLU
    |
    v
Dropout
    |
    v
Binary Classification Layer
    |
    v
Cancer Logit
    |
    v
Sigmoid
    |
    v
Cancer Probability
```

The ResNet18 encoder is initialized using pretrained ImageNet weights and adapted for binary classification.

---

# Transfer Learning

Transfer learning is used to leverage visual features learned by ResNet18 on the ImageNet dataset.

Instead of training the entire CNN from random initialization, the pretrained convolutional layers provide a useful starting point for learning histopathology-specific patterns.

The final classification layers are adapted for the binary classification task:

```text
Normal
   vs
Cancer
```

This approach reduces the amount of training required compared with training a deep CNN entirely from scratch.

---

# Image Preprocessing

The original PCam images have dimensions:

```text
96 × 96 × 3
```

During training, images are resized to:

```text
128 × 128
```

The preprocessing pipeline includes normalization and conversion to PyTorch tensors.

Training augmentation includes:

- Horizontal flipping
- Vertical flipping
- 90-degree rotations
- Small image rotations
- Shift and scale transformations
- Color jitter
- Image normalization

Validation and evaluation images use deterministic preprocessing without random augmentation.

The purpose of augmentation is to expose the model to different spatial orientations and color variations while reducing overfitting.

---

# Data Pipeline

The dataset is implemented using a custom PyTorch Dataset.

The HDF5 image files are accessed lazily rather than loading the entire image dataset into memory.

The general pipeline is:

```text
HDF5 Dataset
     |
     v
Lazy Image Loading
     |
     v
Image Augmentation
     |
     v
Normalization
     |
     v
PyTorch Tensor
     |
     v
DataLoader
     |
     v
Model
```

The dataset pipeline also uses stratified splitting so that the class distribution remains approximately consistent across the training, validation, and evaluation subsets.

A weighted random sampler is used during training to reduce potential class imbalance effects.

---

# Training Configuration

The main training configuration is controlled through:

```text
config.yaml
```

Important parameters include:

```yaml
model:
  backbone: "resnet18"
  pretrained: true
  dropout: 0.4
  hidden_dim: 128
  num_classes: 1
```

Training configuration:

```yaml
training:
  epochs: 1
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: "adamw"
  scheduler: "cosine"
  grad_clip: 1.0
```

The current configuration uses one epoch because the project was developed and tested on a CPU-based environment using a 2,000-image working subset.

The training pipeline itself supports multiple epochs and can be extended to larger datasets and longer training runs.

---

# Loss Function

Binary classification is performed using:

```text
BCEWithLogitsLoss
```

The model outputs a raw logit for each image.

The sigmoid function is then applied to convert the logit into a probability:

```text
P(Cancer) = Sigmoid(Logit)
```

A probability threshold can then be used to determine the predicted class.

---

# Optimization

The training pipeline uses the **AdamW optimizer**.

AdamW combines adaptive gradient optimization with decoupled weight decay, which can improve regularization during neural network training.

The configured learning rate is:

```text
0.0001
```

Weight decay:

```text
0.0001
```

Gradient clipping is also enabled to limit excessively large gradients during optimization.

---

# Training Results

The model was trained on the 2,000-image working subset.

Training split:

```text
1,600 images
```

Validation split:

```text
200 images
```

Evaluation split:

```text
200 images
```

After one training epoch:

```text
Train Loss : 0.6350
Train Acc  : 0.6200

Val Loss   : 0.5229
Val Acc    : 0.7350
```

The best model checkpoint is saved to:

```text
outputs/checkpoints/resnet18_pcam_best.pth
```

---

# Model Evaluation

The trained model was evaluated on the 200-image development test split.

The following metrics were calculated:

- Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1 Score
- ROC-AUC

## Results

```text
Accuracy    : 0.7850
Precision   : 0.7413
Sensitivity : 0.9464
Specificity : 0.5795
F1 Score    : 0.8314
ROC-AUC     : 0.9012
```

These results should be interpreted as a development-stage evaluation because the experiment uses a relatively small subset and only one training epoch.

---

# Classification Report

```text
              precision    recall  f1-score   support

Normal          0.89      0.58      0.70        88
Cancer          0.74      0.95      0.83       112

accuracy                            0.79       200
macro avg       0.82      0.76      0.77       200
weighted avg    0.81      0.79      0.78       200
```

The model achieved high recall for the cancer class.

In a medical image classification context, sensitivity is an important metric because missing a cancerous patch can be more problematic than incorrectly flagging a normal patch.

However, these results are experimental and should not be interpreted as clinical performance.

---

# Confusion Matrix

The resulting confusion matrix was:

```text
                 Predicted
                 Normal  Cancer

Actual Normal       51      37
Actual Cancer        6     106
```

This corresponds to:

```text
True Negative  : 51
False Positive : 37
False Negative : 6
True Positive   : 106
```

The model correctly identified most cancerous patches in this development evaluation subset, resulting in a sensitivity of approximately 94.64%.

The confusion matrix visualization is generated automatically and saved to:

```text
outputs/results/confusion_matrix.png
```

---

# ROC-AUC

The ROC curve is generated to evaluate the model's ability to distinguish between normal and cancerous patches across different classification thresholds.

The development evaluation produced:

```text
ROC-AUC = 0.9012
```

The ROC curve is saved to:

```text
outputs/results/roc_curve.png
```

A higher ROC-AUC indicates better ranking of positive and negative examples across classification thresholds.

---

# Explainable AI with Grad-CAM

## Why Explainability?

Deep learning models can achieve strong classification performance while remaining difficult to interpret.

In medical imaging, understanding which regions influenced a prediction can be useful for model analysis and debugging.

This project therefore implements **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM produces a heatmap showing image regions that contributed strongly to the model's prediction.

The workflow is:

```text
Input Histopathology Patch
          |
          v
       ResNet18
          |
          v
    Model Prediction
          |
          v
      Gradients
          |
          v
Activation Maps
          |
          v
     Grad-CAM
          |
          v
Heatmap Overlay
```

---

# Automatic Grad-CAM Layer Selection

Different CNN architectures expose their convolutional layers using different internal module structures.

Instead of hard-coding a specific layer name, the implementation automatically searches the model and selects the final convolutional layer.

For the ResNet18 model used in this project, the selected layer was:

```text
encoder.7.1.conv2
```

This makes the Grad-CAM implementation more robust to changes in the model architecture.

---

# Grad-CAM Example

One sample prediction produced:

```text
Actual Label       : Cancer
Predicted Label    : Cancer
Tumor Probability  : 0.8540
```

The generated Grad-CAM visualization highlights the regions of the histopathology patch that contributed most strongly to the cancer prediction.

The visualization is saved to:

```text
outputs/gradcam/gradcam_sample.png
```

Grad-CAM is used here as a model interpretation and debugging technique. It should not be considered a clinical diagnostic tool.

---

# Project Structure

```text
histopathology-wsi-cancer-detector/
│
├── app.py
├── config.yaml
├── dataset.py
├── evaluate.py
├── evaluate_model.py
├── gradcam.py
├── model.py
├── slide_inference.py
├── stain_normalization.py
├── train.py
├── wsi_tiling.py
├── README.md
├── requirements.txt
│
├── data/
│   └── pcamv1/
│       ├── camelyonpatch_level_2_split_valid_x.h5
│       ├── camelyonpatch_level_2_split_valid_y.h5
│       └── ...
│
└── outputs/
    ├── checkpoints/
    │   └── resnet18_pcam_best.pth
    │
    ├── gradcam/
    │   └── gradcam_sample.png
    │
    ├── results/
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    │
    └── logs/
```

---

# File Descriptions

## `model.py`

Defines the deep learning architecture.

Responsibilities include:

- ResNet-based feature extraction
- EfficientNet support
- Classification head
- Binary prediction
- Loss function
- Optimizer
- Learning-rate scheduler
- Encoder freezing/unfreezing

---

## `dataset.py`

Handles the PCam dataset.

Responsibilities include:

- HDF5 image loading
- Label loading
- Data augmentation
- Image normalization
- Train/validation/test splitting
- Weighted sampling
- PyTorch DataLoader creation

---

## `train.py`

Main training script.

Responsibilities include:

- Loading configuration
- Creating datasets and DataLoaders
- Initializing the model
- Training the network
- Validation
- Checkpoint saving

---

## `evaluate_model.py`

Evaluates a trained model.

It calculates:

- Accuracy
- Precision
- Sensitivity
- Specificity
- F1 Score
- ROC-AUC
- Classification report
- Confusion matrix

It also generates evaluation plots.

---

## `gradcam.py`

Implements Grad-CAM visualization.

It:

1. Loads a trained checkpoint.
2. Selects a sample image.
3. Performs inference.
4. Identifies the final convolutional layer.
5. Computes gradients.
6. Generates a Grad-CAM heatmap.
7. Overlays the heatmap on the original image.

---

## `wsi_tiling.py`

Contains functionality related to dividing Whole-Slide Images into smaller patches.

Conceptually:

```text
Whole-Slide Image
       |
       v
Tiling
       |
       v
Small Image Patches
       |
       v
CNN Classification
```

The WSI-related components provide a foundation for extending the patch-level classifier toward larger digital pathology workflows.

---

## `slide_inference.py`

Contains functionality for performing inference across multiple patches from a slide and aggregating patch-level predictions.

This provides a conceptual bridge between:

```text
Patch-level classification
```

and

```text
Slide-level analysis
```

The current project primarily evaluates the patch-level classification pipeline.

---

## `stain_normalization.py`

Provides functionality for handling color variation in histopathology images.

Stain normalization can be useful because histopathology slides can differ in appearance due to differences in:

- staining procedures
- scanners
- laboratories
- image acquisition conditions

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/KinglerFaizan/histopathology-wsi-cancer-detector.git
cd histopathology-wsi-cancer-detector
```

If you are using the adapted version of the project, use your own repository URL instead.

---

## 2. Create a Virtual Environment

For Python 3.10:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The major dependencies include:

```text
PyTorch
Torchvision
Timm
Albumentations
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
PyYAML
h5py
tqdm
```

---

# Dataset Setup

The project expects the PCam data under:

```text
data/pcamv1/
```

The working dataset uses:

```text
camelyonpatch_level_2_split_valid_x.h5
camelyonpatch_level_2_split_valid_y.h5
```

The original PCam dataset can be obtained from the official PCam repository:

```text
https://github.com/basveeling/pcam
```

The complete PCam dataset is substantially larger than the development subset used in this project.

---

# Configuration

Training and dataset parameters are controlled through:

```text
config.yaml
```

Example:

```yaml
data:
  dataset: "pcam"
  data_dir: "./data/pcamv1"
  img_size: 128
  patch_size: 96
  overlap: 0
  tissue_threshold: 0.5
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_workers: 0

model:
  backbone: "resnet18"
  pretrained: true
  dropout: 0.4
  hidden_dim: 128
  num_classes: 1

training:
  epochs: 1
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 0
  pos_weight: 1.0
  grad_clip: 1.0
  seed: 42
```

---

# Running the Project

## Train the Model

Run:

```bash
python train.py
```

The script will:

1. Load the dataset.
2. Create the training, validation and evaluation splits.
3. Initialize ResNet18.
4. Apply image preprocessing and augmentation.
5. Train the model.
6. Evaluate validation performance.
7. Save the best checkpoint.

The checkpoint is saved under:

```text
outputs/checkpoints/
```

---

# Evaluate the Model

Run:

```bash
python evaluate_model.py
```

The evaluation script loads the saved checkpoint and produces:

```text
Accuracy
Precision
Sensitivity
Specificity
F1 Score
ROC-AUC
Classification Report
Confusion Matrix
ROC Curve
```

Generated files:

```text
outputs/results/confusion_matrix.png
outputs/results/roc_curve.png
```

---

# Generate Grad-CAM

Run:

```bash
python gradcam.py
```

The script loads the trained model and generates an explanation for a sample prediction.

The result is saved to:

```text
outputs/gradcam/gradcam_sample.png
```

---

# Example Prediction

Example output from the Grad-CAM pipeline:

```text
============================================================
Grad-CAM Histopathology Demo
============================================================

Device: cpu
Checkpoint loaded successfully.

Patch index: 0
Actual label: Cancer
Predicted: Cancer
Tumor probability: 0.8540

Automatically selected Grad-CAM layer:
encoder.7.1.conv2

Grad-CAM saved to:
./outputs/gradcam/gradcam_sample.png

============================================================
Grad-CAM completed successfully!
============================================================
```

---

# Technologies Used

## Programming Language

```text
Python
```

## Deep Learning

```text
PyTorch
Torchvision
Timm
```

## Computer Vision

```text
OpenCV
Pillow
Albumentations
```

## Machine Learning

```text
Scikit-learn
```

## Data Processing

```text
NumPy
Pandas
HDF5 / h5py
```

## Visualization

```text
Matplotlib
Seaborn
```

## Explainable AI

```text
Grad-CAM
```

## Development Tools

```text
Git
GitHub
Jupyter Notebook
VS Code
```

---

# Digital Pathology Relevance

The project is motivated by applications in **AI-assisted Digital Pathology**.

Histopathology images can be extremely large, making direct analysis computationally challenging.

A scalable digital pathology pipeline can therefore divide a Whole-Slide Image into smaller regions:

```text
Whole-Slide Image
        |
        v
Patch Extraction
        |
        v
Patch-Level Classification
        |
        v
Prediction Aggregation
        |
        v
Slide-Level Analysis
```

The classification component developed in this project provides the foundation for such a workflow.

The project also explores explainability through Grad-CAM, which can help visualize image regions associated with model predictions.

---

# Why ResNet18?

ResNet18 was selected because it provides a good balance between:

- Model capacity
- Computational efficiency
- Training speed
- Transfer learning capability
- Compatibility with Grad-CAM

A lighter architecture is particularly useful during development when experiments are performed on CPU hardware.

The architecture can later be replaced with deeper ResNet variants or other CNN architectures for larger-scale experiments.

---

# Limitations

The current implementation has several limitations.

### 1. Small Development Dataset

Only 2,000 images were used for the reported experiment rather than the complete PCam dataset.

### 2. Limited Training

The reported experiment was trained for one epoch.

Longer training with appropriate validation and hyperparameter tuning would likely provide a stronger model.

### 3. Development Evaluation Split

The evaluation images were sampled from the PCam validation split.

Therefore, the results should not be compared directly with official PCam benchmark results.

### 4. No Clinical Validation

The model has not been validated on clinical data and should not be used for medical diagnosis.

### 5. Patch-Level Evaluation

The primary experiment focuses on individual histopathology patches.

Full Whole-Slide Image inference and slide-level performance have not been benchmarked in the current experiment.

### 6. Computational Constraints

The project was developed and tested in a CPU-based environment.

Training on the complete PCam dataset would benefit significantly from GPU acceleration.

---

# Future Improvements

Several improvements can be explored in future versions.

## 1. Train on the Complete PCam Dataset

Use the complete PCam training split instead of the 2,000-image development subset.

This would provide a more representative training experiment.

---

## 2. Longer Training

Train the model for multiple epochs with:

- Early stopping
- Learning-rate scheduling
- Hyperparameter tuning
- Model checkpointing

---

## 3. Stronger CNN Backbones

Experiment with:

```text
ResNet50
EfficientNet
EfficientNet-B4
DenseNet
ConvNeXt
```

and compare their performance.

---

## 4. Hard Negative Mining

Identify difficult normal patches that are incorrectly classified as cancer and use them for additional training.

This can help improve specificity.

---

## 5. Stain Normalization

Apply stain normalization techniques to reduce differences caused by staining and image acquisition conditions.

---

## 6. Whole-Slide Image Analysis

Extend the patch-level model into a complete WSI pipeline:

```text
WSI
 |
 v
Tissue Detection
 |
 v
Tiling
 |
 v
Patch Classification
 |
 v
Prediction Heatmap
 |
 v
Slide-Level Aggregation
```

---

## 7. Advanced Aggregation

Instead of simple averaging, experiment with:

- Top-K aggregation
- Attention-based pooling
- Multiple Instance Learning
- Weighted patch aggregation

---

## 8. Improved Explainability

Extend Grad-CAM analysis with:

- Grad-CAM++
- Integrated Gradients
- Saliency maps
- Attention visualization

---

## 9. External Validation

Evaluate the trained model on an independent histopathology dataset to assess generalization.

---

# Applications

Potential research applications include:

- Histopathology image classification
- Cancer tissue detection
- Digital pathology research
- AI-assisted pathology workflows
- Tissue-region analysis
- Medical image explainability research
- Computer-aided diagnosis research

These applications are intended as research directions and do not imply clinical validation of the current model.

---

# Project Highlights

The project demonstrates an end-to-end computer vision workflow:

```text
Dataset
   |
   v
HDF5 Data Pipeline
   |
   v
Preprocessing
   |
   v
Data Augmentation
   |
   v
Transfer Learning
   |
   v
ResNet18
   |
   v
Binary Classification
   |
   v
Model Evaluation
   |
   v
ROC-AUC / Confusion Matrix
   |
   v
Grad-CAM Explainability
```

Key development outcomes include:

```text
Development Dataset : 2,000 images
Training Images     : 1,600
Validation Images   : 200
Evaluation Images   : 200

Accuracy            : 78.50%
Precision           : 74.13%
Sensitivity         : 94.64%
Specificity         : 57.95%
F1 Score            : 83.14%
ROC-AUC             : 90.12%
```

---

# References

## PatchCamelyon

Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., & Welling, M.

"Rotation Equivariant CNNs for Digital Pathology."

Official PCam repository:

```text
https://github.com/basveeling/pcam
```

---

## Grad-CAM

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.

"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."

The method is used in this project to visualize regions contributing to CNN predictions.

---

# Project Focus

This project focuses on applying **deep learning and explainable computer vision to histopathology image analysis**.

The implementation demonstrates how a CNN can be trained for cancerous tissue classification and how Grad-CAM can be used to inspect the visual evidence behind model predictions.

The current version is a development and research project rather than a clinically validated diagnostic system.

---

# Acknowledgements

This project was developed by adapting and extending an existing open-source histopathology cancer detection repository.

The implementation was modified to create a reproducible development pipeline using the PCam dataset, ResNet18, PyTorch, model evaluation, checkpointing, and Grad-CAM-based explainability.

The project is intended for educational, research, and portfolio purposes.
