# 🌱 Farm Shield AI: Himalayan Edition

An offline-first deep learning system for plant disease diagnosis, developed as the AI engine for **Farm Shield Pro: Himalayan Edition**.

The project uses **transfer learning with MobileNetV3-Small** to identify diseases across multiple crops and exports the trained model to **TensorFlow Lite** for on-device inference in a Flutter application.

---

# Project Overview

Farm Shield AI is designed to provide farmers with fast, accurate, and offline crop disease diagnosis directly on their smartphones.

Unlike cloud-based solutions, inference runs completely on-device using TensorFlow Lite, making the application suitable for remote Himalayan regions where internet connectivity is limited.

The AI model serves as the diagnosis engine behind the Farm Shield Pro mobile application.

---

# Features

- MobileNetV3-Small transfer learning
- Offline TensorFlow Lite inference
- Supports 35 disease classes
- Covers 6 major crops
- End-to-end PyTorch training pipeline
- Automated checkpoint saving
- Model evaluation and classification reports
- Confusion matrix generation
- LiteRT Torch deployment pipeline
- TensorFlow Lite model export
- Deployment verification pipeline

---

# Supported Crops

- Apple
- Capsicum
- Maize
- Potato
- Tomato
- Wheat

---

# Dataset

The model was trained using a combined dataset built from:

- PlantVillage Dataset
- Wheat Plant Disease Dataset

## Dataset Statistics

| Metric | Value |
|---------|------:|
| Total Images | 40,977 |
| Crops | 6 |
| Disease Classes | 35 |

---

# Model Architecture

**Base Model**

- MobileNetV3-Small
- ImageNet pretrained weights

**Training Strategy**

- Transfer Learning
- Frozen feature extractor
- Custom classification head
- CrossEntropyLoss
- Adam Optimizer

---

# Training Pipeline

```
Dataset
        │
        ▼
FarmShieldDataset
        │
        ▼
Data Augmentation
        │
        ▼
MobileNetV3
        │
        ▼
Training
        │
        ▼
Validation
        │
        ▼
Checkpoint
        │
        ▼
Evaluation
```

---

# Deployment Pipeline

```
PyTorch Checkpoint (.pth)
                │
                ▼
LiteRT Torch
                │
                ▼
TensorFlow Lite (.tflite)
                │
                ▼
LiteRT Runtime Verification
                │
                ▼
Flutter Application
```

---

# Repository Structure

```
farm_shield_ai/

├── deployment/
│   ├── config.py
│   ├── export.py
│   ├── model_loader.py
│   └── verify.py
│
├── documentation/
│   └── labels.json
│
├── exports/
│   ├── farmshield_v1.pth
│   ├── farmshield_v1.tflite
│   └── confusion_matrix.png
│
├── training/
│   ├── datasets/
│   ├── models/
│   ├── utils/
│   ├── checkpoint.py
│   ├── engine.py
│   ├── evaluate.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

# Results

## Validation Accuracy

**91.3%**

### Evaluation Metrics

- Classification Report
- Per-class Precision
- Per-class Recall
- Per-class F1 Score
- Confusion Matrix

The trained TensorFlow Lite model was successfully verified using the LiteRT runtime before deployment.

---

# Deployment

The exported model:

```
farmshield_v1.tflite
```

is optimized for:

- Flutter
- Android
- Offline inference
- Mobile deployment

---

# Future Improvements

- Fine-tune entire MobileNetV3 backbone
- Quantization-aware training
- Model pruning
- Knowledge distillation
- Larger real-world Himalayan dataset
- Additional crop support
- Explainable AI (Grad-CAM)
- Continuous learning pipeline

---

# Tech Stack

### AI

- PyTorch
- TorchVision
- LiteRT Torch
- TensorFlow Lite
- NumPy
- Scikit-learn

### Deployment

- LiteRT Runtime
- Flutter (Farm Shield Pro)

**Note:** The training datasets are not included in this repository due to their size. The repository contains the complete training and deployment pipeline, while datasets should be downloaded separately.

---

# Integration

This repository provides the AI engine for:

**Farm Shield Pro: Himalayan Edition**

The Flutter application performs:

- Image capture
- Image preprocessing
- TensorFlow Lite inference
- Disease diagnosis
- Remedy recommendation
- Voice guidance
- Farm record management

---

# License

This project was developed as part of the **TechnoHub Internship Program** for educational and research purposes.