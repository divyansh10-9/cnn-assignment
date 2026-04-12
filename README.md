# Multi-Task Visual Perception Pipeline (Oxford-IIIT Pet Dataset)

This project implements a **multi-task visual perception pipeline** using PyTorch.  
The system simultaneously performs:

- Image Classification (Pet Breed)
- Object Localization (Bounding Box)
- Semantic Segmentation (Trimap Mask)

The pipeline is built using a **shared VGG11 backbone** and three task-specific heads.

---

# Project Structure

├── checkpoints
│ └── checkpoints.md
├── datasets
│ └── pets_dataset.py
├── inference.py
├── losses
│ └── iou_loss.py
├── models
│ ├── classification.py
│ ├── layers.py
│ ├── localization.py
│ ├── multitask.py
│ ├── segmentation.py
│ └── vgg11.py
├── train.py
├── requirements.txt
└── README.md

### 3. Segmentation Head

Architecture:
- U-Net style decoder
- Transposed Convolutions
- Skip connections

Loss:
- Dice Loss

---

# Multi-Task Model

The unified model outputs:

- Classification logits
- Bounding box predictions
- Segmentation masks

Single forward pass:

---

# Dataset

This project uses the **Oxford-IIIT Pet Dataset**:

- 37 Pet Breeds
- Bounding Boxes
- Segmentation Masks (Trimaps)

Dataset Link:
https://www.robots.ox.ac.uk/~vgg/data/pets/

---

# Model Architecture

## Shared Backbone

- VGG11 (Implemented from scratch)
- Batch Normalization
- Custom Dropout

## Task Heads

### 1. Classification Head

Output:
- 37-class pet breed classification

Loss:
- Cross Entropy Loss

---

### 2. Localization Head

Output:
- Bounding box:  
  `[x_center, y_center, width, height]`

Loss:
- MSE Loss  
- Custom IoU Loss

---

### 3. Segmentation Head

Architecture:
- U-Net style decoder
- Transposed Convolutions
- Skip connections

Loss:
- Dice Loss

---

# Multi-Task Model

The unified model outputs:

- Classification logits
- Bounding box predictions
- Segmentation masks

Single forward pass:


classification, localization, segmentation = model(image)


---

# Training

Train each module individually:

## Classification


python train.py --data_root ./datasets --task classification


## Localization


python train.py --data_root ./datasets --task localization


## Segmentation


python train.py --data_root ./datasets --task segmentation


---

# Inference


python inference.py --data_root ./datasets


---

# Weights & Biases Report

Training experiments are logged using Weights & Biases:

WandB Project Link:

https://wandb.ai/divyanshusingh2605910-indian-institute-of-technology-madras/Oxford-IIIT-Pet-Multitask

Tracked Metrics:

- Classification Loss
- Localization Loss
- Segmentation Dice Loss
- Training Curves

---

# Model Checkpoints

Model checkpoints are automatically downloaded from Google Drive.

Files:

- classifier.pth
- localizer.pth
- unet.pth

These are loaded automatically inside `multitask.py`.

---

# Requirements

Install dependencies:


pip install -r requirements.txt


Main libraries:

- torch
- numpy
- matplotlib
- scikit-learn
- wandb
- albumentations

---

# Author

Divyanshu Singh  
Indian Institute of Technology Madras

---

# Notes

- VGG11 implemented from scratch
- Custom Dropout implemented
- Custom IoU Loss implemented
- Multi-task learning architecture
- End-to-end perception pipeline

---


## wandb report link - https://wandb.ai/divyanshusingh2605910-indian-institute-of-technology-madras/Oxford-IIIT-Pet-Multitask/reports/DA6401-ASSIGNMENT2--VmlldzoxNjQ5NjIzNw


# github link - https://github.com/divyansh10-9/cnn-assignment