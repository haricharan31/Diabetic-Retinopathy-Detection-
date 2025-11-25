# Diabetic Retinopathy Detection Using Transformer-Based Architectures

This repository contains the complete implementation of a Diabetic Retinopathy (DR) classification system using Transformer-based deep learning models. The project evaluates four architectures: a standalone Vision Transformer (ViT), two fusion-based Transformer models integrating Medical Transformers (MedT), and a staged Model Freezing Pipeline. The goal is to leverage both global and local retinal feature representations to achieve high-accuracy DR grading.

---

##  Project Overview

Diabetic Retinopathy is a leading cause of blindness globally. Early detection is critical, and fundus image analysis using deep learning has shown promising results. This project explores how Transformer architectures can be applied and fused for improved DR severity classification across four classes: **No DR**, **Mild**, **Moderate**, and **Severe**.

---

##  Models Implemented

### 1. **Vision Transformer (ViT)**
- Extracts global contextual features from fundus images using self-attention.
- Serves as the baseline model.
- Achieved **87.50%** accuracy.

### 2. **Fusion Transformer Model 1 (MedT + ViT Concatenation)**
- Combines localized features from Medical Transformer (MedT) with global ViT features.
- Fusion is performed through vector concatenation.
- Achieved **86.50%** accuracy.

### 3. **Fusion Transformer Model 2 (Token-Level Fusion)**
- Stacks local (MedT) and global (ViT CLS) vectors as tokens.
- A fusion Transformer performs cross-token self-attention.
- Achieved **88.79%** accuracy.

### 4. **Model Freezing Pipeline**
- Staged training: freeze all Transformer backbones initially, train classifier head, then unfreeze with a higher LR.
- Provides excellent training stability and generalization.
- Achieved **88.94%** accuracy (best model).

---

##  Final Results

| Model                               | Accuracy (%) |
|-------------------------------------|--------------|
| Vision Transformer (ViT)            | 87.50        |
| Fusion Transformer Model 1          | 86.50        |
| Fusion Transformer Model 2          | 88.79        |
| Model Freezing Pipeline             | **88.94**    |

---

##  Dataset

- APTOS 2019 Blindness Detection dataset.
- Contains retinal fundus images labeled with DR severity.
- Preprocessing includes resizing, normalization, augmentation (rotation, flips).

---

##  Training Configuration

- **Batch Size:** 64  
- **Epochs:** 14  
- **Framework:** PyTorch  
- **Image Size:** 224×224  
- **Optimizer:** AdamW  
- **Loss Function:** CrossEntropy  

Learning rates:
- ViT baseline: `5e-5`
- Fusion Models: `3e-5`
- Freezing model base LR: `3e-5`
- Freezing model head LR: `1e-3`

