# DA6401 Assignment 2 – Building a Complete Visual Perception Pipeline
**Ayman Hamza Haji (GE26Z814)
---

## Project Overview

This project implements a complete multi-task visual perception pipeline using the Oxford-IIIT Pet Dataset. The system performs three main computer vision tasks:

- Image Classification (37 pet breeds)
- Object Localization (Bounding Box Prediction)
- Semantic Segmentation (Pixel-wise Mask)

All components are combined into a single unified model that performs all tasks in one forward pass.

---

## Task 1: VGG11 Classification with Custom Regularization

A VGG11 architecture was implemented from begining using PyTorch.

### Key Features:
- Custom VGG11 implementation (no pretrained models used)
- Batch Normalization added to stabilize training
- Custom Dropout layer implemented manually (not using torch.nn.Dropout)

### Result:
Batch Normalization improves training stability and speed, while Dropout reduces overfitting and improves generalization.

---

## Task 2: Object Localization

The classification model was extended to predict bounding boxes.

### Implementation:
- VGG11 encoder used as feature extractor
- Regression head added to predict:
  ```
  (x_center, y_center, width, height)
  ```

### Loss Functions:
- Mean Squared Error (MSE)
- Custom Intersection over Union (IoU) Loss

### Result:
Fine-tuning the backbone improves localization performance compared to freezing it.

---

## Task 3: Semantic Segmentation (U-Net)

A U-Net style architecture was implemented using the VGG11 encoder.

### Key Components:
- Encoder: VGG11 convolutional layers
- Decoder: Transposed Convolutions for upsampling
- Skip Connections for combining low-level and high-level features

### Loss Function:
- Cross-Entropy Loss for pixel-wise classification

### Result:
Skip connections help recover spatial information lost during downsampling, improving segmentation quality

---

## Task 4: Unified Multi-Task Pipeline

All three tasks were combined into a single model.

### Single Forward Pass Outputs:
1. Classification logits (37 classes)
2. Bounding box coordinates
3. Segmentation mask

### Advantages:
- Shared feature learning
- Reduced computation
- Better generalization across tasks

---

## Weights & Biases (W&B) Experiments

All experiments were tracked and visualized using W&B.

---

### 2.1 Regularization Effect of BatchNorm

Batch Normalization resulted in:
- Faster convergence
- More stable training
- Ability to use higher learning rates

---

### 2.2 Dropout Analysis

Three setups were compared:
- No Dropout
- Dropout (p = 0.2)
- Dropout (p = 0.5)

Observation:
Higher dropout improved generalization but slowed convergence.

---

### 2.3 Transfer Learning Strategies

Compared:
- Frozen backbone
- Partial fine-tuning
- Full fine-tuning

Result:
Full fine-tuning achieved the best performance due to better task adaptation.

---

### 2.4 Feature Map Visualization

- Early layers capture edges and textures
- Deeper layers capture semantic structures such as faces and shapes

---

### 2.5 Object Detection Analysis

W&B table includes:
- Ground truth bounding boxes
- Predicted bounding boxes
- IoU scores

Failure Case:
Images with multiple objects, or complex backgrounds lead to low IoU even when confidence is high.

---

### 2.6 Segmentation Evaluation

Metrics used:
- Pixel Accuracy
- Dice Score

Observation:
Pixel Accuracy can be misleading due to background dominance, while Dice Score gives a better evaluation.

---

### 2.7 Final Pipeline Showcase

The pipeline was tested on 3 unseen images from the internet.

Observations:
- Works well on clear, single-object images
- Bounding boxes become less accurate in cluttered scenes
- Segmentation struggles with poor lighting or complex backgrounds

---

### 2.8 Meta-Analysis and Reflection

#### Architectural Design:
Batch Normalization improved stability, while Dropout reduced overfitting.

#### Encoder Strategy:
Fine-tuning improved performance but introduced slight interference between tasks.

#### Loss Functions:
IoU Loss improved localization accuracy, while Dice-based evaluation provided better segmentation understanding.



## 📂 Project Structure
```
DA6401_ASSIGNMENT_2/
│
├── checkpoints/
├── data/
├── losses/
├── models/
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   ├── multitask.py
│   └── vgg11.py
│
├── train.py
├── inference.py
├── segmentation_wandb.py
├── bbox_wandb_table.py
├── feature_maps.py
├── final_pipeline_showcase.py
│
├── requirements.txt
└── README.md

```
---
## Installation

pip install -r requirements.txt

---

## How to Run

### Train Models
```
python train.py --data_root data/oxford-iiit-pet --task classification --epochs 15 --batch_size 32 --lr 1e-4 
python train.py --data_root data/oxford-iiit-pet --task classification --epochs 15 --batch_size 32 --lr 1e-4 --no_batchnorm  
```
---

### Run Evaluation and Visualization
```
python segmentation_wandb.py  
python bbox_wandb_table.py  
python feature_maps.py  
python final_pipeline_showcase.py  
```
---

## Weights & Biases (W&B)

Add your report link here after uploading:
W&B Report: https://wandb.ai/ge26z814-iitm-india/da6401_assignment_2/reports/Building-a-Complete-Visual-Perception-Pipeline--VmlldzoxNjM4MjE5MA?accessToken=jylaaaupcezpqbulsgnhugmu2ravopx1n8t4buugqp0j20aal9n44g4sb33vkl6s

---

## GitHub Repository

GitHub Repo: https://github.com/Melanin08/da6401_assignment_2.git 

---

## Important Notes

- All models are implemented from scratch
- Only allowed libraries are used
- Dataset splitting avoids data leakage
- Custom Dropout and IoU Loss are correctly implemented

---

## Conclusion

This project successfully builds a unified visual perception pipeline capable of classification, localization, and segmentation. The model performs well on structured images but shows limitations in handling multiple objects, and complex real-world scenarios.









