# Domain Adaptation Enhancement Module (DAEM) for Cross-Domain Few-Shot Object Detection

## Overview

The Domain Adaptation Enhancement Module (DAEM) is designed for cross-domain few-shot object detection tasks, focusing on improving model generalization to target domains. This module addresses domain differences through multiple mechanisms, enabling effective knowledge transfer from source to target domains, even with limited target domain samples.

## Key Features

- **Batch Enhancement**: Diversifies training samples through style transfer and data augmentation
- **Feature Alignment**: Reduces domain feature distribution differences using Maximum Mean Discrepancy (MMD) loss
- **Improved Generalization**: Better performance on cross-domain detection tasks

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.7+
- PyTorch 1.13+

### Setting Up the Environment

```bash
# Create and activate conda environment
conda create -n cdfsod python=3.9
conda activate cdfsod

# Install PyTorch with CUDA support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt

# Install detectron2
python -m pip install -e detectron2

# Build project
python setup.py develop
```

## Directory Structure

```
|- lib/
|  |- daem/           # Domain Adaptation Enhancement Module
|  |- dinov2/         # DINOv2 implementation
|
|- tools/
|  |- train_enhanced_net.py  # Training script with DAEM
|  |- train_net.py           # Baseline training script
|
|- configs/           # Configuration files for different datasets
|
|- datasets/          # Directory structure for datasets
|  |- dataset1/
|  |- dataset2/
|  |- dataset3/
|
|- weights/           # Pre-trained weights
|  |- background/     # Background prototypes
|  |- trained/        # Pre-trained models
```

## Dataset Preparation

Place your datasets in the corresponding directories:
- `datasets/dataset1/` - First dataset
- `datasets/dataset2/` - Second dataset
- `datasets/dataset3/` - Third dataset

Each dataset should follow the COCO format with the following structure:
```
datasets/dataset1/
  ├── annotations/
  │   └── instances_train.json
  │   └── instances_val.json
  ├── train/
  │   └── images...
  └── val/
      └── images...
```

## Running the Code

### 1. Basic Training (without DAEM)

```bash
python tools/train_net.py \
  --num-gpus 1 \
  --config-file configs/dataset1/vitl_shot1_dataset1_finetune.yaml \
  MODEL.WEIGHTS weights/trained/vitl_0089999.pth \
  DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
  OUTPUT_DIR output/vitl_base/dataset1_1shot/
```

### 2. Training with DAEM

```bash
python tools/train_enhanced_net.py \
  --num-gpus 1 \
  --config-file configs/dataset1/vitl_shot1_dataset1_finetune_daem.yaml \
  MODEL.WEIGHTS weights/trained/vitl_0089999.pth \
  DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
  OUTPUT_DIR output/daem_vitl/dataset1_1shot/
```

### 3. Using Shell Scripts

For convenience, you can use the provided shell scripts:

```bash
# Run baseline models on all datasets
bash main_results.sh

# Run DAEM-enhanced training on specific configuration
bash run_daem.sh configs/dataset1/vitl_shot1_dataset1_finetune_daem.yaml
```

## Configuration Options

The DAEM module can be customized through configuration files in the `configs/` directory. Key parameters include:

```yaml
DAEM:
  ENABLED: True            # Enable/disable DAEM
  STRENGTH: 0.7            # Strength of domain adaptation enhancement
  FEATURE_ALIGNMENT: True  # Enable feature alignment
  MMD_WEIGHT: 0.1          # Weight for MMD loss
  STYLE_WEIGHT: 0.3        # Weight for style consistency loss
```

## Pre-trained Models

Place the pre-trained models in the corresponding directories:
- Background prototypes: `weights/background/`
- Pre-trained ViT models: `weights/trained/`

## License

This project is released under the MIT License. 