# 🔬 SegFramework — Semantic Segmentation for Medical & Histopathology Imaging

> A modular, production-ready deep learning framework for pixel-level image segmentation — supporting CNN and Transformer architectures with a unified training pipeline.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Models](https://img.shields.io/badge/Models-8%20Architectures-purple?style=flat-square)](#implemented-models)

---

## Table of Contents

- [What is Semantic Segmentation?](#what-is-semantic-segmentation)
  - [Binary Segmentation](#binary-segmentation-2-class)
  - [Multi-Class Segmentation](#multi-class-segmentation-3-class-or-more)
- [Signet Ring Cell Region Detection](#signet-ring-cell-region-detection)
- [Framework Pipeline](#framework-pipeline)
- [Implemented Models](#implemented-models)
- [Key Features](#key-features)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Organise Your Data](#2-organise-your-data)
  - [3. Configure](#3-configure)
  - [4. Train](#4-train)
  - [5. Evaluate](#5-evaluate)
- [Configuration Reference](#configuration-reference)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
  - [Schedulers](#schedulers)
- [Augmentations](#augmentations)
- [Adding a New Model](#adding-a-new-model)
- [Output Files](#output-files)

---

## What is Semantic Segmentation?

Semantic segmentation is a pixel-level image analysis task where each pixel in an image is assigned to a specific class label. Unlike image classification, semantic segmentation identifies the exact spatial regions corresponding to objects or tissues within the image.

### Binary Segmentation (2-Class)

Binary segmentation separates the image into two classes: foreground and background

Example: Tumor vs non-tumor tissue segmentation. 

### Multi-Class Segmentation (3-Class or More)

Multi-class segmentation assigns each pixel to one of multiple categories.

Example: Tumor, non-tumor, and background

This enables detailed tissue or object-level understanding within complex images.


| Task | Description | Example |
|---|---|---|
| **Binary (2-class)** | Foreground vs. background | Tumor vs. non-tumor |
| **Multi-class (≥3)** | Multiple tissue or object categories | Tumor · non-tumor · background |

---

## Signet Ring Cell Region Detection

The following images show semantic segmentation for signet ring cell (SRC) carcinoma region detection from histopathology whole-slide images. Signet ring cells are characterized by mucin-filled cytoplasm and displaced nuclei, making precise localization important for pathological assessment and disease characterization.

The segmentation framework enables pixel-level identification of SRC regions using annotated binary masks and deep learning-based semantic segmentation models.

## Framework Pipeline

### Input
- RGB image size: `512 × 512`
- Ground truth annotation:
  - Binary mask
  - Pixel-wise segmentation mask
  - Mask pixel values: integer class indices (`0`, `1`, `2`, ...)
 
## Input

| RGB Image | Ground Truth Mask |
|:---------:|:-----------------:|
| ![Input image](images/Ground_truth_1.png) | ![Ground truth mask](images/Mask_1.png) |
| ![Input image](images/Ground_truth_2.png) | ![Ground truth mask](images/Mask_2.png) | 
 
### Output
- Predicted segmentation mask
- Pixel-level classification map

| Predicted Segmentation Mask |
|:--------------------------:|
| ![Predicted output](images/predictions_grid.png) |

--- 

## Implemented Models

| # | Architecture | Year | Type | Notes |
|---|---|---|---|---|
| 1 | **U-Net** | 2015 | CNN | Classic encoder-decoder with skip connections |
| 2 | **SegNet** | 2015 | CNN | Encoder-decoder with max-pooling indices |
| 3 | **Attention U-Net** | 2018 | CNN + Attention | Gated attention on skip connections |
| 4 | **UNet++** | 2018 | CNN | Nested dense skip connections |
| 5 | **nnU-Net** | 2021 | CNN | Self-configuring baseline |
| 6 | **SegFormer** | 2021 | Transformer | Hierarchical ViT with lightweight MLP decoder |
| 7 | **Swin-UNet** | 2021 | Transformer | Pure Swin Transformer U-shaped network |
| 8 | **TransUNet** | 2024 | Hybrid | CNN encoder + Transformer bottleneck |
| 9 | **SAM2-UNet** | 2024 | Foundation Model | Frozen ViT-B/16 encoder + adapters + UNet decoder |
| 10 | **KongNet** | 2025 | CNN + Attention | SCSE attention; 1st place MIDOG/PUMA/MONKEY 2025 |
| 11 | **MCP-MedSAM** | 2025 | Lightweight ViT | Lightweight SAM variant; single-GPU trainable |
| 12 | **Swin-UMamba** | 2025 | Hybrid Mamba | Swin Transformer encoder + Mamba VSS decoder |
| 13 | **SAMba-UNet** | 2026 | Hybrid Mamba | Frozen ViT encoder + Mamba bottleneck + conv decoder |

---

## Key Features

- ✅ Binary and multi-class segmentation
- ✅ CNN and Transformer architecture support
- ✅ Modular plug-and-play model registry
- ✅ YAML-driven configuration - no code changes to switch models
- ✅ Joint image + mask augmentation pipeline
- ✅ Comprehensive pixel-wise evaluation metrics (IoU, Dice, Pixel Accuracy)
- ✅ K-fold cross-validation with structured logging
- ✅ GPU training with checkpoint save/resume
- ✅ Inference mode (no ground-truth required)

---

## Applications

- Medical image segmentation
- Histopathology analysis
- Tumor region detection
- Organ segmentation
- Biomedical image analysis
- General computer vision tasks

--- 

## Project Structure

```
seg_framework/
├── configs/                        ← Per-model YAML configurations
│   ├── unet.yaml
│   └── segformer.yaml
│
├── datasets/
│   ├── images/                     ← Input images (.jpg .png .tif)
│   └── ground_truths/              ← Masks with matching filenames (integer class indices)
│
├── modules/
│   ├── __init__.py                 ← MODEL_REGISTRY + get_model()
│   ├── unet/
│   │   └── model.py
│   └── segformer/
│       └── model.py
│
├── utils/
│   ├── augmentations.py            ← Joint image+mask augmentations
│   ├── config.py                   ← load_config(), ConfigDict, validation
│   ├── dataset.py                  ← SegmentationDataset + build_dataloaders()
│   ├── logger.py                   ← Console/file logger + CSVLogger
│   ├── metrics.py                  ← IoU, Dice, Pixel Accuracy, MetricTracker
│   └── train_utils.py              ← Loss functions and training utilities
│
├── logs/
│   └── <model>_<dataset>/
│       └── <timestamp>/
│           ├── fold_1/
│           │   ├── checkpoints/
│           │   ├── metrics.csv
│           │   ├── model_fold1.log
│           │   └── test_results.csv
│           ├── fold_2/ … fold_5/
│           └── summary.csv         ← Per-epoch metrics across all folds
│
├── train.py
├── test.py
├── infer.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or  

```bash 
conda env create -f environment.yml
conda activate seg_framework
```

### 2. Organise Your Data

```
datasets/images/          →  image001.png  image002.png  ...
datasets/ground_truths/   →  image001.png  image002.png  ...
```

> ⚠️ Mask filenames must match their corresponding image filenames exactly.  
> Pixel values should be integer class indices: `0, 1, 2, ...`

### 3. Configure

Edit `configs/unet.yaml`. Key fields:

| Field | Description |
|---|---|
| `model.n_classes` | Number of segmentation classes |
| `model.n_channels` | Input channels (`3` = RGB) |
| `training.loss` | `cross_entropy`, `dice`, or `dice_ce` |
| `training.learning_rate` | Initial learning rate |
| `dataset.augment` | `true` to enable joint augmentations |
| `logging.log_dir` | Directory for logs and checkpoints |

### 4. Train

```bash
# Train with U-Net
python train.py --config configs/unet.yaml

# Train with SegFormer
python train.py --config configs/segformer.yaml

# Resume from a checkpoint
python train.py --config configs/unet.yaml \
    --resume logs/unet_dataset/checkpoints/best_model.pth

# Specify GPU device
python train.py --config configs/unet.yaml --device cuda:1
```

### 5. Evaluate

```bash
# Standard evaluation (Pixel Accuracy, Mean IoU, Dice)
python test.py --config configs/unet.yaml \
    --checkpoint logs/unet_dataset/checkpoints/best_model.pth

# Save predicted mask PNGs
python test.py --config configs/unet.yaml \
    --checkpoint logs/unet_dataset/checkpoints/best_model.pth \
    --save_preds --output_dir outputs/predictions

# Inference only — no ground-truth needed
python test.py --config configs/unet.yaml \
    --checkpoint logs/unet_dataset/checkpoints/best_model.pth \
    --images_dir /path/to/test/images \
    --save_preds
```

---

## Configuration Reference

### Loss Functions

| Value | Description | Best For |
|---|---|---|
| `cross_entropy` | Standard pixel-wise cross-entropy | Balanced classes |
| `dice` | Soft Dice loss | Small structures |
| `dice_ce` | Dice + Cross-Entropy combined | **Recommended** for imbalanced classes |

### Optimizers

| Value | Notes |
|---|---|
| `adam` | Default; reliable general-purpose choice |
| `adamw` | Adam with decoupled weight decay; good for Transformers |
| `sgd` | Requires `optimizer.momentum`; often better final accuracy |

### Schedulers

| Value | Notes |
|---|---|
| `cosine` | `CosineAnnealingLR` — use with Adam/AdamW |
| `step` | `StepLR` — configure `step_size` and `gamma` |
| `plateau` | `ReduceLROnPlateau` — good with SGD |

---

## Augmentations

Enable per-config with `dataset.augment: true`. Applied to the **training split only**.

| Transform | Default Probability |
|---|---|
| Horizontal flip | `p = 0.5` |
| Vertical flip | `p = 0.3` |
| Random rotation ±15° | `p = 0.4` |
| Random crop + resize | `scale = (0.75, 1.0)`, `p = 0.4` |
| Color jitter (brightness/contrast/saturation/hue) | configurable |
| Gaussian blur | `radius = 1.0`, `p = 0.2` |

> Custom augmentations can be added to `utils/augmentations.py` by subclassing `JointTransform`.

---

## Adding a New Model

Only **6 steps** — `train.py` and `test.py` require **zero modifications**.

```
Step 1  modules/<model>/<model>_model.py    →  Define MyModel(nn.Module)
Step 2  modules/<model>/<model>_parts.py   →  Building blocks (optional)
Step 3  modules/__init__.py                →  Add "mymodel": MyModel to MODEL_REGISTRY
Step 4  process/<model>/<model>.py         →  class MyModelProcess(BaseProcess): pass
Step 5  process/__init__.py                →  Add "mymodel": MyModelProcess to PROCESS_REGISTRY
Step 6  configs/<model>.yaml              →  Copy unet.yaml, set model.name: mymodel
```

Then run:

```bash
python train.py --config configs/mymodel.yaml
```

---

## Output Files

| File | Description |
|---|---|
| `logs/.../metrics.csv` | Epoch-level train/val metrics — ready to plot |
| `logs/.../best_model.pth` | Best checkpoint by validation loss |
| `logs/.../*.log` | Timestamped training log |
| `logs/.../summary.csv` | Per-epoch metrics aggregated across folds |
| `outputs/predictions/*.png` | Predicted mask PNGs (if `--save_preds`) |

---

# K-fold validation split creation

Utilities for split generation and data augmentation, located in `utils/`.

---

1. [Split Generation — Random](#1-split-generation--random)
2. [Split Generation — Patient-Level Stratified](#2-split-generation--patient-level-stratified)
3. [Data Augmentation — Copy-Paste](#3-data-augmentation--copy-paste)
4. [Output Structure](#4-output-structure)

---

## 1. Split Generation — Random

**`utils/split_generator_random.py`**

Scans an image folder, collects all image filenames, and writes per-fold CSV split files with column names train, val, and test based on the split type. All models train on identical samples across folds.

### Usage

```bash
# 5-fold cross-validation (all 4 eval modes)
python utils/split_generator_random.py \
    --images_dir  /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/1.dataset/patches \
    --output_dir  splits/dataset_original \
    --dataset_name SRC \
    --folds       5 \
    --val_split   0.1 \
    --test_split  0.2 \
    --seed        42

# Single fold, specific mode only
python utils/split_generator_random.py \
    --images_dir  /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/1.dataset/patches \
    --output_dir  splits/dataset_original \
    --dataset_name SRC \
    --folds       1 \
    --modes       train_val_test or train_val
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--images_dir` | required | Folder containing input image patches |
| `--output_dir` | required | Root folder where CSV splits are written |
| `--dataset_name` | required | Label used in output filenames (e.g. `SRC`) |
| `--folds` | `5` | Number of folds (`1` = single random split) |
| `--val_split` | `0.1` | Validation fraction (used when `folds=1`) |
| `--test_split` | `0.2` | Test holdout fraction |
| `--seed` | `42` | Random seed for reproducibility |
| `--modes` | all 4 | Subset of eval modes to generate |

---

## 2. Split Generation — Patient-Level Stratified

**`utils/split_generator_patient_level_stratified.py`**

Same interface as the random generator but splits at the **slide level** — all tiles from the same slide always go to the same split, guaranteeing zero data leakage.

### Filename pattern required (depends on the target file type, given is an example)

```
{slide_name}_x{XXXXXX}_y{YYYYYY}.ext

Examples:
  S08-38628 G11_x067376_y034486.png       →  slide: S08-38628 G11
  SP-22-078912 A11-1_x012345_y067890.png  →  slide: SP-22-078912 A11-1
```

### Usage

```bash
python utils/split_generator_patient_level_stratified.py \
    --images_dir  /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/1.dataset/patches \
    --output_dir  splits/dataset_original \
    --dataset_name SRC \
    --folds       5 \
    --val_split   0.1 \
    --test_split  0.2 \
    --seed        42
```

Arguments are identical to the random generator above.

> **Use this generator** when your dataset contains multiple tiles per slide (WSI patches). Use the random generator only when each image is an independent sample.

---

## 3. Data Augmentation — Copy-Paste

**`utils/data_augmentation_copy_paste.py`**

Augments mask–patch pairs by copying white-region tiles (foreground) into black-region areas (background) within the same image, to address class imbalance when masks contain >60% background.

### Usage

```bash
python utils/data_augmentation_copy_paste.py \
    --masks_dir       /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/training/masks \
    --patches_dir     /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/training/patches \
    --output_dir      /data_64T_3/Raja/CDH1/src_segmentation/dataset_augmentation/augmented_copy_paste \
    --tile_size       64 \
    --black_thresh    50 \
    --apply_pct       100 \
    --copies_per_mask 2 \
    --seed            42
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--masks_dir` | required | Folder containing binary mask images |
| `--patches_dir` | required | Folder with corresponding input patches |
| `--output_dir` | required | Destination for augmented mask + patch pairs |
| `--tile_size` | `64` | Size of the n×n copy tile in pixels |
| `--black_thresh` | `60` | Min % of black pixels in mask to trigger augmentation |
| `--apply_pct` | `100` | % of qualifying masks to augment (0–100) |
| `--copies_per_mask` | `1` | Max white→black copy operations per mask |
| `--seed` | `42` | Random seed for reproducibility |

### How it works

1. For each mask with `> black_thresh %` black pixels, finds white-region tiles (foreground source) and black-region locations (paste destinations)
2. Copies the white tile and its corresponding patch region into the black area
3. Tracks source and destination positions on separate occupancy grids to guarantee non-overlapping placements across all copies
4. Saves augmented mask and patch pair to `--output_dir`

---

## 4. Output Structure

### Split files (assumed already created)

```
splits/
├── dataset_summary_SRC.csv                          ← slide inventory (stratified only)
├── split_train_val_test/
│   ├── split_train_val_test_SRC_fold1.csv           ← columns: train, val, test
│   ├── split_train_val_test_SRC_fold2.csv
│   ├── split_train_val_test_SRC_fold3.csv
│   ├── split_train_val_test_SRC_fold4.csv
│   └── split_train_val_test_SRC_fold5.csv
├── split_train_val/
│   └── split_train_val_SRC_fold{1..5}.csv           ← columns: train, val
├── split_train_test/
│   └── split_train_test_SRC_fold{1..5}.csv          ← columns: train, test
└── split_training_only/
    └── split_training_only_SRC_fold{1..5}.csv       ← column: train
```

Each CSV has one column per split. Rows are image filenames. Shorter columns are padded with `""` so all columns have equal length.

### Augmented data

```
augmented_copy_paste/
├── masks/
│   ├── original_patch_aug1.png
│   └── ...
└── patches/
    ├── original_patch_aug1.png
    └── ...
```

Augmented files are saved alongside original filenames with an `_aug{N}` suffix. Pass the augmented folder path to `--images_dir` in the split generator to include augmented samples in training splits.

---

## Quick Reference

```bash
# Step 1 — Generate splits (patient-level, recommended for WSI tiles)
python utils/split_generator_patient_level_stratified.py \
    --images_dir /path/to/patches --output_dir splits/ \
    --dataset_name SRC --folds 5 --seed 42

# Step 2 — Augment training data (optional)
python utils/data_augmentation_copy_paste.py \
    --masks_dir /path/to/masks --patches_dir /path/to/patches \
    --output_dir /path/to/augmented --copies_per_mask 2

# Step 3 — Train any model using the generated splits
python train.py --config configs/unet.yaml \
    --splits_dir splits/split_train_val_test \
    --output_dir Results --device cuda:0
```


## Applications

- 🧬 Histopathology tissue analysis
- 🔬 Tumor region detection and delineation
- 🫁 Organ segmentation in medical imaging
- 🧫 Biomedical image analysis
- 🖼️ General-purpose computer vision segmentation

--- 
