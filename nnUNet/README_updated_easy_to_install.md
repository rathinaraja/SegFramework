# nnU-Net — Setup and Implementation Guide

Binary semantic segmentation of H&E histopathology patches using nnU-Net v2 with pre-defined 5-fold cross-validation splits.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Environment Variables](#2-environment-variables)
3. [Dataset Preparation — Part 1](#3-dataset-preparation--part-1)
4. [Dataset Preparation — Part 2 (Fold Splits)](#4-dataset-preparation--part-2-fold-splits)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Output Structure](#7-output-structure)

---

## 1. Installation

```bash
conda create -n nnUNet python=3.10
conda activate nnUNet
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

---

## 2. Environment Variables

nnU-Net requires three environment variables set before any command. Add to `~/.bashrc`:

```bash
export nnUNet_raw="/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_preprocessed"
export nnUNet_results="/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results"

mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
source ~/.bashrc
conda activate nnUNet
```

This creates:

```
/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/
├── nnUNet_preprocessed/
├── nnUNet_raw/
└── nnUNet_results/
```

> ⚠️ **Important:** The dataset ID (`101`) and name (`SRC`) in `Dataset101_SRC` must be consistent across all scripts and commands. Edit paths carefully wherever this appears.

---

## 3. Dataset Preparation — Part 1

### Input data layout

```
/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/
├── training/
│   ├── patches/    (440 images — used for train + val)
│   └── masks/      (440 images — used for train + val)
└── test/
    ├── patches/    (86 images — used for inference)
    └── masks/      (86 images — used for evaluation after training)
```

### Configure `prepare_dataset.py`

Set the following paths inside `prepare_dataset.py`:

```python
TRAIN_PATCHES_DIR = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/training/patches"
TRAIN_MASKS_DIR   = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/training/masks"
TEST_PATCHES_DIR  = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/patches"
OUT_ROOT          = Path("/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_raw/Dataset101_SRC")
```

### Run

```bash
python prepare_dataset.py
```

### Output

```
nnUNet_raw/Dataset101_SRC/
├── dataset.json
├── imagesTr/              ← from training/patches  (named {case_id}_0000.png)
│   ├── case_001_0000.png
│   └── ...
├── imagesTs/              ← from test/patches       (named {case_id}_0000.png)
│   ├── case_001_0000.png
│   └── ...
└── labelsTr/              ← from training/masks     (named {case_id}.png, values 0/1)
    ├── case_001.png
    └── ...
```

---

## 4. Dataset Preparation — Part 2 (Fold Splits)

Associates the pre-defined 5-fold CSV splits with the nnU-Net dataset.

### Fold split files

```
/home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/
├── split_train_val_test_SRC_fold1.csv
├── split_train_val_test_SRC_fold2.csv
├── split_train_val_test_SRC_fold3.csv
├── split_train_val_test_SRC_fold4.csv
└── split_train_val_test_SRC_fold5.csv
```

Each CSV has columns: `train`, `val`, `test` with image filenames.

### Run

```bash
python nnunet_prepare.py \
    --patches_dir  /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset/patches \
    --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset/masks \
    --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test \
    --dataset_id   101 \
    --dataset_name SRC
```

### Fix dataset channels and verify integrity

```bash
# Fix dataset.json channel config (R/G/B) and numTraining count
python fix.py

# Verify dataset integrity — checks image count, channel consistency, mask values
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity

# Copy custom fold splits to preprocessed folder
cp $nnUNet_raw/Dataset101_SRC/splits_final.json \
   $nnUNet_preprocessed/Dataset101_SRC/splits_final.json
```

### Output after preparation

```
nnUNet_raw/Dataset101_SRC/
├── imagesTr/
├── imagesTs/
├── labelsTr/
├── dataset.json
├── splits_final.json       ← fold train/val assignments for nnUNet trainer
└── test_splits.json        ← fold test case IDs for evaluation script

nnUNet_preprocessed/Dataset101_SRC/
├── gt_segmentations/
├── nnUNetPlans_2d/
├── dataset.json
├── dataset_fingerprint.json
├── nnUNetPlans.json
└── splits_final.json       ← copied from nnUNet_raw
```

---

## 5. Training

### Option A — Sequential (single GPU)

```bash
chmod +x nnunet_train.sh
./nnunet_train.sh
```

### Option B — Parallel (multi-GPU)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 0 --npz > fold0.log &
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 1 --npz > fold1.log &
CUDA_VISIBLE_DEVICES=2 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 2 --npz > fold2.log &
CUDA_VISIBLE_DEVICES=3 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 3 --npz > fold3.log &
CUDA_VISIBLE_DEVICES=4 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 4 --npz > fold4.log &
wait
```

### CUDA / driver error fix

If you get a CUDA driver version mismatch error:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Training output

```
nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/
├── fold_0/
│   ├── checkpoint_best.pth
│   ├── checkpoint_final.pth
│   ├── progress.png
│   ├── training_log_*.txt
│   └── validation/
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

---

## 6. Evaluation

Runs inference on the test set for each fold separately, computes metrics, saves visualisations, and appends results to `summary.csv`.

> **Note:** `--fold` uses 0-based nnUNet indexing (`fold0`–`fold4`). `--splits_dir` points to the corresponding 1-based CSV file (`fold1.csv`–`fold5.csv`).

```bash
python nnunet_evaluate.py \
 --dataset_id   101 \
 --dataset_name SRC \
 --fold         fold0 \
 --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold1.csv \
 --output_dir   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device       cuda:1 \
 --checkpoint   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth

python nnunet_evaluate.py \
 --dataset_id   101 \
 --dataset_name SRC \
 --fold         fold1 \
 --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold2.csv \
 --output_dir   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device       cuda:1 \
 --checkpoint   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_1/checkpoint_best.pth

python nnunet_evaluate.py \
 --dataset_id   101 \
 --dataset_name SRC \
 --fold         fold2 \
 --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold3.csv \
 --output_dir   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device       cuda:1 \
 --checkpoint   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_2/checkpoint_best.pth

python nnunet_evaluate.py \
 --dataset_id   101 \
 --dataset_name SRC \
 --fold         fold3 \
 --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold4.csv \
 --output_dir   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device       cuda:1 \
 --checkpoint   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_3/checkpoint_best.pth

python nnunet_evaluate.py \
 --dataset_id   101 \
 --dataset_name SRC \
 --fold         fold4 \
 --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold5.csv \
 --output_dir   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device       cuda:1 \
 --checkpoint   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_4/checkpoint_best.pth
```

### `summary.csv` format

Results are appended per fold. After all 5 folds, run the mean/std script to add summary rows.

```
fold  seed  train_loss  train_acc  train_iou  train_dice  val_loss  val_acc  val_iou  val_dice  test_loss  test_acc  test_iou  test_dice
fold0  42   0.4046      0.9017     0.7883     0.8736      0.4605    0.8807   0.7527   0.8503    0.4179     0.9008    0.7769    0.8667
fold1  42   0.8333      0.7658     0.6054     0.7402      0.9958    0.7006   0.4514   0.5651    0.6863     0.8320    0.6630    0.7822
fold2  42   0.2892      0.9318     0.8469     0.9129      0.5203    0.8695   0.7291   0.8350    0.4937     0.8749    0.7308    0.8341
fold3  42   0.4169      0.8979     0.7890     0.8764      0.5315    0.8734   0.7115   0.8134    0.5282     0.8755    0.7230    0.8241
fold4  42   0.3947      0.9035     0.7950     0.8790      0.4524    0.8925   0.7830   0.8748    0.4968     0.8779    0.7381    0.8394
mean   -    0.4677      0.8801     0.7649     0.8564      0.5921    0.8433   0.6856   0.7877    0.5246     0.8723    0.7264    0.8293
std    -    0.1884      0.0584     0.0827     0.0599      0.2043    0.0718   0.1195   0.1131    0.0886     0.0223    0.0367    0.0275
```

---

## 7. Output Structure

Full directory tree after all steps complete:

```
/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/
│
├── nnUNet_raw/
│   └── Dataset101_SRC/
│       ├── imagesTr/              ← training patches  ({case}_0000.png)
│       ├── imagesTs/              ← test patches      ({case}_0000.png)
│       ├── labelsTr/              ← training masks    ({case}.png, 0/1 pixels)
│       ├── dataset.json
│       ├── splits_final.json      ← fold train/val split for nnUNet
│       └── test_splits.json       ← fold test case IDs for evaluation
│
├── nnUNet_preprocessed/
│   └── Dataset101_SRC/
│       ├── gt_segmentations/      ← ground truth masks (preprocessed)
│       ├── nnUNetPlans_2d/        ← preprocessed .b2nd + .pkl files
│       ├── dataset.json
│       ├── dataset_fingerprint.json
│       ├── nnUNetPlans.json
│       └── splits_final.json      ← copied from nnUNet_raw
│
├── nnUNet_results/
│   └── Dataset101_SRC/
│       └── nnUNetTrainer__nnUNetPlans__2d/
│           ├── fold_0/
│           │   ├── checkpoint_best.pth
│           │   ├── checkpoint_final.pth
│           │   ├── progress.png
│           │   ├── training_log_*.txt
│           │   └── validation/
│           │       ├── *.npz
│           │       ├── *.pkl
│           │       └── summary.json
│           ├── fold_1/
│           ├── fold_2/
│           ├── fold_3/
│           ├── fold_4/
│           ├── dataset.json
│           ├── dataset_fingerprint.json
│           └── plans.json
│
└── nnUNet_validation_results/     ← output of nnunet_evaluate.py
    ├── fold0/
    │   ├── checkpoints/
    │   │   ├── best_model.pth     ← copied from nnUNet_results
    │   │   └── last_model.pth
    │   ├── predictions/           ← predicted masks from nnUNetv2_predict
    │   │   ├── *.png
    │   │   ├── dataset.json
    │   │   ├── plans.json
    │   │   └── predict_from_raw_data_args.json
    │   ├── visualizations/        ← 5-panel: patch|GT|overlay_GT|pred|overlay_pred
    │   │   └── *.png
    │   └── metrics.csv            ← per-case IoU, Dice, Acc + mean row
    ├── fold1/
    ├── fold2/
    ├── fold3/
    ├── fold4/
    └── summary.csv                ← all folds + mean/std rows
```
