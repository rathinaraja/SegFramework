#!/bin/bash
# nnunet_train.sh
# ----------------
# Trains all 5 folds sequentially on a single GPU.
# Edit DATASET_ID, CONFIG, and CUDA_DEVICE as needed.
#
# Usage:
#   chmod +x nnunet_train.sh
#   ./nnunet_train.sh
#
# To run folds in parallel across multiple GPUs:
#   CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 101 2d 0 --npz &
#   CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 101 2d 1 --npz &
#   ...

DATASET_ID=101
CONFIG=2d          # 2d | 3d_fullres | 3d_lowres
CUDA_DEVICE=1      # GPU index

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "======================================================"
echo "Training Dataset $DATASET_ID | Config: $CONFIG | GPU: $CUDA_DEVICE"
echo "======================================================"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "------------------------------------------------------"
    echo "Training Fold $FOLD  ($(date))"
    echo "------------------------------------------------------"
    nnUNetv2_train $DATASET_ID $CONFIG $FOLD --npz
done

echo ""
echo "======================================================"
echo "All 5 folds complete. ($(date))"
echo "======================================================"
echo ""
echo "Next step: run nnunet_evaluate.py"
