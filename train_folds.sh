#!/bin/bash
# Script to train MHSA-cGAN (Swin) for remaining folds
# Usage: bash train_folds.sh

DATASET_PATH="/Users/mehmetgokalpkoreken/Desktop/thesis123/msc_thesis_work/brain"
RESULT_NAME="swin_brain_slice_v1"

for fold in 0 1 2 4
do
  echo "========================================"
  echo "Starting training for Fold $fold"
  echo "========================================"
  
  /Users/mehmetgokalpkoreken/Desktop/thesis123/msc_thesis_work/venv/bin/python -m models.cgan_model \
    --dataset_path "$DATASET_PATH" \
    --dataset_version 23 \
    --region_filter B \
    --fold $fold \
    --result "$RESULT_NAME" \
    --n_epochs 100 \
    --batch_size_train 32 \
    --use_swin True
    
  echo "Finished training for Fold $fold"
done
