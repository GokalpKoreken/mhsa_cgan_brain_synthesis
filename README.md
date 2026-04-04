# Enhanced U-Net Architecture with Multi-Head Self-Attention (MHSA) Blocks for Brain MRI-to-CT Synthesis

This repository contains the PyTorch implementation of the **MHSA-enhanced U-Net cGAN**, proposed for highly accurate brain MRI-to-CT image synthesis. The architecture integrates Multi-Head Self-Attention (MHSA) blocks into the deep bottlenecks ($8\times8$, $4\times4$, $2\times2$) of a U-Net Generator to capture long-range anatomical dependencies (like the brain-skull interface) without the prohibitive $\mathcal{O}(N^2)$ computational cost of full-resolution vision transformers.

## Quantitative Results (Ensemble 5-Fold Cross Validation)
Evaluated on the brain region dataset of the **SynthRAD2023** grand challenge.
*   **Mean Absolute Error (MAE):** 132.1 ± 16.7 HU
*   **Structural Similarity Index (SSIM):** 0.88 ± 0.01
*   **Peak Signal-to-Noise Ratio (PSNR):** 20.4 ± 0.4 dB

## Prerequisites & Environment Setup

The codebase is built on Python 3.11+, PyTorch, and MONAI. 
Ensure you have the required dependencies installed:

```bash
pip install torch torchvision torchaudio
pip install monai numpy scipy scikit-learn pydicom click timm
```

## Dataset Structure
The dataloader expects the SynthRAD2023 dataset structure. The `DATASET_PATH` should point to a directory containing:
*   `brain/`: Brain MRI/CT paired volumes.

## Training the Model (5-Fold Cross Validation)

The training pipeline uses a 5-fold stratified cross-validation strategy. We provide a shell script that sequentially handles the training across all 5 folds.

### 1. Launching the Training

Navigate to the repository and execute the bash script. Ensure you update the `DATASET_PATH` inside the script to point to your local SynthRAD2023 dataset directory.

```bash
bash train_folds.sh
```

Alternatively, you can manually trigger training for a specific fold (e.g., Fold 0):

```bash
python -m models.cgan_model \
  --dataset_path "/path/to/dataset" \
  --dataset_version 23 \
  --region_filter B \
  --fold 0 \
  --result "mhsa_brain_synthesis_v1" \
  --n_epochs 100 \
  --batch_size_train 32 \
  --use_swin True
```
> **Note on Naming:** The flag `--use_swin True` activates the inclusion of our specialized deep-bottleneck attention blocks (the MHSA blocks described in the paper).

### 2. Checkpoints
Checkpoints for each fold will be saved under:
`results/SynthRAD2023/cGAN/validations/saved_models/SynthRAD2023/cGAN/mhsa_brain_synthesis_v1/fold_{X}`

## Inference and Ensemble Testing

Once all 5 folds are trained, you can evaluate the ensemble performance. The `test_ensemble.py` script loads the best weights from all 5 folds, generates synthetic CTs for the hold-out test set, averages the predictions (ensemble), and calculates the final MAE, PSNR, and SSIM metrics.

```bash
python test_ensemble.py \
  --dataset_path "/path/to/dataset" \
  --dataset_version 23 \
  --region_filter B \
  --result "mhsa_brain_synthesis_v1" \
  --use_swin True
```

This script will output the structural metrics evaluated specifically within the patient anatomy mask, excluding the surrounding air, to provide a clinically honest structural evaluation.

## Acknowledgements
Data used in this repository is provided by the [SynthRAD2023 Grand Challenge](https://synthrad2023.grand-challenge.org/).
