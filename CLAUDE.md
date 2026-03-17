# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Check Change Log First
Before starting any task, read the `.memory/` folder for recent change records. Files are named `YYYY-MM-DD_description.md` and contain context on major changes made to this project.

## Project Overview

Fork of nnU-Net v2 for brain aneurysm detection and segmentation from MRA (TOF-MRA) images. All comments and documentation are in Traditional Chinese. The project uses two datasets:
- **Dataset550_MRA_Vessel**: 4-class vessel segmentation
- **Dataset127_DeepAneurysm**: Aneurysm detection/segmentation (uses vessel masks to constrain training region)

## Repository Structure

The repo contains multiple nnU-Net variants as subdirectories. The **active development variant** is `nnResUNet-location-cls-upsample/`. Other directories (`nnResUNet/`, `nnResUNet-classifier/`, `nnResUNet-upsample/`) are earlier versions.

Root-level scripts (`pipeline_aneurysm_torch.py`, `pipline_aneurysm.ipynb`, `gpu_aneurysm.py`, `make_normalized_img.py`) handle the full inference pipeline and preprocessing.

## Setup and Commands

### Installation
```bash
conda create -n aneurysm python=3.10 && conda activate aneurysm
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd nnResUNet-location-cls-upsample && pip install -e .
```

### Required Environment Variables
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Key CLI Commands
```bash
# Preprocessing
nnUNetv2_plan_and_preprocess -d DATASET_ID -c 3d_fullres -np 16

# Training (single fold)
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p PLANS_NAME \
  --num_iterations_per_epoch 500 --enable_early_stopping

# Prediction
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d DATASET_ID -c 3d_fullres -f FOLD

# Resume training
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p PLANS_NAME --c

# Validation only
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p PLANS_NAME --val
```

### Tests
Integration tests only (no pytest framework). Located at `nnResUNet-location-cls-upsample/nnunetv2/tests/integration_tests/`. Run as standalone scripts.

## Architecture: Key Customizations from Standard nnU-Net

### Custom Network Architectures (`nnunetv2/utilities/unet_v2.py`)
Six custom architectures adding classifier heads to the base ResidualEncoderUNet:
- `ResidualEncoderUNetClassifier` / `ResidualEncoderUNetClassifier2D`
- `ResidualEncoderUNetAttentionClassifier` / `ResidualEncoderUNetAttentionClassifier2D`
- `ResidualEncoderUNetGuidedClassifier` / `ResidualEncoderUNetGuidedClassifier2D`

These are registered in the `mapping` dict in `get_network_from_plans.py`. The architecture is selected by setting `UNet_class_name` in the plans JSON file.

### Category-Based Weighted Sampling (`nnunetv2/training/dataloading/utils.py`)
`build_sampling_probabilities()` weights case sampling by category (1-4) from `splits_final.json`. Two modes:
- `"multiplier"`: direct weight multiplication per case
- `"target_proportion"`: auto-adjusts weights by category count for balanced representation

Configured via CLI: `--sampling_category_weights "2:1:1:1" --sampling_category_weight_mode target_proportion --enable_sampling_weights`

### Enhanced Trainer (`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`)
Custom additions to the standard trainer:
- AdamW optimizer (default, replaces SGD) + CosineAnnealingLR scheduler
- Early stopping with configurable patience/min_delta
- MLflow experiment tracking integration
- All hyperparameters configurable via CLI args in `run/run_training.py`

### CLI Parameter Parsing (`nnunetv2/run/run_training.py`)
`parse_sampling_category_weights()` accepts multiple formats: ratio (`"2:1:1:1"`), key=value (`"1=2,2=1"`), or JSON (`'{"1": 2}'`).

### Data Flow
Plans JSON (`UNet_class_name`) → `get_network_from_plans.py` (mapping dict) → architecture class from `unet_v2.py`. The `splits_final.json` file contains both fold splits and `sampling_categories` for weighted sampling.

## Data Format
- NIfTI (.nii.gz) images following nnU-Net naming: `{CASE_ID}_0000.nii.gz` (image), `{CASE_ID}.nii.gz` (label)
- Dataset127 has additional `vesselsTr/` (vessel masks constraining sampling region) and `dilationsTr/` (dilated labels for sampling region expansion)
