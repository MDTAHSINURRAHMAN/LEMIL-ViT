# Kaggle Environment Setup Documentation

## Overview

This document details the setup and configuration of the Kaggle environment for the LeMiL-ViT project. The setup is automated through a Jupyter notebook (`kaggle_setup.ipynb`) that ensures reproducibility and proper configuration of all necessary components.

## Environment Components

### 1. Dependencies Installation

The notebook automatically installs and verifies the following required packages:

- PyTorch and torchvision for deep learning
- timm for Vision Transformer models
- transformers for transformer-based architectures
- pandas for data manipulation
- scikit-learn for metrics and evaluation
- matplotlib for visualization
- Pillow for image processing

### 2. GPU Configuration

The setup includes comprehensive GPU configuration:

- Automatic detection of CUDA availability
- GPU device information display (name and count)
- Device selection (CUDA if available, CPU fallback)
- CUDA settings verification

### 3. Random Seed Configuration

For reproducibility, the notebook sets consistent random seeds across:

- Python's random module
- NumPy
- PyTorch (both CPU and CUDA operations)
- CUDA backend configurations (deterministic mode)
  Fixed seed value: 42

### 4. Directory Structure

The notebook establishes and verifies the following directory structure:

```
/kaggle/input/lemit-vit/
├── train_small/
│   └── train_small/          # Contains training images (3,757 images)
└── val_small/
    └── val_small/            # Contains validation images (9,528 images)
```

### 5. Dataset Verification

The setup performs comprehensive dataset verification:

- Validates existence of training and validation directories
- Counts and verifies image files in each split
- Compares against expected counts:
  - Training set: 3,757 images
  - Validation set: 9,528 images
- Displays sample image names for verification
- Checks directory structure integrity

### 6. Output Directory

Establishes working directory for outputs:

- Location: `/kaggle/working/lemit-vit`
- Automatically created if not present
- Used for storing model checkpoints, logs, and results

## Usage Instructions

1. **Dataset Upload**:

   - Upload the dataset to Kaggle datasets
   - Structure should match the specified directory layout
   - Ensure all PNG images are present in respective directories

2. **Notebook Setup**:

   - Import the `kaggle_setup.ipynb` notebook
   - Run all cells in sequence
   - Verify successful completion of all checks

3. **Verification**:
   - Confirm GPU availability status
   - Check dataset structure verification results
   - Ensure all package installations are successful

## Important Considerations

1. **GPU Requirements**:

   - The notebook automatically adapts to available GPU resources
   - Works on both GPU and CPU environments
   - Optimized for Kaggle's P100/T4 GPU instances

2. **Data Consistency**:

   - Verifies exact image counts for data integrity
   - Reports any mismatches or missing files
   - Ensures dataset completeness before training

3. **Reproducibility**:
   - Fixed random seeds ensure consistent results
   - Deterministic CUDA operations when possible
   - Controlled environment for research validation

## Troubleshooting

Common issues and solutions:

1. **Missing Images**:

   - Verify dataset upload completeness
   - Check directory structure matches specification
   - Ensure all PNG files are properly transferred

2. **GPU Issues**:

   - Confirm Kaggle GPU runtime is selected
   - Check CUDA availability in output logs
   - Verify GPU acceleration is enabled in notebook settings

3. **Package Installation**:
   - Internet connectivity required for first run
   - Check Kaggle package versions if conflicts occur
   - Monitor installation logs for any errors

## Version Information

- Initial Release: October 2025
- Tested on Kaggle Notebooks Platform
- Compatible with Python 3.7+
- Verified with PyTorch 2.0+
