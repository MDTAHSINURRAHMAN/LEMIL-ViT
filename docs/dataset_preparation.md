# ChestX-ray14 Dataset Subset Creation Documentation

## Overview

This document details the process of creating a balanced subset of the ChestX-ray14 dataset while maintaining statistical properties and addressing class imbalance. The goal was to create a smaller, yet representative dataset suitable for initial experiments and model validation.

## Original Dataset

- **Source**: ChestX-ray14 (NIH Chest X-ray Dataset)
- **Total Images**: ~112,000 chest X-ray images
- **Disease Classes**: 14 thoracic disease classes
- **Label Type**: Multi-label (multiple diseases can be present in one image)

### Disease Classes

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## Methodology

### 1. Data Split Requirements

- **Test Split**: Preserved the official patient-wise test split unchanged
- **Validation Split**: ~10-12% of remaining patients
- **Training Split**: Target of 3.5k-4k images from remaining patients
- **Key Constraint**: No patient overlap between splits (patient-wise splitting)

### 2. Sampling Strategy

Implemented a two-phase iterative stratified sampling approach at the patient level:

#### Phase 1: Minimum Representation

- Prioritized diseases based on rarity
- Ensured minimum image counts per class:
  - Training: 40 images minimum per class
  - Validation: 8 images minimum per class
- Special handling for rare diseases (e.g., Hernia)

#### Phase 2: Balance Optimization

- Iteratively added patients while maintaining:
  - Class prevalence within ±20% of source distribution
  - Multi-label co-occurrence patterns
  - Overall size constraints

### 3. Implementation Details

- Used Python with scikit-learn and pandas
- Set random seed (42) for reproducibility
- Implemented custom metrics for co-occurrence drift (Jensen-Shannon divergence)

## Results

### 1. Dataset Statistics

| Split      | Images | Percentage |
| ---------- | ------ | ---------- |
| Test       | 25,596 | ~66%       |
| Validation | 9,528  | ~25%       |
| Training   | 3,757  | ~9%        |
| **Total**  | 38,881 | 100%       |

### 2. Class Distribution Analysis

#### Training Set (n=3,757)

| Disease            | Count | Prevalence | Rel. Diff |
| ------------------ | ----- | ---------- | --------- |
| Atelectasis        | 373   | 0.099      | +3.7%     |
| Cardiomegaly       | 77    | 0.020      | +3.9%     |
| Effusion           | 376   | 0.100      | +0.0%     |
| Infiltration       | 595   | 0.158      | -0.6%     |
| Mass               | 176   | 0.047      | +0.5%     |
| Nodule             | 203   | 0.054      | -0.7%     |
| Pneumonia          | 42    | 0.011      | +10.4%    |
| Pneumothorax       | 133   | 0.035      | +16.2%    |
| Consolidation      | 139   | 0.037      | +12.2%    |
| Edema              | 60    | 0.016      | +0.3%     |
| Emphysema          | 68    | 0.018      | +10.1%    |
| Fibrosis           | 65    | 0.017      | +19.7%    |
| Pleural Thickening | 115   | 0.031      | +18.1%    |
| Hernia             | 9     | 0.002      | +47.0%    |

#### Validation Set (n=9,528)

| Disease            | Count | Prevalence | Rel. Diff |
| ------------------ | ----- | ---------- | --------- |
| Atelectasis        | 913   | 0.096      | +0.1%     |
| Cardiomegaly       | 188   | 0.020      | +0.0%     |
| Effusion           | 972   | 0.102      | +1.9%     |
| Infiltration       | 1,525 | 0.160      | +0.5%     |
| Mass               | 443   | 0.046      | -0.3%     |
| Nodule             | 524   | 0.055      | +1.1%     |
| Pneumonia          | 116   | 0.012      | +20.3%    |
| Pneumothorax       | 324   | 0.034      | +11.6%    |
| Consolidation      | 365   | 0.038      | +16.2%    |
| Edema              | 154   | 0.016      | +1.5%     |
| Emphysema          | 188   | 0.020      | +20.0%    |
| Fibrosis           | 165   | 0.017      | +19.8%    |
| Pleural Thickening | 278   | 0.029      | +12.6%    |
| Hernia             | 19    | 0.002      | +22.4%    |

### 3. Quality Metrics

#### Class Balance

- Most classes maintained prevalence within ±20% of source distribution
- Exception: Hernia (rare class) with slightly higher deviation due to extreme rarity

#### Co-occurrence Preservation

Jensen-Shannon divergence from source distribution:

- Training set: 0.0877
- Validation set: 0.0520
  (Lower values indicate better preservation of label co-occurrence patterns)

## Output Files

Generated three CSV files with consistent format:

1. `test_official.csv`: Official test split
2. `val_small.csv`: Validation subset
3. `train_small.csv`: Training subset

### File Format

All CSVs contain three columns:

- Image Index
- Patient ID
- Finding Labels (pipe-separated list of diseases)

## Notes and Limitations

1. **Rare Disease Handling**: Hernia class has fewer than target minimum images due to its extreme rarity in the source dataset
2. **Patient-wise Splitting**: Strict adherence to patient-wise splitting may result in slightly imbalanced class distributions
3. **Co-occurrence Patterns**: While preserved overall, some minor variations in disease co-occurrence patterns exist

## Reproducibility

- Random seed set to 42
- All preprocessing steps documented in `create_subset.py`
- Environment requirements: Python with pandas, numpy, scikit-learn, and scipy
