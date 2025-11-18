# TBMF Phantom Dataset Statistical Analysis

Statistical analysis and retraining of the TBMF (Tensor-Based Multi-Parameter Fitting) model on phantom data. This repository contains the code for reproducing the model training and validation using phantom datasets, as opposed to the human data used in the original TBMF paper.

## Overview

This project focuses on statistical analysis and model retraining using **phantom data** rather than human/clinical data. The TBMF methodology has been applied to synthetic phantom datasets to:
- Validate model performance on controlled, reproducible data
- Conduct statistical analysis with known ground truth parameters
- Compare results with the original human-based TBMF paper

The use of phantom datasets allows for precise evaluation of model accuracy and reliability with full knowledge of underlying parameters.

## Dataset

- **Phantom Data**: Synthetic MRI phantoms with known parameter values (used in this analysis)
- **Original Paper**: Human/clinical data (TBMF reference work)

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Preprocess
cd src/preprocessing
python pre_process.py

# Train Model 2
cd src/training
python model2_example_training_fs_ksw.py

# Inference
cd src/inference
python analysis.py
```

## Project Structure
- `src/preprocessing/` - SAM segmentation & data prep
- `src/models/` - Vision Transformer architecture
- `src/training/` - Training scripts (Model 1 & 2)
- `src/inference/` - Inference & evaluation
- `src/utils/` - Helper functions
- `configs/` - Configuration files for training and model parameters
- `model_checkpoints/` - Trained model weights
- `phantoms_clean/` - Processed phantom dataset
- `predictions_model/` - Model inference results

## Models
- **Model 1**: Whole phantom reconstruction
- **Model 2**: Parameter-specific (pH/mM, T1/T2, ksw/fs, B0/B1)

## Methodology

The analysis employs Vision Transformer-based deep learning for MRI parameter mapping on phantom datasets. The training pipeline includes:
1. **Data Preprocessing**: SAM segmentation and phantom data preparation
2. **Model Training**: Supervised learning on phantom parameters with known ground truth
3. **Statistical Analysis**: Validation metrics (ICC, Pearson correlation)
4. **Inference & Evaluation**: Parameter estimation and performance assessment

This controlled environment enables rigorous statistical validation before application to clinical data.

## Performance
- ICC: 0.938-0.967
- Pearson r: 0.957-0.978

## License
MIT
