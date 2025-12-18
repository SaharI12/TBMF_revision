# Multi-Contrast Generation and Quantitative MRI using a Transformer-Based Framework with RF Excitation Embeddings

[![Nature Communications](https://img.shields.io/badge/Published-Nature_Communications-blue)](https://www-nature-com.bengurionu.idm.oclc.org/articles/s42003-025-09371-3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository contains the **revisions, validations, and analysis code** for the paper **"Multi-contrast generation and quantitative MRI using a transformer-based framework with RF excitation embeddings"** published in Nature Communications.

This is the **revision and validation repository** containing comprehensive testing, phantom model development, and statistical analysis that strengthened the methodology.

**Estimated Parameters:**
- KSSW (Steady-State Signal)
- MT% (Magnetization Transfer Percentage)
- B₀ (Main Magnetic Field)
- B₁ (Radiofrequency Field)
- T₁ (Spin-Lattice Relaxation)
- T₂ (Spin-Spin Relaxation)

## Repository Contents

This repository is organized into two main analysis components:

### 📁 `/human/` - Test-Retest Study & Human Data Validation

Contains test-retest analysis on human/clinical MRI data with comprehensive validation and state-of-the-art comparisons.

**Key Files:**
- `main.py` - Main test-retest pipeline and analysis
- `transformer_architecture_prod.py` - ViT model for inference
- `noise_addition.py` - AWGN robustness testing
- `attention_analsys.py` - Attention weight visualization
- `shap.py` - SHAP explainability analysis
- `ICC_compare_method.ipynb` - Statistical validation (ICC)
- `ICC_and_ccomperssin_wm_gm_gt.ipynb` - Tissue-specific analysis
- `images_and_box_plot_model_2.py` - Results visualization

See [`/human/README.md`](human/README.md) for detailed documentation.

### 📁 `/phantoms/` - Phantom Model Development & Validation

Contains complete phantom modeling pipeline with preprocessing, training, and comprehensive statistical validation on synthetic data with known ground truth.

**Key Directories:**
- `src/preprocessing/` - SAM segmentation and phantom data preparation
- `src/training/` - Training scripts (Model 1 & 2 variations)
- `src/inference/` - Inference and evaluation
- `src/models/` - Vision Transformer architecture
- `src/utils/` - Helper functions and utilities
- `configs/` - Configuration files
- `notebooks/` - Jupyter notebooks for analysis

**Performance Metrics:**
- ICC: 0.938-0.967
- Pearson r: 0.957-0.978

See [`/phantoms/README.md`](phantoms/README.md) for detailed documentation.

## My Contributions to This Paper

I performed all of the following:

### ✅ All Revisions
Complete methodology updates, refinements, and improvements to strengthen the paper.

### ✅ Test-Retest Study
Rigorous evaluation of model consistency and reliability across multiple acquisitions on human subject data.

### ✅ Comparison to State-of-the-Art Methods
Comprehensive benchmark comparison with existing quantitative MRI methodologies to validate performance advantages.

### ✅ Phantom Modeling (Complete Pipeline)
- Dataset preparation and synthetic phantom generation
- SAM-based automatic segmentation implementation
- Data preprocessing and normalization
- Model training on controlled, ground-truth phantom data

### ✅ Statistical Analysis
- Intraclass Correlation (ICC) reliability analysis
- Pearson correlation validation
- Tissue-specific performance analysis (white matter, gray matter)
- Noise robustness assessment
- Complete statistical validation framework

All of this analysis code is located in this folder.

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)
- Git LFS (for large data files)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SaharI12/TBMF.git
cd TBMF
```

2. Install Git LFS:
```bash
git lfs install
git lfs pull
```

3. Choose your analysis pathway:

**Test-Retest Study (Human Data):**
```bash
cd human
pip install -r requirements.txt
python main.py
```

**Phantom Model Validation:**
```bash
cd phantoms
pip install -r requirements.txt
# Preprocessing
python src/preprocessing/pre_process.py
# Train Model 2
python src/training/model2_example_training_fs_ksw.py
# Run inference
python src/inference/analysis.py
```

## Key Analysis Workflows

### Test-Retest & Robustness Analysis (Human)
```bash
cd human

# ICC statistical validation
jupyter notebook ICC_compare_method.ipynb

# Tissue-specific analysis
jupyter notebook ICC_and_ccomperssin_wm_gm_gt.ipynb

# Noise robustness testing
python noise_addition.py

# Attention visualization
python attention_analsys.py

# SHAP explainability
python shap.py

# Prediction visualizations
python images_and_box_plot_model_2.py
```

### Phantom Validation (Synthetic Data)
```bash
cd phantoms

# Preprocess phantom data
python src/preprocessing/pre_process.py

# Train on phantom data with known ground truth
python src/training/model1_phantoms_not_segmented_training.py
python src/training/model2_example_training_fs_ksw.py

# Run inference and evaluation
python src/inference/analysis.py
```

## Publication

**"Multi-contrast generation and quantitative MRI using a transformer-based framework with RF excitation embeddings"**

Published in: *Nature Communications*

**Link:** https://www-nature-com.bengurionu.idm.oclc.org/articles/s42003-025-09371-3

## Official Implementation Repository

For the original TBMF implementation, visit the official repository:

**GitHub:** https://github.com/momentum-laboratory/tbmf

## Configuration

Both subdirectories use configuration files for managing:
- Data paths and directories
- Model architecture parameters
- Training hyperparameters
- Validation settings

Refer to configuration files in each subdirectory for details.

## Dependencies

### Core Libraries
- **PyTorch** - Deep learning framework
- **NumPy, SciPy** - Scientific computing
- **Pandas** - Data manipulation
- **h5py** - HDF5 file format support
- **Matplotlib, Seaborn** - Visualization
- **SHAP** - Model explainability
- **TorchMetrics** - Evaluation metrics
- **Segment Anything Model (SAM)** - Segmentation

See `requirements.txt` in each subdirectory for complete dependency lists.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{TBMF2025,
  title={Multi-contrast generation and quantitative MRI using a transformer-based framework with RF excitation embeddings},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s42003-025-09371-3}
}
```

## Acknowledgments

This work builds upon:
- Vision Transformer (ViT) architecture
- Segment Anything Model (SAM) for segmentation
- SHAP library for model interpretability
- PyTorch ecosystem for deep learning

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Segment Anything](https://arxiv.org/abs/2304.02643)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

---

**Note:** This repository contains the revision, validation, and analysis code for the published Nature Communications paper.
