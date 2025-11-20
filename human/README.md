# Vision Transformer for MRI Parameter Estimation

## Overview

This project implements a Vision Transformer (ViT) based deep learning model for quantitative MRI parameter estimation. The model predicts six quantitative MRI parameters from high-resolution 2D MRI images using a transformer-based architecture with a convolutional decoder.

**Estimated Parameters:**
- KSSW (Steady-State Signal)
- MT% (Magnetization Transfer Percentage)
- B₀ (Main Magnetic Field)
- B₁ (Radiofrequency Field)
- T₁ (Spin-Lattice Relaxation)
- T₂ (Spin-Spin Relaxation)

## Key Features

- **Vision Transformer Architecture**: State-of-the-art transformer-based model with 6 layers and multi-head attention
- **Interpretability**: Integrated tools for:
  - Attention weight visualization
  - SHAP-based feature importance analysis
  - Model uncertainty quantification
- **Robustness Testing**: Noise simulation and generalization assessment
- **Statistical Validation**: ICC (Intraclass Correlation) analysis for reliability assessment
- **Comprehensive Metrics**: PSNR, SSIM, NRMSE evaluation

## Project Structure

```
├── transformer_architecture_prod.py  # Core ViT model architecture
├── functions_prod.py                 # Production utilities for training/inference
├── functions.py                      # Utility functions and dataset loaders
├── noise_addition.py                 # AWGN simulation for robustness testing
├── attention_analsys.py              # Attention weight extraction & visualization
├── shap.py                           # SHAP explainability analysis
├── images_and_box_plot_model_2.py   # Prediction visualization
├── create_images_for_the_same_subject.py  # Subject-specific prediction generation
├── main.py                           # Main pipeline entry point
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration, optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SaharI12/TBMF.git
cd TBMF/human
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the project:
```bash
# Copy the example configuration file
cp config.example.yaml config.yaml

# Edit config.yaml with your local paths
# Update:
# - data.root_dir: path to your MRI data
# - model.checkpoint_dir: path to saved model checkpoints
# - Other paths and parameters as needed
nano config.yaml  # or use your preferred editor
```

5. Prepare data and models:
```bash
# Create the data and checkpoints directories
mkdir -p data checkpoints predictions

# Place your training data in ./data/
# Place model checkpoints in ./checkpoints/
# Note: These directories are ignored by Git (see .gitignore)
```

## Configuration

This project uses YAML-based configuration for managing paths and parameters. All scripts should reference the `config.yaml` file instead of hardcoding paths.

### Setup Configuration

1. **Copy the example configuration:**
   ```bash
   cp config.example.yaml config.yaml
   ```

2. **Edit `config.yaml` with your paths:**
   ```yaml
   data:
     root_dir: "/path/to/your/data"
     data_dir: "/path/to/your/data/axial"

   model:
     checkpoint_dir: "./checkpoints"
     model1_path: "./checkpoints/model1.pt"

   training:
     batch_size: 16
     learning_rate: 0.0001
   ```

3. **Load configuration in your scripts:**
   ```python
   from config_loader import load_config

   config = load_config()
   data_dir = config.get('data.root_dir')
   batch_size = config.get('training.batch_size')
   ```

### Configuration File Structure

- **data**: Paths to training/validation data
- **model**: Model architecture parameters and checkpoint paths
- **training**: Training hyperparameters (batch size, learning rate, epochs)
- **validation**: Validation settings and test subjects
- **normalization**: Data scaling factors and channel ranges
- **analysis**: Settings for SHAP, attention analysis, and visualization
- **logging**: Logging configuration

See `config.example.yaml` for all available options.

## Usage

### Basic Inference

```python
import torch
from transformer_architecture_prod import Model
from functions_prod import load_checkpoint, TryDataset_v2

# Load model
model = Model(in_channels=6, out_channels=6)
model = load_checkpoint(model, checkpoint_path='checkpoints/model1.pt')
model.eval()

# Load data
dataset = TryDataset_v2(root_dir='./data')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# Run inference
with torch.no_grad():
    for images, targets in dataloader:
        predictions = model(images)
        print(f"Predictions shape: {predictions.shape}")
```

### Running the Full Pipeline

```bash
python main.py
```

### Generating Visualizations

```python
# Attention analysis
python attention_analsys.py

# SHAP analysis
python shap.py

# Prediction visualizations
python images_and_box_plot_model_2.py
```

## Model Architecture

### Input
- **Shape**: (Batch, 6, H, W) - 6-channel MRI images
- **Channels**: Multiple quantitative contrast weightings

### Architecture
1. **Patch Embedding**: Divides images into 9×9 patches and projects to 768-dim embeddings
2. **Positional Encoding**: Learnable positional embeddings for spatial information
3. **Transformer Encoder**: 3 layers with 4 attention heads
4. **Convolutional Decoder**: Upsampling and refinement to output resolution

### Output
- **Shape**: (Batch, 6, H, W) - 6-channel parameter maps

## Model Performance

The model demonstrates:
- Strong correlation with ground truth parameters
- Robust performance under AWGN (Additive White Gaussian Noise)
- Reliable ICC scores indicating consistency
- Interpretable predictions via attention weights and SHAP analysis

## Validation and Analysis

### Statistical Validation
Run ICC analysis for reliability assessment:
```bash
jupyter notebook ICC_compare_method.ipynb
jupyter notebook ICC_and_ccomperssin_wm_gm_gt.ipynb
```

### Robustness Testing
Test model robustness to noise:
```python
from noise_addition import add_awgn
noisy_images = add_awgn(images, snr_db=20)
```

### Interpretability
Analyze model decisions:
```python
# Attention visualization
from attention_analsys import AttentionExtractor
extractor = AttentionExtractor(model)
attention_maps = extractor.extract_attention(images)

# Feature importance
python shap.py  # Generates SHAP analysis
```

## Dependencies

See `requirements.txt` for full dependency list. Key dependencies:
- **PyTorch**: Deep learning framework
- **NumPy, SciPy**: Scientific computing
- **h5py**: HDF5 file format support
- **Matplotlib, Seaborn**: Visualization
- **SHAP**: Model explainability
- **TorchMetrics**: Model evaluation metrics

## Configuration

### Data Configuration
Update paths and parameters in each script:
- Input data directory: `root_dir`
- Model checkpoint paths: `checkpoint_path`
- Normalization factors: See data loading functions

### Model Configuration
Modify model hyperparameters in `transformer_architecture_prod.py`:
```python
model_config = {
    'in_channels': 6,
    'out_channels': 6,
    'img_size': 192,
    'patch_size': 9,
    'embedding_dim': 768,
    'num_heads': 4,
    'depth': 3
}
```

## Training

To train the model:
```python
from transformer_architecture_prod import Model
from functions_prod import train_epoch, TryDataset_v2

model = Model(in_channels=6, out_channels=6)
dataset = TryDataset_v2(root_dir='./data', split='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(num_epochs):
    train_epoch(model, dataloader, optimizer, criterion)
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{tbmf2024,
  title={Vision Transformer for Quantitative MRI Parameter Estimation},
  author={Your Name},
  year={2024},
  url={https://github.com/SaharI12/TBMF}
}
```

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

This work builds upon:
- Vision Transformer (ViT) architecture from Dosovitskiy et al.
- SHAP library for model interpretability
- PyTorch ecosystem for deep learning

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [Quantitative MRI: From Basics to Clinical Applications](https://www.wiley.com/en-us/Quantitative+MRI-p-9781119058304)

---

**Note**: This repository contains research code. For production use, additional error handling, validation, and optimization may be needed.