# 
Deep learning framework for MRI parameter mapping using Vision Transformers.

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
- `model_checkpoints/` - Trained model weights

## Models
- **Model 1**: Whole phantom reconstruction
- **Model 2**: Parameter-specific (pH/mM, T1/T2, ksw/fs, B0/B1)

## Performance
- ICC: 0.938-0.967
- Pearson r: 0.957-0.978

## License
MIT
