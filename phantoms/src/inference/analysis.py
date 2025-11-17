import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

# Fix this import - should be relative
from src.utils.functions import *

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           'configs', 'inference_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f"✓ Loaded config from: {config_path}")

# Get base path (project root)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Convert relative paths to absolute paths
if config['model_type'] == 'model1':
    config['model1']['checkpoint'] = os.path.join(BASE_PATH, config['model1']['checkpoint'])
    config['model1']['dataset_path'] = os.path.join(BASE_PATH, config['model1']['dataset_path'])
else:
    # For model2, convert checkpoint paths
    for param_type in config['model2']['checkpoints']:
        config['model2']['checkpoints'][param_type] = os.path.join(
            BASE_PATH, config['model2']['checkpoints'][param_type]
        )
    config['model2']['dataset_path'] = os.path.join(
        BASE_PATH,
        config['model2']['dataset_path'],
        config['model2']['parameter_type']
    )

# For backward compatibility with your existing code, create the old variable names:
if config['model_type'] == 'model1':
    checkpoint_path = config['model1']['checkpoint']
    path_to_dataset = config['model1']['dataset_path']
else:
    checkpoint_path = config['model2']['checkpoints'][config['model2']['parameter_type']]
    path_to_dataset = config['model2']['dataset_path']

# Model parameters
model_parameters = config['model_parameters']
model_parameters['device'] = torch.device(config['device']) if torch.cuda.is_available() else torch.device("cpu")

# FIXED: Use checkpoint_path instead of config["model1_checkpoint"]
model = load_model(checkpoint_path, model_parameters)
print("Model loaded successfully")

# Load data paths
data_paths = sorted(glob.glob(os.path.join(path_to_dataset, r"*/*/*/data/*.h5")))
params_paths = sorted(glob.glob(os.path.join(path_to_dataset, r"*/*/*/params/*.h5")))
labels_paths = sorted(glob.glob(os.path.join(path_to_dataset, r"*/*/*/labels/*.h5")))
print(f"data paths len {len(data_paths)}, labels len: {len(labels_paths)}, params len : {len(params_paths)}")

# Create dataset
if config['model_type'] == "model1":
    config['model'] = model
    # Model 1 dataset
    dataset = TryDataset_v3(
        data_dir=data_paths,
        param_dir=params_paths,
        labels_dir=labels_paths,
        scale_data=config['scaling']['scale_max_data'],  # FIXED: Access from scaling section
        scale_param=config['scaling']['scale_max_params']  # FIXED: Access from scaling section
    )
else:
    # Model 2 dataset
    dataset = TryDataset_v0(
        data_dir=data_paths,
        param_dir=params_paths,
        labels_dir=labels_paths,
        scale_data=config['model2_scaling_options'][config['model2']['parameter_type']]['scale_data'],
        scale_param=config['model2_scaling_options'][config['model2']['parameter_type']]['scale_param'],
        model_type=config['model2']['parameter_type']  # FIXED: Use parameter_type
    )

    # FIXED: checkpoint_path already has the right checkpoint
    config['model'] = create_model_v0(model_parameters, checkpoint_path)

# Create dataloader
test_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True
)

# Run inference
if config['model_type'] == "model1":
    # FIXED: Access from output section
    if config['output']["create_images"]:
        create_images(test_dataloader, config)
    if config['metrics']["calculate"]:
        create_images_with_metrics(test_dataloader, config)
    if config['metrics']["calculate_mto4m"]:
        create_m_to_4m(test_dataloader, config)

if config['model_type'] == "model2":
    # FIXED: Access from output section
    if config['output']["create_images"]:
        predictions, target = run_inference_model2(config, test_dataloader)
        cmaps = setup_colormaps(config)
        plt_results_model2(config, predictions, target, cmaps)