import os
import torch
from functions import *
from config_loader import load_config

# Load configuration from config.yaml
try:
    config_loader = load_config('config.yaml')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Build configuration dictionary for runtime
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config = {
    "model_checkpoint": config_loader.get('model.model2_path', 'checkpoints/model2.pt'),
    "data_paths": config_loader.get('data.data_dir', './data/axial'),
    "out_path": config_loader.get('analysis.predictions_dir', './predictions'),
    "device": device,
    "scale_data": config_loader.get('normalization.scale_data', 4578.9688),
    "scale_params": config_loader.get('normalization.scale_params', 13.9984),
    "wanted_output": {
        "image_plotting": True,
        "metrics": True
    },
    "plot_specific_seq": 0,
}

os.makedirs(config["out_path"], exist_ok=True)

model_hyperparameters = {
    "device": device,
    "image_size": config_loader.get('model.img_size', 144),
    "sequence_len": config_loader.get('model.in_channels', 6),
    "patch_size": config_loader.get('model.patch_size', 9),
    "embedding_dim": config_loader.get('model.embedding_dim', 768),
    "dropout": config_loader.get('model.dropout', 0),
    "mlp_size": config_loader.get('model.mlp_size', 3072),
    "num_transformer_layers": config_loader.get('model.num_transformer_layers', 3),
    "num_heads": config_loader.get('model.num_heads', 4),
}

# Create dataset and dataloader
dataset = Load_Dataset(config)
print(f"Dataset length: {len(dataset)}")

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Set shuffle=False for consistent indexing

# Load model
model = create_model_v0(model_hyperparameters, config['model_checkpoint'])
model.to(model_hyperparameters["device"])
model.eval()  # Set to evaluation mode

# IMPORTANT: Pass DataLoader to config
config['dataset'] = dataloader
config['model'] = model

print("Model loaded successfully")
#
create_images = Plot_Metrics(config)
if config["wanted_output"]['image_plotting']:
    create_images.load_and_predict()
    # ADD THIS SECTION FOR SUMMARY STATISTICS
    if config["wanted_output"]['metrics']:
        print("\n" + "=" * 50)
        print("GENERATING SUMMARY STATISTICS")
        print("=" * 50)

        # Generate complete summary report (includes statistics, table, and boxplots)
        create_images.generate_summary_report()


