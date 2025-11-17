import os
from functions import *
import torch

config = {
    "model_checkpoint": "/home/sahar/Models/Dinor_revision/personal_git/human/checkpoints/model2.pt",
    "data_paths": "/home/sahar/Models/Dinor_revision/personal_git/human/data/axial",
    "out_path": "/home/sahar/Models/Dinor_revision/personal_git/human/predictions",
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "scale_data": 4578.9688,
    "scale_params": 13.9984,
    "wanted_output": {
        "image_plotting": True,
        "metrics" : True
    },

    "plot_specific_seq": 0,
}
os.makedirs(config["out_path"], exist_ok=True)

model_hyperparameters = {
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "image_size": 144,
    "sequence_len": 6,
    "patch_size": 9,
    "embedding_dim": 768,
    "dropout": 0,
    "mlp_size": 3072,
    "num_transformer_layers": 3,
    "num_heads": 4,
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

# IMPORTANT: Pass DataLoader to config, not raw Dataset
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


