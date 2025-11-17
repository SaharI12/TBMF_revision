import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

from Dinor_revision.git_tbmf.phantoms.src.utils.functions import *


config = {
    'model1_checkpoint' : "/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/model_checkpoints/model1/not_segmented.pt", # change if needed
    "model_type" : 'model2',

    "path_to_dataset": { "model1" : r"/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/phantoms_clean/model1/scan12",
                         "model2" : r"/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/phantoms_clean/model2/scan12"},
    "scale_max_params": 4,
    "scale_max_data": 1,
    "num_workers" : 2,
    "device": torch.device("cuda") if torch.cuda.is_available() else "cpu",
    # Create images section
    "create_images": True,
    "images_path": "/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/predictions_model",
    "images_option": 'sequence', # options: 'sequence', 'all'.
    "metrics" : True, # Set true to calculate PSNR, NRMSE and SSIM for the dataset
    "mto4m" : True, # Set true to calculate MTO4M for the dataset
    "parameter_map": np.array(
        [[2, 2, 1.7, 1.5, 1.2, 1.2, 3, 0.5, 3, 1, 2.2, 3.2, 1.5, 0.7, 1.5, 2.2, 2.5, 1.2, 3, 0.2,
          1.5, 2.5, 0.7, 4, 3.2, 3.5, 1.5, 2.7, 0.7, 0.5],
         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]),

    # Model2 scaling
    "model2_scaling" :  "pH_mM",  # Options: pH_mM, T1_T2, ksw_fs, B0_B1
    "model2_checkpoint" : {"pH_mM" :"/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/model_checkpoints/model2/model2_pH_mM.pt",
                           "T1_T2": "/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/model_checkpoints/model2/model2_T1_T2.pt",
                           "ksw_fs": "/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/model_checkpoints/model2/model2_fs_ksw.pt",
                            "B0_B1": "/home/sahar/Models/Dinor_revision/git_tbmf/phantoms/model_checkpoints/model2/model2_b.pt"},

}
model_parameters = {
        "image_size": 80,
        "patch_size": 5,
        "device": torch.device("cuda") if torch.cuda.is_available() else "cpu",
        "num_workers": 1,
        "sequence_len": 6,
        "dropout": 0.0,
        "num_transformer_layers": 3,
        "num_heads": 4,
        "embedding_dim": 768,
        "mlp_size": 3072,
    }

model = load_model(config["model1_checkpoint"], model_parameters)
print("Model loaded successfully")

if config['model_type'] == "model1":
    path_to_dataset = config["path_to_dataset"][config['model_type']]
else:
    path_to_dataset = os.path.join(config["path_to_dataset"][config['model_type']], config["model2_scaling"])

data_paths = sorted(glob.glob(os.path.join(path_to_dataset, r"*/*/*/data/*.h5")))
params_paths = sorted(glob.glob(os.path.join(path_to_dataset, r"*/*/*/params/*.h5")))
labels_paths = sorted(glob.glob(os.path.join(path_to_dataset, r"*/*/*/labels/*.h5")))
print(f"data paths len {len(data_paths)}, labels len: {len(labels_paths)}, params len : {len(params_paths)}")

if config['model_type'] == "model1":
    config['model'] = model
    # Create dataset and dataloader
    # Model 1
    dataset = TryDataset_v3(
        data_dir=data_paths,
        param_dir=params_paths,
        labels_dir=labels_paths,
        scale_data=config.get("scale_max_data", 1),
        scale_param=config["scale_max_params"]
    )
else:
    dataset = TryDataset_v0(data_dir=data_paths,
                            param_dir=params_paths,
                            labels_dir=labels_paths,
                            scale_data=1,
                            scale_param=4,
                            model_type=config['model2_scaling'])

    config['model'] = create_model_v0(model_parameters, config['model2_checkpoint'][config["model2_scaling"]])


test_dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=config["num_workers"],
                            pin_memory=True)

if config['model_type'] == "model1":
    if config["create_images"]:
        create_images(test_dataloader, config)
    if config["metrics"]:
        create_images_with_metrics(test_dataloader, config)
    if config["mto4m"]:
        create_m_to_4m(test_dataloader, config)

if config['model_type'] == "model2":
    if config["create_images"]:
        predictions, target = run_inference_model2(config, test_dataloader)
        cmaps = setup_colormaps(config)
        plt_results_model2(config, predictions, target, cmaps)

