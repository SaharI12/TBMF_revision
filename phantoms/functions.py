from transformer_architecture_prod import Model
import torch
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torch.nn.functional import mse_loss
from torch import nn
from tqdm import tqdm
import matplotlib.colors as mcolors

def load_model(checkpoint_path, config):
    """
    Load the trained model from checkpoint.
    """
    try:
        # Initialize model with the provided configuration
        model = Model(
            img_size=config["image_size"],
            num_channels=config["sequence_len"],
            patch_size=config["patch_size"],
            embedding_dim=768,
            dropout=config["dropout"],
            mlp_size=3072,
            num_transformer_layers=config["num_transformer_layers"],
            num_heads=config["num_heads"]
        ).to(config["device"])

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])

        # If checkpoint contains 'model_state_dict', use that, otherwise assume it's the state dict directly
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {checkpoint_path}")
        # print model architecture for debugging
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None



def create_model_v0(args, weights_path):
    model = Model(img_size=args["image_size"],
                  num_channels=args["sequence_len"],
                  patch_size=args["patch_size"],
                  embedding_dim=args["embedding_dim"],
                  dropout=args["dropout"],
                  mlp_size=args["mlp_size"],
                  num_transformer_layers=args["num_transformer_layers"],
                  num_heads=args["num_heads"])

    model.conv_layers[9] = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    model.conv_layers[10] = nn.BatchNorm2d(num_features=32)
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=16))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=8))
    model.conv_layers.append(nn.ReLU())

    #model.conv_layers.append(nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.Sigmoid())

    model.load_state_dict(torch.load(weights_path))

    return model

class TryDataset_v3(Dataset):
    def __init__(self, data_dir, param_dir, labels_dir, scale_data, scale_param):
        self.data_paths = data_dir
        self.param_paths = param_dir
        self.label_paths = labels_dir

        self.scale_data = scale_data
        self.scale_params = scale_param

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):

        dataset_idx = h5py.File(self.data_paths[index])['res'][:] / self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels_idx = h5py.File(self.label_paths[index])['res'][:] / self.scale_data

        return dataset_idx.astype('float32'), params_idx.astype('float32'), labels_idx.astype('float32')


class TryDataset_v0(Dataset):

    def __init__(self, data_dir, param_dir, labels_dir, scale_data, scale_param, model_type):
        self.data_paths = data_dir
        self.param_paths = param_dir
        self.label_paths = labels_dir
        self.model_type = model_type
        self.scale_data = scale_data
        self.scale_params = scale_param

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):

        dataset_idx = h5py.File(self.data_paths[index])['res'][:] / self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels_idx = h5py.File(self.label_paths[index])['res'][:]

        if self.model_type == "pH_mM":
            labels_idx[0, :, :] = labels_idx[0, :, :] / 6  # pH
            labels_idx[1, :, :] = labels_idx[1, :, :] / 100  # mM
        elif self.model_type == "T1_T2":
            labels_idx[0, :, :] = labels_idx[0, :, :] / 3500  # T1
            labels_idx[1, :, :] = labels_idx[1, :, :] / 1000  # T2

        elif self.model_type == "b":
            labels_idx[0, :, :] = (labels_idx[0, :, :] + 1) / (1 + 1.7)  # b0
            labels_idx[1, :, :] = labels_idx[1, :, :] / 1.5  # B1
        else:
            labels_idx[0, :, :] = (labels_idx[0, :, :] + 0.001) / 0.005  # fs 110,000 /3 for display
            labels_idx[1, :, :] = (labels_idx[1, :, :] + labels_idx[1, :, :].min()) / labels_idx[1, :, :].max() - \
                                  labels_idx[1, :, :].min()  # ksw

        labels_idx[np.isnan(labels_idx)] = 0
        dataset_idx[np.isnan(dataset_idx)] = 0

        return dataset_idx.astype(np.float32), params_idx.astype(np.float32), labels_idx.astype(np.float32)

def get_user_slice_selection():
    """Ask user which slice they want to visualize (0-18)"""
    while True:
        try:
            slice_num = int(input("Which slice would you like to visualize? (0-18): "))
            if 0 <= slice_num <= 18:
                return slice_num
            else:
                print("Please enter a number between 0 and 18.")
        except ValueError:
            print("Error, plotting only first sequence")
            return 0


def plot_model1(y_pred, y, save_path):
    # Convert tensors to numpy and move to CPU
    vmin = 0.20
    vmax = 0.65
    y_pred = y_pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # Create custom colormap: viridis but starting with black
    import matplotlib.colors as mcolors
    viridis = plt.cm.viridis
    viridis_colors = viridis(np.linspace(0, 1, 256))
    viridis_colors[0] = [0, 0, 0, 1]  # Set first color to black (RGBA)
    custom_viridis = mcolors.ListedColormap(viridis_colors)
    custom_viridis.set_under('black')  # Values below vmin will be black

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    fig.set_facecolor("black")
    fig.suptitle('Model Predictions vs Ground Truth', color='white')

    # Plot ground truth (first row) and predictions (second row)
    for i in range(6):
        # Ground truth
        im1 = axes[0, i].imshow(y[0, i], cmap=custom_viridis, vmin=vmin, vmax=vmax)
        axes[0, i].axis('off')

        # Predictions
        im2 = axes[1, i].imshow(y_pred[0, i], cmap=custom_viridis, vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')

        # Calculate centered position for colorbar
        subplot_width = 0.775 / 6  # Total width divided by 6 columns
        subplot_left = 0.125 + i * subplot_width  # Left edge of subplot
        cbar_width = subplot_width * 0.6  # Colorbar width (60% of subplot width)
        cbar_left = subplot_left + (subplot_width - cbar_width) / 2  # Center the colorbar

        # Make colorbar taller and add proper tick labels
        cbar_ax = fig.add_axes([cbar_left, 0.05, cbar_width, 0.06])
        cbar = plt.colorbar(im1, cax=cbar_ax, orientation='horizontal')

        # Configure colorbar ticks and labels
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_ticklabels([f'{vmin:.2f}', f'{(vmin + vmax) / 2:.2f}', f'{vmax:.2f}'])
        cbar.ax.tick_params(labelsize=8, colors='white')

    plt.subplots_adjust(bottom=0.1)  # More room for colorbars with labels
    plt.savefig(save_path, dpi=300, facecolor='black')
    plt.close()
    print(f"Saved visualization to {save_path}")


def create_images(dataloader, config):
    model = config["model"]
    sequence = None
    images_path = os.path.join(config["images_path"], config["model_type"],"images")
    os.makedirs(images_path, exist_ok=True)
    if config["images_option"] == "sequence":
        sequence = get_user_slice_selection()

    if config["model_type"] == "model1":
        with torch.no_grad():
            for batch_idx, (X, p, y) in enumerate(dataloader):
                # Process batches at intervals: 0,19,38... or 1,20,39... etc.
                if sequence is not None and batch_idx % 19 != sequence:
                    continue
                print(f"Processing batch {batch_idx} for sequence {sequence}")
                X = X.to(config["device"])
                p = p.to(config["device"])
                y_pred = model(X, p)
                y_pred = y_pred.permute(0, 3, 1, 2)
                path = os.path.join(images_path,config["model_type"], f"seq{sequence}_batch{batch_idx}.png")
                plot_model1(y_pred, y, path)



def calc_psnr(y_pred, y_true):
    psnr = PeakSignalNoiseRatio(data_range=1).to(device=torch.device("cuda"))
    slices = y_pred.shape[1]
    if slices == 6:
        psnr_val = 0
        for i in range(slices):
            psnr_val += psnr(y_pred[:, i, :, :].unsqueeze(1),
                             y_true[:, i, :, :].unsqueeze(1))

        psnr_val = psnr_val / slices

    else:  # 30 slice
        psnr_val = 0
        for i in range(24):
            psnr_val += psnr(y_pred[:, i + 6, :, :].unsqueeze(1),
                             y_true[:, i + 6, :, :].unsqueeze(1))

        psnr_val = psnr_val / 24

    return psnr_val


def calc_nrmse(y_pred, y_true):
    slices = y_pred.shape[1]
    if slices == 6:
        nrmse_val = 0
        for i in range(slices):
            y_pred_idx = y_pred[:, i, :, :].unsqueeze(1)
            y_true_idx = y_true[:, i, :, :].unsqueeze(1)
            mse = mse_loss(y_pred_idx, y_true_idx)
            rmse = torch.sqrt(mse)

            nrmse_val += rmse / (torch.max(y_true_idx) - torch.min(y_true_idx))

        nrmse_val = nrmse_val / slices

    else:
        nrmse_val = 0
        for i in range(24):
            y_pred_idx = y_pred[:, i + 6, :, :].unsqueeze(1)
            y_true_idx = y_true[:, i + 6, :, :].unsqueeze(1)
            mse = mse_loss(y_pred_idx, y_true_idx)
            rmse = torch.sqrt(mse)

            nrmse_val += rmse / (torch.max(y_true_idx) - torch.min(y_true_idx))

        nrmse_val = nrmse_val / 24

    return nrmse_val


def calc_ssim(y_pred, y_true):
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1).to('cuda')
    slices = y_pred.shape[1]
    if slices == 6:
        ssim_val = 0
        for i in range(slices):
            ssim_val += ssim_fn(y_pred[:, i, :, :].unsqueeze(1),
                                y_true[:, i, :, :].unsqueeze(1))

        ssim_val = ssim_val / 6

    else:
        ssim_val = 0
        for i in range(24):
            ssim_val += ssim_fn(y_pred[:, i + 6, :, :].unsqueeze(1),
                                y_true[:, i + 6, :, :].unsqueeze(1))

        ssim_val = ssim_val / 24

    return ssim_val



def create_images_with_metrics(dataloader, config):
    """
    Single function to create model predictions and calculate metrics (PSNR, NRMSE, SSIM) using torch.
    Saves prediction visualizations and creates one summary table with combined checkpoints.
    """
    model = config["model"]

    # Create directories
    os.makedirs(config['images_path'] +f"/{config["model_type"]}" + f"/{config["model_type"]}", exist_ok=True)

    # Initialize torchmetrics on the same device
    device = config["device"]

    # Collect all metrics across all batches
    all_psnr_values = []
    all_nrmse_values = []
    all_ssim_values = []


    if config["model_type"] == "model1":
        with torch.no_grad():
            for batch_idx, (X, p, y) in enumerate(dataloader):

                print(f"Processing batch {batch_idx}")
                X = X.to(config["device"])
                p = p.to(config["device"])
                y = y.to(config["device"])

                y_pred = model(X, p)
                y_pred = y_pred.permute(0, 3, 1, 2)

                # Add to overall collection
                all_psnr_values.append(calc_psnr(y_pred,y))
                all_nrmse_values.append(calc_nrmse(y_pred, y))
                all_ssim_values.append(calc_ssim(y_pred,y))


        # Calculate overall statistics across all batches and channels
        if all_psnr_values:  # Make sure we have data
            psnr_tensor = torch.tensor(all_psnr_values, device=device)
            nrmse_tensor = torch.tensor(all_nrmse_values, device=device)
            ssim_tensor = torch.tensor(all_ssim_values, device=device)

            mean_psnr = torch.mean(psnr_tensor).item()
            mean_nrmse = torch.mean(nrmse_tensor).item()
            mean_ssim = torch.mean(ssim_tensor).item()

            std_psnr = torch.std(psnr_tensor).item()
            std_nrmse = torch.std(nrmse_tensor).item()
            std_ssim = torch.std(ssim_tensor).item()

            # Create summary table visualization
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            fig.set_facecolor("black")
            ax.axis('off')
            ax.set_facecolor('black')

            # Create summary table with combined checkpoints
            table_data = [
                ['Metric', 'Mean', 'Std'],
                ['PSNR (dB)', f'{mean_psnr:.3f}', f'{std_psnr:.3f}'],
                ['NRMSE', f'{mean_nrmse:.3f}', f'{std_nrmse:.3f}'],
                ['SSIM', f'{mean_ssim:.3f}', f'{std_ssim:.3f}']
            ]

            table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                             colWidths=[0.3, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(14)
            table.scale(1.5, 2.5)

            # Style the table
            for i in range(len(table_data)):
                for j in range(len(table_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#404040')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#202020')
                        cell.set_text_props(color='white')
                    cell.set_edgecolor('white')

            # Add title
            total_samples = len(all_psnr_values)
            title = f'Overall Model Performance Summary\n({total_samples} samples across all sequences)'
            ax.set_title(title, color='white', fontsize=16, y=0.8)

            # Save summary table
            plt.tight_layout()
            summary_path = os.path.join(config['images_path'] ,config["model_type"], 'metrics_summary.png')
            plt.savefig(summary_path, dpi=300, facecolor='black', bbox_inches='tight')
            plt.close()

            print(f"\n=== OVERALL METRICS SUMMARY ===")
            print(f"Total samples processed: {total_samples}")
            print(f"PSNR: {mean_psnr:.3f} ± {std_psnr:.3f} dB")
            print(f"NRMSE: {mean_nrmse:.3f} ± {std_nrmse:.3f}")
            print(f"SSIM: {mean_ssim:.3f} ± {std_ssim:.3f}")
            print(f"Summary table saved to: {summary_path}")


def create_m_to_4m(dataloader, config):
    """
    Single function to create model predictions and calculate metrics (PSNR, NRMSE, SSIM) using torch.
    Saves prediction visualizations and creates one summary table with combined checkpoints.
    """

    save_path = os.path.join(config['images_path'], config['model_type'] , 'm_to_4m')
    model = config["model"]
    parameter_map = config["parameter_map"]
    p0 = parameter_map[:, 0:12]
    p0 = np.expand_dims(p0, axis=0)
    p0 = torch.from_numpy(p0).float().to(config["device"])

    p1 = parameter_map[:, 6:18]
    p1 = np.expand_dims(p1, axis=0)
    p1 = torch.from_numpy(p1).float().to(config["device"])
    p2 = parameter_map[:, 12:24]
    p2 = np.expand_dims(p2, axis=0)
    p2 = torch.from_numpy(p2).float().to(config["device"])
    p3 = parameter_map[:, 18:]
    p3 = np.expand_dims(p3, axis=0)
    p3 = torch.from_numpy(p3).float().to(config["device"])
    # Create directories
    os.makedirs(save_path, exist_ok=True)

    # Initialize torchmetrics on the same device
    device = config["device"]

    # Collect all metrics across all batches
    all_psnr_values = []
    all_nrmse_values = []
    all_ssim_values = []

    # First pass: collect y values from batches where batch_idx % 18 == 0
    y_true_storage = {}

    with torch.no_grad():
        for batch_idx, (X, p, y) in enumerate(dataloader):
            if batch_idx % 18 == 0:
                y_true_storage[batch_idx] = y.to(config["device"])

    # Second pass: process predictions and compare to stored y_true
    for batch_idx, (X, p, y) in enumerate(dataloader):
        if batch_idx % 19 != 0:
            continue

        print(f"Processing batch {batch_idx}")
        X = X.to(config["device"])
        y = y.to(config["device"])

        y_pred_1 = model(X, p0)
        y_pred_1 = y_pred_1.permute(0, 3, 1, 2)

        y_pred_2 = model(y_pred_1, p1)
        y_pred_2 = y_pred_2.permute(0, 3, 1, 2)

        y_pred_3 = model(y_pred_2, p2)
        y_pred_3 = y_pred_3.permute(0, 3, 1, 2)

        y_pred_4 = model(y_pred_3, p3)
        y_pred_4 = y_pred_4.permute(0, 3, 1, 2)

        # Find the corresponding y_true from batch where batch_idx % 18 == 0
        # You can choose which one to use based on your logic
        target_batch_idx = (batch_idx // 18) * 18  # Gets the nearest lower multiple of 18

        if target_batch_idx in y_true_storage:
            y_true = y_true_storage[target_batch_idx]


            plot_model1(y_pred_4, y_true, os.path.join(save_path, f"batch{batch_idx}.png"))

            # Compare y_pred_4 to y_true from the stored batch
            all_psnr_values.append(calc_psnr(y_pred_4, y_true))
            all_nrmse_values.append(calc_nrmse(y_pred_4, y_true))
            all_ssim_values.append(calc_ssim(y_pred_4, y_true))




    # Calculate overall statistics across all batches and channels
    if all_psnr_values:  # Make sure we have data
        psnr_tensor = torch.tensor(all_psnr_values, device=device)
        nrmse_tensor = torch.tensor(all_nrmse_values, device=device)
        ssim_tensor = torch.tensor(all_ssim_values, device=device)

        mean_psnr = torch.mean(psnr_tensor).item()
        mean_nrmse = torch.mean(nrmse_tensor).item()
        mean_ssim = torch.mean(ssim_tensor).item()

        std_psnr = torch.std(psnr_tensor).item()
        std_nrmse = torch.std(nrmse_tensor).item()
        std_ssim = torch.std(ssim_tensor).item()

        # Create summary table visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fig.set_facecolor("black")
        ax.axis('off')
        ax.set_facecolor('black')

        # Create summary table with combined checkpoints
        table_data = [
            ['Metric', 'Mean', 'Std'],
            ['PSNR (dB)', f'{mean_psnr:.3f}', f'{std_psnr:.3f}'],
            ['NRMSE', f'{mean_nrmse:.3f}', f'{std_nrmse:.3f}'],
            ['SSIM', f'{mean_ssim:.3f}', f'{std_ssim:.3f}']
        ]

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                             colWidths=[0.3, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.5, 2.5)

        # Style the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#404040')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#202020')
                    cell.set_text_props(color='white')
                cell.set_edgecolor('white')

        # Add title
        total_samples = len(all_psnr_values)
        title = f'Overall Model Performance Summary\n({total_samples} samples across all sequences)'
        ax.set_title(title, color='white', fontsize=16, y=0.8)

        # Save summary table
        plt.tight_layout()
        summary_path = os.path.join(save_path, 'metrics_summary.png')
        plt.savefig(summary_path, dpi=300, facecolor='black', bbox_inches='tight')
        plt.close()



# Model 2 analysis:
def run_inference_model2(config, test_loader, output_dir=None):
    """Run inference on test dataset and collect checkpoints"""
    model = config["model"]
    device = config["device"]
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []
    all_inputs = []

    print("Running inference on test dataset...")

    with torch.no_grad():
        for batch_idx, (X, p, y) in enumerate(tqdm(test_loader, desc="Processing batches")):
            X, p, y = X.to(device), p.to(device), y.to(device)

            # Get predictions
            pred = model(X, p)

            # Convert to numpy and store
            pred_np = pred.cpu().numpy()
            target_np = y.cpu().numpy()

            all_predictions.append(pred_np)
            all_targets.append(target_np)

    # Concatenate all checkpoints
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Transpose predictions to match target format (batch, channels, height, width)
    predictions = predictions.transpose(0, 3, 1, 2)

    print(f"Inference complete. Shape: {predictions.shape}")

    return predictions, targets

def setup_colormaps(config):

    model_type = config["model2_scaling"]
    if model_type == "pH_mM" or model_type == "ksw_fs":

        cmap1 = 'magma'
        original_map = plt.get_cmap('viridis')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0  # minimum value is set to black
        cmap2 = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)
        return [cmap1, cmap2]

    elif model_type == "T1_T2":
        original_map = plt.get_cmap('winter')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        cmap_winter = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = plt.get_cmap('hot')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0  # minimum value is set to black
        cmap_b_hot = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        return [cmap_b_hot, cmap_winter]

    else:
        buda_map = plt.get_cmap('RdBu_r')
        lipari_map = plt.get_cmap('RdGy')
        original_map = buda_map
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        b_bwr = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = lipari_map
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        b_rdgy = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        return [b_bwr, b_rdgy]


def plt_results_model2(config, y_pred, y_true, cmaps):
    """Plot checkpoints and save to file"""
    model_type = config["model2_scaling"]
    cmap1, cmap2 = setup_colormaps(config)

    if model_type == "pH_mM":
        y_pred1 = y_pred[:,0] * 6 # Scale to match original pH values
        y_true1 = y_true[:,0] * 6 # Scale to match original pH values
        print(f" pH values: {y_pred1.max()}, {y_pred1.min()}. for gt: {y_true1.max()}, {y_true1.min()}")
        y_pred2 = y_pred[:,1] * 100 # Scale to match concentration values
        y_true2 = y_true[:,1] * 100# Scale to match concentration values
        vmin = [3,0]
        vmax = [7,120]
        title_1 = "pH "
        title_2 = "concentration (mM)"
        save_path = os.path.join(config['images_path'], config['model_type'], 'pH_mM')
    elif model_type == "ksw_fs":
        y_pred2 = (y_pred[:,0]*0.005 - 0.001) *110000/ 3  # Scale to match original fs values
        y_true2 = (y_true[:,0]*0.005 - 0.001) *110000/ 3  # Scale to match original fs values
        y_pred2 = np.where(y_pred2 < 4, 0, y_pred2)
        y_true2 = np.where(y_true2 < 4, 0, y_true2)

        y_pred1 = y_pred[:,1] * 1500 # Scale to match original ksw values
        y_true1 = y_true[:,1] * 1500 # Scale to match original ksw values
        y_pred1 = np.where(y_pred1 < 5, 0, y_pred1)
        y_true1 = np.where(y_true1 < 5, 0, y_true1)

        vmin = [0,0]
        vmax = [1400,120]
        title_1 = r'k$_{sw}$ (s$^{-1}$)'
        title_2 = r'f$_{s}$(mM)'
        save_path = os.path.join(config['images_path'], config['model_type'], 'ksw_fs')

    elif model_type == "T1_T2":
        y_pred1 = y_pred[:,0] * 3500 # Scale to match original T1 values
        y_true1 = y_true[:,0] * 3500 # Scale to match original T1 values
        y_pred2 = y_pred[:,1] * 1000 # Scale to match original T2 values
        y_true2 = y_true[:,1] * 1000 # Scale to match original T2 values
        vmin = [2500,400]
        vmax = [3400,1050]
        title_1 = 'T1(ms)'
        title_2 = 'T2(ms)'
        save_path = os.path.join(config['images_path'], config['model_type'], 'T1_T2')

    else: # B0 B1
        y_pred1 = (y_pred[:,0] * 2.7) - 1
        y_true1 = (y_true[:,0] * 2.7) - 1

        # Set a threshold to mask more near-zero values as NaN
        threshold = 0.00065  # Adjust this value as needed
        y_true1 = np.where(y_true1 == 0, -0.6, y_true1)
        y_pred1 = np.where(np.abs(y_pred1) < threshold, -0.6, y_pred1)

        y_true2 = y_true[:,1] *1.5
        y_pred2 = y_pred[:,1] * 1.5

        # Set zero values to exactly 0 for proper black background
        y_true2 = np.where(y_true2 == 0, 0, y_true2)
        y_pred2 = np.where(y_pred2 == 0, 0, y_pred2)

        vmin = [-0.6,0.5]
        vmax = [0.6,1.5]
        title_1 = r'B$_{0}$(ppm)'
        title_2 = r'B$_{1}$(rel.)'
        save_path = os.path.join(config['images_path'], config['model_type'], 'B0_B1')

    os.makedirs(save_path, exist_ok=True)

    for batch_idx in range(y_pred.shape[0]):

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('black')  # Set figure background to black

        # Add column headers above the plots
        fig.text(0.32, 0.95, title_1, fontsize=28, color='white', ha='center', va='center', weight='bold')
        fig.text(0.71, 0.95, title_2, fontsize=28, color='white', ha='center', va='center', weight='bold')

        # Add row labels on the left - moved inward from the edge
        fig.text(0.06, 0.75, 'a', fontsize=30, color='white', ha='center', va='center',
                rotation=0, weight='bold')
        fig.text(0.06, 0.35, 'b', fontsize=30, color='white', ha='center', va='center',
                rotation=0, weight='bold')

        # Row 1: Ground Truth pH
        im1 = axes[0, 0].imshow(y_true1[batch_idx], cmap=cmap1, vmin=vmin[0], vmax=vmax[0])
        axes[0, 0].axis('off')
        axes[0, 0].set_facecolor('black')

        # Row 2: Predicted pH
        im2 = axes[1, 0].imshow(y_pred1[batch_idx], cmap=cmap1, vmin=vmin[0], vmax=vmax[0])
        axes[1, 0].axis('off')
        axes[1, 0].set_facecolor('black')

        # Row 1: Ground Truth mM
        im3 = axes[0, 1].imshow(y_true2[batch_idx], cmap=cmap2, vmin=vmin[1], vmax=vmax[1])
        axes[0, 1].axis('off')
        axes[0, 1].set_facecolor('black')

        # Row 2: Predicted mM
        im4 = axes[1, 1].imshow(y_pred2[batch_idx], cmap=cmap2,vmin=vmin[1], vmax=vmax[1])
        axes[1, 1].axis('off')
        axes[1, 1].set_facecolor('black')

        # Add colorbars closer to the images
        # pH colorbar (left column)
        cbar1_ax = fig.add_axes([0.125, 0.08, 0.35, 0.03])  # [left, bottom, width, height]
        cbar1_ax.set_facecolor('black')
        cbar1 = plt.colorbar(im1, cax=cbar1_ax, orientation='horizontal')
        cbar1.ax.tick_params(colors='white', labelsize=16)
        cbar1.ax.xaxis.set_label_position('bottom')
        cbar1.outline.set_edgecolor('white')
        cbar1.outline.set_linewidth(1)

        # mM colorbar (right column)
        cbar2_ax = fig.add_axes([0.525, 0.08, 0.35, 0.03])  # [left, bottom, width, height]
        cbar2_ax.set_facecolor('black')
        cbar2 = plt.colorbar(im3, cax=cbar2_ax, orientation='horizontal')
        cbar2.ax.tick_params(colors='white', labelsize=16)
        cbar2.ax.xaxis.set_label_position('bottom')
        cbar2.outline.set_edgecolor('white')
        cbar2.outline.set_linewidth(1)

        # Adjust layout to accommodate labels and colorbars with more space for images
        plt.subplots_adjust(bottom=0.18, top=0.85, left=0.12, right=0.89, hspace=0.1, wspace=0.1)

        save_folder = os.path.join(save_path, f'sample_{batch_idx}_results.png')
        plt.savefig(save_folder, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

