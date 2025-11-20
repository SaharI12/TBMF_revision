import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from transformer_architecture_prod import *
from functions import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import cm
import time
import os
from tqdm import tqdm
import glob
import torch
from torch.utils.data import DataLoader
from config_loader import load_config


def save_single_prediction(y_pred, y_true, save_path):
    """
    Save a single prediction and ground truth label to .h5 file

    Args:
        y_pred: Single prediction tensor
        y_true: Single ground truth tensor
        save_path: Path where to save the .h5 file
    """
    with h5py.File(save_path, 'w') as f:
        # Save prediction and ground truth
        f.create_dataset('y_pred', data=y_pred.cpu().detach().numpy())
        f.create_dataset('y_true', data=y_true.cpu().detach().numpy())

        # Save metadata
        f.attrs['pred_shape'] = y_pred.shape
        f.attrs['true_shape'] = y_true.shape
        print(f"Saved single prediction and label to: {save_path}")


def rotate_image(image):
    """
    Rotate the image by 90 degrees counter clockwise.
    """
    return np.rot90(image, k=1, axes=(0, 1))


def plot_subject_pred(y_pred_day_1, y_pred_day_2, y_pred_day_3run_1,
                               y_pred_day_run_2, y_pred_day_3_run_3, savefig_path=None):
    """
    Plot brain parameter maps in the style of the reference image with 5 rows,
    keeping the rotate function and making images larger.

    Args:
        y_pred_day_*: Prediction tensors to visualize
        savefig_path: Path to save the figure
    """
    # Convert predictions to numpy arrays
    y_pred_day_1 = y_pred_day_1.cpu().detach().numpy()[0]
    y_pred_day_2 = y_pred_day_2.cpu().detach().numpy()[0]
    y_pred_day_3run_1 = y_pred_day_3run_1.cpu().detach().numpy()[0]
    y_pred_day_run_2 = y_pred_day_run_2.cpu().detach().numpy()[0]
    y_pred_day_3_run_3 = y_pred_day_3_run_3.cpu().detach().numpy()[0]

    # Stack all predictions into a single array
    y_pred = np.stack((y_pred_day_1, y_pred_day_2, y_pred_day_3run_1,
                       y_pred_day_run_2, y_pred_day_3_run_3), axis=0)
    print(f"Visualization array shape: {y_pred.shape}")


    # Define colormaps (using your existing ones)
    buda_map = LinearSegmentedColormap.from_list('buda', cm_data_vik)
    lipari_map = LinearSegmentedColormap.from_list('lipari', cm_data_brok)

    original_map = plt.get_cmap('viridis')
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0  # minimum value is set to black
    b_viridis = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

    original_map = buda_map
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0  # minimum value is set to black
    b_bwr = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

    original_map = lipari_map
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0  # minimum value is set to black
    b_rdgy = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

    original_map = plt.get_cmap('winter')
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0  # minimum value is set to black
    winter = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

    original_map = plt.get_cmap('hot')
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0  # minimum value is set to black
    b_hot = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

    # Use your existing colormaps
    cbar_list = ['magma', b_viridis, b_bwr, b_rdgy, b_hot, winter]

    # Define value ranges for each parameter
    vmin = [0, 0, -0.6, 0.5, 500, 30]
    vmax = [100, 27.27, 0.6, 1.5, 2500, 130]

    # Column titles with units (matching reference image)
    column_titles = [
        r"$K_{ssw}$ (S$^{-1}$)",
        r"$f_s$ (%)",
        r"$B_0$ (ppm)",
        r"$B_1$ (rel.)",
        r"$T_1$ (ms)",
        r"$T_2$ (ms)"
    ]

    # Create figure with improved layout - make it larger for better fit
    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(20, 24),
                            gridspec_kw={'wspace': 0.02, 'hspace': 0.3})

    # Set black background
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)

    # Plot each image
    for i in range(5):
        # Add row label (a, b, c, d, e)
        axs[i, 0].text(-0.2, 0.5, chr(97 + i), fontsize=25, color='white',
                       transform=axs[i, 0].transAxes, va='center', fontweight='bold')

        for j in range(6):
            # Get data and apply appropriate scaling
            if j == 0:  # Kssw
                data = rotate_image(y_pred[i, 20:-20, 20:-20, 0] * 100)
                vrange = [0, 100]
            elif j == 1:  # fs
                data = rotate_image(y_pred[i, 20:-20, 20:-20, 1] * 27.27)
                vrange = [0, 27.27]
            elif j == 2:  # B0
                data = rotate_image(y_pred[i, 20:-20, 20:-20, 2] * 2.7 - 1)
                vrange = [-0.6, 0.6]
            elif j == 3:  # B1
                data = rotate_image(y_pred[i, 20:-20, 20:-20, 3] * 3.4944)
                vrange = [0.5, 1.5]
            elif j == 4:  # T1
                data = rotate_image(y_pred[i, 20:-20, 20:-20, 4] * 10000)
                vrange = [500, 2500]
            else:  # T2
                data = rotate_image(y_pred[i, 20:-20, 20:-20, 5] * 1000)
                vrange = [30, 130]

            # Display with correct colormap and value range
            im = axs[i, j].imshow(data, vmin=vrange[0], vmax=vrange[1], cmap=cbar_list[j])
            axs[i, j].axis('off')

            # Add parameter title only to top row
            if i == 0:
                axs[i, j].set_title(column_titles[j], fontsize=20, color='white', pad=10)

    # Add colorbars at the bottom - matching reference image style
    # Increase the spacing between rows and the colorbar area
    plt.subplots_adjust(bottom=0.12)

    # Create colorbar areas with proper spacing
    cbar_bottom = 0.03  # Position at the bottom of the figure
    cbar_height = 0.02  # Make colorbar height a bit larger

    for j in range(6):
        # Calculate position for each colorbar to align with columns
        ax_pos = axs[0, j].get_position()
        cbar_width = ax_pos.width * 0.9
        cbar_x = ax_pos.x0 + (ax_pos.width - cbar_width) / 2

        # Create axes for colorbar
        cax = fig.add_axes([cbar_x, cbar_bottom, cbar_width, cbar_height])

        # Define colorbar values
        if j == 0:  # Kssw
            vrange = [0, 50, 100]
            cmap = cbar_list[j]
        elif j == 1:  # fs
            vrange = [0, 14, 27]
            cmap = cbar_list[j]
        elif j == 2:  # B0
            vrange = [-0.6, 0, 0.6]
            cmap = cbar_list[j]
        elif j == 3:  # B1
            vrange = [0.5, 0.8, 1.5]
            cmap = cbar_list[j]
        elif j == 4:  # T1
            vrange = [500, 1500, 2500]
            cmap = cbar_list[j]
        else:  # T2
            vrange = [30, 80, 130]
            cmap = cbar_list[j]

        # Create colorbar
        norm = plt.Normalize(vmin=vrange[0], vmax=vrange[2])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Make colorbar horizontal with white tick labels
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(colors='white', labelsize=12)
        cbar.set_ticks([vrange[0], vrange[1], vrange[2]])

        # Format tick labels
        if j == 0 or j == 1 or j == 4 or j == 5:  # Kssw, fs, T1, T2 - integers
            cbar.set_ticklabels([int(vrange[0]), int(vrange[1]), int(vrange[2])])
        else:  # B0, B1 - decimals
            cbar.set_ticklabels([f"{vrange[0]:.1f}", f"{vrange[1]:.1f}", f"{vrange[2]:.1f}"])

    # Save the figure if a path is provided
    if savefig_path:
        plt.savefig(savefig_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to: {savefig_path}")

    return fig

class TryDataset_v2(Dataset):

    def __init__(self, data_dir, param_dir, labels_dir, scale_data, scale_param):
        self.data_paths = data_dir
        self.param_paths = param_dir
        self.label_paths = labels_dir

        self.scale_data = scale_data
        self.scale_params = scale_param

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        dataset_idx =  h5py.File(self.data_paths[index])['res'][:] / self.scale_data
        params_idx =  h5py.File(self.param_paths[index])['res'][:]/ self.scale_params
        labels = h5py.File(self.label_paths[index])['res'][:]
        labels[0, :, :] = labels[0, :, :] / 100  # KSSW
        labels[1, :, :] = labels[1, :, :] * 100 / 27.27  # MT
        labels[2, :, :] = (labels[2, :, :] + 1) / (1.7 + 1)  # B0

        labels[3, :, :] = labels[3, :, :] / 3.4944  # B1
        labels[4, :, :] = labels[4, :, :] / 10000  # T1
        labels[5, :, :] = labels[5, :, :] / 1000  # T2
        return dataset_idx.astype('float32'), params_idx.astype('float32'), labels.astype('float32')



def main(args):
    device = args.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    start_time = time.time()

    # Get output path from args or use default
    savefig_path = args.get("out_dir", "./predictions/single_subject_prediction.png")
    os.makedirs(os.path.dirname(savefig_path), exist_ok=True)

    scale_data = args.get("scale_data", 4578.9688)
    scale_params = args.get("scale_params", 13.9984)

    # Collect paths
    index = 61
    vol = "vol16"
    view = "sagittal"  # axial, coronal, sagittal

    # Get base data directory from args
    data_base_dir = args.get("data_dir", "./data")

    data_paths = glob.glob(
        os.path.join(data_base_dir, vol, '*', view, str(index), 'dataset', f'slice_{index}_image_0.h5'))
    data_paths = sorted(data_paths)
    param_paths = glob.glob(
        os.path.join(data_base_dir, vol, '*', view, str(index), 'params', f'slice_{index}_image_0.h5'))
    param_paths = sorted(param_paths)
    label_paths = glob.glob(
        os.path.join(data_base_dir, vol, '*', view, str(index), 'labels', f'slice_{index}_image_0.h5'))
    label_paths = sorted(label_paths)
    print(f"Data paths: {data_paths}")
    # data_paths[1] = fr'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/combined_output/{vol}/day2/{view}/70/dataset/slice_70_image_0.h5'
    # param_paths[1] = fr'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/combined_output/{vol}/day2/{view}/70/params/slice_70_image_0.h5'
    # data_paths[2] = fr'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/combined_output/{vol}/day3_run1/{view}/70/dataset/slice_70_image_0.h5'
    # param_paths[2] = fr'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/combined_output/{vol}/day3_run1/{view}/70/params/slice_70_image_0.h5'
    # data_paths[4] = r'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/combined_output/vol16/day3_run3/axial/70/dataset/slice_70_image_0.h5'
    # param_paths[4] = r'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/combined_output/vol16/day3_run3/axial/70/params/slice_70_image_0.h5'
    # Create the dataset
    dataset = TryDataset_v2(data_dir=data_paths,
                            param_dir=param_paths,
                            labels_dir=label_paths,
                            scale_data=scale_data,
                            scale_param=scale_params)

    # Load the model
    model = create_model_v0(args, weights_path=args["new_model_weight_path"]).to(args["device"])
    model.eval()

    # Create data loader
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)

    # Iterate over the dataset
    # Process each batch
    with torch.no_grad():  # Add no_grad for inference
        for batch, (X, p, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            X = X.to(args["device"])
            p = p.to(args["device"])
            y = y.to(args["device"])
            # Forward pass
            predicted_labels = model(X, p)
            if batch == 0:
                y_pred_day_1 = predicted_labels
                h5_save_dir = args.get("h5_save_dir", "./predictions/h5_results")
                os.makedirs(h5_save_dir, exist_ok=True)
                h5_save_path = os.path.join(h5_save_dir, f'vol16_day_1_sagittal_tmbf.h5')
                save_single_prediction(y_pred_day_1, y, h5_save_path)
            elif batch == 1:
                y_pred_day_2 = predicted_labels
            elif batch == 2:
                y_pred_day_3_run_1 = predicted_labels
            elif batch == 3:
                y_pred_day_3_run_2 = predicted_labels
            elif batch == 4:
                y_pred_day_3_run_3 = predicted_labels

    plot_subject_pred(y_pred_day_1, y_pred_day_2, y_pred_day_3_run_1,
                      y_pred_day_3_run_2, y_pred_day_3_run_3, savefig_path)





if __name__ == '__main__':
    arguments = dict()
    arguments["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    arguments["image_size"] = 144
    arguments["sequence_len"] = 6
    arguments["patch_size"] = 9

    arguments["embedding_dim"] = 768

    arguments["dropout"] = 0
    arguments["mlp_size"] = 3072
    arguments["num_transformer_layers"] = 3
    arguments["num_heads"] = 4

    # Load configuration from config.yaml
    try:
        config_loader = load_config('config.yaml')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    arguments["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    arguments["data_dir"] = config_loader.get('data.root_dir', './data')
    arguments["new_model_weight_path"] = config_loader.get('model.model2_path', 'checkpoints/model2.pt')
    arguments["out_dir"] = config_loader.get('analysis.predictions_dir', './predictions/single_subject_prediction.png')
    arguments["h5_save_dir"] = config_loader.get('analysis.predictions_dir', './predictions/h5_results')
    arguments["scale_data"] = config_loader.get('normalization.scale_data', 4578.9688)
    arguments["scale_params"] = config_loader.get('normalization.scale_params', 13.9984)

    main(arguments)

