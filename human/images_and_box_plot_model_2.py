from transformer_architecture_prod import *
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from functions_prod import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import cm
import time
import os
import scipy.io as sio
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sns

CHANNEL_RANGES = {
    0: 100.0,   # Kssw
    1: 27.27,   # MT_perc
    2: 1.7,     # B0
    3: 1.0,     # B1
    4: 3000.0,  # T1
    5: 1000.0,  # T2
}

def calc_psnr(psnr, y_pred, y_true):
    """
    Calculate PSNR for each metric channel and overall average

    Parameters:
    -----------
    psnr : torchmetrics.PeakSignalNoiseRatio
        The PSNR calculation function
    y_pred : torch.Tensor
        Predicted tensor of shape [B, C, H, W]
    y_true : torch.Tensor
        Ground truth tensor of shape [B, C, H, W]
    Returns:
    --------
    psnr_per_metric : list
        List of PSNR values for each metric channel
    psnr_val : torch.Tensor
        Overall average PSNR value
    """
    slices = y_pred.shape[1]
    psnr_per_metric = []
    psnr_val = 0

    for i in range(slices):
        metric_psnr = psnr(y_pred[:, i, :, :].unsqueeze(1),
                         y_true[:, i, :, :].unsqueeze(1))
        psnr_val += metric_psnr
        psnr_per_metric.append(metric_psnr.item())  # Convert tensor to scalar

    psnr_val = psnr_val / slices

    return psnr_per_metric, psnr_val


def calc_nrmse(y_pred, y_true, eps=1e-2):
    """
    Calculate normalized RMSE for each metric channel and overall average

    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted tensor of shape [B, C, H, W]
    y_true : torch.Tensor
        Ground truth tensor of shape [B, C, H, W]
    eps : float
        Small epsilon to avoid division by zero

    Returns:
    --------
    nrmse_per_metric : list
        List of NRMSE values for each metric channel
    nrmse_val : torch.Tensor
        Overall average NRMSE value
    """
    slices = y_pred.shape[1]
    nrmse_per_metric = []
    nrmse_val = 0

    for i in range(slices):
        y_pred_idx = y_pred[:, i, :, :].unsqueeze(1)
        y_true_idx = y_true[:, i, :, :].unsqueeze(1)
        mse = mse_loss(y_pred_idx, y_true_idx)
        rmse = torch.sqrt(mse)

        # Normalize by the range of the specific channel
        normalized_rmse = rmse / CHANNEL_RANGES[i]

        nrmse_val += normalized_rmse
        nrmse_per_metric.append(normalized_rmse.item())  # Convert tensor to scalar

    nrmse_val = nrmse_val / slices

    return nrmse_per_metric, nrmse_val


def calc_ssim(ssim_fn, y_pred, y_true):
    """
    Calculate SSIM for each metric channel and overall average

    Parameters:
    -----------
    ssim_fn : torchmetrics.StructuralSimilarityIndexMeasure
        The SSIM calculation function
    y_pred : torch.Tensor
        Predicted tensor of shape [B, C, H, W]
    y_true : torch.Tensor
        Ground truth tensor of shape [B, C, H, W]

    Returns:
    --------
    ssim_per_metric : list
        List of SSIM values for each metric channel
    ssim_val : torch.Tensor
        Overall average SSIM value
    """
    slices = y_pred.shape[1]
    ssim_per_metric = []
    ssim_val = 0

    for i in range(slices):
        metric_ssim = ssim_fn(y_pred[:, i, :, :].unsqueeze(1),
                             y_true[:, i, :, :].unsqueeze(1))
        ssim_val += metric_ssim
        ssim_per_metric.append(metric_ssim.item())  # Convert tensor to scalar

    ssim_val = ssim_val / slices

    return ssim_per_metric, ssim_val


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


        dataset_idx = sio.loadmat(self.data_paths[index])['res'] / self.scale_data
        params_idx = sio.loadmat(self.param_paths[index])['res'] / self.scale_params
        labels = sio.loadmat(self.label_paths[index])['res']
        labels[0,:,:] = labels[0,:,:] / 100 # KSSW
        labels[1,:,:] = labels[1,:,:]*100 / 27.27 # MT
        labels[2,:,:] = (labels[2,:,:]+1) / (1.7+1) # B0

        labels[3,:,:] = labels[3,:,:] / 3.4944 # B1
        labels[4,:,:] = labels[4,:,:] / 10000 # T1
        labels[5,:,:] = labels[5,:,:] / 1000 # T2



        return dataset_idx.astype('float32'), params_idx.astype('float32'), labels.astype('float32')



def plot_true_and_pred_sequences(test_labels, predicted_labels, save_path):

    predicted_labels = predicted_labels.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()
    # check if there are nan in the test labels


    ######################### Define the colormaps #########################
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

    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))

    current_image = test_labels[0, 0, :, :]  * 100
    axs[0, 0].imshow(current_image,
                     vmin=0,
                     vmax=100,
                     cmap='magma')

    axs[0, 0].set_title(label="Kssw", fontsize=20)
    axs[0, 0].set_ylabel(ylabel="Ground\nTruth", rotation=0, labelpad=50, fontsize=20)
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])

    current_image = test_labels[0, 1, :, :] * 27.27
    axs[0, 1].imshow(current_image,
                     vmin=0,
                     vmax=27.27,
                     cmap=b_viridis)
    axs[0, 1].set_title(label="MT_perc", fontsize=20)
    axs[0, 1].axis('off')

    current_image = (test_labels[0, 2, :, :] * 2.7)-1
    current_image = current_image.copy()
    current_image[current_image == 0] = np.nan

    # 3) make a local copy of your cmap and set NaNs ? black
    cmap = b_bwr.copy()
    cmap.set_bad('black')


    # plot masked data with your same vmin/vmax + cmap
    axs[0, 2].imshow(current_image,
                     vmin=-0.6,
                     vmax=0.6,
                     cmap=cmap)

    axs[0, 2].set_title(label=u'B\u2080', fontsize=20)
    axs[0, 2].axis('off')

    current_image = test_labels[0, 3, :, :] * 3.4944
    axs[0, 3].imshow(current_image,
                     vmin=0.5,
                     vmax=1.5,
                     cmap=b_rdgy)
    axs[0, 3].set_title(label=u'B\u2081', fontsize=20)
    axs[0, 3].axis('off')

    current_image = test_labels[0, 4, :, :] * 10000
    axs[0, 4].imshow(current_image,
                     vmin=0,
                     vmax=3000,
                     cmap='hot')

    axs[0, 4].set_title(label=u'T\u2081', fontsize=20)
    axs[0, 4].axis('off')

    current_image = test_labels[0, 5, :, :] * 1000
    axs[0, 5].imshow(current_image,
                     vmin=0,
                     vmax=200,
                     cmap=winter,
                     )
    axs[0, 5].set_title(label=u'T\u2082', fontsize=20)
    axs[0, 5].axis('off')

    current_image = predicted_labels[0, :, :, 0] * 100
    axs[1, 0].imshow(current_image,
                     vmin=0,
                     vmax=100,
                     cmap='magma')
    axs[1, 0].set_ylabel(ylabel="Predicted\nImages", rotation=0, labelpad=50, fontsize=20)
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xticks([])

    current_image = predicted_labels[0, :, :, 1] * 27.27
    axs[1, 1].imshow(current_image,
                     vmin=0,
                     vmax=27.27,
                     cmap=b_viridis)
    axs[1, 1].axis('off')

    current_image = (predicted_labels[0, :, :, 2] * 2.7) - 1
    axs[1, 2].imshow(current_image,
                     vmin=-0.6,
                     vmax=0.6,
                     cmap=b_bwr,
                     )
    axs[1, 2].axis('off')

    current_image =predicted_labels[0, :, :, 3] * 3.4944
    axs[1, 3].imshow(current_image,
                     vmin=0.5,
                     vmax=1.5,
                     cmap=b_rdgy)
    axs[1, 3].axis('off')

    current_image = predicted_labels[0, :, :, 4] * 10000
    axs[1, 4].imshow(current_image,
                     vmin=0,
                     vmax=3000,
                     cmap='hot')
    axs[1, 4].axis('off')

    current_image = predicted_labels[0, :, :, 5] * 1000
    axs[1, 5].imshow(current_image,
                     vmin=0,
                     vmax=200,
                     cmap=winter)
    axs[1, 5].axis('off')

    if current_image.shape == (116, 116):
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.65, wspace=0)
    elif current_image.shape == (116, 88):
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.5, wspace=0)

    elif current_image.shape == (126, 88):
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.4, wspace=0)
    elif current_image.shape == (144, 88):
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.5, wspace=0)
    elif current_image.shape == (126, 144):
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=-0.7, wspace=0)

    cbar_list = ['magma', b_viridis, b_bwr, b_rdgy, 'hot', winter]

    vmin = [0, 0, -0.6, 0.5, 0, 0]
    vmax = [100, 27.27, 0.6, 1.5, 3000, 1000]
    for j in range(6):
        cax = fig.add_axes(
            [axs[0, j].get_position().x0 + 0.01, 0.1, axs[0, j].get_position().width , 0.02])  # 0.2
        sm = cm.ScalarMappable(cmap=cbar_list[j], norm=plt.Normalize(vmin=vmin[j], vmax=vmax[j]))
        sm.set_array([])
        plt.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.linspace(vmin[j], vmax[j], 5))
    plt.savefig(save_path)
    plt.close()


def create_per_metric_boxplots(metrics_df, save_dir):
    """
    Create boxplots showing metrics for each individual channel (Kssw, MT_perc, B0, B1, T1, T2)

    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing the metrics data with per-metric columns
    save_dir : str
        Directory where to save the boxplot images
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pandas as pd
    import numpy as np

    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Set style
    sns.set(style="whitegrid")

    # Channel names
    channel_names = ['Kssw', 'MT_perc', 'B0', 'B1', 'T1', 'T2']

    # Create boxplots for SSIM per channel
    ssim_cols = [f'ssim_metric_{i}' for i in range(6)]
    psnr_cols = [f'psnr_metric_{i}' for i in range(6)]
    nrmse_cols = [f'nrmse_metric_{i}' for i in range(6)]

    if all(col in metrics_df.columns for col in ssim_cols):  # If we have all 6 metrics
        # Reshape data for seaborn
        ssim_data = []
        for i, col in enumerate(ssim_cols):
            for value in metrics_df[col]:
                ssim_data.append({
                    'Channel': channel_names[i],
                    'SSIM': value
                })

        ssim_df = pd.DataFrame(ssim_data)

        # Create boxplot - Fixed to avoid deprecation warning
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Channel', y='SSIM', data=ssim_df, hue='Channel', legend=False)
        plt.title('SSIM by Channel', fontsize=16)
        plt.xlabel('Channel', fontsize=14)
        plt.ylabel('SSIM', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ssim_per_channel_boxplot.png'), dpi=300)
        plt.close()

    if all(col in metrics_df.columns for col in psnr_cols):  # If we have all 6 metrics
        # Reshape data for seaborn
        psnr_data = []
        for i, col in enumerate(psnr_cols):
            for value in metrics_df[col]:
                psnr_data.append({
                    'Channel': channel_names[i],
                    'PSNR': value
                })

        psnr_df = pd.DataFrame(psnr_data)

        # Create boxplot - Fixed to avoid deprecation warning
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Channel', y='PSNR', data=psnr_df, hue='Channel', legend=False)
        plt.title('PSNR by Channel', fontsize=16)
        plt.xlabel('Channel', fontsize=14)
        plt.ylabel('PSNR (dB)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'psnr_per_channel_boxplot.png'), dpi=300)
        plt.close()

    if all(col in metrics_df.columns for col in nrmse_cols):  # If we have all 6 metrics
        # Reshape data for seaborn
        nrmse_data = []
        for i, col in enumerate(nrmse_cols):
            for value in metrics_df[col]:
                nrmse_data.append({
                    'Channel': channel_names[i],
                    'NRMSE': value
                })

        nrmse_df = pd.DataFrame(nrmse_data)

        # Create boxplot - Fixed to avoid deprecation warning
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Channel', y='NRMSE', data=nrmse_df, hue='Channel', legend=False)
        plt.title('NRMSE by Channel', fontsize=16)
        plt.xlabel('Channel', fontsize=14)
        plt.ylabel('NRMSE', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'nrmse_per_channel_boxplot.png'), dpi=300)
        plt.close()

    # Combined metrics in one figure
    if (all(col in metrics_df.columns for col in ssim_cols) and
        all(col in metrics_df.columns for col in psnr_cols) and
        all(col in metrics_df.columns for col in nrmse_cols)):

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # SSIM - Fixed to avoid deprecation warning
        sns.boxplot(x='Channel', y='SSIM', data=ssim_df, ax=axes[0], hue='Channel', legend=False)
        axes[0].set_title('SSIM by Channel', fontsize=14)
        axes[0].set_xlabel('Channel', fontsize=12)
        axes[0].set_ylabel('SSIM', fontsize=12)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # PSNR - Fixed to avoid deprecation warning
        sns.boxplot(x='Channel', y='PSNR', data=psnr_df, ax=axes[1], hue='Channel', legend=False)
        axes[1].set_title('PSNR by Channel', fontsize=14)
        axes[1].set_xlabel('Channel', fontsize=12)
        axes[1].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # NRMSE - Fixed to avoid deprecation warning
        sns.boxplot(x='Channel', y='NRMSE', data=nrmse_df, ax=axes[2], hue='Channel', legend=False)
        axes[2].set_title('NRMSE by Channel', fontsize=14)
        axes[2].set_xlabel('Channel', fontsize=12)
        axes[2].set_ylabel('NRMSE', fontsize=12)
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)

        plt.suptitle('Performance Metrics by Channel', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'combined_metrics_per_channel.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Per-channel metric boxplots saved to {save_dir}")


def create_metrics_boxplots(metrics_df, save_dir):
    """
    Create boxplots for SSIM, PSNR, and NRMSE metrics per volunteer.

    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing the metrics data with columns 'volunteer', 'ssim', 'psnr', and 'nrmse'
    save_dir : str
        Directory where to save the boxplot images

    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt


    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Create boxplot for SSIM
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='volunteer', y='ssim', data=metrics_df)
    plt.title('SSIM Distribution by Volunteer', fontsize=16)
    plt.xlabel('Volunteer', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ssim_boxplot.png'), dpi=300)
    plt.close()

    # Create boxplot for PSNR
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='volunteer', y='psnr', data=metrics_df)
    plt.title('PSNR Distribution by Volunteer', fontsize=16)
    plt.xlabel('Volunteer', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psnr_boxplot.png'), dpi=300)
    plt.close()

    # Create boxplot for NRMSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='volunteer', y='nrmse', data=metrics_df)
    plt.title('NRMSE Distribution by Volunteer', fontsize=16)
    plt.xlabel('Volunteer', fontsize=14)
    plt.ylabel('NRMSE', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nrmse_boxplot.png'), dpi=300)
    plt.close()

    # Create a combined metrics plot
    plt.figure(figsize=(15, 10))

    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot SSIM
    sns.boxplot(x='volunteer', y='ssim', data=metrics_df, ax=axes[0])
    axes[0].set_title('SSIM by Volunteer', fontsize=14)
    axes[0].set_xlabel('Volunteer', fontsize=12)
    axes[0].set_ylabel('SSIM', fontsize=12)

    # Plot PSNR
    sns.boxplot(x='volunteer', y='psnr', data=metrics_df, ax=axes[1])
    axes[1].set_title('PSNR by Volunteer', fontsize=14)
    axes[1].set_xlabel('Volunteer', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)

    # Plot NRMSE
    sns.boxplot(x='volunteer', y='nrmse', data=metrics_df, ax=axes[2])
    axes[2].set_title('NRMSE by Volunteer', fontsize=14)
    axes[2].set_xlabel('Volunteer', fontsize=12)
    axes[2].set_ylabel('NRMSE', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_metrics_boxplot.png'), dpi=300)
    plt.close()

    # Create boxplots grouped by day for each volunteer
    days = metrics_df['day'].unique()

    # Filter out None values if any
    days = [day for day in days if day is not None]

    if len(days) > 1:  # Only create these plots if we have multiple days
        plt.figure(figsize=(12, 8))

        # SSIM by volunteer and day
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='volunteer', y='ssim', hue='day', data=metrics_df)
        plt.title('SSIM by Volunteer and Day', fontsize=16)
        plt.xlabel('Volunteer', fontsize=14)
        plt.ylabel('SSIM', fontsize=14)
        plt.legend(title='Day')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ssim_by_day_boxplot.png'), dpi=300)
        plt.close()

        # PSNR by volunteer and day
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='volunteer', y='psnr', hue='day', data=metrics_df)
        plt.title('PSNR by Volunteer and Day', fontsize=16)
        plt.xlabel('Volunteer', fontsize=14)
        plt.ylabel('PSNR (dB)', fontsize=14)
        plt.legend(title='Day')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'psnr_by_day_boxplot.png'), dpi=300)
        plt.close()

        # NRMSE by volunteer and day
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='volunteer', y='nrmse', hue='day', data=metrics_df)
        plt.title('NRMSE by Volunteer and Day', fontsize=16)
        plt.xlabel('Volunteer', fontsize=14)
        plt.ylabel('NRMSE', fontsize=14)
        plt.legend(title='Day')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'nrmse_by_day_boxplot.png'), dpi=300)
        plt.close()

    print(f"All metric boxplots saved to {save_dir}")



def create_combined_metrics_boxplots(metrics_df, save_dir):
    """
    Create a combined boxplot showing overall metrics for all volunteers together
    with reduced outliers, analyzing data by day first.

    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing the metrics data with columns 'volunteer', 'day', 'ssim', 'psnr', and 'nrmse'
    save_dir : str
        Directory where to save the boxplot images
    """
    # Make sure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Set style
    sns.set(style="whitegrid")

    # Group data by day and calculate the daily metrics
    # This creates a new DataFrame with metrics aggregated by day
    daily_metrics = metrics_df.groupby(['volunteer', 'day']).agg({
        'ssim': 'mean',
        'psnr': 'mean',
        'nrmse': 'mean'
    }).reset_index()

    # Create a figure for the combined boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Customize flier (outlier) properties to reduce visual clutter
    flierprops = dict(
        marker='o',
        markerfacecolor='gray',
        markersize=3,
        linestyle='none',
        alpha=0.5,
        markeredgecolor='gray'
    )

    # SSIM boxplot for daily metrics
    sns.boxplot(y=daily_metrics['ssim'], ax=axes[0], flierprops=flierprops, color='#6495ED')
    axes[0].set_title('Overall SSIM', fontsize=14)
    axes[0].set_ylabel('SSIM', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # PSNR boxplot for daily metrics
    sns.boxplot(y=daily_metrics['psnr'], ax=axes[1], flierprops=flierprops, color='#6495ED')
    axes[1].set_title('Overall PSNR', fontsize=14)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # NRMSE boxplot for daily metrics
    sns.boxplot(y=daily_metrics['nrmse'], ax=axes[2], flierprops=flierprops, color='#6495ED')
    axes[2].set_title('Overall NRMSE', fontsize=14)
    axes[2].set_ylabel('NRMSE', fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a title
    plt.suptitle('Combined Performance Metrics Across All Volunteers', fontsize=16, y=1.05)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'combined_metrics_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate and print summary statistics on the daily metrics
    print("\nOverall Statistics (by day):")
    print(f"SSIM: Mean = {daily_metrics['ssim'].mean():.4f}, Median = {daily_metrics['ssim'].median():.4f}")
    print(f"PSNR: Mean = {daily_metrics['psnr'].mean():.4f}, Median = {daily_metrics['psnr'].median():.4f}")
    print(f"NRMSE: Mean = {daily_metrics['nrmse'].mean():.4f}, Median = {daily_metrics['nrmse'].median():.4f}")

    # Also create a version that includes scan information if available
    if 'scan' in metrics_df.columns:
        scan_metrics = metrics_df.groupby(['volunteer', 'day', 'scan']).agg({
            'ssim': 'mean',
            'psnr': 'mean',
            'nrmse': 'mean'
        }).reset_index()

        # Create another figure for scan-level metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # SSIM boxplot for scan metrics
        sns.boxplot(y=scan_metrics['ssim'], ax=axes[0], flierprops=flierprops, color='#6495ED')
        axes[0].set_title('Overall SSIM (by scan)', fontsize=14)
        axes[0].set_ylabel('SSIM', fontsize=12)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # PSNR boxplot for scan metrics
        sns.boxplot(y=scan_metrics['psnr'], ax=axes[1], flierprops=flierprops, color='#6495ED')
        axes[1].set_title('Overall PSNR (by scan)', fontsize=14)
        axes[1].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # NRMSE boxplot for scan metrics
        sns.boxplot(y=scan_metrics['nrmse'], ax=axes[2], flierprops=flierprops, color='#6495ED')
        axes[2].set_title('Overall NRMSE (by scan)', fontsize=14)
        axes[2].set_ylabel('NRMSE', fontsize=12)
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)

        plt.suptitle('Combined Performance Metrics By Scan', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'combined_metrics_by_scan_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_time = time.time()

    save_path = '/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/Results_human/GM/axial'
    os.makedirs(save_path, exist_ok=True)
    image_save_path = os.path.join(save_path, "images")
    os.makedirs(image_save_path, exist_ok=True)

    scale_data = 4578.9688
    scale_params = 13.9984

    # Collect paths
    data_paths = glob.glob(r'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/subjects_model_2/*/*/axial/*/dataset/*.mat')
    data_paths = sorted(data_paths)
    param_paths = glob.glob(r'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/subjects_model_2/*/*/axial/*/params/*.mat')
    param_paths = sorted(param_paths)
    label_paths = glob.glob(r'/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/subjects_model_2y/*/*/axial/*/labels/*.mat')
    label_paths = sorted(label_paths)

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

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    channel_names = ['Kssw', 'MT_perc', 'B0', 'B1', 'T1', 'T2']
    # Metrics storage
    metrics = {
        'volunteer': [],
        'day': [],
        'batch': [],
        'ssim': [],
        'psnr': [],
        'nrmse': []
    }
        # Add per-metric columns
    for i in range(len(channel_names)):
        metrics[f'ssim_metric_{i}'] = []
        metrics[f'psnr_metric_{i}'] = []
        metrics[f'nrmse_metric_{i}'] = []



    # Process each batch
    with torch.no_grad():  # Add no_grad for inference
        for batch, (X, p, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # Extract volunteer and day info from the path
            path = data_paths[batch]
            path_parts = path.split('/')

            # Find the volunteer part (volXX)
            volunteer = None
            day = None
            for part in path_parts:
                if part.startswith('vol'):
                    volunteer = part
                # Look for day1, day2, or day3_runX pattern
                elif part.startswith('day'):
                    day = part

            # Process data
            X, p, y = X.to(device), p.to(device), y.to(device)

            # Forward pass
            y_pred = model(X, p)

            # Adjust dimensions for metrics calculation
            y_pred_perm = y_pred.permute(0, 3, 1, 2)  # [B, C, H, W]

            ssim_per_metric, ssim_val = calc_ssim(ssim_fn, y_pred_perm, y)
            psnr_per_metric, psnr_val = calc_psnr(psnr_fn, y_pred_perm, y)
            nrmse_per_metric, nrmse_val = calc_nrmse(y_pred_perm, y)

            # Store metrics - IMPORTANT: Move tensors to CPU and convert to Python scalar values
            metrics['volunteer'].append(volunteer)
            metrics['day'].append(day)
            metrics['batch'].append(batch)
            # Convert tensor to scalar value
            metrics['ssim'].append(ssim_val.cpu().item())
            metrics['psnr'].append(psnr_val.cpu().item())
            metrics['nrmse'].append(nrmse_val.cpu().item())

            # Per-metric values are already scalar values from the calc_* functions
            for i in range(len(channel_names)):
                metrics[f'ssim_metric_{i}'].append(ssim_per_metric[i])
                metrics[f'psnr_metric_{i}'].append(psnr_per_metric[i])
                metrics[f'nrmse_metric_{i}'].append(nrmse_per_metric[i])

            # Save visualization


            # Print progress
            if batch % 25 == 0:
                save_img_path = os.path.join(image_save_path, f"{volunteer}_{day}_batch_{batch}.jpg")
                plot_true_and_pred_sequences(y, y_pred, save_img_path)
                print(f"Batch {batch}: Volunteer={volunteer}, Day={day}, SSIM={ssim_val:.4f}, PSNR={psnr_val:.4f}, NRMSE={nrmse_val:.4f}")

    # Calculate average metrics per volunteer
    volunteers = set(metrics['volunteer'])
    print("\n=== Metrics Summary ===")

    for vol in volunteers:
        vol_indices = [i for i, v in enumerate(metrics['volunteer']) if v == vol]
        vol_ssim = sum([metrics['ssim'][i] for i in vol_indices]) / len(vol_indices)
        vol_psnr = sum([metrics['psnr'][i] for i in vol_indices]) / len(vol_indices)
        vol_nrmse = sum([metrics['nrmse'][i] for i in vol_indices]) / len(vol_indices)

        print(f"{vol}: SSIM={vol_ssim:.4f}, PSNR={vol_psnr:.4f}, NRMSE={vol_nrmse:.4f}")

    # Calculate overall average metrics
    avg_ssim = sum(metrics['ssim']) / len(metrics['ssim'])
    avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
    avg_nrmse = sum(metrics['nrmse']) / len(metrics['nrmse'])

    print(f"\nOverall: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.4f}, NRMSE={avg_nrmse:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(save_path, "metrics_results.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")

    # Create boxplots for visualization
    plots_dir = os.path.join(save_path, "boxplots")
    create_metrics_boxplots(metrics_df, plots_dir)
    create_combined_metrics_boxplots(metrics_df, plots_dir)

    # Create per-metric boxplots (new function)
    create_per_metric_boxplots(metrics_df, plots_dir)

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

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

    arguments["data_dir"] = "/home/sahar/Models/Dinor_revision/new_phantom/dinor_train/code_addition/human/subjects_model_2"
    arguments[
        "new_model_weight_path"] = "/home/sahar/tbmf/tbmf/final_scripts_for_publition_lab_git/model2_weights/with_pretrained_weights/checkpoint_epoch_348.pt"

    main(arguments)