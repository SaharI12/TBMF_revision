import scipy.io as sio

from transformer_architecture_prod import *
from functions import *
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch.nn.functional as F
import time
import glob
from torch.utils.data import Dataset, DataLoader
import warnings
import numpy as np
import h5py
from config_loader import load_config

warnings.filterwarnings('ignore')

# Constants and configuration
CHANNEL_RANGES = {
    0: 100.0,  # Kssw
    1: 27.27,  # MT_perc
    2: 1.7,  # B0
    3: 1.0,  # B1
    4: 3000.0,  # T1
    5: 1000.0,  # T2
}

CHANNEL_NAMES = ['KSSW', 'MT%', 'B₀', 'B₁', 'T₁', 'T₂']
CHANNEL_SCALES = [100.0, 27.27, 2.7, 3.4944, 10000.0, 1000.0]


def awgn(signal, snr_db, signal_power_mode='measured', signal_power_db=None):
    """
    Python equivalent of MATLAB's awgn function

    Args:
        signal (torch.Tensor): Input signal
        snr_db (float): Signal-to-noise ratio in dB
        signal_power_mode (str): 'measured' to measure signal power, 'specified' to use provided power
        signal_power_db (float): Signal power in dB (only used if signal_power_mode='specified')

    Returns:
        tuple: (noisy_signal, actual_snr_db, signal_power_db, noise_power_db)

    """
    torch.manual_seed(42)  # For reproducibility
    if snr_db == float('inf'):
        return signal, float('inf'), float('inf'), float('-inf')

    # Measure or specify signal power
    if signal_power_mode == 'measured':
        # Calculate signal power: E[|x|^2]
        signal_power_linear = torch.mean(signal ** 2)
        signal_power_db_calc = 10 * torch.log10(signal_power_linear)
    elif signal_power_mode == 'specified' and signal_power_db is not None:
        signal_power_db_calc = signal_power_db
        signal_power_linear = 10 ** (signal_power_db / 10)
    else:
        raise ValueError("Invalid signal_power_mode or missing signal_power_db")

    # Calculate required noise power
    # SNR_dB = 10*log10(P_signal/P_noise)
    # P_noise = P_signal / 10^(SNR_dB/10)
    noise_power_linear = signal_power_linear / (10 ** (snr_db / 10))
    noise_power_db = 10 * torch.log10(noise_power_linear)

    # Generate white Gaussian noise
    noise_std = torch.sqrt(noise_power_linear)
    noise = torch.randn_like(signal) * noise_std

    # Add noise to signal
    noisy_signal = signal + noise

    # Calculate actual SNR for verification
    actual_noise_power = torch.mean(noise ** 2)
    actual_snr_linear = signal_power_linear / actual_noise_power
    actual_snr_db = 10 * torch.log10(actual_snr_linear)

    return noisy_signal, actual_snr_db.item(), signal_power_db_calc.item(), noise_power_db.item()


class TryDataset_v0(Dataset):
    def __init__(self, data_dir, param_dir, labels_dir, scale_data, scale_param):
        self.data_paths = data_dir
        self.param_paths = param_dir
        self.label_paths = labels_dir
        self.scale_data = scale_data
        self.scale_params = scale_param

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        if 'bay4_volunteer3_mar_7_2022' in self.data_paths[index]:
            dataset_idx = sio.loadmat(self.data_paths[index])['res']
        else:
            dataset_idx = sio.loadmat(self.data_paths[index])['res'] / self.scale_data

        params_idx = sio.loadmat(self.param_paths[index])['res'] / self.scale_params
        labels_idx = sio.loadmat(self.label_paths[index])['res']

        labels_idx[0, :, :] = labels_idx[0, :, :] / 100  # KSSW
        labels_idx[1, :, :] = labels_idx[1, :, :]  / 27.27  # MT_perc
        labels_idx[2, :, :] = (labels_idx[2, :, :] + 1) / (1.7 + 1)  # B0
        labels_idx[3, :, :] = labels_idx[3, :, :] / 3.4944  # B1
        labels_idx[4, :, :] = labels_idx[4, :, :] / 10000  # T1

        if ('bay1_volunteer1' in self.data_paths[index] or
                'bay4_06_22_2020_volunteer1' in self.data_paths[index] or
                'bay4_volunteer2_2020_07_17' in self.data_paths[index]):
            labels_idx[5, :, :] = labels_idx[5, :, :] * 1
        else:
            labels_idx[5, :, :] = labels_idx[5, :, :] / 1000

        labels_idx[np.isnan(labels_idx)] = 0
        dataset_idx[np.isnan(dataset_idx)] = 0

        return dataset_idx.astype(np.float32), params_idx.astype(np.float32), labels_idx.astype(np.float32)

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
        if 'bay4_volunteer3_mar_7_2022' in self.data_paths[index]:
            dataset_idx = h5py.File(self.data_paths[index])['res'][:]
        else:
            dataset_idx = h5py.File(self.data_paths[index])['res'] [:]/ self.scale_data



        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels_idx = h5py.File(self.label_paths[index])['res'][:]

        labels_idx[0, :, :] = labels_idx[0, :, :] / 100  # KSSW
        labels_idx[1, :, :] = labels_idx[1, :, :] *100/ 27.27  # MT_perc
        labels_idx[2, :, :] = (labels_idx[2, :, :] + 1) / (1.7 + 1)  # B0
        labels_idx[3, :, :] = labels_idx[3, :, :] / 3.4944  # B1
        labels_idx[4, :, :] = labels_idx[4, :, :] / 10000  # T1

        if ('bay1_volunteer1' in self.data_paths[index] or
                'bay4_06_22_2020_volunteer1' in self.data_paths[index] or
                'bay4_volunteer2_2020_07_17' in self.data_paths[index]):
            labels_idx[5, :, :] = labels_idx[5, :, :] * 1
        else:
            labels_idx[5, :, :] = labels_idx[5, :, :] / 1000
        labels_idx[np.isnan(labels_idx)] = 0
        dataset_idx[np.isnan(dataset_idx)] = 0

        return dataset_idx.astype(np.float32), params_idx.astype(np.float32), labels_idx.astype(np.float32)


class AdvancedNoiseRobustnessAnalyzer:
    """Advanced MRI Noise Robustness Analyzer using MATLAB-equivalent AWGN function"""

    def __init__(self, device, num_parameters=6):
        self.device = device
        self.num_parameters = num_parameters
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)

    def add_awgn_noise(self, signal, snr_db, signal_power_mode='measured'):
        """Add AWGN noise using MATLAB-equivalent awgn function"""
        return awgn(signal, snr_db, signal_power_mode)

    def calculate_metrics_batch(self, y_pred, y_true):
        """Calculate all metrics efficiently for a batch"""
        batch_results = {'ssim': [], 'psnr': [], 'nrmse': []}

        for ch in range(self.num_parameters):
            # Extract channel data
            y_ch = y_true[:, ch:ch + 1, :, :]
            pred_ch = y_pred[:, ch:ch + 1, :, :]

            if ch ==2 :
                # current_image = (test_labels[0, 2, :, :] * 2.7) - 1
                # current_image = current_image.copy()
                y_ch[y_ch == 1/2.7] = 0

            # Normalize for SSIM/PSNR
            y_min, y_max = y_ch.min(), y_ch.max()
            if y_max - y_min > 1e-8:
                y_ch_norm = (y_ch - y_min) / (y_max - y_min)
                pred_ch_norm = torch.clamp((pred_ch - y_min) / (y_max - y_min), 0, 1)
            else:
                y_ch_norm = torch.zeros_like(y_ch)
                pred_ch_norm = torch.zeros_like(pred_ch)


            # Calculate metrics
            ssim_val = self.ssim_fn(pred_ch_norm, y_ch_norm)
            psnr_val = self.psnr_fn(pred_ch_norm, y_ch_norm)


            # NRMSE calculation
            mse = F.mse_loss(pred_ch, y_ch)
            rmse = torch.sqrt(mse)
            actual_rmse = rmse * CHANNEL_SCALES[ch]
            nrmse = actual_rmse / CHANNEL_RANGES[ch]

            batch_results['ssim'].append(ssim_val.item())
            batch_results['psnr'].append(psnr_val.item())
            batch_results['nrmse'].append(nrmse.item())

        return batch_results

    def analyze_noise_robustness(self, model, test_loader, snr_values_db=None, debug=True):
        """Advanced noise robustness analysis using AWGN from 5 to 80 dB"""
        if snr_values_db is None:
            # Linear progression from 80 to 5 dB in steps of 5
            snr_values_db = list(range(80, 0, -5))  # [80, 75, 70, ..., 10, 5]

        print(f"\nAnalyzing noise robustness using AWGN for {len(snr_values_db)} SNR levels")
        print(f"SNR range: {snr_values_db[0]} to {snr_values_db[-1]} dB")

        # Initialize checkpoints storage
        results = {
            'snr_db': snr_values_db,
            'ssim': np.zeros((self.num_parameters, len(snr_values_db))),
            'psnr': np.zeros((self.num_parameters, len(snr_values_db))),
            'nrmse': np.zeros((self.num_parameters, len(snr_values_db))),
            'actual_snr_db': [],
            'signal_power_db': [],
            'noise_power_db': []
        }

        model.eval()

        with torch.inference_mode():
            for snr_idx, target_snr_db in enumerate(tqdm(snr_values_db, desc="Processing SNR levels")):

                batch_metrics = {'ssim': [], 'psnr': [], 'nrmse': []}
                actual_snrs_db = []
                signal_powers_db = []
                noise_powers_db = []

                for batch_idx, (X, p, y) in enumerate(test_loader):
                    X, p, y = X.to(self.device), p.to(self.device), y.to(self.device)

                    # Add AWGN noise
                    X_noisy, actual_snr_db, signal_power_db, noise_power_db = self.add_awgn_noise(X, target_snr_db)

                    actual_snrs_db.append(actual_snr_db)
                    signal_powers_db.append(signal_power_db)
                    noise_powers_db.append(noise_power_db)

                    # Get predictions
                    predictions = model(X_noisy, p)
                    predictions = torch.permute(predictions, (0, 3, 1, 2))

                    # Calculate metrics
                    metrics = self.calculate_metrics_batch(predictions, y)

                    for metric_type in batch_metrics:
                        batch_metrics[metric_type].append(metrics[metric_type])

                    # Debug output for first batch of key SNR values
                    if batch_idx == 0 and debug and target_snr_db in [80, 60, 40, 20, 10, 5]:
                        print(f"\nSNR={target_snr_db}dB - Mean SSIM: {np.mean(metrics['ssim']):.4f}, "
                              f"PSNR: {np.mean(metrics['psnr']):.2f}dB, "
                              f"NRMSE: {np.mean(metrics['nrmse']):.4f}")
                        print(
                            f"  Signal Power: {signal_power_db:.2f}dB, Noise Power: {noise_power_db:.2f}dB, Actual SNR: {actual_snr_db:.2f}dB")

                # Average across all batches
                for metric_type in ['ssim', 'psnr', 'nrmse']:
                    results[metric_type][:, snr_idx] = np.mean(batch_metrics[metric_type], axis=0)

                results['actual_snr_db'].append(np.mean(actual_snrs_db))
                results['signal_power_db'].append(np.mean(signal_powers_db))
                results['noise_power_db'].append(np.mean(noise_powers_db))

        return results

    def save_numpy_results(self, results, save_dir):
        """
        Save numpy arrays for individual parameter values and average values

        Args:
            results (dict): Results dictionary from analyze_noise_robustness
            save_dir (str): Directory to save the numpy files
        """
        print(f"\nSaving numpy checkpoints to: {save_dir}")

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save SNR values (common for all files)
        snr_array = np.array(results['snr_db'])
        np.save(os.path.join(save_dir, 'snr_values_db.npy'), snr_array)
        print(f"Saved SNR values: {os.path.join(save_dir, 'snr_values_db.npy')}")

        # Save individual parameter checkpoints
        for param_idx, param_name in enumerate(CHANNEL_NAMES):
            # Create safe filename (replace special characters)
            safe_param_name = param_name.replace('₀', '0').replace('₁', '1').replace('%', 'perc')

            # Save individual parameter metrics
            param_data = {
                'snr_db': snr_array,
                'ssim': results['ssim'][param_idx, :],
                'psnr': results['psnr'][param_idx, :],
                'nrmse': results['nrmse'][param_idx, :]
            }

            filename = f'parameter_{param_idx}_{safe_param_name}_metrics.npy'
            np.save(os.path.join(save_dir, filename), param_data)
            print(f"Saved {param_name} metrics: {os.path.join(save_dir, filename)}")

        # Calculate and save average values
        avg_ssim = np.mean(results['ssim'], axis=0)
        avg_psnr = np.mean(results['psnr'], axis=0)
        avg_nrmse = np.mean(results['nrmse'], axis=0)

        average_data = {
            'snr_db': snr_array,
            'avg_ssim': avg_ssim,
            'avg_psnr': avg_psnr,
            'avg_nrmse': avg_nrmse,
            'individual_ssim': results['ssim'],  # Include individual data for reference
            'individual_psnr': results['psnr'],
            'individual_nrmse': results['nrmse']
        }

        avg_filename = 'average_metrics_all_parameters.npy'
        np.save(os.path.join(save_dir, avg_filename), average_data)
        print(f"Saved average metrics: {os.path.join(save_dir, avg_filename)}")

        # Save additional analysis data
        additional_data = {
            'snr_db': snr_array,
            'actual_snr_db': np.array(results['actual_snr_db']),
            'signal_power_db': np.array(results['signal_power_db']),
            'noise_power_db': np.array(results['noise_power_db']),
            'channel_names': CHANNEL_NAMES,
            'channel_ranges': list(CHANNEL_RANGES.values()),
            'channel_scales': CHANNEL_SCALES
        }

        additional_filename = 'additional_analysis_data.npy'
        np.save(os.path.join(save_dir, additional_filename), additional_data)
        print(f"Saved additional data: {os.path.join(save_dir, additional_filename)}")


    def plot_results(self, results, save_path=None):
        """Enhanced plotting with SNR in dB and individual parameter plots"""
        snr_values_db = results['snr_db'].copy()

        # Create figure with 1x3 subplots for individual parameters
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, len(CHANNEL_NAMES)))

        # Plot SSIM
        ax = axes[0]
        for i, param_name in enumerate(CHANNEL_NAMES):
            ax.plot(snr_values_db, results['ssim'][i, :],
                    marker='o', color=colors[i], linewidth=2.5, markersize=5,
                    alpha=0.8, label=param_name)

        ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('SSIM', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_ylim([0, 1])
        ax.set_xlim([min(snr_values_db) - 2, max(snr_values_db) + 2])

        # Plot PSNR
        ax = axes[1]
        for i, param_name in enumerate(CHANNEL_NAMES):
            ax.plot(snr_values_db, results['psnr'][i, :],
                    marker='s', color=colors[i], linewidth=2.5, markersize=5,
                    alpha=0.8, label=param_name)

        ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_xlim([min(snr_values_db) - 2, max(snr_values_db) + 2])

        # Plot NRMSE
        ax = axes[2]
        for i, param_name in enumerate(CHANNEL_NAMES):
            ax.plot(snr_values_db, results['nrmse'][i, :],
                    marker='^', color=colors[i], linewidth=2.5, markersize=5,
                    alpha=0.8, label=param_name)

        ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('NRMSE', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.set_xlim([min(snr_values_db) - 2, max(snr_values_db) + 2])

        plt.tight_layout()

        if save_path:
            individual_save_path = save_path.replace('.png', '_individual_parameters.png')
            plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
            print(f"\nIndividual parameters plot saved to: {individual_save_path}")

        plt.show()

        # Create separate figure for averages
        self.plot_average_performance(results, save_path)

    def plot_average_performance(self, results, save_path=None):
        """Create separate figure with 3 subplots for average performance with SNR in dB"""
        snr_values_db = results['snr_db']

        # Calculate averages across all parameters
        avg_ssim = np.mean(results['ssim'], axis=0)
        avg_psnr = np.mean(results['psnr'], axis=0)
        avg_nrmse = np.mean(results['nrmse'], axis=0)

        # Create figure with 1x3 subplots for averages
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot Average SSIM
        ax = axes[0]
        ax.plot(snr_values_db, avg_ssim,
                color='tab:blue', marker='o', linewidth=3, markersize=6, alpha=0.8)

        ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('SSIM', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([min(snr_values_db) - 2, max(snr_values_db) + 2])

        # Plot Average PSNR
        ax = axes[1]
        ax.plot(snr_values_db, avg_psnr,
                color='tab:orange', marker='s', linewidth=3, markersize=6, alpha=0.8)

        ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(snr_values_db) - 2, max(snr_values_db) + 2])

        # Plot Average NRMSE
        ax = axes[2]
        ax.plot(snr_values_db, avg_nrmse,
                color='tab:green', marker='^', linewidth=3, markersize=6, alpha=0.8)

        ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
        ax.set_ylabel('NRMSE', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xlim([min(snr_values_db) - 2, max(snr_values_db) + 2])

        plt.tight_layout()

        if save_path:
            average_save_path = save_path.replace('.png', '_average_performance.png')
            plt.savefig(average_save_path, dpi=300, bbox_inches='tight')
            print(f"Average performance plot saved to: {average_save_path}")

        plt.show()



class EnhancedVisualizationTools:
    """Enhanced visualization tools with AWGN noise addition"""

    def __init__(self, device):
        self.device = device
        self.setup_colormaps()

    def setup_colormaps(self):
        """Setup colormaps exactly as in the provided visualization code"""
        try:
            self.buda_map = plt.get_cmap('RdBu_r')
            self.lipari_map = plt.get_cmap('RdGy')
        except:
            self.buda_map = plt.get_cmap('RdBu_r')
            self.lipari_map = plt.get_cmap('RdGy')

        # Create modified colormaps with black minimum exactly as in provided code
        original_map = plt.get_cmap('viridis')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0  # minimum value is set to black
        self.b_viridis = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = self.buda_map
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        self.b_bwr = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = self.lipari_map
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        self.b_rdgy = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = plt.get_cmap('winter')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        self.winter = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

    def visualize_noise_effects(self, model, test_loader, snr_values_db=None,
                                sample_idx=0, save_path=None):
        """Enhanced noise effect visualization with AWGN and SNR in dB"""
        if snr_values_db is None:
            snr_values_db = [float('inf'),60, 40, 20, 10, 5]

        model.eval()

        # Get sample
        for i, (X, p, y) in enumerate(test_loader):
            if i == sample_idx:
                X, p, y = X.to(self.device), p.to(self.device), y.to(self.device)
                break

        analyzer = AdvancedNoiseRobustnessAnalyzer(self.device)

        for target_snr_db in snr_values_db:
            # Handle clean case
            if target_snr_db == float('inf'):
                snr_str = "clean"
            else:
                snr_str = f"{target_snr_db}dB"
                print(f"\nProcessing AWGN SNR: {snr_str} for sample index {sample_idx}")

            # Update save path
            if save_path:
                current_save_path = save_path.replace('.png', f'_AWGN_SNR_{snr_str}.png')
            else:
                current_save_path = None

            self._create_noise_visualization_3x6(model, analyzer, X, p, y, target_snr_db,
                                                 sample_idx, current_save_path)

    def _create_noise_visualization_3x6(self, model, analyzer, X, p, y, target_snr_db,
                                        sample_idx, save_path):
        """Create 3x6 subplot visualization with AWGN noise"""
        # Generate noisy input and prediction
        if target_snr_db == float('inf'):
            X_noisy, actual_snr_db = X, float('inf')
            title_snr = "Clean Signal"
        else:
            X_noisy, actual_snr_db, signal_power_db, noise_power_db = analyzer.add_awgn_noise(X, target_snr_db)
            title_snr = f"AWGN: Target SNR = {target_snr_db} dB (Actual: {actual_snr_db:.1f} dB)"

        with torch.inference_mode():
            prediction = model(X_noisy, p)
            prediction = torch.permute(prediction, (0, 3, 1, 2))

        # Convert to numpy for visualization
        test_labels = y.cpu().detach().numpy()
        predicted_labels = prediction.cpu().detach().numpy()
        predicted_labels = np.transpose(predicted_labels, (0, 2, 3, 1))
        X_noisy_np = X_noisy.cpu().numpy()

        # Create 3x6 subplots with black background - INCREASED SIZE
        fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(30, 18))  # Increased from (24, 15)

        # Set black background
        fig.patch.set_facecolor('black')
        for i in range(3):
            for j in range(6):
                axs[i, j].set_facecolor('black')

        # Row 1: Ground Truth
        current_image = test_labels[0, 0, :, :] * 100
        axs[2, 0].imshow(current_image, vmin=0, vmax=100, cmap='magma')
        axs[2, 0].set_title(label="Kssw", fontsize=16, color='white')  # Reduced from 20
        axs[2, 0].set_ylabel(ylabel="Ground\nTruth", rotation=0, labelpad=60, fontsize=16,
                             color='white')  # Reduced from 20
        axs[2, 0].set_yticks([])
        axs[2, 0].set_xticks([])

        current_image = test_labels[0, 1, :, :] * 27.27
        axs[2, 1].imshow(current_image, vmin=0, vmax=27.27, cmap=self.b_viridis)
        axs[2, 1].set_title(label="MT_perc", fontsize=16, color='white')  # Reduced from 20
        axs[2, 1].axis('off')

        current_image = (test_labels[0, 2, :, :] * 2.7) - 1
        current_image = current_image.copy()
        current_image[current_image == 0] = np.nan
        cmap = self.b_bwr.copy()
        cmap.set_bad('black')
        axs[2, 2].imshow(current_image, vmin=-0.6, vmax=0.6, cmap=cmap)
        axs[2, 2].set_title(label=u'B\u2080', fontsize=16, color='white')  # Reduced from 20
        axs[2, 2].axis('off')

        current_image = test_labels[0, 3, :, :] * 3.4944
        axs[2, 3].imshow(current_image, vmin=0.5, vmax=1.5, cmap=self.b_rdgy)
        axs[2, 3].set_title(label=u'B\u2081', fontsize=16, color='white')  # Reduced from 20
        axs[2, 3].axis('off')

        current_image = test_labels[0, 4, :, :] * 10000
        axs[2, 4].imshow(current_image, vmin=0, vmax=3000, cmap='hot')
        axs[2, 4].set_title(label=u'T\u2081', fontsize=16, color='white')  # Reduced from 20
        axs[2, 4].axis('off')

        current_image = test_labels[0, 5, :, :] * 1000
        axs[2, 5].imshow(current_image, vmin=0, vmax=200, cmap=self.winter)
        axs[2, 5].set_title(label=u'T\u2082', fontsize=16, color='white')  # Reduced from 20
        axs[2, 5].axis('off')

        # Row 2: Predictions
        current_image = predicted_labels[0, :, :, 0] * 100
        axs[1, 0].imshow(current_image, vmin=0, vmax=100, cmap='magma')
        axs[1, 0].set_ylabel(ylabel="Predicted\nImages", rotation=0, labelpad=60, fontsize=16,
                             color='white')  # Reduced from 20
        axs[1, 0].set_yticks([])
        axs[1, 0].set_xticks([])

        current_image = predicted_labels[0, :, :, 1] * 27.27
        axs[1, 1].imshow(current_image, vmin=0, vmax=27.27, cmap=self.b_viridis)
        axs[1, 1].axis('off')

        current_image = (predicted_labels[0, :, :, 2] * 2.7) - 1
        axs[1, 2].imshow(current_image, vmin=-0.6, vmax=0.6, cmap=self.b_bwr)
        axs[1, 2].axis('off')

        current_image = predicted_labels[0, :, :, 3] * 3.4944
        axs[1, 3].imshow(current_image, vmin=0.5, vmax=1.5, cmap=self.b_rdgy)
        axs[1, 3].axis('off')

        current_image = predicted_labels[0, :, :, 4] * 10000
        axs[1, 4].imshow(current_image, vmin=0, vmax=3000, cmap='hot')
        axs[1, 4].axis('off')

        current_image = predicted_labels[0, :, :, 5] * 1000
        axs[1, 5].imshow(current_image, vmin=0, vmax=200, cmap=self.winter)
        axs[1, 5].axis('off')

        # Row 3: Input (middle slice)
        input_slice_idx = X_noisy_np.shape[-1] // 2
        for param_idx in range(6):
            input_image = X_noisy_np[0, :, :, input_slice_idx]
            axs[0, param_idx].imshow(input_image, cmap='gray')

            if param_idx == 0:
                axs[0, param_idx].set_ylabel("Input", rotation=0, labelpad=60,
                                             fontsize=16, fontweight='bold', color='white')  # Reduced from 20
                axs[0, param_idx].set_yticks([])
                axs[0, param_idx].set_xticks([])
            else:
                axs[0, param_idx].axis('off')

        # Adjust subplot spacing
        current_image_shape = predicted_labels[0, :, :, 0].shape
        if current_image_shape == (116, 116):
            plt.subplots_adjust(top=0.92, bottom=0.20, hspace=-0.65, wspace=0.05)  # Increased bottom margin
        elif current_image_shape == (116, 88):
            plt.subplots_adjust(top=0.92, bottom=0.20, hspace=-0.5, wspace=0.05)
        elif current_image_shape == (126, 88):
            plt.subplots_adjust(top=0.92, bottom=0.20, hspace=-0.4, wspace=0.05)
        elif current_image_shape == (144, 88):
            plt.subplots_adjust(top=0.92, bottom=0.20, hspace=-0.5, wspace=0.05)
        elif current_image_shape == (126, 144):
            plt.subplots_adjust(top=0.92, bottom=0.20, hspace=-0.7, wspace=0.05)
        else:
            plt.subplots_adjust(top=0.92, bottom=0.20, hspace=0.1, wspace=0.05)

        # Add colorbars with better spacing
        cbar_list = ['magma', self.b_viridis, self.b_bwr, self.b_rdgy, 'hot', self.winter]
        vmin = [0, 0, -0.6, 0.5, 0, 0]
        vmax = [100, 27.27, 0.6, 1.5, 3000, 200]

        # Calculate colorbar dimensions with spacing
        total_width = 0.85  # Total width for all colorbars
        n_bars = 6
        gap_width = 0.02  # Gap between colorbars
        total_gap_width = gap_width * (n_bars - 1)
        cbar_width = (total_width - total_gap_width) / n_bars
        start_x = 0.075  # Starting x position

        for j in range(6):
            # Calculate x position for each colorbar with gaps
            x_pos = start_x + j * (cbar_width + gap_width)

            cax = fig.add_axes([x_pos, 0.08, cbar_width, 0.025])  # Reduced height and moved up
            cax.set_facecolor('black')
            sm = cm.ScalarMappable(cmap=cbar_list[j], norm=plt.Normalize(vmin=vmin[j], vmax=vmax[j]))
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
                                ticks=np.linspace(vmin[j], vmax[j], 3))  # Reduced ticks from 5 to 3
            cbar.ax.tick_params(colors='white', labelsize=10)  # Reduced font size

        plt.suptitle(f'{title_snr} - Sample {sample_idx}',
                     fontsize=20, fontweight='bold', color='white')  # Kept title size larger

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"Visualization saved to: {save_path}")

        plt.show()
        plt.close()


def main(args):
    """Enhanced main function with AWGN noise analysis"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_time = time.time()

    # Setup paths
    savefig_path = 'output/Noise_addition'
    os.makedirs(savefig_path, exist_ok=True)

    # Configuration
    scale_data = 4578.9688
    scale_params = 13.9984

    # Load data paths
    base_path = r"data/axial"
    data_paths = sorted(glob.glob(os.path.join(base_path, r'*/dataset/*.h5')))
    param_paths = sorted(glob.glob(os.path.join(base_path, r'*/params/*.h5')))
    label_paths = sorted(glob.glob(os.path.join(base_path, r'*/labels/*.h5')))

    print(f"Found {len(data_paths)} data files, {len(param_paths)} param files, {len(label_paths)} label files")

    # Create dataset and model
    dataset = TryDataset_v2(
        data_dir=data_paths,
        param_dir=param_paths,
        labels_dir=label_paths,
        scale_data=scale_data,
        scale_param=scale_params
    )

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = create_model_v0(args, weights_path=args["new_model_weight_path"]).to(device)
    model.eval()

    # Initialize analyzer and visualization tools
    analyzer = AdvancedNoiseRobustnessAnalyzer(device)
    visualizer = EnhancedVisualizationTools(device)

    # Define SNR values:  60 to 10 dB in steps of 2
    snr_values_db =  list(range(60, 5, -2))
    print(f"Testing {len(snr_values_db)} SNR levels: Clean + {snr_values_db[1]} to {snr_values_db[-1]} dB")

    # Create visualizations for key SNR levels
    print("\n" + "=" * 60)
    print("CREATING AWGN NOISE EFFECT VISUALIZATIONS")
    print("=" * 60)

    vis_snr_values_db = [60, 40, 30, 25, 20, 10]
    visualizer.visualize_noise_effects(
        model=model,
        test_loader=test_loader,
        snr_values_db=vis_snr_values_db,
        sample_idx=50,
        save_path=os.path.join(savefig_path, 'awgn_noise_effects_sample_1000.png')
    )

    # Run comprehensive noise robustness analysis
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE AWGN NOISE ROBUSTNESS ANALYSIS")
    print("=" * 60)

    results = analyzer.analyze_noise_robustness(
        model=model,
        test_loader=test_loader,
        snr_values_db=snr_values_db,
        debug=True
    )

    # Generate and save plots
    save_plot_path = os.path.join(savefig_path, 'awgn_noise_robustness_60_to_10_dB.png')
    analyzer.plot_results(
        results=results,
        save_path=save_plot_path
    )

    numpy_save_path = os.path.join(savefig_path, 'numpy_results')
    analyzer.save_numpy_results(results, numpy_save_path)


if __name__ == '__main__':
    # Load configuration from config.yaml
    try:
        config_loader = load_config('config.yaml')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    arguments = {
        "device": device,
        "image_size": config_loader.get('model.img_size', 144),
        "sequence_len": config_loader.get('model.in_channels', 6),
        "patch_size": config_loader.get('model.patch_size', 9),
        "embedding_dim": config_loader.get('model.embedding_dim', 768),
        "dropout": config_loader.get('model.dropout', 0),
        "mlp_size": config_loader.get('model.mlp_size', 3072),
        "num_transformer_layers": config_loader.get('model.num_transformer_layers', 3),
        "num_heads": config_loader.get('model.num_heads', 4),
        "new_model_weight_path": config_loader.get('model.model2_path', 'checkpoints/model2.pt')
    }

    main(arguments)