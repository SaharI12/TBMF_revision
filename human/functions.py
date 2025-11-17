import glob
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import torch
import numpy as np
from transformer_architecture_prod import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.nn.functional import mse_loss

#--------Data loading----------------
class Load_Dataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.data_folder = self.config["data_paths"]



        self.data_paths = sorted(glob.glob(os.path.join(self.data_folder, "*/dataset/*.h5")))
        self.param_paths = sorted(glob.glob(os.path.join(self.data_folder, "*/params/*.h5")))
        self.label_paths = sorted(glob.glob(os.path.join(self.data_folder, "*/labels/*.h5")))

        self.scale_data = self.config["scale_data"]
        self.scale_params = self.config["scale_params"]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):

        dataset_idx = h5py.File(self.data_paths[index])['res'] [:]/ self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels = h5py.File(self.label_paths[index])['res'][:]

        labels[0, :, :] = labels[0, :, :] / 100  # KSSW
        labels[1, :, :] = labels[1, :, :]*100 / 27.27  # MT
        labels[2, :, :] = (labels[2, :, :] + 1) / (1.7 + 1)  # B0
        labels[3, :, :] = labels[3, :, :] / 3.4944  # B1
        labels[4, :, :] = labels[4, :, :] / 10000  # T1
        labels[5, :, :] = labels[5, :, :] / 1000  # T2


        return (torch.from_numpy(dataset_idx.astype(np.float32)),
                torch.from_numpy(params_idx.astype(np.float32)),
                torch.from_numpy(labels.astype(np.float32)))

#-----------Model Loading --------------------------------------
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

    model.conv_layers.append(nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.Sigmoid())

    try:
        model.load_state_dict(torch.load(weights_path, map_location=args["device"]))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    return model

#-------- Plot model predictions----------
class Plot_Metrics():
    def __init__(self, config):
        self.config = config
        self.dataset = config["dataset"]
        self.model = config['model']
        self.out_path_images = os.path.join(config["out_path"], "images")
        self.out_path_metrics = os.path.join(config["out_path"], "metrics")
        self.device = config["device"]
        self.seq = config["plot_specific_seq"]
        os.makedirs(self.out_path_images, exist_ok=True)
        os.makedirs(self.out_path_metrics, exist_ok=True)

        self.psnr = PeakSignalNoiseRatio(data_range=1).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.channel_names = ['Kssw', 'MT_perc', 'B0', 'B1', 'T1', 'T2']

        # Initialize metrics storage
        self.metrics = {'batch': [], 'ssim': [], 'psnr': [], 'nrmse': []}
        for i in range(len(self.channel_names)):
            self.metrics.update({
                f'ssim_metric_{i}': [], f'psnr_metric_{i}': [], f'nrmse_metric_{i}': []
            })

    def _create_colormaps(self):
        """Create and return colormaps for visualization"""
        buda_map = LinearSegmentedColormap.from_list('buda', cm_data_vik)
        lipari_map = LinearSegmentedColormap.from_list('lipari', cm_data_brok)

        maps = {}
        for name, cmap in [('viridis', plt.get_cmap('viridis')), ('buda', buda_map),
                           ('lipari', lipari_map), ('winter', plt.get_cmap('winter'))]:
            color_mat = cmap(np.arange(cmap.N))
            color_mat[0, 0:3] = 0
            maps[f'b_{name}'] = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        return maps

    def _plot_true_and_pred_sequences(self, test_labels, predicted_labels, save_path):
        pred, true = predicted_labels.cpu().detach().numpy(), test_labels.cpu().detach().numpy()
        maps = self._create_colormaps()

        fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))

        # Define parameters for each channel
        channels = [
            (100, 0, 100, 'magma'), (27.27, 0, 27.27, maps['b_viridis']),
            (2.7, -0.6, 0.6, maps['b_buda']), (3.4944, 0.5, 1.5, maps['b_lipari']),
            (10000, 0, 3000, 'hot'), (1000, 0, 200, maps['b_winter'])
        ]

        titles = ["Kssw", "MT_perc", u'B\u2080', u'B\u2081', u'T\u2081', u'T\u2082']

        for i, (scale, vmin, vmax, cmap) in enumerate(channels):
            # Ground truth
            img = true[0, i, :, :] * scale
            if i == 2:  # B0 special handling
                img = img - 1
                img = img.copy()
                img[img == 0] = np.nan
                cmap = cmap.copy()
                cmap.set_bad('black')

            axs[0, i].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0, i].set_title(titles[i], fontsize=20)
            axs[0, i].axis('off')

            # Predictions
            pred_img = pred[0, :, :, i] * scale
            if i == 2:
                pred_img = pred_img - 1
            axs[1, i].imshow(pred_img, vmin=vmin, vmax=vmax, cmap=cmap)
            axs[1, i].axis('off')

        # Set labels
        axs[0, 0].set_ylabel("Ground\nTruth", rotation=0, labelpad=50, fontsize=20)
        axs[1, 0].set_ylabel("Predicted\nImages", rotation=0, labelpad=50, fontsize=20)
        for ax in [axs[0, 0], axs[1, 0]]:
            ax.set_yticks([])
            ax.set_xticks([])

        # Adjust layout based on image shape
        shape_adjustments = {
            (116, 116): -0.65, (116, 88): -0.5, (126, 88): -0.4,
            (144, 88): -0.5, (126, 144): -0.7
        }
        hspace = shape_adjustments.get(pred_img.shape, -0.5)
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=hspace, wspace=0)

        # Add colorbars
        cbar_data = [('magma', 0, 100), (maps['b_viridis'], 0, 27.27), (maps['b_buda'], -0.6, 0.6),
                     (maps['b_lipari'], 0.5, 1.5), ('hot', 0, 3000), (maps['b_winter'], 0, 1000)]

        for j, (cmap, vmin, vmax) in enumerate(cbar_data):
            cax = fig.add_axes([axs[0, j].get_position().x0 + 0.01, 0.1,
                                axs[0, j].get_position().width, 0.02])
            sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, cax=cax, orientation='horizontal',
                         ticks=np.linspace(vmin, vmax, 5))

        plt.savefig(save_path)
        plt.close()

    def _calc_metrics(self, y_pred, y_true):
        """Calculate all metrics for each channel and overall average with B0 special handling"""
        slices = y_pred.shape[1]
        metrics = {'ssim': [], 'psnr': [], 'nrmse': []}

        for i in range(slices):
            pred_slice = y_pred[:, i, :, :].unsqueeze(1)
            true_slice = y_true[:, i, :, :].unsqueeze(1)

            # Special handling for B0 (channel 2)
            if i == 2:  # B0 channel
                # Apply the same transformations as in plotting
                # Scale and subtract 1 (assuming the same scaling as in plotting: * 2.7 - 1)
                pred_slice_processed = pred_slice * 2.7 - 1
                true_slice_processed = true_slice * 2.7 - 1

                # Create masks for valid (non-zero) values
                pred_mask = pred_slice_processed != 0
                true_mask = true_slice_processed != 0
                combined_mask = pred_mask & true_mask

                # Only calculate metrics on valid pixels (non-zero values)
                if combined_mask.sum() > 0:  # Ensure we have valid pixels
                    # Apply mask to both tensors
                    pred_masked = pred_slice_processed * combined_mask.float()
                    true_masked = true_slice_processed * combined_mask.float()

                    # For regions where mask is 0, set to a background value to avoid affecting metrics
                    # Use the minimum valid value as background
                    if combined_mask.sum() > 1:  # Need at least 2 pixels for meaningful metrics
                        min_val = torch.min(true_masked[combined_mask])
                        pred_masked[~combined_mask] = min_val
                        true_masked[~combined_mask] = min_val

                        # Calculate metrics on processed data
                        try:
                            ssim_val = self.ssim(pred_masked, true_masked).item()
                            psnr_val = self.psnr(pred_masked, true_masked).item()

                            # NRMSE calculation
                            mse = mse_loss(pred_masked, true_masked)
                            rmse = torch.sqrt(mse)
                            nrmse_val = (rmse / (torch.max(true_masked) - torch.min(true_masked))).item()

                        except:
                            # Fallback to original values if processing fails
                            ssim_val = self.ssim(pred_slice, true_slice).item()
                            psnr_val = self.psnr(pred_slice, true_slice).item()
                            mse = mse_loss(pred_slice, true_slice)
                            rmse = torch.sqrt(mse)
                            nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()
                    else:
                        # Fallback for insufficient valid pixels
                        ssim_val = self.ssim(pred_slice, true_slice).item()
                        psnr_val = self.psnr(pred_slice, true_slice).item()
                        mse = mse_loss(pred_slice, true_slice)
                        rmse = torch.sqrt(mse)
                        nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()
                else:
                    # No valid pixels, use original calculation
                    ssim_val = self.ssim(pred_slice, true_slice).item()
                    psnr_val = self.psnr(pred_slice, true_slice).item()
                    mse = mse_loss(pred_slice, true_slice)
                    rmse = torch.sqrt(mse)
                    nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()
            else:
                # Standard calculation for all other channels
                ssim_val = self.ssim(pred_slice, true_slice).item()
                psnr_val = self.psnr(pred_slice, true_slice).item()

                # NRMSE
                mse = mse_loss(pred_slice, true_slice)
                rmse = torch.sqrt(mse)
                nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()

            metrics['ssim'].append(ssim_val)
            metrics['psnr'].append(psnr_val)
            metrics['nrmse'].append(nrmse_val)

        # Calculate averages
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return metrics, avg_metrics

    def load_and_predict(self):
        with torch.no_grad():
            for batch, (X, p, y) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
                if self.seq is not None and batch % 19 != self.seq:
                    continue

                X, p, y = X.to(self.device), p.to(self.device), y.to(self.device)
                y_pred = self.model(X, p)
                y_pred_perm = y_pred.permute(0, 3, 1, 2)

                if self.config["wanted_output"]['image_plotting']:
                    save_path = os.path.join(self.out_path_images, f"batch_{batch}.jpg")
                    self._plot_true_and_pred_sequences(y, y_pred, save_path)

                if self.config["wanted_output"]['metrics']:
                    per_metric, avg_metrics = self._calc_metrics(y_pred_perm, y)

                    # Store checkpoints
                    self.metrics['batch'].append(batch)
                    for metric in ['ssim', 'psnr', 'nrmse']:
                        self.metrics[metric].append(avg_metrics[metric])
                        for i in range(len(self.channel_names)):
                            self.metrics[f'{metric}_metric_{i}'].append(per_metric[metric][i])

    def calculate_summary_statistics(self):
        """Calculate overall and per-parameter statistics"""
        import numpy as np

        if not self.metrics['ssim']:
            print("No metrics data. Run load_and_predict() first.")
            return None

        # Overall stats
        overall = {metric: {'mean': np.mean(self.metrics[metric]),
                            'std': np.std(self.metrics[metric])}
                   for metric in ['ssim', 'psnr', 'nrmse']}

        # Per-parameter stats
        per_param = {}
        for i, channel in enumerate(self.channel_names):
            per_param[channel] = {
                metric.upper(): {'mean': np.mean(self.metrics[f'{metric}_metric_{i}']),
                                 'std': np.std(self.metrics[f'{metric}_metric_{i}'])}
                for metric in ['ssim', 'psnr', 'nrmse']
            }

        # Print checkpoints
        print("\n" + "=" * 60 + "\nSUMMARY STATISTICS\n" + "=" * 60)
        print("\n1. OVERALL STATISTICS")
        print("-" * 50)
        for metric in ['ssim', 'psnr', 'nrmse']:
            unit = " dB" if metric == 'psnr' else ""
            print(f"{metric.upper()}: {overall[metric]['mean']:.4f} ± {overall[metric]['std']:.4f}{unit}")

        print("\n2. PER-PARAMETER STATISTICS")
        print("-" * 50)
        for channel, stats in per_param.items():
            print(f"\n{channel}:")
            for metric in ['SSIM', 'PSNR', 'NRMSE']:
                unit = " dB" if metric == 'PSNR' else ""
                print(f"  {metric}: {stats[metric]['mean']:.4f} ± {stats[metric]['std']:.4f}{unit}")

        return {'overall_stats': overall, 'per_parameter_stats': per_param}

    def create_summary_table(self):
        """Create formatted summary statistics table"""
        import numpy as np

        if not self.metrics['ssim']:
            print("No metrics data. Run load_and_predict() first.")
            return

        # Prepare table data
        table_data = []

        # Overall row
        overall_data = [np.mean(self.metrics[m]) for m in ['ssim', 'psnr', 'nrmse']]
        overall_std = [np.std(self.metrics[m]) for m in ['ssim', 'psnr', 'nrmse']]
        table_data.append(['Overall'] + [f'{mean:.4f} ± {std:.4f}'
                                         for mean, std in zip(overall_data, overall_std)])

        # Per-parameter rows
        for i, channel in enumerate(self.channel_names):
            row_data = []
            for metric in ['ssim', 'psnr', 'nrmse']:
                mean = np.mean(self.metrics[f'{metric}_metric_{i}'])
                std = np.std(self.metrics[f'{metric}_metric_{i}'])
                row_data.append(f'{mean:.4f} ± {std:.4f}')
            table_data.append([channel] + row_data)

        # Create table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data, colLabels=['Parameter', 'SSIM', 'PSNR (dB)', 'NRMSE'],
                         cellLoc='center', loc='center')

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # Header styling
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(1, i)].set_facecolor('#E8F5E8')
            table[(1, i)].set_text_props(weight='bold')

        # Alternate row colors
        for i in range(2, len(table_data) + 1):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(4):
                table[(i, j)].set_facecolor(color)

        plt.title('Summary Statistics: Mean ± Standard Deviation', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(os.path.join(self.out_path_metrics, 'summary_statistics_table.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Summary table saved to {self.out_path_metrics}")

    def create_boxplots(self):
        """Create boxplots for per-parameter metrics"""
        import seaborn as sns
        import pandas as pd

        if not self.metrics['ssim']:
            print("No metrics data. Run load_and_predict() first.")
            return

        sns.set(style="whitegrid")

        # Create combined plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, metric in enumerate(['ssim', 'psnr', 'nrmse']):
            data = []
            for i in range(6):
                for value in self.metrics[f'{metric}_metric_{i}']:
                    data.append({'Channel': self.channel_names[i], 'Value': value})

            df = pd.DataFrame(data)
            sns.boxplot(x='Channel', y='Value', data=df, ax=axes[idx], hue='Channel', legend=False)

            title = f'{metric.upper()}'
            if metric == 'psnr':
                title += ' (dB)'
            axes[idx].set_title(f'{title} by Channel', fontsize=14)
            axes[idx].set_xlabel('Channel', fontsize=12)
            axes[idx].set_ylabel(title, fontsize=12)
            axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

        plt.suptitle('Performance Metrics by Channel', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path_metrics, 'metrics_per_channel_boxplot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplots saved to {self.out_path_metrics}")

    def generate_summary_report(self):
        """Generate complete summary report"""
        print("Generating summary report...")

        if not self.calculate_summary_statistics():
            return

        self.create_summary_table()
        self.create_boxplots()

        print(f"Complete summary report saved to {self.out_path_metrics}")


