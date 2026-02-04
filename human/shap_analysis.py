import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import h5py
from transformer_architecture_prod import create_model_v0
from glob import glob
import os
import shap
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config_loader import load_config


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
        dataset_idx = h5py.File(self.data_paths[index])['res'][:] / self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels = h5py.File(self.label_paths[index])['res'][:]
        labels[0, :, :] = labels[0, :, :] / 100  # KSSW
        labels[1, :, :] = labels[1, :, :] / 27.27  # MT
        labels[2, :, :] = (labels[2, :, :] + 1) / (1.7 + 1)  # B0
        labels[3, :, :] = labels[3, :, :] / 3.4944  # B1
        labels[4, :, :] = labels[4, :, :] / 10000  # T1
        labels[5, :, :] = labels[5, :, :] / 1000  # T2
        return dataset_idx.astype('float32'), params_idx.astype('float32'), labels.astype('float32')


class WrappedModel(torch.nn.Module):
    def __init__(self, model, output_mode='mean', channel_idx=None, spatial_idx=None):
        super().__init__()
        self.model = model.eval()
        self.output_mode = output_mode
        self.channel_idx = channel_idx
        self.spatial_idx = spatial_idx

    def forward(self, x, p):
        output = self.model(x, p)  # [batch, H, W, channels]
        if self.output_mode == 'mean':
            return output.mean(dim=(1, 2))  # [batch, 6]
        elif self.output_mode == 'channel_mean':
            if self.channel_idx is None:
                raise ValueError("channel_idx must be specified for channel_mean mode")
            return output[:, :, :, self.channel_idx].mean(dim=(1, 2)).unsqueeze(1)  # [batch, 1]
        elif self.output_mode == 'specific_pixel':
            if self.channel_idx is None or self.spatial_idx is None:
                raise ValueError("Both channel_idx and spatial_idx must be specified for specific_pixel mode")
            h, w = self.spatial_idx
            return output[:, h, w, self.channel_idx].unsqueeze(1)  # [batch, 1]
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")


def main(args):
    device = args["device"]
    print(f"Using device: {device}")
    save_path = args["out_dir"]
    os.makedirs(save_path, exist_ok=True)
    print(args["data_dir"])

    model = create_model_v0(args, weights_path=args["new_model_weight_path"]).to(device)
    model.eval()

    # Gather all data paths (using the axial view data structure from second file)
    data_paths, param_paths, label_paths = [], [], []
    data_tests, param_tests, label_tests = [], [], []

    for scan in os.listdir(os.path.join(args["data_dir"], "axial")):
        for i in range(8):  # Limit for RAM saving/debug
            # Training data
            for j in range(1, 18):  # Images 1-17
                data_glob = glob(os.path.join(args["data_dir"], "axial", scan, 'dataset', f'slice_{scan}_image_{j}.h5'))
                param_glob = glob(os.path.join(args["data_dir"], "axial", scan, 'params', f'slice_{scan}_image_{j}.h5'))
                label_glob = glob(os.path.join(args["data_dir"], "axial", scan, 'labels', f'slice_{scan}_image_{j}.h5'))
                data_paths.extend(data_glob)
                param_paths.extend(param_glob)
                label_paths.extend(label_glob)
            # Test data
        for i in range(20):
            for j in range(1):
                data_test = glob(os.path.join(args["data_dir"], "axial", scan, 'dataset', f'slice_{scan}_image_0.h5'))
                param_test = glob(os.path.join(args["data_dir"], "axial", scan, 'params', f'slice_{scan}_image_0.h5'))
                label_test = glob(os.path.join(args["data_dir"], "axial", scan, 'labels', f'slice_{scan}_image_0.h5'))
                data_tests.extend(data_test)
                param_tests.extend(param_test)
                label_tests.extend(label_test)

    print(f"Found {len(data_paths)} training data files")
    print(f"Found {len(data_tests)} test data files")

    # Create datasets & loaders
    dataset = TryDataset_v2(
        data_dir=data_paths,
        param_dir=param_paths,
        labels_dir=label_paths,
        scale_data=4578.9688,
        scale_param=13.9984
    )
    dataset_2 = TryDataset_v2(
        data_dir=data_tests,
        param_dir=param_tests,
        labels_dir=label_tests,
        scale_data=4578.9688,
        scale_param=13.9984
    )

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    test_loader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False, pin_memory=True)

    # Load background data
    x_list, p_list, _ = [], [], []
    for batch, (X, p, y) in enumerate(tqdm.tqdm(test_loader, desc="Loading background data for SHAP")):
        x_list.append(X)
        p_list.append(p)
        if batch >= 50:
            break
    x_all = torch.cat(x_list, dim=0).to(device)
    p_all = torch.cat(p_list, dim=0).to(device)
    print(f"Background data loaded: x_all shape {x_all.shape}, p_all shape {p_all.shape}")

    # Load test data
    x_test, p_test, _ = [], [], []
    for batch, (X, p, y) in enumerate(tqdm.tqdm(test_loader_2, desc="Loading test data for SHAP")):
        x_test.append(X)
        p_test.append(p)
        if batch >= 3:
            break
    x_all_test = torch.cat(x_test, dim=0).to(device)
    p_all_test = torch.cat(p_test, dim=0).to(device)
    print(f"Test data loaded: x_all_test shape {x_all_test.shape}, p_all_test shape {p_all_test.shape}")

    # Prepare SHAP data
    background_x = x_all.requires_grad_()
    background_p = p_all.requires_grad_()
    print(f"DEBUG: background_x shape={background_x.shape}, background_p shape={background_p.shape}")

    test_x = x_all_test.requires_grad_()
    test_p = p_all_test.requires_grad_()
    print(f"DEBUG: test_x shape={test_x.shape}, test_p shape={test_p.shape}")

    background = [background_x, background_p]
    test = [test_x, test_p]

    output_channels = ['KSSW', 'MT', 'B0', 'B1', 'T1', 'T2']
    param_names = [
        'B1_1', 'B1_2', 'B1_3', 'B1_4', 'B1_5', 'B1_6', 'B1_7', 'B1_8', 'B1_9', 'B1_10', 'B1_11', 'B1_12',
        'PPM_1', 'PPM_2', 'PPM_3', 'PPM_4', 'PPM_5', 'PPM_6', 'PPM_7', 'PPM_8', 'PPM_9', 'PPM_10', 'PPM_11', 'PPM_12'
    ]

    # Debug information
    print(f"DEBUG: param_names length={len(param_names)}, num_param_features={test_p.numel() // test_p.shape[0]}")
    test_p_flat = test_p.reshape(test_p.shape[0], -1)

    print("==== PARAM/IMAGE STATISTICS ====")
    print("test_p std (per feature):", np.std(test_p.detach().cpu().numpy(), axis=0))
    print("background_p std (per feature):", np.std(background_p.detach().cpu().numpy(), axis=0))
    print("test_x mean/std:", test_x.mean().item(), test_x.std().item())
    print("background_x mean/std:", background_x.mean().item(), background_x.std().item())

    # Test model output
    with torch.no_grad():
        torch.cuda.empty_cache()
        out_real = model(test_x, test_p)
        print("Model output (real, min/max):", out_real.min().item(), out_real.max().item())
        torch.cuda.empty_cache()

    # SHAP Analysis for each output channel
    all_shap_values_x = []
    all_shap_values_p = []

    for out_c in range(6):
        print(f"\n===== SHAP for output channel {out_c} ({output_channels[out_c]}) =====")
        torch.cuda.empty_cache()

        wrapped_model_channel = WrappedModel(model, output_mode='channel_mean', channel_idx=out_c).to(device)
        explainer_channel = shap.GradientExplainer(wrapped_model_channel, background, batch_size=10)
        shap_values_channel = explainer_channel.shap_values(test)

        print(f"DEBUG: SHAP values shape for channel {out_c}: {shap_values_channel[0].shape}, {shap_values_channel[1].shape}")

        all_shap_values_x.append(shap_values_channel[0])
        all_shap_values_p.append(shap_values_channel[1])

        # PARAMS: SHAP summary plot (from second file)
        shap_p = shap_values_channel[1].reshape(test_p.shape[0], -1)
        try:
            shap.summary_plot(
                shap_p,
                test_p_flat,
                feature_names=param_names,
                show=False,
                max_display=24,
            )
            plt.title(f"Param SHAP - Output Channel: {output_channels[out_c]}, Axial View")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"shap_summary_p_output_ch{out_c}_{output_channels[out_c]}_axial_images_1_to_6.png"))
            plt.close()
        except Exception as e:
            print(f"Error in plotting SHAP for p, channel {out_c}: {e}")

    # ===== INPUT SECTION ANALYSIS (from first file) =====
    print("\n===== Creating input section analysis =====")

    # Analyze SHAP values per input section (input channel)
    for input_chan in range(6):  # 6 input MRI images
        print(f"\nAnalyzing input section {input_chan}")

        # Collect SHAP values for this input channel across all output channels
        input_section_shaps = []

        for out_c in range(6):
            # Extract SHAP values for this input channel: [samples, H, W]
            shap_for_input = all_shap_values_x[out_c][:, :, :, input_chan, 0]  # Remove last dimension
            input_section_shaps.append(shap_for_input)

        # Create comprehensive visualization for this input section - SMALLER SQUARES
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'SHAP Analysis - Input MRI Section {input_chan + 1} (Axial View)', fontsize=16, fontweight='bold')

        # Plot 1-6: Individual output channel SHAP maps (
        for out_c in range(6):
            row = out_c // 3
            col = out_c % 3
            ax = axes[row, col]

            # Use first sample for visualization
            shap_map = input_section_shaps[out_c][0]  # [H, W]
            print(f"input section map = {shap_map.shape}")

            # Symmetric color scale
            vmax = max(abs(shap_map.min()), abs(shap_map.max()))
            vmin = -vmax

            im = ax.imshow(shap_map, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'→ {output_channels[out_c]}', fontweight='bold', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, shrink=0.6)

        # Plot 7: Sum of absolute SHAP values across all output channels
        ax_sum = axes[0, 3]
        # Sum absolute SHAP values across all output channels for this input
        sum_abs_shap = np.sum([np.abs(shap_map) for shap_map in
                               [input_section_shaps[i][0] for i in range(6)]], axis=0)

        im_sum = ax_sum.imshow(sum_abs_shap, cmap='hot')
        ax_sum.set_title('Sum |SHAP| all outputs', fontweight='bold', fontsize=10)
        ax_sum.axis('off')
        plt.colorbar(im_sum, ax=ax_sum, shrink=0.6)

        # Plot 8: Mean absolute SHAP across samples
        ax_mean = axes[1, 3]
        # Average across all samples for this input-output combination
        mean_abs_shap = np.mean([np.abs(input_section_shaps[i]) for i in range(6)], axis=(0, 1))

        im_mean = ax_mean.imshow(mean_abs_shap, cmap='hot')
        ax_mean.set_title('Mean |SHAP| samples', fontweight='bold', fontsize=10)
        ax_mean.axis('off')
        plt.colorbar(im_mean, ax=ax_mean, shrink=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"shap_input_section_{input_chan + 1}_comprehensive_axial.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


if __name__ == '__main__':
    # Load configuration from config.yaml
    try:
        config_loader = load_config('config.yaml')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Build arguments dictionary from configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = {
        "device": device,
        "image_size": config_loader.get('model.img_size', 144),
        "sequence_len": config_loader.get('model.in_channels', 6),
        "patch_size": config_loader.get('model.patch_size', 9),
        "embedding_dim": config_loader.get('model.embedding_dim', 768),
        "dropout": config_loader.get('model.dropout', 0),
        "mlp_size": config_loader.get('model.mlp_size', 3072),
        "num_transformer_layers": config_loader.get('model.num_transformer_layers', 3),
        "num_heads": config_loader.get('model.num_heads', 4),
        "data_dir": config_loader.get('data.root_dir', './data'),
        "new_model_weight_path": config_loader.get('model.model2_path', 'checkpoints/model2.pt'),
        "out_dir": config_loader.get('analysis.predictions_dir', './predictions/shap'),
    }
    main(args)