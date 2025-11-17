import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import cm
import os
from typing import Dict, List, Tuple
import scipy.io as sio
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import glob
import h5py
# Import custom modules
from transformer_architecture_prod import *
from functions_prod import *
from images_and_box_plot_model_2 import plot_true_and_pred_sequences


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
        dataset_idx = h5py.File(self.data_paths[index])['res'] [:]/ self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels = h5py.File(self.label_paths[index])['res'][:]
        labels[0, :, :] = labels[0, :, :] / 100  # KSSW
        labels[1, :, :] = labels[1, :, :] / 27.27  # MT
        labels[2, :, :] = (labels[2, :, :] + 1) / (1.7 + 1)  # B0
        labels[3, :, :] = labels[3, :, :] / 3.4944  # B1
        labels[4, :, :] = labels[4, :, :] / 10000  # T1
        labels[5, :, :] = labels[5, :, :] / 1000  # T2
        return dataset_idx.astype('float32'), params_idx.astype('float32'), labels.astype('float32')


class AttentionExtractor:
    """Extract attention weights from the trained model without modifying it"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def extract_attention_weights(self, x, p):
        """
        Extract attention weights by accessing the transformer layers directly
        """
        self.model.eval()
        attention_dict = {}

        with torch.no_grad():
            # Permute input
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

            # 1. Patch Embedding
            x_patches = self.model.patch_embedding(x)  # [B, num_patches, embed_dim]

            # 2. Add positional embedding
            x_embed = self.model.positional_embedding + x_patches

            # 3. Dropout (in eval mode, this is identity)
            x_embed = self.model.embedding_dropout(x_embed)

            # 4. Parameter embedding
            p_embed = self.model.linear_layer(p)  # [B, embed_dim]
            x_embed = torch.cat((x_embed, p_embed), dim=1)  # [B, num_patches+1, embed_dim]

            # 5. Pass through transformer and capture attention
            transformer_input = x_embed

            for i, layer in enumerate(self.model.transformer_encoder.layers):
                # Get attention weights using the built-in forward with need_weights=True
                attn_output, attn_weights = self._get_attention_from_layer(layer, transformer_input)
                attention_dict[f'layer_{i}'] = attn_weights.cpu().numpy()
                transformer_input = attn_output

            # 6. Continue with conv layers for final output
            x_conv = transformer_input.unsqueeze(dim=1)
            x_conv = self.model.conv_layers(x_conv)
            output = x_conv.permute(0, 2, 3, 1)

        return output, attention_dict

    def _get_attention_from_layer(self, layer, x):
        """Extract attention from a transformer encoder layer"""
        # Handle norm_first
        if layer.norm_first:
            x2 = layer.norm1(x)
        else:
            x2 = x

        # Get attention output and weights
        attn_output, attn_weights = layer.self_attn(x2, x2, x2, need_weights=True, average_attn_weights=False)

        # Complete the forward pass
        x = x + layer.dropout1(attn_output)

        # MLP
        if layer.norm_first:
            x2 = layer.norm2(x)
        else:
            x = layer.norm1(x)
            x2 = x

        x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
        x = x + layer.dropout2(x2)

        if not layer.norm_first:
            x = layer.norm2(x)

        return x, attn_weights


class AttentionVisualizer:
    """Visualize attention maps and their impact on output parameters"""

    def __init__(self, image_size=144, patch_size=9):
        self.image_size = image_size
        self.patch_size = patch_size
        # Calculate actual number of patches from your PatchEmbedding
        self.patches_per_side = image_size // patch_size  # 144//9 = 16
        self.num_patches = self.patches_per_side ** 2  # 16*16 = 256
        self.channel_names = ['Kssw', 'MT_perc', 'B0', 'B1', 'T1', 'T2']

    def visualize_attention_analysis(self, attention_dict, input_data, predictions,
                                     ground_truth, save_path, batch_idx=0, sample_idx=0):
        """
        Comprehensive visualization of attention patterns with ground truth comparison
        """
        os.makedirs(save_path, exist_ok=True)
        # Extract data for visualization
        input_images = input_data[sample_idx].cpu().numpy()  # [H, W, C]
        pred_maps = predictions[sample_idx].cpu().numpy()  # [H, W, 6]
        gt_maps = ground_truth[sample_idx].cpu().numpy()  # [6, H, W]

        # Get colormaps
        colormaps = self._get_colormaps()

        # 1. Visualize attention maps for each layer
        for layer_name, attn_weights in attention_dict.items():
            layer_path = os.path.join(save_path, layer_name)
            os.makedirs(layer_path, exist_ok=True)
            print(f"Visualizing attention for layer {layer_name} at {layer_path},{attn_weights.shape} ")

            # attn_weights shape: [batch, num_heads, seq_len, seq_len]
            if len(attn_weights.shape) == 4:
                layer_attn = attn_weights[sample_idx]  # [num_heads, seq_len, seq_len]
            else:
                # If attention is already averaged across heads
                layer_attn = attn_weights[sample_idx:sample_idx + 1]  # Keep dims
            print(f"Layer attention shape: {layer_attn.shape} - Working")
            self._visualize_layer_attention_heads(
                layer_attn,
                layer_path
            )
            print("Finished visualizing attention heads")
            # Visualize attention overlaid on predictions
            self._visualize_attention_on_outputs(
                layer_attn,
                pred_maps,
                gt_maps,
                layer_path,
                colormaps
            )
            print("Finished visualizing attention on outputs")

        # 2. Create attention evolution across layers
        self._visualize_attention_evolution(
            attention_dict,
            sample_idx,
            os.path.join(save_path, 'attention_evolution.png')
        )
        print("Finished visualizing attention evolution")

    def _visualize_layer_attention_heads(self, attn_weights, save_path):
        """Visualize individual attention heads"""
        num_heads = attn_weights.shape[0]

        # Create grid layout
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for head_idx in range(num_heads):
            row, col = head_idx // cols, head_idx % cols
            ax = axes[row, col]

            # Get attention for this head
            head_attn = attn_weights[head_idx]  # Shape: (258, 258)

            # Check if we have parameter tokens (if shape is larger than expected)
            if head_attn.shape[0] == self.num_patches + 2:  # 256 patches + 2 param tokens
                # Exclude parameter tokens from both dimensions
                head_attn = head_attn[:-2, :-2]  # Now shape: (256, 256)
            elif head_attn.shape[0] == self.num_patches + 1:  # 256 patches + 1 param token
                # Exclude parameter token from both dimensions
                head_attn = head_attn[:-1, :-1]  # Now shape: (256, 256)
            else:
                print(f"No parameter tokens to exclude for head {head_idx + 1}")

            # Average over query positions (which patches are being attended to)
            avg_attn = head_attn.mean(axis=0)  # Shape: (256,)

            # Reshape to 2D
            if avg_attn.shape[0] == self.num_patches:
                attn_map = avg_attn.reshape(
                    self.patches_per_side,  # Use patches_per_side instead of calculating
                    self.patches_per_side
                )
            else:
                # Handle unexpected sizes
                print(f"Warning: Unexpected attention size {avg_attn.shape[0]}, expected {self.num_patches}")
                grid_size = int(np.sqrt(avg_attn.shape[0]))
                if grid_size * grid_size != avg_attn.shape[0]:
                    print(f"Error: Cannot reshape {avg_attn.shape[0]} into square grid")
                    continue
                attn_map = avg_attn.reshape(grid_size, grid_size)

            # Upsample to image size
            from scipy.ndimage import zoom
            zoom_factor = self.image_size / attn_map.shape[0]
            attn_map_upsampled = zoom(attn_map, zoom_factor, order=1)

            im = ax.imshow(attn_map_upsampled, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Remove empty subplots
        for idx in range(num_heads, rows * cols):
            row, col = idx // cols, idx % cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'attention_heads.png'), dpi=150)
        plt.close()

    def _visualize_attention_on_outputs(self, attn_weights, pred_maps, gt_maps, save_path, colormaps):
        """Overlay attention on output predictions"""
        # Average attention across all heads
        avg_attn = attn_weights.mean(axis=0)  # Shape: (258, 258)

        # Get patch-to-patch attention (exclude parameter tokens if present)
        if avg_attn.shape[0] == self.num_patches + 2:
            avg_attn = avg_attn[:-2, :-2]  # Exclude 2 param tokens
        elif avg_attn.shape[0] == self.num_patches + 1:
            avg_attn = avg_attn[:-1, :-1]  # Exclude 1 param token

        # Average over queries
        avg_attn = avg_attn.mean(axis=0)  # Shape: (256,)

        # Reshape and upsample
        if avg_attn.shape[0] == self.num_patches:
            attn_map = avg_attn.reshape(
                self.patches_per_side,
                self.patches_per_side
            )
        else:
            grid_size = int(np.sqrt(avg_attn.shape[0]))
            if grid_size * grid_size != avg_attn.shape[0]:
                print(f"Error: Cannot reshape {avg_attn.shape[0]} into square grid")
                return
            attn_map = avg_attn.reshape(grid_size, grid_size)

        # Rest of the method remains the same...
    def _scale_channel(self, data, channel_idx):
        """Scale channel data to original range"""
        if channel_idx == 0:  # Kssw
            return data * 100
        elif channel_idx == 1:  # MT_perc
            return data * 27.27
        elif channel_idx == 2:  # B0
            return (data * 2.7) - 1
        elif channel_idx == 3:  # B1
            return data * 3.4944
        elif channel_idx == 4:  # T1
            return data * 10000
        elif channel_idx == 5:  # T2
            return data * 1000
        return data

    def _visualize_attention_evolution(self, attention_dict, sample_idx, save_path):
        """Show how attention patterns evolve across layers"""
        num_layers = len(attention_dict)

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]

        for i, (layer_name, attn_weights) in enumerate(attention_dict.items()):
            # Get attention for this sample
            if len(attn_weights.shape) == 4:
                layer_attn = attn_weights[sample_idx]
            else:
                layer_attn = attn_weights[sample_idx:sample_idx + 1]

            # Average across heads
            avg_attn = layer_attn.mean(axis=0)

            # Exclude parameter token if present
            if avg_attn.shape[0] == self.num_patches + 2:
                avg_attn = avg_attn[:-2, :-2]  # Exclude 2 param tokens
            elif avg_attn.shape[0] == self.num_patches + 1:
                avg_attn = avg_attn[:-1, :-1]  # Exclude 1 param token

            # Average over queries
            avg_attn = avg_attn.mean(axis=0)

            # Reshape and upsample
            if avg_attn.shape[0] == self.num_patches:
                attn_map = avg_attn.reshape(
                    self.image_size // self.patch_size,
                    self.image_size // self.patch_size
                )
            else:
                grid_size = int(np.sqrt(avg_attn.shape[0]))
                attn_map = avg_attn.reshape(grid_size, grid_size)

            from scipy.ndimage import zoom
            zoom_factor = self.image_size / attn_map.shape[0]
            attn_map_upsampled = zoom(attn_map, zoom_factor, order=1)

            im = axes[i].imshow(attn_map_upsampled, cmap='hot')
            axes[i].set_title(layer_name)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        plt.suptitle('Attention Evolution Across Transformer Layers', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _get_colormaps(self):
        """Get colormaps for each output channel"""
        # Using the colormaps from your cm_data_vik and cm_data_brok
        buda_map = LinearSegmentedColormap.from_list('buda', cm_data_vik)
        lipari_map = LinearSegmentedColormap.from_list('lipari', cm_data_brok)

        original_map = plt.get_cmap('viridis')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        b_viridis = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = buda_map
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        b_bwr = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = lipari_map
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        b_rdgy = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        original_map = plt.get_cmap('winter')
        color_mat = original_map(np.arange(original_map.N))
        color_mat[0, 0:3] = 0
        winter = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        return ['magma', b_viridis, b_bwr, b_rdgy, 'hot', winter]

# Metrics and importance analysis functions

def compute_gradient_based_importance(model, X, p, device):
    """Further enhanced gradient analysis with better scaling"""
    model.eval()

    X_grad = X.clone().detach().requires_grad_(True)
    p_no_grad = p.clone().detach()

    # Forward pass
    output = model(X_grad, p_no_grad)

    importance_matrix = np.zeros((6, 6))

    for output_param in range(6):
        # Clear gradients
        if X_grad.grad is not None:
            X_grad.grad.zero_()

        # Get parameter output
        if output.shape[-1] == 6:  # [B, H, W, 6]
            param_output = output[0, :, :, output_param]
        else:  # [B, 6, H, W]
            param_output = output[0, output_param, :, :]

        # Use sum instead of variance for gradient computation
        loss = param_output.sum()
        loss.backward(retain_graph=True)

        # Get gradients for each input image
        if X_grad.grad is not None:
            input_grads = X_grad.grad[0]  # [H, W, 6]

            for input_img in range(6):
                # Use mean absolute gradient instead of L2 norm
                grad_magnitude = torch.abs(input_grads[:, :, input_img]).mean().item()
                importance_matrix[output_param, input_img] = grad_magnitude

    # Row-wise normalization
    for i in range(6):
        row_sum = importance_matrix[i].sum()
        if row_sum > 0:
            importance_matrix[i] /= row_sum

    return importance_matrix


def compute_input_ablation_importance(model, X, p, device):
    """Enhanced ablation with mean substitution instead of zero ablation"""
    model.eval()

    with torch.no_grad():
        # Get baseline prediction
        baseline_pred = model(X, p)

        importance_matrix = np.zeros((6, 6))

        # Calculate channel means for better ablation
        channel_means = X.mean(dim=(0, 1, 2), keepdim=True)  # [1, 1, 1, 6]

        for input_idx in range(6):
            # Use mean ablation instead of zero ablation
            X_ablated = X.clone()
            X_ablated[:, :, :, input_idx] = channel_means[:, :, :, input_idx]

            # Get prediction without this input
            ablated_pred = model(X_ablated, p)

            # Measure impact on each output with parameter-specific normalization
            for output_idx in range(6):
                if baseline_pred.shape[-1] == 6:  # [B, H, W, 6]
                    baseline_val = baseline_pred[0, :, :, output_idx]
                    ablated_val = ablated_pred[0, :, :, output_idx]
                else:  # [B, 6, H, W]
                    baseline_val = baseline_pred[0, output_idx, :, :]
                    ablated_val = ablated_pred[0, output_idx, :, :]

                # Use relative change in variance (more robust metric)
                baseline_var = baseline_val.var().item()
                ablated_var = ablated_val.var().item()

                if baseline_var > 1e-6:
                    importance_matrix[output_idx, input_idx] = abs(baseline_var - ablated_var) / baseline_var
                else:
                    # Fallback to absolute difference
                    diff = torch.abs(baseline_val - ablated_val).mean().item()
                    importance_matrix[output_idx, input_idx] = diff

        # Row-wise normalization
        for i in range(6):
            row_sum = importance_matrix[i].sum()
            if row_sum > 0:
                importance_matrix[i] /= row_sum

    return importance_matrix
def compute_perturbation_importance(model, X, p, device, perturbation_strength=0.1):
    """Use small perturbations instead of gradients"""
    model.eval()

    with torch.no_grad():
        # Get baseline prediction
        baseline_pred = model(X, p)

        importance_matrix = np.zeros((6, 6))

        for input_idx in range(6):
            # Add small noise to this input channel
            X_perturbed = X.clone()

            # Get the standard deviation of this channel for realistic perturbation
            channel_std = X[:, :, :, input_idx].std()
            noise = torch.randn_like(X[:, :, :, input_idx]) * channel_std * perturbation_strength
            X_perturbed[:, :, :, input_idx] += noise

            # Get prediction with perturbed input
            perturbed_pred = model(X_perturbed, p)

            # Measure impact on each output
            for output_idx in range(6):
                if baseline_pred.shape[-1] == 6:  # [B, H, W, 6]
                    baseline_val = baseline_pred[0, :, :, output_idx]
                    perturbed_val = perturbed_pred[0, :, :, output_idx]
                else:  # [B, 6, H, W]
                    baseline_val = baseline_pred[0, output_idx, :, :]
                    perturbed_val = perturbed_pred[0, output_idx, :, :]

                # Use relative change in standard deviation
                baseline_std = baseline_val.std().item()
                perturbed_std = perturbed_val.std().item()

                if baseline_std > 1e-6:
                    importance_matrix[output_idx, input_idx] = abs(baseline_std - perturbed_std) / baseline_std
                else:
                    # Fallback to mean absolute difference
                    diff = torch.abs(baseline_val - perturbed_val).mean().item()
                    importance_matrix[output_idx, input_idx] = diff

        # Row-wise normalization
        for i in range(6):
            row_sum = importance_matrix[i].sum()
            if row_sum > 0:
                importance_matrix[i] /= row_sum

    return importance_matrix


def diagnose_model_sensitivity(model, X, p, device):
    """Check if model is actually sensitive to different inputs"""
    model.eval()

    sensitivity_results = {}

    with torch.no_grad():
        baseline_pred = model(X, p)

        print("Testing model sensitivity...")
        for input_idx in range(6):
            # Zero out one input completely
            X_test = X.clone()
            X_test[:, :, :, input_idx] = 0

            pred_without_input = model(X_test, p)

            # Calculate overall change
            total_change = torch.abs(baseline_pred - pred_without_input).mean().item()
            print(f"Input {input_idx+1}: Total change when zeroed = {total_change:.6f}")

            # Per-parameter change
            param_changes = []
            for out_idx in range(6):
                if baseline_pred.shape[-1] == 6:
                    param_change = torch.abs(
                        baseline_pred[0, :, :, out_idx] - pred_without_input[0, :, :, out_idx]
                    ).mean().item()
                else:
                    param_change = torch.abs(
                        baseline_pred[0, out_idx, :, :] - pred_without_input[0, out_idx, :, :]
                    ).mean().item()

                param_changes.append(param_change)
                print(f"  Output {out_idx+1}: {param_change:.6f}")

            sensitivity_results[f'input_{input_idx+1}'] = {
                'total_change': total_change,
                'param_changes': param_changes
            }

    return sensitivity_results

class CombinedAttentionAnalyzer:
    """Analyze attention patterns across multiple samples with detailed head and layer analysis"""

    def __init__(self, attention_visualizer):
        self.visualizer = attention_visualizer
        self.accumulated_attention = {}
        self.head_attention_accumulator = {}
        self.layer_attention_accumulator = {}
        self.sample_count = 0

    def accumulate_attention(self, attention_dict):
        """Accumulate attention weights across samples for both combined and detailed analysis"""
        for layer_name, attn_weights in attention_dict.items():
            # Original combined analysis accumulation
            if layer_name not in self.accumulated_attention:
                self.accumulated_attention[layer_name] = []

            # Average across batch and heads for this sample
            if len(attn_weights.shape) == 4:  # [batch, heads, seq, seq]
                avg_attn = attn_weights.mean(axis=(0, 1))  # Average batch and heads
            else:
                avg_attn = attn_weights.mean(axis=0) # [head, seq, seq]

            self.accumulated_attention[layer_name].append(avg_attn)

            #  Detailed head and layer analysis accumulation
            if len(attn_weights.shape) == 4:
                batch_size, num_heads, seq_len, _ = attn_weights.shape

                # Store head-level attention
                if layer_name not in self.head_attention_accumulator:
                    self.head_attention_accumulator[layer_name] = {
                        f'head_{i}': [] for i in range(num_heads)
                    }

                # Store layer-level attention (average across heads)
                if layer_name not in self.layer_attention_accumulator:
                    self.layer_attention_accumulator[layer_name] = []

                # Process each head separately
                for head_idx in range(num_heads):
                    head_attn = attn_weights[0, head_idx]  # Take first sample in batch
                    self.head_attention_accumulator[layer_name][f'head_{head_idx}'].append(head_attn)

                # Average across heads for layer-level analysis
                layer_avg_attn = attn_weights[0].mean(axis=0)  # Average across heads
                self.layer_attention_accumulator[layer_name].append(layer_avg_attn)

        self.sample_count += 1

    def create_combined_attention_maps(self, save_path):
        """Create combined attention visualizations including detailed head and layer analysis"""
        os.makedirs(save_path, exist_ok=True)

        # 1. Average attention across all samples
        avg_attention = {}
        for layer_name in self.accumulated_attention:
            attention_stack = np.stack(self.accumulated_attention[layer_name])
            avg_attention[layer_name] = attention_stack.mean(axis=0)

        # 2. Original visualizations
        self._visualize_combined_layer_patterns(avg_attention, save_path)
        self._visualize_attention_focus_areas(avg_attention, save_path)
        self._analyze_layer_specialization(avg_attention, save_path)

        # 3. NEW: Detailed head and layer analysis
        self._visualize_head_averages(save_path)
        self._visualize_layer_averages(save_path)

    def _visualize_combined_layer_patterns(self, avg_attention, save_path):
        """Show what each layer focuses on across all samples"""
        num_layers = len(avg_attention)
        fig, axes = plt.subplots(2, num_layers, figsize=(5 * num_layers, 10))

        if num_layers == 1:
            axes = axes.reshape(2, 1)

        for i, (layer_name, attn_weights) in enumerate(avg_attention.items()):
            # Remove parameter tokens if present
            if attn_weights.shape[0] == self.visualizer.num_patches + 2:
                attn_weights = attn_weights[:-2, :-2]
            elif attn_weights.shape[0] == self.visualizer.num_patches + 1:
                attn_weights = attn_weights[:-1, :-1]

            # 1. Average attention map (what patches are attended to)
            avg_attn_map = attn_weights.mean(axis=0)
            attn_2d = avg_attn_map.reshape(
                self.visualizer.patches_per_side,
                self.visualizer.patches_per_side
            )

            # Upsample to image size
            from scipy.ndimage import zoom
            zoom_factor = self.visualizer.image_size / attn_2d.shape[0]
            attn_upsampled = zoom(attn_2d, zoom_factor, order=1)

            im1 = axes[0, i].imshow(attn_upsampled, cmap='hot', interpolation='bilinear')
            axes[0, i].set_title(f'{layer_name}\nAttention Focus')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            # 2. Attention diversity (how spread out attention is)
            attention_entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-8), axis=1)
            entropy_2d = attention_entropy.reshape(
                self.visualizer.patches_per_side,
                self.visualizer.patches_per_side
            )
            entropy_upsampled = zoom(entropy_2d, zoom_factor, order=1)

            im2 = axes[1, i].imshow(entropy_upsampled, cmap='viridis', interpolation='bilinear')
            axes[1, i].set_title(f'{layer_name}\nAttention Diversity')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

        plt.suptitle(f'Combined Attention Analysis Across {self.sample_count} Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_attention_patterns.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_attention_focus_areas(self, avg_attention, save_path):
        """Identify and visualize consistent attention focus areas"""
        # Create attention intensity map across layers
        combined_attention = np.zeros((self.visualizer.image_size, self.visualizer.image_size))

        for layer_name, attn_weights in avg_attention.items():
            # Process attention weights
            if attn_weights.shape[0] == self.visualizer.num_patches + 2:
                attn_weights = attn_weights[:-2, :-2]
            elif attn_weights.shape[0] == self.visualizer.num_patches + 1:
                attn_weights = attn_weights[:-1, :-1]

            avg_attn_map = attn_weights.mean(axis=0)
            attn_2d = avg_attn_map.reshape(
                self.visualizer.patches_per_side,
                self.visualizer.patches_per_side
            )

            from scipy.ndimage import zoom
            zoom_factor = self.visualizer.image_size / attn_2d.shape[0]
            attn_upsampled = zoom(attn_2d, zoom_factor, order=1)
            combined_attention += attn_upsampled

        # Normalize
        combined_attention /= len(avg_attention)

        # Find top attention areas
        threshold = np.percentile(combined_attention, 90)  # Top 10% most attended areas
        high_attention_mask = combined_attention > threshold

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(combined_attention, cmap='hot')
        plt.title('Combined Attention Intensity')
        plt.colorbar(fraction=0.046)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(high_attention_mask, cmap='binary')
        plt.title('High Attention Areas (Top 10%)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # Show attention intensity distribution
        plt.hist(combined_attention.flatten(), bins=50, alpha=0.7, color='red')
        plt.axvline(threshold, color='black', linestyle='--', label=f'90th percentile: {threshold:.3f}')
        plt.xlabel('Attention Intensity')
        plt.ylabel('Frequency')
        plt.title('Attention Distribution')
        plt.legend()

        plt.suptitle(f'Attention Focus Analysis ({self.sample_count} samples)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'attention_focus_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()



    def _analyze_layer_specialization(self, avg_attention, save_path):
        """Analyze what each layer specializes in"""
        layer_stats = {}

        for layer_name, attn_weights in avg_attention.items():
            if attn_weights.shape[0] == self.visualizer.num_patches + 2:
                attn_weights = attn_weights[:-2, :-2]
            elif attn_weights.shape[0] == self.visualizer.num_patches + 1:
                attn_weights = attn_weights[:-1, :-1]

            # Calculate various attention statistics
            stats = {
                'mean_attention': attn_weights.mean(),
                'attention_std': attn_weights.std(),
                'max_attention': attn_weights.max(),
                'attention_entropy': -np.sum(attn_weights * np.log(attn_weights + 1e-8)),
                'sparsity': (attn_weights < 0.01).sum() / attn_weights.size,  # Fraction of near-zero attention
                'concentration': np.sum(attn_weights > np.percentile(attn_weights, 95))  # Highly attended patches
            }
            layer_stats[layer_name] = stats

        # Create comparison plot
        metrics = list(layer_stats[list(layer_stats.keys())[0]].keys())
        layers = list(layer_stats.keys())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [layer_stats[layer][metric] for layer in layers]
            axes[i].bar(layers, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)

        plt.suptitle('Layer Specialization Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'layer_specialization.png'), dpi=150, bbox_inches='tight')
        plt.close()

        return layer_stats

        # NEW METHODS FOR DETAILED ANALYSIS


    def _visualize_head_averages(self, save_path):
        """Show average attention pattern for each head across all samples"""

        for layer_name in self.head_attention_accumulator:
            layer_path = os.path.join(save_path, f'{layer_name}_heads')
            os.makedirs(layer_path, exist_ok=True)

            num_heads = len(self.head_attention_accumulator[layer_name])

            # Create subplot for all heads in this layer
            fig, axes = plt.subplots(2, num_heads, figsize=(5 * num_heads, 10))
            if num_heads == 1:
                axes = axes.reshape(2, 1)

            for head_idx in range(num_heads):
                head_key = f'head_{head_idx}'
                head_attentions = self.head_attention_accumulator[layer_name][head_key]

                # Average across all samples
                avg_head_attention = np.mean(head_attentions, axis=0)

                # Remove parameter tokens if present
                if avg_head_attention.shape[0] == self.visualizer.num_patches + 2:
                    patch_attention = avg_head_attention[:-2, :-2]
                elif avg_head_attention.shape[0] == self.visualizer.num_patches + 1:
                    patch_attention = avg_head_attention[:-1, :-1]
                else:
                    patch_attention = avg_head_attention

                # Calculate head statistics for display
                mean_attention = patch_attention.mean()
                entropy = -np.sum(patch_attention * np.log(patch_attention + 1e-8))

                # 1. Attention focus map
                avg_attention_map = patch_attention.mean(axis=0)
                attn_2d = avg_attention_map.reshape(
                    self.visualizer.patches_per_side,
                    self.visualizer.patches_per_side
                )

                # Upsample to image size
                from scipy.ndimage import zoom
                zoom_factor = self.visualizer.image_size / attn_2d.shape[0]
                attn_upsampled = zoom(attn_2d, zoom_factor, order=1)

                im1 = axes[0, head_idx].imshow(attn_upsampled, cmap='hot', interpolation='bilinear')
                axes[0, head_idx].set_title(f'Head {head_idx}\nMean: {mean_attention:.4f}')
                axes[0, head_idx].axis('off')
                plt.colorbar(im1, ax=axes[0, head_idx], fraction=0.046)

                # 2. Attention diversity (entropy)
                attention_entropy = -np.sum(patch_attention * np.log(patch_attention + 1e-8), axis=1)
                entropy_2d = attention_entropy.reshape(
                    self.visualizer.patches_per_side,
                    self.visualizer.patches_per_side
                )
                entropy_upsampled = zoom(entropy_2d, zoom_factor, order=1)

                im2 = axes[1, head_idx].imshow(entropy_upsampled, cmap='viridis', interpolation='bilinear')
                axes[1, head_idx].set_title(f'Head {head_idx} Diversity\nEntropy: {entropy:.2f}')
                axes[1, head_idx].axis('off')
                plt.colorbar(im2, ax=axes[1, head_idx], fraction=0.046)

            plt.suptitle(f'{layer_name} - All Heads Average ({self.sample_count} samples)', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(layer_path, f'{layer_name}_all_heads.png'), dpi=150, bbox_inches='tight')
            plt.close()


    def _visualize_layer_averages(self, save_path):
        """Show average attention pattern for each layer"""

        num_layers = len(self.layer_attention_accumulator)
        fig, axes = plt.subplots(2, num_layers, figsize=(6 * num_layers, 12))

        if num_layers == 1:
            axes = axes.reshape(2, 1)

        for i, (layer_name, layer_attentions) in enumerate(self.layer_attention_accumulator.items()):
            # Average across all samples
            avg_layer_attention = np.mean(layer_attentions, axis=0)

            # Remove parameter tokens if present
            if avg_layer_attention.shape[0] == self.visualizer.num_patches + 2:
                patch_attention = avg_layer_attention[:-2, :-2]
            elif avg_layer_attention.shape[0] == self.visualizer.num_patches + 1:
                patch_attention = avg_layer_attention[:-1, :-1]
            else:
                patch_attention = avg_layer_attention

            # Calculate layer statistics for display
            mean_attention = patch_attention.mean()
            max_attention = patch_attention.max()
            effective_rank = self._calculate_effective_rank(patch_attention)

            # 1. Attention focus map
            avg_attention_map = patch_attention.mean(axis=0)
            attn_2d = avg_attention_map.reshape(
                self.visualizer.patches_per_side,
                self.visualizer.patches_per_side
            )

            from scipy.ndimage import zoom
            zoom_factor = self.visualizer.image_size / attn_2d.shape[0]
            attn_upsampled = zoom(attn_2d, zoom_factor, order=1)

            im1 = axes[0, i].imshow(attn_upsampled, cmap='hot', interpolation='bilinear')
            axes[0, i].set_title(f'{layer_name}\nMean: {mean_attention:.4f}\nMax: {max_attention:.4f}')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            # 2. Attention concentration map
            concentration = patch_attention.max(axis=1)  # Max attention each patch gives
            concentration_2d = concentration.reshape(
                self.visualizer.patches_per_side,
                self.visualizer.patches_per_side
            )
            concentration_upsampled = zoom(concentration_2d, zoom_factor, order=1)

            im2 = axes[1, i].imshow(concentration_upsampled, cmap='plasma', interpolation='bilinear')
            axes[1, i].set_title(f'{layer_name} Concentration\nEffective Rank: {effective_rank:.1f}')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

        plt.suptitle(f'Layer-wise Average Attention Patterns ({self.sample_count} samples)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'layer_averages.png'), dpi=150, bbox_inches='tight')
        plt.close()


    def _calculate_effective_rank(self, attention_matrix):
        """Calculate effective rank of attention matrix"""
        try:
            eigenvals = np.linalg.eigvals(attention_matrix)
            eigenvals = eigenvals[eigenvals > 1e-10]
            if len(eigenvals) == 0:
                return 0
            eigenvals = eigenvals / eigenvals.sum()
            entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
            return np.exp(entropy)
        except:
            return 0

        # NEW: Add the separate heatmap visualization method


    def _create_separate_importance_heatmaps(self, grad_importance, ablation_importance, permutation_importance  ,save_path):
        """Create 2 separate heatmap visualizations for better clarity"""

        output_names = ['Kssw', 'MT%', 'B₀', 'B₁', 'T₁', 'T₂']
        input_names = [f'Input {i + 1}' for i in range(6)]

        # 1. Gradient-based importance heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(grad_importance,
                    annot=True,
                    fmt='.3f',
                    cmap='Reds',
                    xticklabels=input_names,
                    yticklabels=output_names,
                    cbar_kws={'label': 'Gradient Importance'})
        plt.title(f'Gradient-based Input Importance ({self.sample_count} samples)', fontsize=16)
        plt.xlabel('Input Image', fontsize=12)
        plt.ylabel('Output Parameter', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '1_gradient_importance_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Ablation-based importance heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(ablation_importance,
                    annot=True,
                    fmt='.3f',
                    cmap='Blues',
                    xticklabels=input_names,
                    yticklabels=output_names,
                    cbar_kws={'label': 'Ablation Importance'})
        plt.title(f'Ablation-based Input Importance ({self.sample_count} samples)', fontsize=16)
        plt.xlabel('Input Image', fontsize=12)
        plt.ylabel('Output Parameter', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '2_ablation_importance_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Permutation-based importance heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(permutation_importance,
                    annot=True,
                    fmt='.3f',
                    cmap='Greens',
                    xticklabels=input_names,
                    yticklabels=output_names,
                    cbar_kws={'label': 'Permutation Importance'})
        plt.title(f'Permutation-based Input Importance ({self.sample_count} samples)', fontsize=16)
        plt.xlabel('Input Image', fontsize=12)
        plt.ylabel('Output Parameter', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '3_Permutation_importance_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()



    def _create_method_correlation_analysis(self, grad_importance, ablation_importance,
                                          perturbation_importance, save_path):
        """Analyze correlations between different methods"""

        # Flatten all matrices for correlation analysis
        grad_flat = grad_importance.flatten()
        ablation_flat = ablation_importance.flatten()
        perturbation_flat = perturbation_importance.flatten()

        # Create correlation matrix
        methods_data = np.column_stack([grad_flat, ablation_flat, perturbation_flat])
        correlation_matrix = np.corrcoef(methods_data.T)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Correlation heatmap
        method_names = ['Gradient', 'Ablation', 'Perturbation']
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    xticklabels=method_names, yticklabels=method_names,
                    center=0, vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title('Method Correlation Matrix', fontsize=14)

        # 2. Scatter plots
        axes[1].scatter(ablation_flat, grad_flat, alpha=0.6, label='Gradient vs Ablation', s=30)
        axes[1].scatter(ablation_flat, perturbation_flat, alpha=0.6, label='Perturbation vs Ablation', s=30)
        axes[1].scatter(grad_flat, perturbation_flat, alpha=0.6, label='Perturbation vs Gradient', s=30)
        axes[1].set_xlabel('Importance Score (Method 1)')
        axes[1].set_ylabel('Importance Score (Method 2)')
        axes[1].set_title('Method Comparison Scatter Plot', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'method_correlation_analysis.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        return correlation_matrix

def main_attention_analysis():
    """Main function with comprehensive analysis including all four methods"""

    args = {
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "image_size": 144,
        "sequence_len": 6,
        "patch_size": 9,
        "embedding_dim": 768,
        "dropout": 0,
        "mlp_size": 3072,
        "num_transformer_layers": 3,
        "num_heads": 4,
        "data_dir": "data",
        "new_model_weight_path": 'checkpoints/model2.pt'
    }

    device = args["device"]
    print(f"Using device: {device}")

    # Setup paths
    save_path = 'output/attention_analysis'
    os.makedirs(save_path, exist_ok=True)

    # Load model
    print("Loading model...")
    model = create_model_v0(args, weights_path=args["new_model_weight_path"]).to(device)
    model.eval()
    print("Model loaded successfully!")

    # Load data
    data_paths = sorted(
        glob.glob(os.path.join(args["data_dir"], "*" , 'dataset', '*.h5'))
    )
    param_paths = sorted(
        glob.glob(os.path.join(args["data_dir"],  "*" , 'params', '*.h5'))
    )
    label_paths = sorted(
        glob.glob(os.path.join(args["data_dir"],  "*" , 'labels', '*.h5'))
    )

    print(f"Found {len(data_paths)} data files")

    # Create dataset
    dataset = TryDataset_v2(
        data_dir=data_paths,
        param_dir=param_paths,
        labels_dir=label_paths,
        scale_data=4578.9688,
        scale_param=13.9984
    )

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Initialize analyzers
    attention_extractor = AttentionExtractor(model, device)
    attention_visualizer = AttentionVisualizer(
        image_size=args["image_size"],
        patch_size=args["patch_size"]
    )
    combined_analyzer = CombinedAttentionAnalyzer(attention_visualizer)

    print("\nAnalyzing attention patterns...")

    # Process samples for attention analysis
    num_samples_to_analyze = 1000  # Limit for faster processing
    for batch_idx, (X, p, y) in enumerate(tqdm(test_loader, desc="Processing attention samples")):
        if batch_idx >= num_samples_to_analyze:
            break

        # Skip if sample is mostly empty
        if torch.count_nonzero(X[0, :, :, 0]) < 750:
            continue

        X, p, y = X.to(device), p.to(device), y.to(device)
        y[:, 1, :, :] = y[:, 1, :, :] * 100  # MT%

        try:
            # Extract attention weights
            predictions, attention_dict = attention_extractor.extract_attention_weights(X, p)
            combined_analyzer.accumulate_attention(attention_dict)

            # Create visualizations for selected samples
            if batch_idx % 250 == 0:
                batch_save_path = os.path.join(save_path, f'sample_{batch_idx}')
                attention_visualizer.visualize_attention_analysis(
                    attention_dict, X, predictions, y, batch_save_path, batch_idx=batch_idx)
                plot_true_and_pred_sequences(y, predictions, batch_save_path)

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    # Create combined attention analysis
    print("\nCreating combined attention analysis...")
    combined_save_path = os.path.join(save_path, 'combined_analysis')
    combined_analyzer.create_combined_attention_maps(combined_save_path)

    # **NEW: Comprehensive Input Importance Analysis**
    print("\nRunning comprehensive input importance analysis...")

    # Collect checkpoints from all four methods
    gradient_matrices = []
    ablation_matrices = []
    perturbation_matrices = []
    sensitivity_results_list = []

    # Analyze importance using all methods
    num_importance_samples = 300  # Reduce number for faster computation
    for batch_idx, (X, p, y) in enumerate(tqdm(test_loader, desc="Processing importance samples")):
        if batch_idx >= num_importance_samples:
            break

        # Skip empty samples
        if torch.count_nonzero(X[0, :, :, 0]) < 750:
            continue

        X, p = X.to(device), p.to(device)

        try:
            # 1. Original gradient-based importance
            grad_imp = compute_gradient_based_importance(model, X, p, device)
            gradient_matrices.append(grad_imp)

            # 2. Ablation-based importance
            abl_imp = compute_input_ablation_importance(model, X, p, device)
            ablation_matrices.append(abl_imp)

            # 3. NEW: Perturbation-based importance
            pert_imp = compute_perturbation_importance(model, X, p, device)
            perturbation_matrices.append(pert_imp)

            # 4. NEW: Sensitivity analysis (only for first few samples to save time)
            if batch_idx < 100:
                sens_results = diagnose_model_sensitivity(model, X, p, device)
                sensitivity_results_list.append(sens_results)

        except Exception as e:
            print(f"Error in importance analysis for batch {batch_idx}: {e}")

    # Average checkpoints and create comprehensive analysis
    if gradient_matrices:
        avg_gradient = np.mean(gradient_matrices, axis=0)
        avg_ablation = np.mean(ablation_matrices, axis=0)
        avg_perturbation = np.mean(perturbation_matrices, axis=0)


        correlation_matrix = combined_analyzer._create_method_correlation_analysis(
            avg_gradient, avg_ablation, avg_perturbation, save_path)

        # Create individual method heatmaps (your existing function)
        combined_analyzer._create_separate_importance_heatmaps(avg_gradient, avg_ablation, avg_perturbation, save_path)

        # Print summary statistics
        print("\n" + "="*60)
        print("COMPREHENSIVE IMPORTANCE ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nMethod Correlations:")
        print(f"Gradient-Ablation: {correlation_matrix[0,1]:.3f}")
        print(f"Gradient-Perturbation: {correlation_matrix[0,2]:.3f}")
        print(f"Ablation-Perturbation: {correlation_matrix[1,2]:.3f}")

        print(f"\nMost sensitive inputs (by sensitivity analysis):")


        print(f"\nSaved comprehensive analysis to {save_path}")

    print(f"\nAnalysis complete! Results saved to {save_path}")


if __name__ == '__main__':
    main_attention_analysis()