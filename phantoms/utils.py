import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL.ImageChops import offset
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from tqdm import tqdm
import h5py
import cv2
from matplotlib.patches import Circle
from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate




# Loading the original mat file

def load_mat_volume(file_path):
    """
    Loads the .mat file and returns the first variable that is not a meta key.
    Assumes the .mat file contains one main 4D array.

    Parameters:
        file_path (str): Path to the .mat file

    Returns:
        numpy.ndarray: The 4D volume data
    """
    try:
        mat_contents = sio.loadmat(file_path)
        # Filter out meta keys such as __header__, __version__, and __globals__
        keys = [key for key in mat_contents.keys() if not key.startswith('__')]
        if not keys:
            raise ValueError(f"No valid data variable found in {file_path}")
        volume = mat_contents[keys[0]]
        return volume
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        raise


def load_labels_volume(file_path):
    """
    Loads the labels .mat file and concatenates the two arrays into a 4D volume.

    Parameters:
        file_path (str): Path to the labels .mat file

    Returns:
        numpy.ndarray: The 4D volume data with shape [2, slices, H, W]
    """
    try:
        mat_contents = sio.loadmat(file_path)

        # Extract the two specific arrays
        if 'AL_fB_Im' not in mat_contents or 'AL_kBA_Im' not in mat_contents:
            raise ValueError(f"Required keys 'AL_fB_Im' and 'AL_kBA_Im' not found in {file_path}")

        al_fb_im = mat_contents['AL_fB_Im']  # Shape: (100, 116, 116)
        al_kba_im = mat_contents['AL_kBA_Im']  # Shape: (100, 116, 116)

        print(f"Loaded AL_fB_Im with shape: {al_fb_im.shape}")
        print(f"Loaded AL_kBA_Im with shape: {al_kba_im.shape}")

        # Stack them to create 4D array [2, slices, H, W]
        labels_volume = np.stack([al_fb_im, al_kba_im], axis=0)

        print(f"Combined labels volume shape: {labels_volume.shape}")
        return labels_volume

    except Exception as e:
        print(f"Error loading labels from {file_path}: {str(e)}")
        raise

# Removal of slices with less data inside

def remove_black_slices(volume : np.ndarray, axis: int, threshold=30 , min_keep_percentage=5) -> np.ndarray:
    """
    Generic function to remove black slices along any specified axis of a 4D volume.

    Parameters:
        volume (numpy.ndarray): 4D volume with shape [scans, slice, H, W]
        axis (int): Axis along which to remove slices (1 for axial, 2 for coronal, 3 for sagittal))
        threshold (int): Minimum number of non-zero pixels required to keep a slice
        min_keep_percentage (float): Minimum percentage of slices to keep to avoid excessive trimming

    Returns:
        numpy.ndarray: Volume with black slices removed along the specified axis
    """
    # Remove the fist scan (localizer)
    if volume.shape[0] == 31:
        volume = volume[1:, :, :, :]
    keep_indices = []
    axis_length = volume.shape[axis]
    slice_to_check = 1
    # Iterate over slices along the specified axis
    for i in range(axis_length):
        keep_slice = False
        # Take a slice along the specified axis for each scan
        if axis == 1:
            slice_data = volume[slice_to_check, i, :, :]
        elif axis == 2:
            slice_data = volume[slice_to_check, :, i, :]
        elif axis ==3:
            slice_data = volume[slice_to_check, :, :, i]
        else:
            raise ValueError(f"Unsupported axis: {axis}. Must be 1 (axial), 2 (coronal), or 3 (sagittal).")

        if np.count_nonzero(slice_data) >= threshold:
            keep_indices.append(i)

    # Safety check: ensure we keep at least a minimum percentage of slices
    min_slices = max(1, int(axis_length * min_keep_percentage / 100))
    if len(keep_indices) < min_slices:
        print(f"Warning: Only {len(keep_indices)} slices would be kept along axis {axis}.")
        print(f"This is below the minimum of {min_slices} slices ({min_keep_percentage}%).")
        print(f"Keeping the {min_slices} most non-empty slices instead.")

        # Find the slices with the most non-zero pixels
        slice_counts = []
        for i in range(axis_length):
            total_nonzero = 0
            for j in range(volume.shape[0]):
                if axis == 1:
                    slice_data = volume[j, i, :, :]
                elif axis == 2:
                    slice_data = volume[j, :, i, :]
                else:
                    slice_data = volume[j, :, :, i]
                total_nonzero += np.count_nonzero(slice_data)
            slice_counts.append((i, total_nonzero))

        keep_indices = [x[0] for x in slice_counts[:min_slices]]

    # Create a new volume with only the kept slices
    if axis == 1:
        new_volume = volume[:, keep_indices, :, :]
    elif axis==2:  # axis == 1
        new_volume = volume[:, :, keep_indices, :]
    else:
        new_volume = volume[:, :, :, keep_indices]

    return new_volume

def visualize_slices(volume, base_name = None, output_folder=None):
    """
    Visualize and optionally save representative slices from the volume.

    Parameters:
        volume (numpy.ndarray): 4D volume with shape [H, slices, W, scans]
        base_name (str): Base filename for the plot titles
        output_folder (str, optional): Folder to save visualizations, if None, only displays
    """
    if volume.shape[2] == 0:
        print("Cannot visualize: Volume has zero-sized dimensions")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Normalize the plots for better visualization
    def normalize_for_plot(img):
        if np.max(img) == np.min(img):
            return img
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    # Coronal view: take a middle slice from axis 0
    mid_coronal = volume.shape[1] // 2
    example_coronal = normalize_for_plot(volume[0, mid_coronal, :, :])
    im0 = axes[0].imshow(example_coronal, cmap='gray')
    axes[0].set_title(f"Coronal Slice {mid_coronal}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Sagittal view: take a middle slice from axis 2
    mid_sagittal = volume.shape[2] // 2
    example_sagittal = normalize_for_plot(volume[0, :, mid_sagittal, :])
    im1 = axes[1].imshow(example_sagittal, cmap='gray')
    axes[1].set_title(f"Sagittal Slice {mid_sagittal}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if base_name:
        plt.suptitle(f"{base_name} - Cleaned Volume")

    plt.tight_layout()

    if output_folder:
        fig_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_visualization.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Visualization saved to: {fig_path}")

    plt.show()


class LabelsProcessor:
    """
    A class to handle loading, processing, and saving of phantom labels data.
    Works in conjunction with PhantomSegmenter checkpoints.
    """

    def __init__(self, config):
        """
        Initialize the LabelsProcessor with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.output_folder = config["output_folder"]
        self.labels_name = config.get("labels_name", "labels.mat")
        self.phantom_name = config.get("phantom_name", None)

        # Labels data storage
        self.original_labels = None
        self.processed_labels = None
        self.phantom_masked_labels = None
        self.vial_masked_labels = None

        # Metadata
        self.label_keys = ['AL_fB_Im', 'AL_kBA_Im']
        self.label_shapes = {}

    def load_labels_volume(self, file_path):
        """
        Load the labels .mat file and concatenate the two arrays into a 4D volume.

        Args:
            file_path (str): Path to the labels .mat file

        Returns:
            numpy.ndarray: The 4D volume data with shape [2, slices, H, W]
        """
        try:
            print(f"\nLoading labels from: {file_path}")
            mat_contents = sio.loadmat(file_path)

            # Check for required keys
            missing_keys = [key for key in self.label_keys if key not in mat_contents]
            if missing_keys:
                raise ValueError(f"Required keys {missing_keys} not found in {file_path}")

            # Extract the two specific arrays
            al_fb_im = mat_contents[self.label_keys[0]]  # Shape: (slices, H, W)
            al_kba_im = mat_contents[self.label_keys[1]]  # Shape: (slices, H, W)

            # Store shapes for metadata
            self.label_shapes[self.label_keys[0]] = al_fb_im.shape
            self.label_shapes[self.label_keys[1]] = al_kba_im.shape

            print(f"Loaded {self.label_keys[0]} with shape: {al_fb_im.shape}")
            print(f"Loaded {self.label_keys[1]} with shape: {al_kba_im.shape}")

            # Stack them to create 4D array [2, slices, H, W]
            labels_volume = np.stack([al_fb_im, al_kba_im], axis=0)

            print(f"Combined labels volume shape: {labels_volume.shape}")
            self.original_labels = labels_volume

            return labels_volume

        except Exception as e:
            print(f"Error loading labels from {file_path}: {str(e)}")
            raise

    def apply_slice_selection(self, labels_volume, wanted_slices):
        """
        Apply the same slice selection logic as used for phantom data.

        Args:
            labels_volume (np.ndarray): 4D labels with shape [2, slices, H, W]
            wanted_slices: Slice selection (same format as phantom segmenter)

        Returns:
            numpy.ndarray: Labels volume with selected slices
        """
        if wanted_slices is None:
            return labels_volume

        print(f"Applying slice selection to labels: {wanted_slices}")

        # Apply slice selection logic (adapted from PhantomSegmenter.find_wanted_slices)
        current_slice_range = wanted_slices

        if isinstance(current_slice_range, list) and len(current_slice_range) > 0:
            if isinstance(current_slice_range[0], list):
                # Multiple ranges - need to extract and concatenate them
                label_slices = []
                for slice_range in current_slice_range:
                    if len(slice_range) == 2:  # Two slice objects
                        slice1, slice2 = slice_range
                        label_slices.append(labels_volume[:, slice1, :, :])
                        label_slices.append(labels_volume[:, slice2, :, :])
                    else:
                        label_slices.append(labels_volume[:, slice_range[0], :, :])

                processed_labels = np.concatenate(label_slices, axis=1)  # concatenate along slice dimension
            else:
                # Single slice object in a list
                processed_labels = labels_volume[:, current_slice_range[0], :, :]
        else:
            # Single range
            processed_labels = labels_volume[:, current_slice_range, :, :]

        print(f"Labels shape after slice selection: {processed_labels.shape}")
        self.processed_labels = processed_labels
        return processed_labels

    def apply_masks_to_labels(self, labels_volume, phantom_masks_3d, vial_masks_3d):
        """
        Apply phantom and vial masks to the labels data.

        Args:
            labels_volume (np.ndarray): 4D labels with shape [2, slices, H, W]
            phantom_masks_3d (np.ndarray): 3D phantom masks
            vial_masks_3d (np.ndarray): 3D vial masks

        Returns:
            dict: Dictionary containing masked labels data
        """
        print("\n" + "=" * 50)
        print("APPLYING MASKS TO LABELS DATA")
        print("=" * 50)

        num_label_types, num_slices, height, width = labels_volume.shape

        # Validate mask dimensions
        if phantom_masks_3d.shape != (num_slices, height, width):
            raise ValueError(
                f"Phantom mask shape {phantom_masks_3d.shape} doesn't match labels shape {(num_slices, height, width)}")

        if vial_masks_3d.shape != (num_slices, height, width):
            raise ValueError(
                f"Vial mask shape {vial_masks_3d.shape} doesn't match labels shape {(num_slices, height, width)}")

        # Initialize output arrays
        phantom_masked_labels = np.zeros_like(labels_volume)
        vial_masked_labels = np.zeros_like(labels_volume)

        print(f"Processing {num_label_types} label types across {num_slices} slices...")

        for label_idx in tqdm(range(num_label_types), desc="Processing label types"):
            label_name = self.label_keys[label_idx]

            for slice_idx in range(num_slices):
                # Get masks for this slice
                phantom_mask = phantom_masks_3d[slice_idx, :, :]
                vial_mask = vial_masks_3d[slice_idx, :, :]

                # Get original label slice
                original_label_slice = labels_volume[label_idx, slice_idx, :, :]

                # Apply phantom mask
                phantom_masked_labels[label_idx, slice_idx, :, :] = original_label_slice * phantom_mask

                # Apply vial mask
                vial_masked_labels[label_idx, slice_idx, :, :] = original_label_slice * vial_mask

        # Store checkpoints
        self.phantom_masked_labels = phantom_masked_labels
        self.vial_masked_labels = vial_masked_labels

        results = {
            'phantom_masked_labels': phantom_masked_labels,
            'vial_masked_labels': vial_masked_labels,
            'original_labels': labels_volume
        }

        # Print statistics
        self._print_masking_statistics(labels_volume, phantom_masked_labels, vial_masked_labels)

        return results

    def _print_masking_statistics(self, original, phantom_masked, vial_masked):
        """Print statistics about the masking process."""
        print("\nMasking Statistics:")
        for i, label_name in enumerate(self.label_keys):
            orig_sum = np.sum(original[i])
            phantom_sum = np.sum(phantom_masked[i])
            vial_sum = np.sum(vial_masked[i])

            phantom_retention = (phantom_sum / orig_sum * 100) if orig_sum > 0 else 0
            vial_retention = (vial_sum / orig_sum * 100) if orig_sum > 0 else 0

            print(f"  {label_name}:")
            print(f"    Original sum: {orig_sum:.2f}")
            print(f"    Phantom masked sum: {phantom_sum:.2f} ({phantom_retention:.1f}% retained)")
            print(f"    Vial masked sum: {vial_sum:.2f} ({vial_retention:.1f}% retained)")

    def visualize_labels_comparison(self, slice_idx=0, save_path=None):
        """
        Visualize original vs masked labels for a specific slice.

        Args:
            slice_idx (int): Slice index to visualize
            save_path (str, optional): Path to save the visualization
        """
        if self.processed_labels is None or self.phantom_masked_labels is None or self.vial_masked_labels is None:
            print("Error: No processed labels data available. Run apply_masks_to_labels first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for label_idx, label_name in enumerate(self.label_keys):
            # Original
            im1 = axes[label_idx, 0].imshow(self.processed_labels[label_idx, slice_idx], cmap='viridis')
            axes[label_idx, 0].set_title(f'{label_name} - Original')
            axes[label_idx, 0].axis('off')
            plt.colorbar(im1, ax=axes[label_idx, 0], fraction=0.046, pad=0.04)

            # Phantom masked
            im2 = axes[label_idx, 1].imshow(self.phantom_masked_labels[label_idx, slice_idx], cmap='viridis')
            axes[label_idx, 1].set_title(f'{label_name} - Phantom Masked')
            axes[label_idx, 1].axis('off')
            plt.colorbar(im2, ax=axes[label_idx, 1], fraction=0.046, pad=0.04)

            # Vial masked
            im3 = axes[label_idx, 2].imshow(self.vial_masked_labels[label_idx, slice_idx], cmap='viridis')
            axes[label_idx, 2].set_title(f'{label_name} - Vial Masked')
            axes[label_idx, 2].axis('off')
            plt.colorbar(im3, ax=axes[label_idx, 2], fraction=0.046, pad=0.04)

        plt.suptitle(f'Labels Comparison - Slice {slice_idx + 1}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def save_results(self, phantom_name=None):
        """
        Save all labels checkpoints to HDF5 files.

        Args:
            phantom_name (str, optional): Name for the phantom (used in filenames)
        """
        if phantom_name is None:
            phantom_name = self.phantom_name or "phantom"

        print(f"\nSaving labels checkpoints for {phantom_name}...")

        # Create output directory
        os.makedirs(self.output_folder, exist_ok=True)

        # Save phantom-masked labels
        phantom_labels_path = os.path.join(self.output_folder, f"{phantom_name}_phantom_masked_labels.h5")
        with h5py.File(phantom_labels_path, 'w') as hf:
            for i, key in enumerate(self.label_keys):
                hf.create_dataset(key, data=self.phantom_masked_labels[i], compression='gzip')
        print(f"Saved phantom-masked labels to: {phantom_labels_path}")

        # Save vial-masked labels
        vial_labels_path = os.path.join(self.output_folder, f"{phantom_name}_vial_masked_labels.h5")
        with h5py.File(vial_labels_path, 'w') as hf:
            for i, key in enumerate(self.label_keys):
                hf.create_dataset(key, data=self.vial_masked_labels[i], compression='gzip')
        print(f"Saved vial-masked labels to: {vial_labels_path}")

        # Save original processed labels
        original_labels_path = os.path.join(self.output_folder, f"{phantom_name}_original_labels.h5")
        with h5py.File(original_labels_path, 'w') as hf:
            for i, key in enumerate(self.label_keys):
                hf.create_dataset(key, data=self.processed_labels[i], compression='gzip')
        print(f"Saved original processed labels to: {original_labels_path}")

        # Save metadata
        self._save_metadata(phantom_name)

    def _save_metadata(self, phantom_name):
        """Save metadata about the labels processing."""
        metadata_path = os.path.join(self.output_folder, f"{phantom_name}_labels_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Labels Processing Metadata\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Phantom Name: {phantom_name}\n")
            f.write(f"Label Keys: {self.label_keys}\n")
            f.write(f"Original Label Shapes:\n")
            for key, shape in self.label_shapes.items():
                f.write(f"  {key}: {shape}\n")
            if self.processed_labels is not None:
                f.write(f"Processed Labels Shape: {self.processed_labels.shape}\n")
            if self.phantom_masked_labels is not None:
                f.write(f"Phantom Masked Labels Shape: {self.phantom_masked_labels.shape}\n")
            if self.vial_masked_labels is not None:
                f.write(f"Vial Masked Labels Shape: {self.vial_masked_labels.shape}\n")
        print(f"Saved labels metadata to: {metadata_path}")

    def process_complete_pipeline(self, labels_file_path, phantom_results, wanted_slices=None):
        """
        Complete pipeline for processing labels with phantom segmentation checkpoints.

        Args:
            labels_file_path (str): Path to the labels .mat file
            phantom_results (dict): Results from PhantomSegmenter.process_phantom_complete()
            wanted_slices: Slice selection (same format as used in phantom processing)

        Returns:
            dict: Complete labels processing checkpoints
        """
        print("\n" + "=" * 60)
        print("STARTING COMPLETE LABELS PROCESSING PIPELINE")
        print("=" * 60)

        # Step 1: Load labels
        labels_volume = self.load_labels_volume(labels_file_path)

        labels_volume = np.transpose(labels_volume, (0, 2, 1, 3))
        labels_volume = remove_black_slices(labels_volume, axis=self.config["axis"], threshold=self.config["threshold"],)

        # Step 2: Apply slice selection
        processed_labels = self.apply_slice_selection(labels_volume, wanted_slices)

        # Step 3: Apply masks
        labels_results = self.apply_masks_to_labels(
            processed_labels,
            phantom_results['phantom_masks_3d'],
            phantom_results['vial_masks_3d']
        )

        # Step 4: Save checkpoints
        phantom_name = self.phantom_name or "phantom"
        self.save_results(phantom_name)

        # Step 5: Create visualization
        visualization_path = os.path.join(self.output_folder, f"{phantom_name}_labels_comparison.png")
        self.visualize_labels_comparison(slice_idx=0, save_path=visualization_path)

        print("\n" + "=" * 60)
        print("LABELS PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return labels_results

class PhantomSegmenter:
    def __init__(self, config):
        """
        Initialize the PhantomSegmenter with configuration parameters.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = None
        self.mask_generator = None

        # Extract config parameters
        self.sam_checkpoint = config["sam_checkpoint"]

        # Phantom circle segmentation parameters
        self.phantom_area_threshold_low = config["area_threshold_low"]
        self.phantom_area_threshold_high = config["area_threshold_high"]
        self.default_wanted_slices = config.get("wanted_slices", None)

        # Vial segmentation parameters
        self.vial_area_threshold_low = config.get("vial_area_threshold_low", 22)
        self.vial_area_threshold_high = config.get("vial_area_threshold_high", 75)
        self.tube_radius = config.get("tube_radius", 5)
        self.default_num_vials = config.get("default_num_vials", 6)

        self.output_folder = config["output_folder"]
        self.phantom_name = config.get("phantom_name", None)

        # Manual segmentation variables
        self.manual_points = []
        self.current_slice_img = None
        self.manual_mask = None
        self.user_finished = False

    def initialize_sam_model(self, model_type="vit_h"):
        """Initialize SAM model and mask generator."""
        print(f"Using device: {self.device}")

        self.sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            min_mask_region_area=100,
        )

    def show_phantom_and_get_user_input(self, data):
        """
        Show phantom image to user and get input for number of vials and slices.

        Args:
            data (np.ndarray): 4D phantom data

        Returns:
            tuple: (num_vials, wanted_slices)
        """
        # Show a representative slice from the middle of the volume
        mid_scan = 0  # Use first scan
        mid_slice = data.shape[1] // 2  # Middle slice

        sample_img = data[mid_scan, mid_slice, :, :]

        # Normalize for display
        display_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min())

        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(display_img, cmap='gray')
        plt.title(f'Sample Phantom Image (Scan {mid_scan}, Slice {mid_slice})\n'
                  f'Please examine this image to determine the number of vials')
        plt.axis('off')
        plt.show()

        # Get number of vials from user
        print(f"\n{'=' * 60}")
        print("PHANTOM ANALYSIS - USER INPUT REQUIRED")
        print(f"{'=' * 60}")
        print(f"Please examine the displayed phantom image.")
        print(f"How many vials/tubes do you want to segment?")

        while True:
            try:
                user_input = input(f"Enter number of vials (default: {self.default_num_vials}): ").strip()
                if user_input == "":
                    num_vials = self.default_num_vials
                    break
                else:
                    num_vials = int(user_input)
                    if num_vials > 0:
                        break
                    else:
                        print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

        print(f"Selected number of vials: {num_vials}")

        # Get slice selection from user
        print(f"\nSlice Selection:")
        print(f"Total available slices: {data.shape[1]}")
        print(f"Current default slices: {self.default_wanted_slices}")
        print(f"Options:")
        print(f"1. Use default slice selection")
        print(f"2. Process all slices")
        print(f"3. Enter custom slice ranges")

        while True:
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == "1":
                wanted_slices = self.default_wanted_slices
                break
            elif choice == "2":
                wanted_slices = None  # Process all slices
                break
            elif choice == "3":
                wanted_slices = self.get_custom_slice_input(data.shape[1])
                break
            else:
                print("Please enter 1, 2, or 3.")

        print(f"Selected slices: {wanted_slices}")
        print(f"{'=' * 60}")

        return num_vials, wanted_slices

    def get_custom_slice_input(self, total_slices):
        """Get custom slice ranges from user."""
        print(f"\nCustom Slice Selection:")
        print(f"Total slices available: 0 to {total_slices - 1}")
        print(f"Enter slice ranges in format: start:end,start:end")
        print(f"Example: 10:20,30:40 (will process slices 10-19 and 30-39)")
        print(f"Example: 15:25 (will process slices 15-24)")

        while True:
            try:
                user_input = input("Enter slice ranges: ").strip()
                slice_ranges = []

                for range_str in user_input.split(','):
                    range_str = range_str.strip()
                    if ':' in range_str:
                        start, end = map(int, range_str.split(':'))
                        if 0 <= start < end <= total_slices:
                            slice_ranges.append(slice(start, end))
                        else:
                            raise ValueError(f"Invalid range {start}:{end}")
                    else:
                        # Single slice
                        single_slice = int(range_str)
                        if 0 <= single_slice < total_slices:
                            slice_ranges.append(slice(single_slice, single_slice + 1))
                        else:
                            raise ValueError(f"Invalid slice {single_slice}")

                if slice_ranges:
                    return [slice_ranges] if len(slice_ranges) > 1 else slice_ranges
                else:
                    print("No valid slices entered. Please try again.")

            except ValueError as e:
                print(f"Error: {e}. Please try again.")

    def _norm_2d(self, img):
        """Normalize 2D image"""
        sum_squares = np.sum(img ** 2, axis=0)
        sum_squares[sum_squares == 0] = 1
        norm_input = img / np.sqrt(sum_squares)
        norm_input[np.isnan(norm_input)] = 0
        return norm_input

    def find_wanted_slices(self, data: np.ndarray, slices: list) -> np.ndarray:
        """Extract wanted slices from 4D data with shape [scan, slice, H, W]"""
        if slices is not None:
            current_slice_range = slices
            print(current_slice_range)

            # Handle multiple slice ranges
            if isinstance(current_slice_range, list) and len(current_slice_range) > 0:
                # Check if it's a list of slice objects or a list containing lists of slices
                if isinstance(current_slice_range[0], list):
                    # Multiple ranges - need to extract and concatenate them
                    input_slices = []
                    for slice_range in current_slice_range:
                        if len(slice_range) == 2:  # Two slice objects
                            slice1, slice2 = slice_range
                            input_slices.append(data[:, slice1, :, :])
                            input_slices.append(data[:, slice2, :, :])
                        else:
                            input_slices.append(data[:, slice_range[0], :, :])

                    input_view_data = np.concatenate(input_slices, axis=1)  # concatenate along slice dimension
                else:
                    # Single slice object in a list
                    input_view_data = data[:, current_slice_range[0], :, :]
            else:
                # Single range
                input_view_data = data[:, current_slice_range, :, :]

            return input_view_data
        else:
            return data

    def find_phantom_circle_mask_targeted(self, masks, area_threshold_low, area_threshold_high, image_shape):
        """
        Find the phantom circle mask with targeted area range
        1. Look for masks in the specific area range
        2. Check if the center is zero (background) - if so, invert the mask
        """
        print(f"Total masks generated: {len(masks)}")
        print(f"Looking for masks with area between {area_threshold_low} and {area_threshold_high} pixels")

        if len(masks) == 0:
            return np.zeros(image_shape, dtype=np.uint8)

        # Sort masks by area to get info
        mask_areas = []
        for i, mask in enumerate(masks):
            area = np.sum(mask['segmentation'])
            mask_areas.append((i, area, mask))

        # Sort by area (descending)
        mask_areas.sort(key=lambda x: x[1], reverse=True)

        # Print all mask areas for debugging
        print("All mask areas:")
        for i, (idx, area, mask) in enumerate(mask_areas):
            status = "TARGET RANGE" if area_threshold_low < area < area_threshold_high else "outside range"
            print(f"  Mask {i}: Area = {area} {status}")
            if i >= 10:  # Limit output
                print(f"  ... and {len(mask_areas) - 10} more masks")
                break

        # Strategy 1: Look for masks specifically in our target area range
        target_masks = []
        for idx, area, mask in mask_areas:
            if area_threshold_low < area < area_threshold_high:
                target_masks.append((idx, area, mask))

        if target_masks:
            print(f"Found {len(target_masks)} masks in target range!")

            # Take the largest mask in the target range
            target_masks.sort(key=lambda x: x[1], reverse=True)  # Sort by area, largest first
            idx, area, mask = target_masks[0]

            selected_mask = mask['segmentation'].astype(np.uint8)

            # Check if center is zero (indicating background mask)
            center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
            center_region = selected_mask[center_y - 2:center_y + 3, center_x - 2:center_x + 3]  # 5x5 region around center
            center_value = np.mean(center_region)

            print(f"Using target range mask with area {area}, center value: {center_value:.2f}")

            if center_value < 0.5:  # Center is mostly zero (background)
                print("Center is zero - inverting mask (was selecting background)")
                selected_mask = 1 - selected_mask  # Invert the mask
            else:
                print("Center is non-zero - using mask as is (selecting phantom)")

            return selected_mask

        # Strategy 2: If no masks in target range, look for masks close to the range
        print("No masks in target range, looking for masks close to the range...")

        # Expand the search range slightly
        expanded_low = area_threshold_low - 500
        expanded_high = area_threshold_high + 500

        close_masks = []
        for idx, area, mask in mask_areas:
            if expanded_low < area < expanded_high:
                close_masks.append((idx, area, mask))

        if close_masks:
            print(f"Found {len(close_masks)} masks in expanded range ({expanded_low}-{expanded_high})")

            # Take the largest mask in the expanded range
            close_masks.sort(key=lambda x: x[1], reverse=True)
            idx, area, mask = close_masks[0]

            selected_mask = mask['segmentation'].astype(np.uint8)

            # Check center and invert if needed
            center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
            center_region = selected_mask[center_y - 2:center_y + 3, center_x - 2:center_x + 3]
            center_value = np.mean(center_region)

            print(f"Using expanded range mask with area {area}, center value: {center_value:.2f}")

            if center_value < 0.5:  # Center is mostly zero
                print("Center is zero - inverting mask")
                selected_mask = 1 - selected_mask
            else:
                print("Center is non-zero - using mask as is")

            return selected_mask

        # Strategy 3: Fallback - take any reasonable mask
        print("No masks in expanded range, using fallback...")
        reasonable_masks = [(idx, area, mask) for idx, area, mask in mask_areas if area > 1000]

        if reasonable_masks:
            idx, area, mask = reasonable_masks[0]  # Largest reasonable mask
            selected_mask = mask['segmentation'].astype(np.uint8)

            # Check center and invert if needed
            center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
            center_region = selected_mask[center_y - 2:center_y + 3, center_x - 2:center_x + 3]
            center_value = np.mean(center_region)

            print(f"Fallback: Using mask with area {area}, center value: {center_value:.2f}")

            if center_value < 0.5:
                print("Center is zero - inverting mask")
                selected_mask = 1 - selected_mask
            else:
                print("Center is non-zero - using mask as is")

            return selected_mask

        print("No suitable masks found at all!")
        return np.zeros(image_shape, dtype=np.uint8)

    def find_unique_tubes(self, masks, target_tubes):
        """
        Find unique tubes avoiding duplicate center locations, keeping only the largest tube in each region
        """
        # Filter tubes by area first
        valid_tubes = []
        for mask in masks:
            area = np.sum(mask['segmentation'])
            if self.vial_area_threshold_low < area < self.vial_area_threshold_high:
                # Calculate center coordinates
                y_indices, x_indices = np.where(mask['segmentation'])
                if len(y_indices) > 0:  # Check if mask is not empty
                    center_y = np.mean(y_indices)
                    center_x = np.mean(x_indices)

                    valid_tubes.append({
                        'mask': mask,
                        'center_y': center_y,
                        'center_x': center_x,
                        'area': area
                    })

        # Sort tubes by area (largest first)
        valid_tubes.sort(key=lambda x: x['area'], reverse=True)

        # Find unique tubes
        unique_tubes = []
        used_centers = []

        for tube in valid_tubes:
            # Check if this tube is in a new region
            is_new_region = True
            for existing_center in used_centers:
                # Check if the tube's center is within 3 pixels of an existing center
                if (abs(tube['center_y'] - existing_center['center_y']) < 3 and
                        abs(tube['center_x'] - existing_center['center_x']) < 3):
                    is_new_region = False
                    break

            # If it's a new region or the first tube, add it
            if is_new_region:
                unique_tubes.append(tube)
                used_centers.append({
                    'center_y': tube['center_y'],
                    'center_x': tube['center_x']
                })

                # Stop when we have enough unique tubes
                if len(unique_tubes) == target_tubes:
                    break

        return [tube['mask'] for tube in unique_tubes]

    def visualize_segmentation(self, original_img, mask, slice_idx, scan_idx, title="Segmentation"):
        """Create visualization of the segmentation"""
        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img, cmap='viridis')
        plt.title('Original image')
        plt.axis('off')

        # Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        # Segmented result
        segmented = original_img.copy()
        segmented[mask == 0] = 0  # Keep only phantom circle region
        plt.subplot(1, 3, 3)
        plt.imshow(segmented, cmap='viridis')
        plt.title('Segmented')
        plt.axis('off')

        plt.suptitle(f'{title} - Slice {slice_idx + 1}')
        plt.tight_layout()
        plt.show()
        plt.close()

        # Print statistics
        print(f"Final mask statistics:")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Mask area: {np.sum(mask)}")
        print(f"  - Mask percentage: {100 * np.sum(mask) / mask.size:.1f}%")

    def ask_user_preference(self):
        """Ask user what to do when automatic segmentation fails"""
        print("\n" + "=" * 60)
        print("AUTOMATIC SEGMENTATION FAILED")
        print("Choose an option:")
        print("1. Open manual segmentation window")
        print("2. Use empty mask for this slice")
        print("3. Use the best available automatic mask (even if incomplete)")
        print("=" * 60)

        while True:
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            print("Invalid choice. Please enter 1, 2, or 3.")

    def on_click(self, event):
        """Handle mouse clicks for manual segmentation"""
        if event.inaxes is not None and event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            self.manual_points.append((x, y))

            # Draw a larger circle at the clicked point
            circle = Circle((x, y), 5, color='red', fill=True)
            event.inaxes.add_patch(circle)
            plt.draw()

            print(f"Point {len(self.manual_points)}: ({x}, {y})")

    def on_key(self, event):
        """Handle keyboard events for manual segmentation"""
        if event.key == 'enter':
            # Finish manual segmentation
            print("Enter pressed - finishing manual segmentation")
            if len(self.manual_points) > 0:
                self.create_manual_mask()
            self.user_finished = True
            plt.close()
        elif event.key == 'r':
            # Reset points
            self.manual_points = []
            plt.clf()
            plt.imshow(self.current_slice_img, cmap='gray')
            plt.title('Manual Segmentation - Click on tube centers, Press R to reset, Enter to finish')
            plt.draw()
            print("Points reset")
        elif event.key == 's':
            # Skip this slice
            print("S pressed - skipping manual segmentation")
            self.manual_points = []
            self.user_finished = True
            plt.close()

    def on_close(self, event):
        """Handle window close event"""
        print("Window closed by user")
        self.user_finished = True

    def create_manual_mask(self):
        """Create a mask from manually selected points"""
        if len(self.manual_points) == 0:
            self.manual_mask = np.zeros(self.current_slice_img.shape, dtype=np.uint8)
            return

        # Create mask with circles around each point
        self.manual_mask = np.zeros(self.current_slice_img.shape, dtype=np.uint8)

        for x, y in self.manual_points:
            # Create a circular mask for each tube
            cv2.circle(self.manual_mask, (x, y), self.tube_radius, 1, -1)

        print(f"Created manual mask with {len(self.manual_points)} tubes")

    def manual_segmentation_window(self, slice_img, slice_idx, target_tubes):
        """Open a window for manual segmentation"""
        # Reset variables
        self.manual_points = []
        self.current_slice_img = slice_img.copy()
        self.manual_mask = None
        self.user_finished = False

        # Set matplotlib to use a GUI backend that supports interaction
        try:
            plt.switch_backend('Qt5Agg')  # Try Qt5 first
        except:
            try:
                plt.switch_backend('TkAgg')  # Fallback to Tk
            except:
                print("Warning: Could not set interactive backend")

        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(f'Manual Segmentation - Slice {slice_idx + 1}\n'
                     f'Click on {target_tubes} tube centers\n'
                     f'Press R to reset, Enter to finish, S to skip')

        # Connect event handlers
        cid_click = fig.canvas.mpl_connect('button_press_event', self.on_click)
        cid_key = fig.canvas.mpl_connect('key_press_event', self.on_key)
        cid_close = fig.canvas.mpl_connect('close_event', self.on_close)

        print(f"\n{'=' * 60}")
        print(f"MANUAL SEGMENTATION MODE")
        print(f"Slice {slice_idx + 1} - Target: {target_tubes} tubes")
        print(f"Instructions:")
        print(f"- Click on tube centers ({target_tubes} total)")
        print(f"- Press 'R' to reset all points")
        print(f"- Press 'Enter' to finish and continue")
        print(f"- Press 'S' to skip this slice")
        print(f"{'=' * 60}")

        # Show the plot and wait for user interaction
        plt.ion()  # Turn on interactive mode
        plt.show()

        # Wait for user to finish
        while not self.user_finished:
            try:
                plt.pause(0.1)  # Small pause to allow events to be processed
                if not plt.get_fignums():  # Check if figure is still open
                    break
            except:
                break

        plt.ioff()  # Turn off interactive mode

        # Disconnect event handlers
        try:
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            fig.canvas.mpl_disconnect(cid_close)
        except:
            pass

        # Create mask from manual points if any were selected
        if self.manual_mask is None and len(self.manual_points) > 0:
            self.create_manual_mask()

        # If no mask was created, make an empty one
        if self.manual_mask is None:
            self.manual_mask = np.zeros(self.current_slice_img.shape, dtype=np.uint8)

        print(f"Manual segmentation completed. Points selected: {len(self.manual_points)}")
        return self.manual_mask

    def segment_phantom_circle(self, data, wanted_slices):
        """
        First step: Segment the phantom circle boundary.

        Args:
            data (np.ndarray): 4D phantom data
            wanted_slices: Slice selection

        Returns:
            tuple: (clean_phantom_data, phantom_masks_3d)
        """
        if self.mask_generator is None:
            self.initialize_sam_model()

        print("\n" + "=" * 50)
        print("STEP 1: PHANTOM CIRCLE SEGMENTATION")
        print("=" * 50)

        # Apply slice selection
        clean_phantom = self.find_wanted_slices(data, wanted_slices)

        # Get dimensions
        num_scans, num_slices, height, width = clean_phantom.shape
        print(f"Processing phantom dimensions: {num_scans} scans, {height}x{width} pixels, {num_slices} slices")

        # Initialize 3D mask array for phantom circles
        phantom_masks_3d = np.zeros((num_slices, height, width), dtype=np.uint8)

        print("Generating phantom circle masks using first scan...")
        for slice_idx in tqdm(range(num_slices), desc="Processing phantom circle masks"):
            img_2d = clean_phantom[0, slice_idx, :, :]

            # Apply normalization and prepare for SAM
            img_normalized = self._norm_2d(img_2d)
            img_min = img_normalized.min()
            img_max = img_normalized.max()

            if img_max > img_min:
                img_sam = ((img_normalized - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
            else:
                img_sam = np.zeros_like(img_normalized, dtype=np.uint8)

            img_rgb = np.stack([img_sam] * 3, axis=-1)

            # Generate masks
            masks = self.mask_generator.generate(img_rgb)

            # Find phantom circle using area thresholds
            phantom_mask = self.find_phantom_circle_mask_targeted(
                masks, self.phantom_area_threshold_low, self.phantom_area_threshold_high, img_2d.shape
            )

            phantom_masks_3d[slice_idx, :, :] = phantom_mask.astype(np.uint8)

            # Show visualization for first slice
            if slice_idx == 0:
                self.visualize_segmentation(img_2d, phantom_mask, slice_idx, 0, "Phantom Circle")

        print(f"Completed phantom circle segmentation for {num_slices} slices")
        return clean_phantom, phantom_masks_3d

    def segment_vials_within_phantom(self, clean_phantom, phantom_masks_3d, num_vials):
        """
        Second step: Segment vials within the phantom circle.

        Args:
            clean_phantom (np.ndarray): Phantom data from step 1
            phantom_masks_3d (np.ndarray): Phantom circle masks from step 1
            num_vials (int): Number of vials to detect

        Returns:
            tuple: (vial_masked_images_4d, vial_masks_3d)
        """
        print("\n" + "=" * 50)
        print("STEP 2: VIAL SEGMENTATION WITHIN PHANTOM")
        print("=" * 50)

        num_scans, num_slices, height, width = clean_phantom.shape

        # Initialize arrays for vial checkpoints
        vial_masks_3d = np.zeros((num_slices, height, width), dtype=np.uint8)
        vial_masked_images_4d = np.zeros_like(clean_phantom, dtype=clean_phantom.dtype)

        print(f"Segmenting {num_vials} vials within phantom circles...")

        for slice_idx in tqdm(range(num_slices), desc="Processing vial segmentation"):
            # Get phantom mask for this slice
            phantom_mask = phantom_masks_3d[slice_idx, :, :]

            # Get image from first scan
            img_2d = clean_phantom[0, slice_idx, :, :]

            # Apply phantom mask to focus only on phantom region
            masked_img = img_2d * phantom_mask

            if np.sum(phantom_mask) == 0:
                print(f"No phantom region found in slice {slice_idx + 1}, skipping vial segmentation")
                vial_masks_3d[slice_idx, :, :] = np.zeros((height, width), dtype=np.uint8)
                continue

            # Normalize masked image for SAM
            if np.max(masked_img) > 0:
                masked_img_normalized = (masked_img / np.max(masked_img) * 255).astype(np.uint8)
            else:
                masked_img_normalized = np.zeros_like(masked_img, dtype=np.uint8)

            # Convert to RGB for SAM
            img_rgb = np.stack([masked_img_normalized] * 3, axis=-1)

            # Generate vial masks
            masks = self.mask_generator.generate(img_rgb)

            # Find unique vials
            unique_vial_masks = self.find_unique_tubes(masks, num_vials)

            print(f"  Found {len(unique_vial_masks)} vials in slice {slice_idx + 1}")

            if len(unique_vial_masks) == num_vials:
                # Create combined vial mask
                combined_vial_mask = np.zeros((height, width), dtype=np.uint8)
                for mask in unique_vial_masks:
                    combined_vial_mask = np.logical_or(combined_vial_mask, mask['segmentation']).astype(np.uint8)

                # Ensure vials are only within phantom boundary
                combined_vial_mask = combined_vial_mask * phantom_mask

                vial_masks_3d[slice_idx, :, :] = combined_vial_mask
                print(f"  Using automatic vial mask for slice {slice_idx + 1}")

            else:
                # Vial segmentation failed - offer options
                print(f"  Vial segmentation failed for slice {slice_idx + 1}")
                print(f"    Expected {num_vials} vials, found {len(unique_vial_masks)}")

                choice = self.ask_user_preference()

                if choice == 1:
                    # Manual vial segmentation within phantom
                    display_img = (masked_img_normalized).astype(np.uint8)
                    manual_mask = self.manual_segmentation_window(display_img, slice_idx, num_vials)

                    if manual_mask is not None and np.sum(manual_mask) > 0:
                        # Ensure manual mask is within phantom boundary
                        manual_mask = manual_mask * phantom_mask
                        vial_masks_3d[slice_idx, :, :] = manual_mask.astype(np.uint8)
                        print(f"  Using manual vial mask for slice {slice_idx + 1}")
                    else:
                        vial_masks_3d[slice_idx, :, :] = np.zeros((height, width), dtype=np.uint8)

                elif choice == 2:
                    vial_masks_3d[slice_idx, :, :] = np.zeros((height, width), dtype=np.uint8)

                elif choice == 3:
                    if len(unique_vial_masks) > 0:
                        combined_vial_mask = np.zeros((height, width), dtype=np.uint8)
                        for mask in unique_vial_masks:
                            combined_vial_mask = np.logical_or(combined_vial_mask, mask['segmentation']).astype(
                                np.uint8)
                        combined_vial_mask = combined_vial_mask * phantom_mask
                        vial_masks_3d[slice_idx, :, :] = combined_vial_mask
                    else:
                        vial_masks_3d[slice_idx, :, :] = np.zeros((height, width), dtype=np.uint8)

        # Apply vial masks to all scans
        print(f"\nApplying vial masks to all {num_scans} scans...")
        for scan_idx in range(num_scans):
            for slice_idx in range(num_slices):
                original_slice = clean_phantom[scan_idx, slice_idx, :, :]
                vial_mask = vial_masks_3d[slice_idx, :, :]
                vial_masked_images_4d[scan_idx, slice_idx, :, :] = original_slice * vial_mask

        torch.cuda.empty_cache()
        return vial_masked_images_4d, vial_masks_3d

    def save_complete_results(self, results):
        """Save all segmentation checkpoints to files."""
        phantom_name = self.phantom_name or "phantom"

        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)

        # Save phantom masks
        phantom_masks_path = os.path.join(self.output_folder, f"{phantom_name}_phantom_masks.h5")
        with h5py.File(phantom_masks_path, 'w') as hf:
            hf.create_dataset('phantom_masks', data=results['phantom_masks_3d'], compression='gzip')
        print(f"Saved phantom masks to: {phantom_masks_path}")

        # Save vial masks
        vial_masks_path = os.path.join(self.output_folder, f"{phantom_name}_vial_masks.h5")
        with h5py.File(vial_masks_path, 'w') as hf:
            hf.create_dataset('vial_masks', data=results['vial_masks_3d'], compression='gzip')
        print(f"Saved vial masks to: {vial_masks_path}")

        # Save final masked images (vials only)
        vial_images_path = os.path.join(self.output_folder, f"{phantom_name}_vial_images.h5")
        with h5py.File(vial_images_path, 'w') as hf:
            hf.create_dataset('vial_images', data=results['vial_masked_images_4d'], compression='gzip')
        print(f"Saved vial images to: {vial_images_path}")

        # Save clean phantom data (within phantom boundary)
        clean_phantom_path = os.path.join(self.output_folder, f"{phantom_name}_clean_phantom.h5")
        with h5py.File(clean_phantom_path, 'w') as hf:
            hf.create_dataset('clean_phantom', data=results['clean_phantom'], compression='gzip')
        print(f"Saved clean phantom to: {clean_phantom_path}")

        # Save metadata
        metadata_path = os.path.join(self.output_folder, f"{phantom_name}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Phantom Name: {phantom_name}\n")
            f.write(f"Number of Vials: {results['num_vials']}\n")
            f.write(f"Wanted Slices: {results['wanted_slices']}\n")
            f.write(f"Phantom Shape: {results['clean_phantom'].shape}\n")
            f.write(f"Phantom Masks Shape: {results['phantom_masks_3d'].shape}\n")
            f.write(f"Vial Masks Shape: {results['vial_masks_3d'].shape}\n")
            f.write(f"Vial Images Shape: {results['vial_masked_images_4d'].shape}\n")
        print(f"Saved metadata to: {metadata_path}")

    def process_phantom_complete(self, data):
        """
        Complete two-step phantom processing pipeline.

        Args:
            data (np.ndarray): 4D phantom data

        Returns:
            dict: Results containing all segmentation outputs
        """
        # Step 0: Get user input
        num_vials, wanted_slices = self.show_phantom_and_get_user_input(data)

        # Step 1: Segment phantom circle
        clean_phantom, phantom_masks_3d = self.segment_phantom_circle(data, wanted_slices)

        # Step 2: Segment vials within phantom
        vial_masked_images_4d, vial_masks_3d = self.segment_vials_within_phantom(
            clean_phantom, phantom_masks_3d, num_vials
        )

        # Save checkpoints
        results = {
            'clean_phantom': clean_phantom,
            'phantom_masks_3d': phantom_masks_3d,
            'vial_masked_images_4d': vial_masked_images_4d,
            'vial_masks_3d': vial_masks_3d,
            'num_vials': num_vials,
            'wanted_slices': wanted_slices
        }

        self.save_complete_results(results)

        print("\n" + "=" * 50)
        print("PHANTOM PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        return results


class PhantomCropper:
    """
    A class to crop phantom data, labels, and masks to target sizes after segmentation processing.
    Target dimensions are determined from config (H, W) and input data shapes (scans, slices).
    """

    def __init__(self, config):
        """
        Initialize the PhantomCropper.

        Args:
            config (dict): Configuration dictionary from the main pipeline
        """
        self.config = config
        self.output_folder = config["output_folder"]
        self.phantom_name = config.get("phantom_name", "phantom")

        # Get target H, W from config
        self.target_height = config.get("target_height", 80)
        self.target_width = config.get("target_width", 80)

        # Target shapes will be calculated dynamically based on input data
        self.target_shapes = None

        # Create cropped output directories
        self.cropped_phantom_path = os.path.join(self.output_folder, "cropped")
        self.cropped_labels_path = os.path.join(self.output_folder, "cropped_labels")
        self.cropped_masks_path = os.path.join(self.output_folder, "cropped_masks")

        # Create output directories
        os.makedirs(self.cropped_phantom_path, exist_ok=True)
        os.makedirs(self.cropped_labels_path, exist_ok=True)
        os.makedirs(self.cropped_masks_path, exist_ok=True)

    def calculate_target_shapes(self, phantom_data):
        """
        Calculate target shapes based on input data dimensions and config.

        Args:
            phantom_data (np.ndarray): Input phantom data with shape [scans, slices, H, W]

        Returns:
            dict: Target shapes for different data types
        """
        input_shape = phantom_data.shape
        num_scans = input_shape[0]  # Get actual number of scans
        num_slices = input_shape[1]  # Get actual number of slices

        target_shapes = {
            'phantom_data': (num_scans, num_slices, self.target_height, self.target_width),
            'labels': (num_slices, self.target_height, self.target_width),
            'masks': (num_slices, self.target_height, self.target_width),
            'vial_images': (num_scans, num_slices, self.target_height, self.target_width),
        }

        print(f"\nCalculated target shapes:")
        print(f"  Input shape: {input_shape}")
        print(f"  Target H: {self.target_height}, Target W: {self.target_width}")
        for key, shape in target_shapes.items():
            print(f"  {key}: {shape}")

        self.target_shapes = target_shapes
        return target_shapes

    def find_phantom_center_and_crop(self, data, target_shape, data_type="phantom"):
        """
        Find the center of mass of the phantom and crop around it.

        Args:
            data (np.ndarray): Input data to crop
            target_shape (tuple): Target shape after cropping
            data_type (str): Type of data for appropriate thresholding

        Returns:
            tuple: (cropped_data, crop_info)
        """
        print(f"  Finding center of mass for {data_type} data...")

        # Create appropriate mask based on data type
        if data_type == "phantom" or data_type == "vial_images":
            # For phantom data, create binary mask using threshold
            threshold = np.mean(data) + 0.5 * np.std(data)
            mask = data > threshold
        elif data_type == "labels":
            # For labels, any non-zero value indicates presence
            mask = data > 0
        elif data_type == "masks":
            # Masks are already binary
            mask = data > 0
        else:
            # Default approach
            mask = data > np.mean(data)

        # Find center of mass
        try:
            com = center_of_mass(mask)
        except:
            # Fallback to geometric center if center_of_mass fails
            com = tuple(s // 2 for s in data.shape)

        # Handle NaN values - set to center of volume if NaN
        com_clean = []
        for i, c in enumerate(com):
            if np.isnan(c) or c is None:
                com_clean.append(data.shape[i] // 2)
            else:
                com_clean.append(int(round(c)))

        com = com_clean
        print(f"    Center of mass: {com}")

        # Crop around center of mass
        current_shape = data.shape
        cropped_data = data.copy()
        crop_info = {"starts": [], "ends": []}

        for dim in range(len(target_shape)):
            if target_shape[dim] < current_shape[dim]:
                # Calculate crop boundaries around center of mass
                half_target = target_shape[dim] // 2
                start = max(0, com[dim] - half_target)
                end = min(current_shape[dim], start + target_shape[dim])

                # Adjust start if we hit the boundary
                if end - start < target_shape[dim]:
                    start = max(0, end - target_shape[dim])

                print(f"    Dim {dim}: cropping from {start} to {end} (around center {com[dim]})")

                # Apply crop
                if dim == 0:
                    cropped_data = cropped_data[start:end, ...]
                elif dim == 1:
                    cropped_data = cropped_data[:, start:end, ...]
                elif dim == 2:
                    cropped_data = cropped_data[:, :, start:end, ...]
                elif dim == 3:
                    cropped_data = cropped_data[:, :, :, start:end]

                crop_info["starts"].append(start)
                crop_info["ends"].append(end)
            else:
                crop_info["starts"].append(0)
                crop_info["ends"].append(current_shape[dim])

        print(f"    Final cropped shape: {cropped_data.shape}")
        return cropped_data, crop_info

    def crop_phantom_results(self, phantom_results):
        """
        Crop all phantom segmentation checkpoints.

        Args:
            phantom_results (dict): Results from PhantomSegmenter.process_phantom_complete()

        Returns:
            dict: Cropped phantom checkpoints
        """
        print("\n" + "=" * 60)
        print("CROPPING PHANTOM SEGMENTATION RESULTS")
        print("=" * 60)

        # Calculate target shapes based on input data
        clean_phantom = phantom_results['clean_phantom']
        self.calculate_target_shapes(clean_phantom)

        cropped_results = {}

        # 1. Crop clean phantom data
        print(f"\nCropping clean phantom data...")
        print(f"  Original shape: {clean_phantom.shape}")

        cropped_phantom, phantom_crop_info = self.find_phantom_center_and_crop(
            clean_phantom, self.target_shapes['phantom_data'], "phantom"
        )
        cropped_results['clean_phantom'] = cropped_phantom
        cropped_results['phantom_crop_info'] = phantom_crop_info

        # 2. Crop phantom masks (use same crop info)
        print(f"\nCropping phantom masks...")
        phantom_masks = phantom_results['phantom_masks_3d']
        print(f"  Original shape: {phantom_masks.shape}")

        # Apply same cropping as phantom data (skip first dimension for masks)
        cropped_phantom_masks = phantom_masks.copy()
        starts = phantom_crop_info["starts"][1:]  # Skip scan dimension
        ends = phantom_crop_info["ends"][1:]

        for dim in range(len(starts)):
            if dim == 0:  # slice dimension
                cropped_phantom_masks = cropped_phantom_masks[starts[dim]:ends[dim], :, :]
            elif dim == 1:  # H dimension
                cropped_phantom_masks = cropped_phantom_masks[:, starts[dim]:ends[dim], :]
            elif dim == 2:  # W dimension
                cropped_phantom_masks = cropped_phantom_masks[:, :, starts[dim]:ends[dim]]

        cropped_results['phantom_masks_3d'] = cropped_phantom_masks
        print(f"  Cropped phantom masks shape: {cropped_phantom_masks.shape}")

        # 3. Crop vial masks (same cropping as phantom masks)
        print(f"\nCropping vial masks...")
        vial_masks = phantom_results['vial_masks_3d']
        print(f"  Original shape: {vial_masks.shape}")

        cropped_vial_masks = vial_masks.copy()
        for dim in range(len(starts)):
            if dim == 0:  # slice dimension
                cropped_vial_masks = cropped_vial_masks[starts[dim]:ends[dim], :, :]
            elif dim == 1:  # H dimension
                cropped_vial_masks = cropped_vial_masks[:, starts[dim]:ends[dim], :]
            elif dim == 2:  # W dimension
                cropped_vial_masks = cropped_vial_masks[:, :, starts[dim]:ends[dim]]

        cropped_results['vial_masks_3d'] = cropped_vial_masks
        print(f"  Cropped vial masks shape: {cropped_vial_masks.shape}")

        # 4. Crop vial images (same cropping as phantom data)
        print(f"\nCropping vial masked images...")
        vial_images = phantom_results['vial_masked_images_4d']
        print(f"  Original shape: {vial_images.shape}")

        cropped_vial_images = vial_images.copy()
        for dim in range(len(phantom_crop_info["starts"])):
            start = phantom_crop_info["starts"][dim]
            end = phantom_crop_info["ends"][dim]

            if dim == 0:  # scan dimension
                cropped_vial_images = cropped_vial_images[start:end, ...]
            elif dim == 1:  # slice dimension
                cropped_vial_images = cropped_vial_images[:, start:end, ...]
            elif dim == 2:  # H dimension
                cropped_vial_images = cropped_vial_images[:, :, start:end, :]
            elif dim == 3:  # W dimension
                cropped_vial_images = cropped_vial_images[:, :, :, start:end]

        cropped_results['vial_masked_images_4d'] = cropped_vial_images
        print(f"  Cropped vial images shape: {cropped_vial_images.shape}")

        # Copy other metadata
        cropped_results['num_vials'] = phantom_results['num_vials']
        cropped_results['wanted_slices'] = phantom_results['wanted_slices']

        return cropped_results

    def crop_labels_results(self, labels_results, phantom_crop_info):
        """
        Crop labels checkpoints using the same cropping info as phantom data.

        Args:
            labels_results (dict): Results from LabelsProcessor
            phantom_crop_info (dict): Cropping information from phantom data

        Returns:
            dict: Cropped labels checkpoints
        """
        print("\n" + "=" * 60)
        print("CROPPING LABELS RESULTS")
        print("=" * 60)

        cropped_labels = {}

        # Use phantom crop info but skip scan dimension for labels
        starts = phantom_crop_info["starts"][1:]  # Skip scan dimension
        ends = phantom_crop_info["ends"][1:]

        for label_type, label_data in labels_results.items():
            if isinstance(label_data, np.ndarray):
                print(f"\nCropping {label_type}...")
                print(f"  Original shape: {label_data.shape}")

                cropped_label = label_data.copy()

                # Apply cropping based on number of dimensions
                if label_data.ndim == 4:  # [2, slices, H, W] format
                    for dim in range(len(starts)):
                        if dim == 0:  # slice dimension
                            cropped_label = cropped_label[:, starts[dim]:ends[dim], :, :]
                        elif dim == 1:  # H dimension
                            cropped_label = cropped_label[:, :, starts[dim]:ends[dim], :]
                        elif dim == 2:  # W dimension
                            cropped_label = cropped_label[:, :, :, starts[dim]:ends[dim]]

                elif label_data.ndim == 3:  # [slices, H, W] format
                    for dim in range(len(starts)):
                        if dim == 0:  # slice dimension
                            cropped_label = cropped_label[starts[dim]:ends[dim], :, :]
                        elif dim == 1:  # H dimension
                            cropped_label = cropped_label[:, starts[dim]:ends[dim], :]
                        elif dim == 2:  # W dimension
                            cropped_label = cropped_label[:, :, starts[dim]:ends[dim]]

                cropped_labels[label_type] = cropped_label
                print(f"  Cropped shape: {cropped_label.shape}")
            else:
                # Copy non-array data as-is
                cropped_labels[label_type] = label_data

        return cropped_labels

    def save_cropped_results(self, cropped_phantom_results, cropped_labels_results=None):
        """
        Save all cropped checkpoints to files.

        Args:
            cropped_phantom_results (dict): Cropped phantom checkpoints
            cropped_labels_results (dict, optional): Cropped labels checkpoints
        """
        print("\n" + "=" * 60)
        print("SAVING CROPPED RESULTS")
        print("=" * 60)

        # Save cropped phantom data
        phantom_files = {
            'clean_phantom': 'clean_phantom_cropped.h5',
            'phantom_masks_3d': 'phantom_masks_cropped.h5',
            'vial_masks_3d': 'vial_masks_cropped.h5',
            'vial_masked_images_4d': 'vial_images_cropped.h5'
        }

        for key, filename in phantom_files.items():
            if key in cropped_phantom_results:
                filepath = os.path.join(self.cropped_phantom_path, f"{self.phantom_name}_{filename}")
                with h5py.File(filepath, 'w') as f:
                    if key == 'phantom_masks_3d':
                        f.create_dataset('phantom_masks', data=cropped_phantom_results[key], compression='gzip')
                    elif key == 'vial_masks_3d':
                        f.create_dataset('vial_masks', data=cropped_phantom_results[key], compression='gzip')
                    elif key == 'vial_masked_images_4d':
                        f.create_dataset('vial_images', data=cropped_phantom_results[key], compression='gzip')
                    else:
                        f.create_dataset('clean_phantom', data=cropped_phantom_results[key], compression='gzip')
                print(f"Saved {key} to: {filepath}")

        # Save cropped labels if available
        if cropped_labels_results:
            for label_type, label_data in cropped_labels_results.items():
                if isinstance(label_data, np.ndarray):
                    if label_data.ndim == 4:  # [2, slices, H, W] format
                        # Save each label type separately
                        label_keys = ['AL_fB_Im', 'AL_kBA_Im']
                        for i, key in enumerate(label_keys):
                            filepath = os.path.join(self.cropped_labels_path,
                                                    f"{self.phantom_name}_{label_type}_{key}_cropped.h5")
                            with h5py.File(filepath, 'w') as f:
                                f.create_dataset(key, data=label_data[i], compression='gzip')
                            print(f"Saved {label_type} {key} to: {filepath}")
                    else:
                        filepath = os.path.join(self.cropped_labels_path,
                                                f"{self.phantom_name}_{label_type}_cropped.h5")
                        with h5py.File(filepath, 'w') as f:
                            f.create_dataset(label_type, data=label_data, compression='gzip')
                        print(f"Saved {label_type} to: {filepath}")

        # Save metadata
        metadata_path = os.path.join(self.cropped_phantom_path, f"{self.phantom_name}_cropped_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Cropped Phantom Processing Metadata\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Phantom Name: {self.phantom_name}\n")
            f.write(f"Target Height: {self.target_height}\n")
            f.write(f"Target Width: {self.target_width}\n")
            f.write(f"Target Shapes: {self.target_shapes}\n")
            f.write(f"Cropped Phantom Shape: {cropped_phantom_results['clean_phantom'].shape}\n")
            f.write(f"Cropped Phantom Masks Shape: {cropped_phantom_results['phantom_masks_3d'].shape}\n")
            f.write(f"Cropped Vial Masks Shape: {cropped_phantom_results['vial_masks_3d'].shape}\n")
            f.write(f"Cropped Vial Images Shape: {cropped_phantom_results['vial_masked_images_4d'].shape}\n")
            if 'phantom_crop_info' in cropped_phantom_results:
                f.write(f"Crop Info: {cropped_phantom_results['phantom_crop_info']}\n")
        print(f"Saved cropped metadata to: {metadata_path}")

    def visualize_cropping_comparison(self, original_results, cropped_results, slice_idx=0):
        """
        Create visualization comparing original vs cropped checkpoints.

        Args:
            original_results (dict): Original phantom checkpoints
            cropped_results (dict): Cropped phantom checkpoints
            slice_idx (int): Slice index to visualize
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original data
        orig_phantom = original_results['clean_phantom'][0, slice_idx, :, :]
        orig_phantom_mask = original_results['phantom_masks_3d'][slice_idx, :, :]
        orig_vial_mask = original_results['vial_masks_3d'][slice_idx, :, :]

        # Cropped data
        crop_phantom = cropped_results['clean_phantom'][0, slice_idx, :, :]
        crop_phantom_mask = cropped_results['phantom_masks_3d'][slice_idx, :, :]
        crop_vial_mask = cropped_results['vial_masks_3d'][slice_idx, :, :]

        # Plot original
        axes[0, 0].imshow(orig_phantom, cmap='gray')
        axes[0, 0].set_title(f'Original Phantom\n{orig_phantom.shape}')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(orig_phantom_mask, cmap='gray')
        axes[0, 1].set_title(f'Original Phantom Mask\n{orig_phantom_mask.shape}')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(orig_vial_mask, cmap='gray')
        axes[0, 2].set_title(f'Original Vial Mask\n{orig_vial_mask.shape}')
        axes[0, 2].axis('off')

        # Plot cropped
        axes[1, 0].imshow(crop_phantom, cmap='gray')
        axes[1, 0].set_title(f'Cropped Phantom\n{crop_phantom.shape}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(crop_phantom_mask, cmap='gray')
        axes[1, 1].set_title(f'Cropped Phantom Mask\n{crop_phantom_mask.shape}')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(crop_vial_mask, cmap='gray')
        axes[1, 2].set_title(f'Cropped Vial Mask\n{crop_vial_mask.shape}')
        axes[1, 2].axis('off')

        plt.suptitle(f'Original vs Cropped Comparison - Slice {slice_idx + 1}')
        plt.tight_layout()

        # Save visualization
        vis_path = os.path.join(self.cropped_phantom_path, f"{self.phantom_name}_cropping_comparison.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {vis_path}")
        plt.show()

    def process_complete_cropping(self, phantom_results, labels_results=None):
        """
        Complete cropping pipeline for all checkpoints.

        Args:
            phantom_results (dict): Results from PhantomSegmenter
            labels_results (dict, optional): Results from LabelsProcessor

        Returns:
            tuple: (cropped_phantom_results, cropped_labels_results)
        """
        # Crop phantom checkpoints
        cropped_phantom_results = self.crop_phantom_results(phantom_results)

        # Crop labels if available
        cropped_labels_results = None
        if labels_results:
            cropped_labels_results = self.crop_labels_results(
                labels_results,
                cropped_phantom_results['phantom_crop_info']
            )

        # Save all checkpoints
        self.save_cropped_results(cropped_phantom_results, cropped_labels_results)

        # Create visualization
        self.visualize_cropping_comparison(phantom_results, cropped_phantom_results)

        print("\n" + "=" * 60)
        print("CROPPING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return cropped_phantom_results, cropped_labels_results


def load_existing_phantom_results(config):
    """
    Load existing phantom segmentation checkpoints from saved files.

    Returns:
        dict: Phantom checkpoints dictionary compatible with process_phantom_complete output
    """
    phantom_name = config.get("phantom_name", "phantom")
    results_path = config.get("existing_results_path", config["output_folder"])

    print(f"\nLoading existing phantom checkpoints for {phantom_name}...")

    try:
        # Load phantom masks
        phantom_masks_path = os.path.join(results_path, f"{phantom_name}_phantom_masks.h5")
        with h5py.File(phantom_masks_path, 'r') as f:
            phantom_masks_3d = f['phantom_masks'][:]

        # Load vial masks
        vial_masks_path = os.path.join(results_path, f"{phantom_name}_vial_masks.h5")
        with h5py.File(vial_masks_path, 'r') as f:
            vial_masks_3d = f['vial_masks'][:]

        # Load vial images
        vial_images_path = os.path.join(results_path, f"{phantom_name}_vial_images.h5")
        with h5py.File(vial_images_path, 'r') as f:
            vial_masked_images_4d = f['vial_images'][:]

        # Load clean phantom
        clean_phantom_path = os.path.join(results_path, f"{phantom_name}_clean_phantom.h5")
        with h5py.File(clean_phantom_path, 'r') as f:
            clean_phantom = f['clean_phantom'][:]

        # Load metadata to get num_vials and wanted_slices
        metadata_path = os.path.join(results_path, f"{phantom_name}_metadata.txt")
        num_vials = 6  # default
        wanted_slices = None  # default

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                content = f.read()
                # Parse metadata
                for line in content.split('\n'):
                    if line.startswith('Number of Vials:'):
                        num_vials = int(line.split(':')[1].strip())
                    elif line.startswith('Wanted Slices:'):
                        wanted_slices_str = line.split(':', 1)[1].strip()
                        if wanted_slices_str != 'None':
                            # This is a simplified parser - you may need to adjust based on your slice format
                            wanted_slices = eval(wanted_slices_str) if wanted_slices_str != 'None' else None

        results = {
            'clean_phantom': clean_phantom,
            'phantom_masks_3d': phantom_masks_3d,
            'vial_masked_images_4d': vial_masked_images_4d,
            'vial_masks_3d': vial_masks_3d,
            'num_vials': num_vials,
            'wanted_slices': wanted_slices
        }

        print(f"Successfully loaded existing phantom checkpoints:")
        print(f"  Clean phantom shape: {clean_phantom.shape}")
        print(f"  Phantom masks shape: {phantom_masks_3d.shape}")
        print(f"  Vial masks shape: {vial_masks_3d.shape}")
        print(f"  Vial images shape: {vial_masked_images_4d.shape}")
        print(f"  Number of vials: {num_vials}")

        return results

    except Exception as e:
        print(f"Error loading existing phantom checkpoints: {str(e)}")
        raise


def load_existing_labels_results(config):
    """
    Load existing labels processing checkpoints from saved files.

    Returns:
        dict: Labels checkpoints dictionary compatible with LabelsProcessor output
    """
    phantom_name = config.get("phantom_name", "phantom")
    results_path = config.get("existing_results_path", config["output_folder"])

    print(f"\nLoading existing labels checkpoints for {phantom_name}...")

    try:
        # Load phantom-masked labels
        phantom_labels_path = os.path.join(results_path, f"{phantom_name}_phantom_masked_labels.h5")
        phantom_masked_labels = {}
        with h5py.File(phantom_labels_path, 'r') as f:
            phantom_masked_labels_4d = np.stack([f['AL_fB_Im'][:], f['AL_kBA_Im'][:]], axis=0)

        # Load vial-masked labels
        vial_labels_path = os.path.join(results_path, f"{phantom_name}_vial_masked_labels.h5")
        with h5py.File(vial_labels_path, 'r') as f:
            vial_masked_labels_4d = np.stack([f['AL_fB_Im'][:], f['AL_kBA_Im'][:]], axis=0)

        # Load original labels
        original_labels_path = os.path.join(results_path, f"{phantom_name}_original_labels.h5")
        with h5py.File(original_labels_path, 'r') as f:
            original_labels = np.stack([f['AL_fB_Im'][:], f['AL_kBA_Im'][:]], axis=0)

        results = {
            'phantom_masked_labels': phantom_masked_labels_4d,
            'vial_masked_labels': vial_masked_labels_4d,
            'original_labels': original_labels
        }

        print(f"Successfully loaded existing labels checkpoints:")
        print(f"  Phantom masked labels shape: {phantom_masked_labels_4d.shape}")
        print(f"  Vial masked labels shape: {vial_masked_labels_4d.shape}")
        print(f"  Original labels shape: {original_labels.shape}")

        return results

    except Exception as e:
        print(f"Error loading existing labels checkpoints: {str(e)}")
        raise


def load_existing_cropped_results(config):
    """
    Load existing cropped checkpoints from saved files.

    Returns:
        tuple: (cropped_phantom_results, cropped_labels_results)
    """
    phantom_name = config.get("phantom_name", "phantom")
    results_path = config.get("existing_results_path", config["output_folder"])
    cropped_path = os.path.join(results_path, "cropped")
    cropped_labels_path = os.path.join(results_path, "cropped_labels")

    print(f"\nLoading existing cropped checkpoints for {phantom_name}...")

    try:
        # Load cropped phantom checkpoints
        cropped_phantom_results = {}

        # Load clean phantom
        clean_phantom_path = os.path.join(cropped_path, f"{phantom_name}_clean_phantom_cropped.h5")
        with h5py.File(clean_phantom_path, 'r') as f:
            cropped_phantom_results['clean_phantom'] = f['clean_phantom'][:]

        # Load phantom masks
        phantom_masks_path = os.path.join(cropped_path, f"{phantom_name}_phantom_masks_cropped.h5")
        with h5py.File(phantom_masks_path, 'r') as f:
            cropped_phantom_results['phantom_masks_3d'] = f['phantom_masks'][:]

        # Load vial masks
        vial_masks_path = os.path.join(cropped_path, f"{phantom_name}_vial_masks_cropped.h5")
        with h5py.File(vial_masks_path, 'r') as f:
            cropped_phantom_results['vial_masks_3d'] = f['vial_masks'][:]

        # Load vial images
        vial_images_path = os.path.join(cropped_path, f"{phantom_name}_vial_images_cropped.h5")
        with h5py.File(vial_images_path, 'r') as f:
            cropped_phantom_results['vial_masked_images_4d'] = f['vial_images'][:]

        # Initialize default values
        cropped_phantom_results['num_vials'] = 6  # default
        cropped_phantom_results['wanted_slices'] = None  # default

        # Load metadata if available - try cropped metadata first
        metadata_path = os.path.join(cropped_path, f"{phantom_name}_cropped_metadata.txt")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                content = f.read()
                # Parse metadata
                for line in content.split('\n'):
                    if line.startswith('Number of Vials:'):
                        try:
                            cropped_phantom_results['num_vials'] = int(line.split(':')[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('Wanted Slices:') and 'wanted_slices' not in cropped_phantom_results:
                        try:
                            wanted_slices_str = line.split(':', 1)[1].strip()
                            if wanted_slices_str != 'None':
                                cropped_phantom_results['wanted_slices'] = eval(wanted_slices_str)
                        except:
                            pass
        else:
            # Fallback: try to load from original metadata
            original_metadata_path = os.path.join(results_path, f"{phantom_name}_metadata.txt")
            if os.path.exists(original_metadata_path):
                with open(original_metadata_path, 'r') as f:
                    content = f.read()
                    # Parse metadata
                    for line in content.split('\n'):
                        if line.startswith('Number of Vials:'):
                            try:
                                cropped_phantom_results['num_vials'] = int(line.split(':')[1].strip())
                            except (ValueError, IndexError):
                                pass
                        elif line.startswith('Wanted Slices:'):
                            try:
                                wanted_slices_str = line.split(':', 1)[1].strip()
                                if wanted_slices_str != 'None':
                                    cropped_phantom_results['wanted_slices'] = eval(wanted_slices_str)
                            except:
                                pass

        # Load cropped labels if available
        cropped_labels_results = None
        if os.path.exists(cropped_labels_path):
            cropped_labels_results = {}

            # Try to load cropped labels files
            for label_type in ['phantom_masked_labels', 'vial_masked_labels', 'original_labels']:
                try:
                    # Load AL_fB_Im
                    fb_path = os.path.join(cropped_labels_path, f"{phantom_name}_{label_type}_AL_fB_Im_cropped.h5")
                    with h5py.File(fb_path, 'r') as f:
                        al_fb_data = f['AL_fB_Im'][:]

                    # Load AL_kBA_Im
                    kba_path = os.path.join(cropped_labels_path, f"{phantom_name}_{label_type}_AL_kBA_Im_cropped.h5")
                    with h5py.File(kba_path, 'r') as f:
                        al_kba_data = f['AL_kBA_Im'][:]

                    # Combine into 4D array
                    cropped_labels_results[label_type] = np.stack([al_fb_data, al_kba_data], axis=0)

                except FileNotFoundError:
                    print(f"Warning: Could not load {label_type} - files not found")

        print(f"Successfully loaded existing cropped checkpoints:")
        print(f"  Cropped phantom shape: {cropped_phantom_results['clean_phantom'].shape}")
        print(f"  Cropped phantom masks shape: {cropped_phantom_results['phantom_masks_3d'].shape}")
        print(f"  Cropped vial masks shape: {cropped_phantom_results['vial_masks_3d'].shape}")
        print(f"  Cropped vial images shape: {cropped_phantom_results['vial_masked_images_4d'].shape}")
        print(f"  Number of vials: {cropped_phantom_results['num_vials']}")

        return cropped_phantom_results, cropped_labels_results

    except Exception as e:
        print(f"Error loading existing cropped checkpoints: {str(e)}")
        raise
# Preprocess and create model1
class Model1_Dataset():
    def __init__(self,config):
        self.config = config
        self.output_folder = config["output_folder"]
        self.phantom_name = config["phantom_name"]

        self.dataset_path = os.path.join(self.output_folder, "model1")
        os.makedirs(self.dataset_path, exist_ok=True)
        if self.config["need_augmentation"]:
            self.angles = [0, 45, 90, 135, 180, 225, 270, 315]
        else:
            self.angles = [0]

        self.window_size = config.get("window_size", 6)
        self.prediction_offset = config.get("prediction_offset", 6)
        self.embedding_window = config.get("embedding_window", 12)
        self.max_samples = config.get("max_training_samples", None)

        self.view_transforms = {
            "axial": (0, 1, 2, 3),  # No transpose
            "coronal": (0, 2, 1, 3),
            "sagittal": (0, 3, 1, 2),
            # Add custom transforms as needed
        }

    def create_data_for_training_and_validation(self,images, out_dir, embedding, slice_num=0):
        out_dir_data = os.path.join(out_dir, "data")
        out_dir_emb = os.path.join(out_dir, "params")
        out_dir_label = os.path.join(out_dir, "labels")
        os.makedirs(out_dir_data, exist_ok=True)
        os.makedirs(out_dir_emb, exist_ok=True)
        os.makedirs(out_dir_label, exist_ok=True)
        index_to_prediction = self.prediction_offset + self.window_size

        for i in range(19):
            data = images[i:i + self.window_size,:, :]
            embedding_pos = embedding[:, i:i + self.embedding_window ]
            label = images[i + self.prediction_offset:i + index_to_prediction,:, :]

            out_data = os.path.join(out_dir_data, f"image_{i}.h5")
            out_emb = os.path.join(out_dir_emb, f"image_{i}.h5")
            out_label = os.path.join(out_dir_label, f"image_{i}.h5")

            with h5py.File(out_data, 'w') as f:
                f.create_dataset('res', data=data, compression="gzip", compression_opts=9)
            with h5py.File(out_emb, 'w') as f:
                f.create_dataset('res', data=embedding_pos, compression="gzip", compression_opts=9)
            with h5py.File(out_label, 'w') as f:
                f.create_dataset('res', data=label, compression="gzip", compression_opts=9)

    def _rotate_image(self,image, angle):
        """Apply rotation to an image with the specified angle"""
        return rotate(image, angle, reshape=False, mode='nearest', order=1)

    def _augment_dataset_with_rotations(self,image, output_dir, param_met,
                                       angles, work_slice=0):
        """Augment the entire dataset using 8-fold rotation, with consistent rotation across protocol slices"""
        # Apply the SAME rotation angle to ALL images in this protocol
        for angle_idx, angle in enumerate(angles):
            out_dir = os.path.join(output_dir, f"angle_{angle}")
            os.makedirs(out_dir, exist_ok=True)

            new_images = np.zeros(image.shape)
            c,H,W = image.shape
            for i in range(c):
                    new_images[i, :, :] = self._rotate_image(image[i, :, :], angle)

            self.create_data_for_training_and_validation(new_images, out_dir, embedding=param_met, slice_num=work_slice)


    def process_data_for_model_1(self, phantom_4D):
        views = self.config["views"]

        for view in views:
            view_output_dir = os.path.join(self.dataset_path, self.phantom_name, view)
            os.makedirs(view_output_dir, exist_ok=True)
            transformed_data = np.transpose(phantom_4D, self.view_transforms[view])

            # normalize the phantom:
            transformed_data = (transformed_data - transformed_data.min()) / (transformed_data.max() - transformed_data.min())

            print(f" image shape for phantom: {self.config['phantom_name']} is {transformed_data.shape} for view {view}")

            ch,s,h,w = transformed_data.shape

            for slice_idx in range(s):
                slice_data = transformed_data[:, slice_idx, :, :]
                slice_output_dir = os.path.join(view_output_dir, str(slice_idx))

                if np.sum(slice_data) > 0:  # Skip empty slices
                    self._augment_dataset_with_rotations(
                        slice_data, slice_output_dir,
                        param_met=self.config['parameter_map'],
                        angles=self.angles, work_slice=slice_idx
                    )



class SegmentedLabelsGenerator:
    """
    A class to generate segmented labels (parameter maps) for model2 training.
    Creates pH, mM, T1, T2, and optionally ksw and fs maps from vial segmentations.
    """

    def __init__(self, config):
        """
        Initialize the SegmentedLabelsGenerator.

        Args:
            config (dict): Configuration dictionary from the main pipeline
        """
        self.config = config
        self.output_folder = config["output_folder"]
        self.phantom_name = config.get("phantom_name", "phantom")

        # Create output directory for parameter maps
        self.maps_output_dir = os.path.join(self.output_folder, "labels_maps")
        os.makedirs(self.maps_output_dir, exist_ok=True)

        # Get default values from config
        self.default_values = config.get("default_parameter_values", {})

        # Parameter values (will be set by user input)
        self.ph_values = None
        self.mM_values = None
        self.T1_values = None
        self.T2_values = None
        self.ksw_values = None
        self.fs_values = None
        self.num_vials = None

    def get_user_parameter_input(self, num_vials):
        """
        Get parameter values from user input or use defaults from config.

        Args:
            num_vials (int): Number of vials to get parameters for

        Returns:
            bool: True if successful, False otherwise
        """
        self.num_vials = num_vials

        print(f"\n{'=' * 60}")
        print("PARAMETER INPUT FOR SEGMENTED LABELS GENERATION")
        print(f"{'=' * 60}")
        print(f"Number of vials detected: {num_vials}")

        # Check if we can offer default values
        if num_vials == 6 and self.default_values:
            print("\nParameter input options:")
            print("1. Use default parameter values from config")
            print("2. Enter custom parameter values manually")
            print("\nDefault values from config:")
            print(f"  pH: {self.default_values.get('ph_values', [])}")
            print(f"  mM: {self.default_values.get('mM_values', [])}")
            print(f"  T1: {self.default_values.get('T1_values', [])}")
            print(f"  T2: {self.default_values.get('T2_values', [])}")
            print(f"  ksw: {self.default_values.get('ksw_values', [])}")
            print(f"  fs: {self.default_values.get('fs_values', [])}")

            while True:
                choice = input("\nEnter your choice (1 for defaults, 2 for manual): ").strip()
                if choice == "1":
                    return self._use_default_values()
                elif choice == "2":
                    return self._get_manual_input(num_vials)
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            if num_vials != 6:
                print(f"\nDefault values are configured for 6 vials.")
                print(f"Since you have {num_vials} vials, please enter values manually.")
            else:
                print(f"\nNo default values found in config.")
                print(f"Please enter values manually.")
            return self._get_manual_input(num_vials)

    def _use_default_values(self):
        """Use default parameter values from config."""
        self.ph_values = self.default_values.get('ph_values', []).copy()
        self.mM_values = self.default_values.get('mM_values', []).copy()
        self.T1_values = self.default_values.get('T1_values', []).copy()
        self.T2_values = self.default_values.get('T2_values', []).copy()

        # Check if all required default values are available
        if not all([self.ph_values, self.mM_values, self.T1_values, self.T2_values]):
            print("Error: Some default values are missing in config. Switching to manual input.")
            return self._get_manual_input(self.num_vials)

        # Ask if user wants to include ksw and fs
        default_ksw = self.default_values.get('ksw_values', [])
        default_fs = self.default_values.get('fs_values', [])

        print(f"\n{'=' * 40}")
        print("OPTIONAL PARAMETERS")
        print(f"{'=' * 40}")

        if default_ksw and default_fs:
            while True:
                include_extra = input("Do you want to include ksw and fs parameters? (y/n): ").strip().lower()
                if include_extra in ['y', 'yes']:
                    self.ksw_values = default_ksw.copy()
                    self.fs_values = default_fs.copy()
                    break
                elif include_extra in ['n', 'no']:
                    self.ksw_values = None
                    self.fs_values = None
                    break
                else:
                    print("Please enter 'y' or 'n'.")
        else:
            print("Default ksw and fs values not found in config. Skipping optional parameters.")
            self.ksw_values = None
            self.fs_values = None

        # Display final values
        self._display_final_values()
        return True

    def _get_manual_input(self, num_vials):
        """Get manual input for parameter values."""
        print(f"\nPlease enter parameter values for {num_vials} vials:")
        print("Enter values separated by commas (e.g., 5.0,5.5,6.0,4.0,5.2,5.8)")

        # Get pH values
        while True:
            try:
                ph_input = input(f"Enter pH values for {num_vials} vials: ").strip()
                self.ph_values = [float(x.strip()) for x in ph_input.split(',')]
                if len(self.ph_values) != num_vials:
                    print(f"Error: Expected {num_vials} values, got {len(self.ph_values)}. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numeric values separated by commas.")

        # Get mM values
        while True:
            try:
                mM_input = input(f"Enter mM values for {num_vials} vials: ").strip()
                self.mM_values = [float(x.strip()) for x in mM_input.split(',')]
                if len(self.mM_values) != num_vials:
                    print(f"Error: Expected {num_vials} values, got {len(self.mM_values)}. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numeric values separated by commas.")

        # Get T1 values
        while True:
            try:
                T1_input = input(f"Enter T1 values for {num_vials} vials: ").strip()
                self.T1_values = [float(x.strip()) for x in T1_input.split(',')]
                if len(self.T1_values) != num_vials:
                    print(f"Error: Expected {num_vials} values, got {len(self.T1_values)}. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numeric values separated by commas.")

        # Get T2 values
        while True:
            try:
                T2_input = input(f"Enter T2 values for {num_vials} vials: ").strip()
                self.T2_values = [float(x.strip()) for x in T2_input.split(',')]
                if len(self.T2_values) != num_vials:
                    print(f"Error: Expected {num_vials} values, got {len(self.T2_values)}. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numeric values separated by commas.")

        # Ask if user wants to enter ksw and fs values
        while True:
            add_extra = input("Do you want to enter ksw and fs values? (y/n): ").strip().lower()
            if add_extra in ['y', 'yes']:
                # Get ksw values
                while True:
                    try:
                        ksw_input = input(f"Enter ksw values for {num_vials} vials: ").strip()
                        self.ksw_values = [float(x.strip()) for x in ksw_input.split(',')]
                        if len(self.ksw_values) != num_vials:
                            print(f"Error: Expected {num_vials} values, got {len(self.ksw_values)}. Please try again.")
                            continue
                        break
                    except ValueError:
                        print("Error: Please enter valid numeric values separated by commas.")

                # Get fs values
                while True:
                    try:
                        fs_input = input(f"Enter fs values for {num_vials} vials: ").strip()
                        self.fs_values = [float(x.strip()) for x in fs_input.split(',')]
                        if len(self.fs_values) != num_vials:
                            print(f"Error: Expected {num_vials} values, got {len(self.fs_values)}. Please try again.")
                            continue
                        break
                    except ValueError:
                        print("Error: Please enter valid numeric values separated by commas.")
                break
            elif add_extra in ['n', 'no']:
                self.ksw_values = None
                self.fs_values = None
                break
            else:
                print("Please enter 'y' or 'n'.")

        # Display final values and get confirmation
        self._display_final_values()

        # Ask for confirmation
        while True:
            confirm = input("Are these values correct? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                print("Please re-enter the parameter values.")
                return self._get_manual_input(num_vials)
            else:
                print("Please enter 'y' or 'n'.")

    def _display_final_values(self):
        """Display the final parameter values for confirmation."""
        print(f"\n{'=' * 40}")
        print("FINAL PARAMETER VALUES:")
        print(f"{'=' * 40}")
        print(f"pH values: {self.ph_values}")
        print(f"mM values: {self.mM_values}")
        print(f"T1 values: {self.T1_values}")
        print(f"T2 values: {self.T2_values}")
        if self.ksw_values:
            print(f"ksw values: {self.ksw_values}")
        if self.fs_values:
            print(f"fs values: {self.fs_values}")
        print(f"{'=' * 40}")

    def find_tubes(self, data, sam, mask_generator, image=None, device=None, num_tubes=6):
        """
        Find tubes in a 2D slice and sort them, avoiding x,y coordinate overlaps.

        Args:
            data: 2D mask data
            sam: SAM model
            mask_generator: SAM mask generator
            image: Original image for visualization
            device: Device for computation
            num_tubes: Number of tubes to find (default: 6)

        Returns:
            List of sorted tube masks with no overlaps
        """
        tubes = []
        if data.ndim == 2:
            img_rgb = np.stack((data,) * 3, axis=-1)
        else:
            img_rgb = data
        img_rgb = img_rgb.astype(np.uint8)

        # Generate segmentation masks using SAM
        masks = mask_generator.generate(img_rgb)

        # Set target number of tubes based on condition
        target_tubes = num_tubes

        # Filter tubes based on area
        for mask in masks:
            area = np.sum(mask['segmentation'])
            if self.config["vial_area_threshold_low"] < area < self.config["vial_area_threshold_high"]:
                # Calculate the y-coordinate of the center of the mask
                y_indices = np.where(mask['segmentation'])[0]
                if len(y_indices) > 0:
                    center_y = np.mean(y_indices)
                    # Also calculate x-coordinate for identification
                    x_indices = np.where(mask['segmentation'])[1]
                    center_x = np.mean(x_indices)
                    # Store the mask along with its center coordinates and area
                    tubes.append({
                        'mask': mask,
                        'center_y': center_y,
                        'center_x': center_x,
                        'segmentation': mask['segmentation'],
                        'area': area
                    })

        # Sort tubes by y-coordinate (top to bottom)
        tubes.sort(key=lambda x: x['center_y'])

        # Check for overlaps and remove them
        non_overlapping_tubes = []
        overlap_threshold_x = 3
        overlap_threshold_y = 3

        for tube in tubes:
            # Check if this tube overlaps with any already selected tube
            is_overlapping = False
            replace_index = -1

            for i, selected_tube in enumerate(non_overlapping_tubes):
                dx = abs(tube['center_x'] - selected_tube['center_x'])
                dy = abs(tube['center_y'] - selected_tube['center_y'])

                if dx < overlap_threshold_x and dy < overlap_threshold_y:
                    is_overlapping = True
                    # If current tube has larger area than the overlapping one, mark for replacement
                    if tube['area'] > selected_tube['area']:
                        replace_index = i
                    break

            # Handle the tube based on overlap check
            if not is_overlapping:
                # No overlap, simply add the tube
                non_overlapping_tubes.append(tube)
            elif replace_index >= 0:
                # Overlap exists but current tube has larger area, replace the existing one
                non_overlapping_tubes[replace_index] = tube

            # If we have enough tubes, stop processing
            if len(non_overlapping_tubes) >= target_tubes:
                break

        # Ensure we have exactly the target number of tubes if possible
        non_overlapping_tubes.sort(key=lambda x: x['center_y'])
        final_tubes = non_overlapping_tubes[:target_tubes]

        return final_tubes

    def process_all_slices(self, mask_3d_data, original_data, sam, mask_generator, device=None):
        """
        Process all slices of a 3D mask and find tubes in each slice.
        Create 3D parameter maps.

        Args:
            mask_3d_data: 3D mask data
            original_data: Original data for visualization
            sam: SAM model
            mask_generator: SAM mask generator
            device: Device for computation (optional)

        Returns:
            dict: Dictionary containing all parameter maps
        """
        print(f"\n{'=' * 60}")
        print("GENERATING SEGMENTED PARAMETER MAPS")
        print(f"{'=' * 60}")

        # Initialize 3D maps
        pH_3D_map = np.zeros(mask_3d_data.shape)
        mM_3D_map = np.zeros(mask_3d_data.shape)
        T1_3D_map = np.zeros(mask_3d_data.shape)
        T2_3D_map = np.zeros(mask_3d_data.shape)
        ksw_3D_map = np.zeros(mask_3d_data.shape)
        fs_3D_map = np.zeros(mask_3d_data.shape)

        print(f"Processing {mask_3d_data.shape[0]} slices...")

        # Process each slice
        for slice_idx in tqdm(range(mask_3d_data.shape[0]), desc="Creating parameter maps"):
            slice_data = mask_3d_data[slice_idx, :, :]
            original_slice = original_data[0, slice_idx, :, :] if original_data.ndim == 4 else original_data[
                slice_idx, :, :]

            slice_tubes = self.find_tubes(
                slice_data, sam, mask_generator,
                original_slice, device,
                num_tubes=self.num_vials
            )

            # Apply parameter values to tubes
            for tube_idx, tube in enumerate(slice_tubes):
                if tube_idx < len(self.ph_values):  # Safety check
                    current_tube = tube['segmentation']
                    pH_3D_map[slice_idx, :, :] += current_tube * self.ph_values[tube_idx]
                    mM_3D_map[slice_idx, :, :] += current_tube * self.mM_values[tube_idx]
                    T1_3D_map[slice_idx, :, :] += current_tube * self.T1_values[tube_idx]
                    T2_3D_map[slice_idx, :, :] += current_tube * self.T2_values[tube_idx]

                    if self.ksw_values is not None:
                        ksw_3D_map[slice_idx, :, :] += current_tube * self.ksw_values[tube_idx]
                    if self.fs_values is not None:
                        fs_3D_map[slice_idx, :, :] += current_tube * self.fs_values[tube_idx]
                else:
                    print(
                        f"Warning: Found more tubes ({len(slice_tubes)}) than parameter values ({len(self.ph_values)}) in slice {slice_idx}")

        # Save the 3D maps
        parameter_maps = self._save_parameter_maps(pH_3D_map, mM_3D_map, T1_3D_map, T2_3D_map, fs_3D_map, ksw_3D_map)

        return parameter_maps

    def _save_parameter_maps(self, pH_3D_map, mM_3D_map, T1_3D_map, T2_3D_map,fs_3D_map, ksw_3D_map ):
        """
        Save all parameter maps to HDF5 files.

        Args:
            pH_3D_map: pH parameter map
            mM_3D_map: mM parameter map
            T1_3D_map: T1 parameter map
            T2_3D_map: T2 parameter map
            ksw_3D_map: ksw parameter map
            fs_3D_map: fs parameter map

        Returns:
            dict: Dictionary containing all parameter maps
        """
        # Define output paths
        maps = {
            'pH': pH_3D_map,
            'mM': mM_3D_map,
            'T1': T1_3D_map,
            'T2': T2_3D_map
        }

        if self.ksw_values is not None:
            maps['ksw'] = ksw_3D_map
        if self.fs_values is not None:
            maps['fs'] = fs_3D_map

        print(f"\nSaving parameter maps to: {self.maps_output_dir}")

        # Save each map
        for map_name, map_data in maps.items():
            if map_data is not None:
                output_path = os.path.join(self.maps_output_dir, f"{self.phantom_name}_{map_name}_map.h5")
                with h5py.File(output_path, 'w') as f:
                    f.create_dataset('mask', data=map_data, compression="gzip", compression_opts=9)
                print(f"Saved {map_name} map to: {output_path}")

        return {
            'pH': pH_3D_map,
            'mM': mM_3D_map,
            'T1': T1_3D_map,
            'T2': T2_3D_map,
            'ksw': ksw_3D_map if self.ksw_values else None,
            'fs': fs_3D_map if self.fs_values else None
        }

    def visualize_parameter_maps(self, parameter_maps, slice_idx=0, save_path=None):
        """
        Visualize parameter maps for a specific slice.

        Args:
            parameter_maps (dict): Dictionary containing parameter maps
            slice_idx (int): Slice index to visualize
            save_path (str, optional): Path to save the visualization
        """
        # Count available maps
        available_maps = [(name, data) for name, data in parameter_maps.items()
                          if data is not None and name.endswith('_map')]

        if not available_maps:
            print("No parameter maps available for visualization.")
            return

        # Create subplot layout
        n_maps = len(available_maps)
        cols = min(3, n_maps)
        rows = (n_maps + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_maps == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if n_maps > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, (map_name, map_data) in enumerate(available_maps):
            param_name = map_name.replace('_3D_map', '')
            slice_data = map_data[slice_idx, :, :]

            im = axes[i].imshow(slice_data, cmap='viridis')
            axes[i].set_title(f'{param_name.upper()} Map - Slice {slice_idx + 1}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        # Hide unused subplots
        for i in range(n_maps, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Parameter Maps - {self.phantom_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter maps visualization saved to: {save_path}")

        plt.show()

    def generate_segmented_labels(self, phantom_results, sam, mask_generator):
        """
        Main function to generate segmented labels for model2.

        Args:
            phantom_results (dict): Results from PhantomSegmenter
            sam: SAM model from PhantomSegmenter
            mask_generator: SAM mask generator from PhantomSegmenter

        Returns:
            dict: Parameter maps checkpoints
        """
        print(f"\n{'=' * 60}")
        print("STARTING SEGMENTED LABELS GENERATION FOR MODEL2")
        print(f"{'=' * 60}")

        # Get user input for parameter values
        num_vials = phantom_results['num_vials']
        success = self.get_user_parameter_input(num_vials)

        if not success:
            print("Failed to get parameter values. Aborting segmented labels generation.")
            return None

        # Get the vial masks and original data
        vial_masks_3d = phantom_results['vial_masks_3d']
        clean_phantom = phantom_results['clean_phantom']

        print(f"Vial masks shape: {vial_masks_3d.shape}")
        print(f"Clean phantom shape: {clean_phantom.shape}")

        # Process all slices to create parameter maps
        parameter_maps = self.process_all_slices(
            vial_masks_3d,
            clean_phantom,
            sam,
            mask_generator
        )

        # Create visualization
        visualization_path = os.path.join(self.maps_output_dir, f"{self.phantom_name}_parameter_maps_visualization.png")
        self.visualize_parameter_maps(parameter_maps, slice_idx=0, save_path=visualization_path)

        print(f"\n{'=' * 60}")
        print("SEGMENTED LABELS GENERATION COMPLETED!")
        print(f"{'=' * 60}")

        return parameter_maps


#
class Model2_Dataset:
    """
    A class to prepare data for Model 2 training with different parameter combinations.
    Model 2 uses parameter maps (pH, mM, T1, T2, ksw, fs) as labels instead of temporal prediction.
    """

    def __init__(self, config):
        self.config = config
        self.output_folder = config["output_folder"]
        self.phantom_name = config["phantom_name"]
        self.params_maps = config["parameter_map"]


        # Dataset output path
        self.dataset_path = os.path.join(self.output_folder, "model2")
        os.makedirs(self.dataset_path, exist_ok=True)

        # Augmentation settings
        if self.config["need_augmentation"]:
            self.angles = [0, 45, 90, 135, 180, 225, 270, 315]
        else:
            self.angles = [0]

        # Window parameters (for input data consistency with Model 1)
        self.window_size = config.get("window_size", 6)
        self.embedding_window = config.get("embedding_window", 12)

        # View transforms
        self.view_transforms = {
            "axial": (0, 1, 2, 3),  # No transpose
            "coronal": (0, 2, 1, 3),
            "sagittal": (0, 3, 1, 2),
        }

    def create_model2_data_for_training(self, images, labels, out_dir, slice_num=0):
        """
        Create Model 2 training data with phantom images as input and parameter maps as labels.

        Args:
            images: Input phantom images [scans, H, W]
            labels: Lbels_paramaps [2,H,W]
            out_dir: Output directory
            slice_num: Current slice number
        """
        out_dir_data = os.path.join(out_dir, "data")
        out_dir_param = os.path.join(out_dir, "params")
        out_dir_label = os.path.join(out_dir, "labels")
        os.makedirs(out_dir_data, exist_ok=True)
        os.makedirs(out_dir_param, exist_ok=True)
        os.makedirs(out_dir_label, exist_ok=True)


        for i in range(19):
            data = images[i:i + self.window_size,:, :]
            embedding_pos = self.params_maps[:, i:i + self.embedding_window ]

            out_data = os.path.join(out_dir_data, f"image_{i}.h5")
            out_emb = os.path.join(out_dir_param, f"image_{i}.h5")
            out_label = os.path.join(out_dir_label, f"image_{i}.h5")

            with h5py.File(out_data, 'w') as f:
                f.create_dataset('res', data=data, compression="gzip", compression_opts=9)
            with h5py.File(out_emb, 'w') as f:
                f.create_dataset('res', data=embedding_pos, compression="gzip", compression_opts=9)
            with h5py.File(out_label, 'w') as f:
                f.create_dataset('res', data=labels, compression="gzip", compression_opts=9)



    def _augment_model2_dataset_with_rotations(self, images, labels, output_dir, angles, slice_idx):
        """
        Augment Model 2 dataset with rotations, applying the same rotation to both input and labels.

        Args:
            images: Input phantom images [6, H, W]
            labels: Labels images [2,80,80]
            output_dir: Output directory
            angles: List of rotation angles
            slice_idx: Current slice index
        """
        for angle_idx, angle in enumerate(angles):
            angle_dir = os.path.join(output_dir, f"angle_{angle}")
            os.makedirs(angle_dir, exist_ok=True)

            # Rotate input images
            rotated_images = np.zeros(images.shape)
            rotated_labels = np.zeros(labels.shape)
            num_scans, H, W = images.shape
            for scan_idx in range(num_scans):
                rotated_images[scan_idx, :, :] = rotate(
                    images[scan_idx, :, :], angle, reshape=False, mode='nearest', order=1
                )
            for scan_idx in range(labels.shape[0]):
                rotated_labels[scan_idx, :, :] = rotate(
                    labels[scan_idx, :, :], angle, reshape=False, mode='nearest', order=1
                )

            # Create training data
            self.create_model2_data_for_training(
                rotated_images, rotated_labels, angle_dir, slice_num=slice_idx
            )


    def process_data_for_model_2(self, phantom_4D, labels_maps, folder_name):
        """
        Process phantom data for Model 2 training.

        Args:
            phantom_4D: 4D phantom data [scans, slices, H, W]
            labels_maps: 4D labels data [parameter, slices, H, W]
            folder_name: str for the folder name to save the data
        """
        views = self.config["views"]

        for view in views:
            print(f"\nProcessing view: {view}")
            view_output_dir = os.path.join(self.dataset_path, self.phantom_name, folder_name, view)
            os.makedirs(view_output_dir, exist_ok=True)

            # Apply view transformation
            transformed_data = np.transpose(phantom_4D, self.view_transforms[view])
            transformed_labels = np.transpose(labels_maps, self.view_transforms[view])

            # Normalize the phantom data
            transformed_data = (transformed_data - transformed_data.min()) / (
                    transformed_data.max() - transformed_data.min() + 1e-8
            )

            print(f"Transformed data shape for {self.config['phantom_name']}: {transformed_data.shape} for view {view}")
            print(f"Transformed labels shape for {self.config['phantom_name']}: {transformed_labels.shape} for view {view}")

            num_scans, num_slices, height, width = transformed_data.shape

            # Process each slice
            for slice_idx in range(num_slices):
                slice_data = transformed_data[:, slice_idx, :, :]  # [scans, H, W]
                labels_slice = transformed_labels[:, slice_idx, :, :]
                slice_output_dir = os.path.join(view_output_dir, str(slice_idx))

                self._augment_model2_dataset_with_rotations(
                    slice_data, labels_slice, slice_output_dir, self.angles, slice_idx
                )
