from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import scipy.io as sio
from torch import nn
import numpy as np
import argparse
import torch
import glob
import os
import h5py
import matplotlib.pyplot as plt


class ssim_huber_loss(nn.Module):
    def __init__(self, ssim_weight, weight_huber, device):
        super(ssim_huber_loss, self).__init__()
        self.weight_ssim = ssim_weight
        self.weight_huber = weight_huber
        self.device = device

    def forward(self, prediction, target):
        loss1 = 1 - StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)  # SSIM Loss
        loss2 = nn.L1Loss()
        return self.weight_ssim * loss1(prediction, target) + self.weight_huber * loss2(prediction, target)


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, (f"Input image size must be divisible by patch size,"
                                                         f" image shape: {image_resolution},"
                                                         f" patch size: {self.patch_size}")

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self,
                 img_size=224,  # from Table 3
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768,  # from Table 1
                 dropout=0.5,
                 mlp_size=3072,  # from Table 1 3072
                 num_transformer_layers=12,  # from Table 1 12
                 num_heads=12):  # 12
        super().__init__()

        # Assert image size is divisible by patch size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."

        # 1. Create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 2. Create class token
        # self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),requires_grad=True)

        # 3. Create positional embedding
        num_patches = (img_size * img_size) // patch_size ** 2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))  # +1 for the class

        # 4. Create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        # 5. Create stack Transformer Encoder layers (stacked single layers)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                  nhead=num_heads,
                                                                                                  dim_feedforward=mlp_size,
                                                                                                  activation="gelu",
                                                                                                  batch_first=True,
                                                                                                  norm_first=True),
                                                         num_layers=num_transformer_layers)  # Stack it N times

        # 6. Align the parameter matrix according to the embedding dimension
        self.linear_layer = nn.Linear(in_features=num_channels * 2,
                                      out_features=embedding_dim)

        # 7. Create the convolutional layers after the transformers - add batch normalization

        self.conv_layers = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7

            # Up-sample to target spatial dimensions (126, 126)
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),  # 8

            # Convolutional layer to get to 4 channels
            nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 10
        )

    def forward(self, x, y):
        # x = x.permute(0, 3, 1, 2)
        # Get some dimensions from x
        batch_size = x.shape[0]

        # Create the patch embedding
        x = self.patch_embedding(x)
        # print(x.shape)

        # First, expand the class token across the batch size
        # class_token = self.class_token.expand(batch_size, -1, -1)  # "-1" means infer the dimension

        # Prepend the class token to the patch embedding
        # x = torch.cat((class_token, x), dim=1)
        # Add the positional embedding to patch embedding with class token
        x = self.positional_embedding + x
        # Dropout on patch + positional embedding
        x = self.embedding_dropout(x)
        # Creating the parameters embedding and concatenating to the data embedding

        y = self.linear_layer(y)
        x = torch.cat((x, y), dim=1)
        # Pass embedding through Transformer Encoder stack
        x = self.transformer_encoder(x).unsqueeze(dim=1)

        x = self.conv_layers(x)
        return x.permute(0, 2, 3, 1)


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
        dataset_idx = h5py.File(self.data_paths[index])['res'][:] / self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels_idx = h5py.File(self.label_paths[index])['res'][:]

        labels_idx[0, :, :] = (labels_idx[0, :, :]+0.00034) / 0.004   # fs 110,000 /3 for display
        labels_idx[1, :, :] = (labels_idx[1, :, :]+12) / (2317 + 12) # ksw

        labels_idx[np.isnan(labels_idx)] = 0
        dataset_idx[np.isnan(dataset_idx)] = 0

        return dataset_idx.astype(np.float32), params_idx.astype(np.float32), labels_idx.astype(np.float32)


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def create_model_v0(args, weights_path):
    model = Model(img_size=args.image_size,
                  num_channels=6,
                  patch_size=args.patch_size,
                  embedding_dim=args.embedding_dim,
                  dropout=args.dropout,
                  mlp_size=args.mlp_size,
                  num_transformer_layers=args.num_transformer_layers,
                  num_heads=args.num_heads)

    model.load_state_dict(torch.load(weights_path))

    """
    for param in model.parameters():
        param.requires_grad = False
    """

    model.conv_layers[9] = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    model.conv_layers[10] = nn.BatchNorm2d(num_features=32)
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=16))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=8))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.Sigmoid())

    return model


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, p, y) in enumerate(dataloader):
            # Send data to target device
            X, p, y = X.to(device), p.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X, p)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits.permute(0, 3, 1, 2), y)
            test_loss += loss.item()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        # Loop through data loader data batches
        for batch, (X, p, y) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")
            # Send data to target device

            X, p, y = X.to(device), p.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X, p)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred.permute(0, 3, 1, 2), y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    return train_loss


def plot_validation_sample(model, val_dataloader, device, epoch, save_dir):
    """Plot 3 different validation samples from separate batches to visualize model performance"""
    model.eval()

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    with torch.inference_mode():
        # Create figure with subplots: 3 samples x 4 plots (target pH, pred pH, target mM, pred mM)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Validation Samples - Epoch {epoch}', fontsize=16)

        # Get 3 different batches (each with batch size 1)
        batch_iter = iter(val_dataloader)

        for sample_idx in range(3):
            try:
                # Get next batch (batch size = 1)
                X, p, y = next(batch_iter)
                X, p, y = X.to(device), p.to(device), y.to(device)

                # Get prediction for this single sample
                pred = model(X, p)

                # Convert to numpy (remove batch dimension since batch_size=1)
                target = y[0].cpu().numpy()  # Shape: (2, height, width)
                prediction = pred[0].cpu().numpy()  # Shape: (height, width, 2)

                # Transpose prediction to match target format
                prediction = prediction.transpose(2, 0, 1)  # Shape: (2, height, width)

                # Plot target pH
                im1 = axes[sample_idx, 0].imshow(target[0] * 0.003, cmap='plasma', vmin=0)
                axes[sample_idx, 0].set_title(f'Sample {sample_idx + 1} - Target pH')
                axes[sample_idx, 0].axis('off')
                plt.colorbar(im1, ax=axes[sample_idx, 0])

                # Plot predicted pH
                im2 = axes[sample_idx, 1].imshow(prediction[0] * 0.003, cmap='plasma', vmin=0)
                axes[sample_idx, 1].set_title(f'Sample {sample_idx + 1} - Predicted pH')
                axes[sample_idx, 1].axis('off')
                plt.colorbar(im2, ax=axes[sample_idx, 1])

                # Plot target mM
                im3 = axes[sample_idx, 2].imshow(target[1] * 2500, cmap='plasma', vmin=0)
                axes[sample_idx, 2].set_title(f'Sample {sample_idx + 1} - Target mM')
                axes[sample_idx, 2].axis('off')
                plt.colorbar(im3, ax=axes[sample_idx, 2])

                # Plot predicted mM
                im4 = axes[sample_idx, 3].imshow(prediction[1] * 2500, cmap='plasma', vmin=0)
                axes[sample_idx, 3].set_title(f'Sample {sample_idx + 1} - Predicted mM')
                axes[sample_idx, 3].axis('off')
                plt.colorbar(im4, ax=axes[sample_idx, 3])

            except StopIteration:
                # If we run out of batches, fill remaining plots with empty plots
                for col in range(4):
                    axes[sample_idx, col].axis('off')
                    axes[sample_idx, col].set_title(f'Sample {sample_idx + 1} - No data')
                print(f"Warning: Only {sample_idx} samples available for plotting")
                break

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(save_dir, f'validation_epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Validation plot with 3 different samples saved to: {save_path}")


def plot_loss_curves(results: Dict[str, List], save_dir: str, epoch: int = None):
    """
    Plot training and validation loss curves and save to specified directory.

    Args:
        results: Dictionary containing 'train_loss' and 'test_loss' lists
        save_dir: Directory to save the plot
        epoch: Current epoch number (optional, for filename)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    epochs_range = range(1, len(results["train_loss"]) + 1)

    plt.plot(epochs_range, results["train_loss"], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, results["test_loss"], 'r-', label='Validation Loss', linewidth=2)

    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add some styling
    plt.tight_layout()

    # Save the plot
    if epoch is not None:
        filename = f'loss_curves_epoch_{epoch:04d}.png'
    else:
        filename = 'loss_curves_final.png'

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Loss curves saved to: {save_path}")


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          lr,
          lr_step_gamma,
          early_stopping_epoch,
          save_checkpoint_dir,
          lr_update_counts,
          device: torch.device) -> Dict[str, List]:
    # Create empty checkpoints dictionary
    results = {"train_loss": [],
               "test_loss": []}

    # Make sure model on target device
    model.to(device)
    best_val_loss = 10000
    best_epoch = -1
    count_lr_reduced = 0
    plots_dir = os.path.join(save_checkpoint_dir, 'validation_plots')

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device,
                                epoch=epoch)

        test_loss = test_step(model=model,
                              dataloader=test_dataloader,
                              loss_fn=loss_fn,
                              device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # Update checkpoints dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        if (epoch) % 25 == 0:
            plot_validation_sample(model, test_dataloader, device, epoch + 1, plots_dir)

        if test_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = test_loss
            print(f"Best epoch:{best_epoch}")
            print(f"Best val loss:{best_val_loss}")
            checkpoint_name = f'checkpoint_epoch_{best_epoch}.pt'
            save_path = os.path.join(save_checkpoint_dir, checkpoint_name)
            checkpoint(model, save_path)

        elif epoch - best_epoch > early_stopping_epoch:
            count_lr_reduced += 1
            lr = lr * lr_step_gamma

            # Updating the learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr

            print(f"Updated the learning rate at epoch: {epoch} to {lr}")
            if count_lr_reduced == lr_update_counts:
                print("Early stopped training at epoch %d" % epoch)
                plot_loss_curves(results, save_checkpoint_dir)
                return results

    # Plot final loss curves
    plot_loss_curves(results, save_checkpoint_dir)


def main(args):
    # Pretrained weights

    # model_weights_path = '/storage/sahar/Results_segmented/checkpoint_epoch_846.pt'
    model_weights_path = '/storage/sahar/model1_results_80/checkpoint_epoch_220.pt'
    checkpoint_dir = '/storage/sahar/model2_results_fs_ksw'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data paths
    root_directory = '/storage/sahar/fs_ksw__model_data'
    data_paths = glob.glob(os.path.join(root_directory, "*/*/*/data/*.h5"))
    param_paths = glob.glob(os.path.join(root_directory, "*/*/*/embedding/*.h5"))
    label_paths = glob.glob(os.path.join(root_directory, "*/*/*/labels/*.h5"))

    dataset = TryDataset_v0(data_dir=data_paths,
                            param_dir=param_paths,
                            labels_dir=label_paths,
                            scale_data=1,
                            scale_param=4)

    print(len(dataset))
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=8
                              )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8
                            )

    # Load the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model_v0(args, weights_path=model_weights_path)
    print("Up until now, the model is created")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_fn = ssim_huber_loss(ssim_weight=0.27,
                              weight_huber=0.75,
                              device=device)  # Originally .6, 0.4



    results = train(model=model,
                    train_dataloader=train_loader,
                    test_dataloader=val_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=args.num_epoch,
                    lr=args.learning_rate,
                    lr_step_gamma=0.8,
                    early_stopping_epoch=2,
                    save_checkpoint_dir=checkpoint_dir,
                    lr_update_counts=30,
                    device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help='Batch size')
    parser.add_argument("--num_epoch", default=1000 , type=int, help='Number of epochs')
    parser.add_argument("--num_gpus", default=1, type=int, help='')
    parser.add_argument("--learning_rate", default=0.002, type=float, help='')  # 0.008
    parser.add_argument("--image_size", default=80, type=int, help='')
    parser.add_argument("--sequence_len", default=6, type=int, help='')
    parser.add_argument("--patch_size", default=5, type=int, help='')
    parser.add_argument("--embedding_dim", default=768, type=int, help='')
    parser.add_argument("--mlp_size", default=3072, type=int, help='')
    parser.add_argument("--num_heads", default=4, type=int, help='')  ###### 4
    parser.add_argument("--num_transformer_layers", default=3, type=int, help='')
    parser.add_argument("--dropout", default=0, type=float, help='')  # 0
    parser.add_argument("--ssim_weight", default=0.6, type=float, help='')
    parser.add_argument("--mae_weight", default=0.4, type=float, help='')
    parser.add_argument("--val_split", default=0.2, type=float, help='')
    parser.add_argument("--num_workers", default=32, type=int, help='Number of workers in the dataloader')

    arguments = parser.parse_args()
    main(arguments)
