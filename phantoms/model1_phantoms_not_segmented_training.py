import os
#
# os.environ["NCCL_P2P_DISABLE"] = "1"
# import os

# NCCL configuration for shared memory issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"  # Disable shared memory transport
os.environ["NCCL_DEBUG"] = "INFO"     # For debugging (remove after fixing)
os.environ["NCCL_IB_DISABLE"] = "1"   # Disable InfiniBand if not available
os.environ["NCCL_NET_GDR_LEVEL"] = "0"  # Disable GPU Direct RDMA

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group
import torch.distributed as dist

from tqdm import tqdm
from torch import nn
import argparse
import torch
import math
import glob
import h5py

#import scipy.io as sio

# class ssim_l1_loss(nn.Module):
#     def __init__(self, ssim_weight, weight_l1, device):
#         super(ssim_l1_loss, self).__init__()
#         self.weight_ssim = ssim_weight
#         self.weight_l1 = weight_l1
#         self.device = device
#
#     def forward(self, prediction, target):
#         loss1 = 1 - StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)  # SSIM Loss
#         loss2 = nn.L1Loss()
#         return self.weight_ssim * loss1(prediction, target) + self.weight_l1 * loss2(prediction, target)

class ssim_l1_loss(nn.Module):
    def __init__(self, ssim_weight, weight_l1, device):
        super(ssim_l1_loss, self).__init__()
        self.weight_ssim = ssim_weight
        self.weight_l1 = weight_l1
        self.device = device

        # Initialize loss functions ONCE during __init__
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction, target):
        # Use the pre-initialized loss functions
        ssim_value = self.ssim_loss(prediction, target)
        l1_value = self.l1_loss(prediction, target)
        return self.weight_ssim * (1 - ssim_value) + self.weight_l1 * l1_value


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
            nn.SyncBatchNorm(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Up-sample to target spatial dimensions (126, 126)
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),

            # Convolutional layer to get to 4 channels
            nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = x.permute(0, 3, 1, 2)
        # Get some dimensions from x
        batch_size = x.shape[0]

        # Create the patch embedding
        x = self.patch_embedding(x)

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


# class TryDataset_v2(Dataset):
#
#     def __init__(self, data_dir, param_dir, labels_dir, scale_data, scale_param):
#         self.data_paths = data_dir
#         self.param_paths = param_dir
#         self.label_paths = labels_dir
#
#         self.scale_data = scale_data
#         self.scale_params = scale_param
#
#     def __len__(self):
#         return len(self.data_paths)
#
#     def __getitem__(self, index: int):
#         dataset_idx = sio.loadmat(self.data_paths[index])['res'] / self.scale_data
#         params_idx = sio.loadmat(self.param_paths[index])['res'] / self.scale_params
#         labels_idx = sio.loadmat(self.label_paths[index])['res'] / self.scale_data
#
#         return dataset_idx.astype('float32'), params_idx.astype('float32'), labels_idx.astype('float32')


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


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            loss_fn: torch.nn.Module,
            lr: float
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss_fn = loss_fn
        self.lr = lr

        # Training parameters
        self.best_val_loss = 10000
        self.best_epoch = -1
        self.count_lr_reduced = 0
        self.early_stopping_epoch = 10
        self.lr_step_gamma = 0.8
        self.lr_update_counts = 30

    def _run_batch(self, X, p, y):
        self.optimizer.zero_grad()
        y_pred = self.model(X, p)
        loss = self.loss_fn(y_pred.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch):
        if self.gpu_id == 0:
            print(f"Starting epoch {epoch + 1}")
        self.model.train()
        self.train_data.sampler.set_epoch(epoch)  # For shuffle!!
        train_loss = torch.zeros(1).to(self.gpu_id)

        if self.gpu_id == 0:
            print(f"About to start training loop with {len(self.train_data)} batches")
        for batch, (X, p, y) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            X, p, y = X.to(self.gpu_id), p.to(self.gpu_id), y.to(self.gpu_id)

            loss_batch = self._run_batch(X, p, y)
            if self.gpu_id == 0 and batch == 0:
                print(f"First batch loss: {loss_batch}")
            train_loss += loss_batch

            torch.cuda.empty_cache()
            del X, p, y
        if self.gpu_id == 0:
            print(f"Finished training loop, about to all_reduce")
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        return train_loss

    def validate(self):
        self.model.eval()
        test_loss = torch.zeros(1).to(self.gpu_id)
        with torch.inference_mode():
            for batch, (X, p, y) in tqdm(enumerate(self.val_data), total=len(self.val_data)):
                X = X.to(self.gpu_id)
                p = p.to(self.gpu_id)
                y = y.to(self.gpu_id)

                test_pred_logits = self.model(X, p)
                loss = self.loss_fn(test_pred_logits.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))
                test_loss += loss.item()

                torch.cuda.empty_cache()
                del X, p, y
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        return test_loss

    def _save_checkpoint(self, file_name):
        ckp = self.model.module.state_dict()
        # Change this line - add the leading slash!
        checkpoint_dir = '/storage/sahar/model1_not_segmented_results'

        os.makedirs(checkpoint_dir, exist_ok=True)
        full_path = os.path.join(checkpoint_dir, file_name)
        print(f"Saving checkpoint to: {full_path}")
        torch.save(ckp, full_path)
        print(f"Checkpoint saved successfully!")

    def train(self, max_epochs: int):

        for epoch in range(max_epochs):
            # train step
            train_loss = self._run_epoch(epoch)

            # Validation step
            # dist.barrier()
            val_loss = self.validate()

            if self.gpu_id == 0:
                print(f"Finished Epoch: {epoch + 1}")
                total_train_loss = train_loss / (5 * len(self.train_data))  # according to the number of gpus
                total_val_loss = val_loss / (5 * len(self.val_data))

                print(
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: {total_train_loss.item():.4f} | "
                    f"test_loss: {total_val_loss.item():.4f} | "
                )

                if total_val_loss < self.best_val_loss:
                    self.best_epoch = epoch
                    self.best_val_loss = total_val_loss.item()
                    print(f"Best epoch:{self.best_epoch + 1}")
                    print(f"Best val loss:{total_val_loss}")
                    checkpoint_name = f'checkpoint_epoch_{self.best_epoch}.pt'

                    self._save_checkpoint(checkpoint_name)
                    print("Checkpoint has been saved")

                elif epoch - self.best_epoch > self.early_stopping_epoch:
                    self.count_lr_reduced += 1
                    self.lr = self.lr * self.lr_step_gamma

                    for g in self.optimizer.param_groups:
                        g['lr'] = self.lr

                    print(f"Updated the learning rate at epoch: {epoch} to {self.lr}")
                    if self.count_lr_reduced == self.lr_update_counts:
                        print("Early stopped training at epoch %d" % epoch)
                        return True


def main(args):
    torch.distributed.init_process_group('nccl')
    rank = args.local_rank
    # Get rank from environment variable (set by torch.distributed.launch)
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Set the device for this process
    torch.cuda.set_device(rank)
    total_epochs = 5000
    batch_size = 16

    num_gpus = args.num_gpus
    image_size = 80
    patch_size = 5
    sequence_len = 6
    embedding_dim = 768
    mlp_size = 3072
    num_heads = 4
    num_transformer_layers = 3
    dropout = 0
    # learning_rate = 0.0004 * math.sqrt(5)
    learning_rate = 0.0004 * math.sqrt(5)
    ssim_weight = 0.2
    weight_l1 = 0.8

    # Creating the dataset
    val_split = 0.2
    scale_max_data = 1
    scale_max_params = 4

    root_directory = '/storage/sahar/model1_train_not_segmented'  # change to the correct data path

    data_paths = glob.glob(os.path.join(root_directory, "*/*/*/data/*.h5"))
    param_paths = glob.glob(os.path.join(root_directory, "*/*/*/embedding/*.h5"))
    label_paths = glob.glob(os.path.join(root_directory, "*/*/*/labels/*.h5"))

    print(f"found {len(data_paths)} data files, {len(param_paths)} params, {len(label_paths)} labels")

    dataset = TryDataset_v3(data_dir=data_paths,
                            param_dir=param_paths,
                            labels_dir=label_paths,
                            scale_data=scale_max_data,
                            scale_param=scale_max_params)

    model = Model(img_size=image_size,
                  num_channels=sequence_len,
                  patch_size=patch_size,
                  embedding_dim=embedding_dim,
                  dropout=dropout,
                  mlp_size=mlp_size,
                  num_transformer_layers=num_transformer_layers,
                  num_heads=num_heads)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Split the datasets into train and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=32,
                              persistent_workers=True,
                              sampler=DistributedSampler(train_dataset, drop_last=True)
                              )

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=32,
                            persistent_workers=True,
                            pin_memory=True,
                            sampler=DistributedSampler(val_dataset, drop_last=True))

    loss_fn = ssim_l1_loss(ssim_weight=ssim_weight, weight_l1=weight_l1, device=rank)
    trainer = Trainer(model, train_loader, val_loader, optimizer, rank, loss_fn, learning_rate)
    # Add this debugging section before trainer.train()
    if rank == 0:
        print("Testing a single forward pass...")
        model.eval()
        with torch.no_grad():
            # Get a single batch
            sample_batch = next(iter(train_loader))
            X, p, y = sample_batch
            X, p, y = X.to(rank), p.to(rank), y.to(rank)

            print(f"Input shapes: X={X.shape}, p={p.shape}, y={y.shape}")

            # Test forward pass
            try:
                output = model(X, p)
                print(f"Forward pass successful! Output shape: {output.shape}")

                # Test loss computation
                loss = loss_fn(output.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))
                print(f"Loss computation successful! Loss: {loss.item()}")

            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()

        print("Starting actual training...")

    # Ensure all processes wait
    dist.barrier()
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", default=0, type=int, help='')
    parser.add_argument("--batch_size", default=64, type=int, help='')
    parser.add_argument("--num_epoch", default=1000, type=int, help='')
    parser.add_argument("--num_gpus", default=5, type=int, help='')
    parser.add_argument("--learning_rate_one_gpu", default=0.0004, type=float, help='')
    parser.add_argument("--image_size", default=144, type=int, help='')
    parser.add_argument("--sequence_len", default=6, type=int, help='')
    parser.add_argument("--patch_size", default=9, type=int, help='')

    args = parser.parse_args()
    main(args)
