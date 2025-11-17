from torch.utils.data import Dataset, DataLoader
from torch import nn
import scipy as sc
import numpy as np
import torch
import glob
import os


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

        # 2. Create positional embedding
        num_patches = (img_size * img_size) // patch_size ** 2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embedding_dim))  # +1 for the class

        # 3. Create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        # 4. Create stack Transformer Encoder layers (stacked single layers)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                     nhead=num_heads,
                                                     dim_feedforward=mlp_size,
                                                     activation="gelu",
                                                     batch_first=True,
                                                     norm_first=True),
            num_layers=num_transformer_layers,
            enable_nested_tensor=False
            )  # Stack it N times

        # 5. Align the parameter matrix according to the embedding dimension
        self.linear_layer = nn.Linear(in_features=num_channels * 2,
                                      out_features=embedding_dim)

        # 6. Create the convolutional layers after the transformers - add batch normalization

        self.conv_layers = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Up-sample to target spatial dimensions (126, 126)
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),

            # Convolutional layer to get to 4 channels
            nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 10
        )

    def forward(self, x, y):
        # x = x.permute(0, 3, 1, 2)

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

        return x.permute(0, 2, 3, 1)  # (n, 144, 144, k)


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
        if 'bay4_volunteer3_mar_7_2022' in self.data_paths[index] or 'cancer' in self.data_paths[index]:
            dataset_idx = sc.io.loadmat(self.data_paths[index])['res']
        else:
            dataset_idx = sc.io.loadmat(self.data_paths[index])['res'] / self.scale_data

        params_idx = sc.io.loadmat(self.param_paths[index])['res'] / self.scale_params
        labels_idx = sc.io.loadmat(self.label_paths[index])['res']

        labels_idx[0, :, :] = labels_idx[0, :, :] / 100  # KSSW
        labels_idx[1, :, :] = labels_idx[1, :, :] / 27.27  # MT_perc
        labels_idx[2, :, :] = (labels_idx[2, :, :] + 1) / (1.7 + 1)  # B0
        labels_idx[3, :, :] = labels_idx[3, :, :] / 3.4944  # B1
        labels_idx[4, :, :] = labels_idx[4, :, :] / 10000  # T1

        if ('bay1_volunteer1' in self.data_paths[index]
                or 'bay4_06_22_2020_volunteer1' in self.data_paths[index]
                or 'bay4_volunteer2_2020_07_17' in self.data_paths[index]
                or 'erlangen_trio_healthy_volunteer' in self.data_paths[index]):
            labels_idx[5, :, :] = labels_idx[5, :, :] * 1

        else:
            labels_idx[5, :, :] = labels_idx[5, :, :] / 1000

        labels_idx[np.isnan(labels_idx)] = 0
        dataset_idx[np.isnan(dataset_idx)] = 0

        return (torch.from_numpy(dataset_idx.astype(np.float32)),
                torch.from_numpy(params_idx.astype(np.float32)),
                torch.from_numpy(labels_idx.astype(np.float32)))


class Model1_dataset(Dataset):
    def __init__(self, x, p, y, scale_data, scale_params, vol_name):
        self.x = x
        self.p = p
        self.y = y

        self.scale_data = scale_data
        self.scale_params = scale_params

        self.vol_name = vol_name

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if 'cancer' in self.vol_name:
            data_idx = self.x[index] * 1
            labels_idx = self.y[index] * 1

        else:
            data_idx = self.x[index] / self.scale_data
            labels_idx = self.y[index] / self.scale_data

        params_idx = self.p[index] / self.scale_params

        return data_idx, params_idx, labels_idx


class Model2_dataset(Dataset):
    def __init__(self, x, p, y, scale_data, scale_params, vol_name):
        self.x = x
        self.p = p
        self.y = y

        self.scale_data = scale_data
        self.scale_params = scale_params

        self.vol_name = vol_name

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.vol_name == 'cancer':
            data_idx = self.x[index] * 1

        else:
            data_idx = self.x[index] / self.scale_data

        params_idx = self.p[index] / self.scale_params
        labels_idx = self.y[index]

        labels_idx[0, :, :] = labels_idx[0, :, :] / 100  # KSSW
        labels_idx[1, :, :] = labels_idx[1, :, :] / 27.27  # MT_perc
        labels_idx[2, :, :] = (labels_idx[2, :, :] + 1) / (1.7 + 1)  # B0
        labels_idx[3, :, :] = labels_idx[3, :, :] / 3.4944  # B1
        labels_idx[4, :, :] = labels_idx[4, :, :] / 10000  # T1

        if self.vol_name == 'erlangen_trio_healthy_volunteer':
            labels_idx[5, :, :] = labels_idx[5, :, :] * 1

        else:
            labels_idx[5, :, :] = labels_idx[5, :, :] / 1000

        labels_idx[torch.isnan(labels_idx)] = 0
        data_idx[torch.isnan(data_idx)] = 0

        return data_idx, params_idx, labels_idx


# Model 2
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


def create_dataloader_per_volunteer(vol_path):
    root_directory = 'model2_test_volunteers'
    data_paths = sorted(glob.glob(os.path.join(root_directory, vol_path, "dataset/*.mat")),
                        key=os.path.getmtime)
    param_paths = sorted(glob.glob(os.path.join(root_directory, vol_path, "params/*.mat")),
                         key=os.path.getmtime)
    label_paths = sorted(glob.glob(os.path.join(root_directory, vol_path, "labels/*.mat")),
                         key=os.path.getmtime)

    test_dataset = TryDataset_v0(data_dir=data_paths,
                                 param_dir=param_paths,
                                 labels_dir=label_paths,
                                 scale_data=4578.9688,
                                 scale_param=13.9984)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)

    return test_loader



