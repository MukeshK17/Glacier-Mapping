import re
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
import rasterio
import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define paths to the rater and mask directories
himachal_raster = ""
himachal_mask = ""

himlad_raster = ""
himlad_mask = ""

sikkim_raster = ""
sikkim_mask = ""

kashmir_raster = ""
kashmir_mask = ""

uttrakhand_raster = ""
uttrakhand_mask = ""

# Function to extract row and column from filename
def extract_row_col(filename):
    match = re.search(r"r(\d+)_c(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)

class MultiBandSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir,  transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.top_bands = top_bands if top_bands else list(range(1,19))
        # self.index_bands = index_bands if index_bands else[]
        self.transform = transform

        # Sort by row/column instead of string order
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.startswith('patch_')],
            key=extract_row_col
        )
        self.mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.startswith('patch_')],
            key=extract_row_col
        )

        assert len(self.image_files) == len(self.mask_files), "Image and mask count mismatch"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Read 18-band image (bands 1 to 18 in rasterio are 1-indexed)
        with rasterio.open(img_path) as src:
            image = src.read(list(range(1, 19)))  # shape: (18, H, W)
            image = np.transpose(image, (1, 2, 0))  # to shape (H, W, C)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # shape: (H, W)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# === Albumentations Transform ===
def get_transform():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.5] * 18, std=[0.5] * 18),
        ToTensorV2(),
    ])


# top_bands = [5, 17, 7, 1, 4, 2]  # Example top bands, can be modified as needed(if required to train on specific bands)
# top_bands = [1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 17]
# index_bands = [3, 5, 7, 8, 15]

# Create datasets for training
# put transform=get_transform() if you want to apply transformations
ds_himachal = MultiBandSegmentationDataset(himachal_raster, himachal_mask,transform = None) 
ds_himlad = MultiBandSegmentationDataset(himlad_raster, himlad_mask,transform = None)
ds_sikkim = MultiBandSegmentationDataset(sikkim_raster, sikkim_mask,transform = None)
ds_kashmir = MultiBandSegmentationDataset(kashmir_raster, kashmir_mask,transform = None)
ds_uttrakhand = MultiBandSegmentationDataset(uttrakhand_raster, uttrakhand_mask,transform = None)


def sample_subset(dataset, fraction):
    n = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), n, replace=False)
    return Subset(dataset, indices)
ds_himachal_sub = sample_subset(ds_himachal, 1)
ds_himlad_sub = sample_subset(ds_himlad, 0.3)
ds_sikkim_sub = sample_subset(ds_sikkim, 0.3)
ds_kashmir_sub = sample_subset(ds_kashmir, 0.3)
ds_uttrakhand_sub = sample_subset(ds_uttrakhand, 0.3)

# Concatenate all datasets
combined_dataset = ConcatDataset([ds_himachal, ds_himlad_sub, ds_sikkim_sub, ds_kashmir_sub, ds_uttrakhand_sub])
# combined_dataset = ConcatDataset([ds_himachal, ds_himlad_sub, ds_uttrakhand_sub])
val_ratio = 0.3
val_size = int(len(combined_dataset) * val_ratio)
train_size = len(combined_dataset) - val_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

def get_dataloaders():
    return train_loader, val_loader

# Create dataset for evaluation/testing
def get_ordered_loader(image_dir, mask_dir, batch_size=32):
    dataset = MultiBandSegmentationDataset(image_dir, mask_dir,transform=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
# Create test loaders for each region
test_loader1 = get_ordered_loader(himachal_raster, himachal_mask, batch_size=8)
test_loader2 = get_ordered_loader(himlad_raster, himlad_mask, batch_size=8)
test_loader3 = get_ordered_loader(sikkim_raster, sikkim_mask, batch_size=8)
test_loader4 = get_ordered_loader(kashmir_raster, kashmir_mask, batch_size=8)
test_loader5 = get_ordered_loader(uttrakhand_raster, uttrakhand_mask, batch_size=8)

def get_test_loaders():
    return test_loader1, test_loader2, test_loader3, test_loader4, test_loader5

