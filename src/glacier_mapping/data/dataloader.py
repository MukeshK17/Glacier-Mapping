import os
import re

import albumentations as a
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split


# Utility: sort patches correctly
def extract_row_col(filename):
    match = re.search(r"r(\d+)_c(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)

# Dataset
class MultiBandSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir,  transform=None, top_bands=None, index_bands=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.top_bands = top_bands if top_bands is not None else list(range(1, 19))
        self.index_bands = index_bands if index_bands is not None else []
        self.transform = transform

        # Sort by row/column
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

        # Read 18-band image
        with rasterio.open(img_path) as src:
            image = src.read(self.top_bands)  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)

        # Read mask
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

# Albumentations Transform
def get_transform():
    return a.Compose([
        a.RandomRotate90(p=0.5),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5),
        a.Normalize(mean=[0.5] * 18, std=[0.5] * 18),
        ToTensorV2(),
    ])


# Helper: sample subset
def sample_subset(dataset, fraction):
    if fraction >= 1.0:
        return dataset
    n = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), n, replace=False)
    return Subset(dataset, indices)

# Create train/val loaders
def create_dataloaders(
    dataset_configs,
    batch_size=16,
    val_ratio=0.3,
    num_workers=4,
    transform=None,
    seed=42,
):
    """
    dataset_configs: dict like:
    {
        "himachal": (img_dir, mask_dir, fraction),
        "sikkim": (img_dir, mask_dir, fraction),
    }
    """

    np.random.seed(seed)

    datasets = []

    for _name, (img_dir, mask_dir, fraction) in dataset_configs.items():
        ds = MultiBandSegmentationDataset(
            img_dir,
            mask_dir,
            transform=transform,
        )
        ds = sample_subset(ds, fraction)
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)

    val_size = int(len(combined_dataset) * val_ratio)
    train_size = len(combined_dataset) - val_size

    train_dataset, val_dataset = random_split(
        combined_dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

# Test loader (ordered)
def get_test_loader(image_dir, mask_dir, batch_size=8, num_workers=4):
    dataset = MultiBandSegmentationDataset(
        image_dir,
        mask_dir,
        transform=None,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
