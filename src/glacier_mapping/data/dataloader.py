import json
import os
import re

import albumentations as a
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

# 18 Standard Himalayan Glaciological Bands
BAND_NAMES = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR",
    "Elevation/DEM",
    "Coherence",
    "dh/dt",
    "Divergence",
    "Persistent Scatterer",
    "Profile Curvature",
    "Slope Angle",
    "Slope Aspect",
    "Tangential Curvature",
    "Thermal Infrared",
    "Unsphericity",
    "Velocity",
    "Backscatter",
]


def extract_row_col(filename: str) -> tuple[int, int]:
    """Extract row and column coordinates from patch filename (e.g., 'patch_r012_c005.tif')."""
    match = re.search(r"r(\d+)_c(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (0, 0)


class MultiBandSegmentationDataset(Dataset):
    """
    Dataset for multi-modal (up to 18-band) satellite imagery and binary glacier masks.
    Handles NaN/inf replacement and optional band filtering.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        top_bands: list[int] | None = None,
        band_means: list[float] | None = None,
        band_stds: list[float] | None = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.top_bands = top_bands if top_bands is not None else list(range(1, 19))
        self.transform = transform
        self.band_means = np.array(band_means, dtype=np.float32) if band_means is not None else None
        self.band_stds = np.array(band_stds, dtype=np.float32) if band_stds is not None else None

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            self.image_files = []
            self.mask_files = []
            return

        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.startswith("patch_") and f.endswith(".tif")],
            key=extract_row_col,
        )
        self.mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.startswith("patch_") and f.endswith(".tif")],
            key=extract_row_col,
        )

        assert len(self.image_files) == len(self.mask_files), (
            f"Image ({len(self.image_files)}) and mask ({len(self.mask_files)}) count mismatch in {image_dir}"
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Read multi-band imagery: (C, H, W) -> (H, W, C)
        with rasterio.open(img_path) as src:
            image = src.read(self.top_bands)
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            image = np.transpose(image, (1, 2, 0)).astype(np.float32)

        # Apply per-channel z-score if available and not handled in albumentations
        if self.band_means is not None and self.band_stds is not None and self.transform is None:
            image = (image - self.band_means) / (self.band_stds + 1e-6)

        # Read binary mask: (H, W)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask = np.nan_to_num(mask, nan=0.0).astype(np.float32)
            mask = (mask > 0.5).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def compute_dataset_band_stats(
    image_dirs: list[str],
    sample_limit: int = 400,
    save_path: str | None = None,
) -> tuple[list[float], list[float]]:
    """
    Computes per-band mean and standard deviation across raw multi-spectral patches.
    Saves to a JSON cache if save_path is provided.
    """
    all_files = []
    for d in image_dirs:
        if os.path.exists(d):
            patches = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("patch_") and f.endswith(".tif")]
            all_files.extend(patches)

    if not all_files:
        # Fallback to neutral defaults if data is not yet downloaded
        return [0.0] * 18, [1.0] * 18

    # Randomly subsample to compute stats fast
    np.random.seed(42)
    selected_files = np.random.choice(all_files, min(len(all_files), sample_limit), replace=False)

    band_sums = np.zeros(18, dtype=np.float64)
    band_sq_sums = np.zeros(18, dtype=np.float64)
    total_pixels = 0

    for path in selected_files:
        with rasterio.open(path) as src:
            img = src.read()  # (18, H, W)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            pixels_per_band = img.shape[1] * img.shape[2]
            band_sums += img.sum(axis=(1, 2))
            band_sq_sums += (img**2).sum(axis=(1, 2))
            total_pixels += pixels_per_band

    means = (band_sums / total_pixels).tolist()
    stds = (np.sqrt(np.maximum(band_sq_sums / total_pixels - np.array(means) ** 2, 1e-6))).tolist()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"means": means, "stds": stds, "bands": BAND_NAMES}, f, indent=2)

    return means, stds


def get_transform(
    band_means: list[float] | None = None,
    band_stds: list[float] | None = None,
    is_train: bool = True,
):
    """
    Returns an Albumentations transform with physically grounded per-band normalization.
    """
    transforms = []
    if is_train:
        transforms.extend([
            a.RandomRotate90(p=0.5),
            a.HorizontalFlip(p=0.5),
            a.VerticalFlip(p=0.5),
        ])

    if band_means is not None and band_stds is not None:
        transforms.append(a.Normalize(mean=band_means, std=band_stds, max_pixel_value=1.0))
    else:
        # Safe fallback standard scaling
        transforms.append(a.Normalize(mean=[0.0] * 18, std=[1.0] * 18, max_pixel_value=1.0))

    transforms.append(ToTensorV2())
    return a.Compose(transforms)


def create_spatial_block_split(
    dataset: MultiBandSegmentationDataset,
    val_ratio: float = 0.2,
) -> tuple[Subset, Subset]:
    """
    Spatial block split: groups patches by spatial row/col blocks
    rather than random sampling, preventing spatial autocorrelation leakage.
    """
    coords = [extract_row_col(f) for f in dataset.image_files]
    if not coords:
        return Subset(dataset, []), Subset(dataset, [])

    rows = np.array([c[0] for c in coords])

    # Block split based on high coordinate region (e.g. top quadrant)
    max_row = rows.max() if len(rows) > 0 else 0
    row_threshold = max_row * (1.0 - val_ratio)

    val_indices = np.where(rows >= row_threshold)[0]
    train_indices = np.where(rows < row_threshold)[0]

    # Fallback if partition is degenerate
    if len(val_indices) == 0 or len(train_indices) == 0:
        n_val = int(len(dataset) * val_ratio)
        indices = np.arange(len(dataset))
        val_indices = indices[-n_val:]
        train_indices = indices[:-n_val]

    return Subset(dataset, train_indices.tolist()), Subset(dataset, val_indices.tolist())


def create_loro_loaders(
    region_paths: dict[str, tuple[str, str]],
    test_region: str,
    batch_size: int = 16,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    band_means: list[float] | None = None,
    band_stds: list[float] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Leave-One-Region-Out (LORO) Dataloaders:
    Strictly reserves one region (e.g. 'sikkim') for zero-shot testing.
    Combines the remaining 4 regions and performs spatial block validation split.
    """
    train_datasets = []
    val_datasets = []

    train_transform = get_transform(band_means, band_stds, is_train=True)
    val_transform = get_transform(band_means, band_stds, is_train=False)

    test_region_key = test_region.lower()
    test_dataset = None

    for name, (img_dir, mask_dir) in region_paths.items():
        if name.lower() == test_region_key:
            test_dataset = MultiBandSegmentationDataset(
                img_dir,
                mask_dir,
                transform=val_transform,
            )
        else:
            ds = MultiBandSegmentationDataset(
                img_dir,
                mask_dir,
                transform=None,  # transform assigned to subsets
            )
            train_sub, val_sub = create_spatial_block_split(ds, val_ratio=val_ratio)

            # Apply separate train/val transforms
            train_sub_ds = MultiBandSegmentationDataset(
                img_dir, mask_dir, transform=train_transform
            )
            val_sub_ds = MultiBandSegmentationDataset(
                img_dir, mask_dir, transform=val_transform
            )

            train_datasets.append(Subset(train_sub_ds, train_sub.indices))
            val_datasets.append(Subset(val_sub_ds, val_sub.indices))

    if test_dataset is None:
        raise ValueError(f"Test region '{test_region}' not found in region_paths keys: {list(region_paths.keys())}")

    combined_train = ConcatDataset(train_datasets) if train_datasets else Subset(test_dataset, [])
    combined_val = ConcatDataset(val_datasets) if val_datasets else Subset(test_dataset, [])

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# Backward-compatible helper functions
def sample_subset(dataset: Dataset, fraction: float) -> Dataset:
    if fraction >= 1.0:
        return dataset
    n = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), n, replace=False)
    return Subset(dataset, indices)


def create_dataloaders(
    dataset_configs: dict,
    batch_size: int = 16,
    val_ratio: float = 0.3,
    num_workers: int = 4,
    transform=None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    np.random.seed(seed)
    datasets = []

    for _name, (img_dir, mask_dir, fraction, *_) in dataset_configs.items():
        ds = MultiBandSegmentationDataset(
            img_dir,
            mask_dir,
            transform=transform if transform else get_transform(is_train=True),
        )
        ds = sample_subset(ds, fraction)
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    val_size = int(len(combined_dataset) * val_ratio)
    train_size = len(combined_dataset) - val_size

    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_test_loader(
    image_dir: str,
    mask_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    transform=None,
) -> DataLoader:
    dataset = MultiBandSegmentationDataset(
        image_dir,
        mask_dir,
        transform=transform if transform else get_transform(is_train=False),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
