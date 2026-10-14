import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_2d_gaussian_window(tile_size: int = 128, sigma_scale: float = 0.25) -> np.ndarray:
    """
    Generates a 2D Gaussian blend window for smooth overlap stitching.
    Center pixels have weight ~1.0; border pixels smoothly taper off.
    """
    center = (tile_size - 1) / 2.0
    sigma = tile_size * sigma_scale
    y, x = np.ogrid[:tile_size, :tile_size]
    gaussian_2d = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    gaussian_2d = np.maximum(gaussian_2d, 1e-4)
    return gaussian_2d.astype(np.float32)


def predict_full_scene(
    model: torch.nn.Module,
    input_raster_path: str,
    output_raster_path: str,
    tile_size: int = 128,
    overlap: float = 0.5,
    batch_size: int = 16,
    device: torch.device | None = None,
    threshold: float = 0.5,
    band_means: list[float] | None = None,
    band_stds: list[float] | None = None,
) -> str:
    """
    Seamless Sliding-Window Inference for arbitrary-sized multi-band satellite rasters.
    Uses 2D Gaussian window blending to eliminate tile boundary artifacts.
    Preserves original GeoTIFF georeferencing, CRS, and transform.
    """
    dev = device or DEVICE
    model = model.to(dev)
    model.eval()

    stride = int(tile_size * (1.0 - overlap))
    stride = max(stride, 1)

    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        raster = src.read()  # (C, H, W)

    raster = np.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Apply per-band normalization if provided
    if band_means is not None and band_stds is not None:
        means = np.array(band_means, dtype=np.float32)[:, None, None]
        stds = np.array(band_stds, dtype=np.float32)[:, None, None]
        raster = (raster - means) / (stds + 1e-6)

    # Pad raster to ensure complete coverage by sliding window
    pad_h = (tile_size - (height - tile_size) % stride) % stride if height > tile_size else (tile_size - height)
    pad_w = (tile_size - (width - tile_size) % stride) % stride if width > tile_size else (tile_size - width)

    padded_raster = np.pad(
        raster,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )
    padded_h, padded_w = padded_raster.shape[1], padded_raster.shape[2]

    prob_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
    window = generate_2d_gaussian_window(tile_size)

    # Collect coordinates of all tiles
    tile_coords = []
    for y in range(0, padded_h - tile_size + 1, stride):
        for x in range(0, padded_w - tile_size + 1, stride):
            tile_coords.append((y, x))

    # Process in batches to balance speed and GPU memory
    with torch.no_grad():
        for i in range(0, len(tile_coords), batch_size):
            batch_coords = tile_coords[i : i + batch_size]
            batch_tiles = [
                padded_raster[:, y : y + tile_size, x : x + tile_size]
                for (y, x) in batch_coords
            ]

            batch_tensor = torch.tensor(np.stack(batch_tiles), dtype=torch.float32, device=dev)
            logits = model(batch_tensor)
            if hasattr(logits, "logits"):
                logits = logits.logits
            if logits.shape[2:] != (tile_size, tile_size):
                logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for idx, (y, x) in enumerate(batch_coords):
                prob_canvas[y : y + tile_size, x : x + tile_size] += probs[idx] * window
                weight_canvas[y : y + tile_size, x : x + tile_size] += window

    # Normalize by overlapping window weights
    weight_canvas = np.maximum(weight_canvas, 1e-6)
    final_prob = (prob_canvas / weight_canvas)[:height, :width]
    binary_mask = (final_prob > threshold).astype(np.uint8)

    # Save output GeoTIFF with original CRS and transform
    profile.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": None,
        "compress": "deflate",
    })

    os.makedirs(os.path.dirname(output_raster_path) or ".", exist_ok=True)
    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(binary_mask, 1)

    print(f"Full-scene segmentation saved to: {output_raster_path} ({width}x{height})")
    return output_raster_path
