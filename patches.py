import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResampleEnum
from rasterio.transform import Affine
import numpy as np
import os

def reproject_mask_to_raster(mask_path, raster_path):
    """Reproject mask to exactly match raster’s CRS, shape, and transform."""
    with rasterio.open(mask_path) as mask_src, rasterio.open(raster_path) as raster_src:
        dst_array = np.zeros((1, raster_src.height, raster_src.width), dtype=mask_src.dtypes[0])

        reproject(
            source=rasterio.band(mask_src, 1),
            destination=dst_array,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=raster_src.transform,
            dst_crs=raster_src.crs,
            resampling=ResampleEnum.nearest
        )

        return dst_array, raster_src.transform, mask_src.profile.copy()

def extract_aligned_patches(raster_path, mask_path, output_dir, patch_size=128):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    with rasterio.open(raster_path) as raster_src:
        raster_data = raster_src.read()
        transform = raster_src.transform
        profile = raster_src.profile
        height, width = raster_src.height, raster_src.width

        # Step 1: Reproject the mask
        mask_data, _, mask_profile = reproject_mask_to_raster(mask_path, raster_path)

        # Step 2: Pad to make divisible by patch size
        pad_h = (patch_size - height % patch_size) % patch_size
        pad_w = (patch_size - width % patch_size) % patch_size

        raster_data = np.pad(raster_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        mask_data = np.pad(mask_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')

        new_height, new_width = raster_data.shape[1:]

        patch_count = 0
        for i in range(0, new_height, patch_size):
            for j in range(0, new_width, patch_size):
                row = i // patch_size
                col = j // patch_size
                patch_id = f"patch_r{row:03d}_c{col:03d}"

                img_patch = raster_data[:, i:i + patch_size, j:j + patch_size]
                mask_patch = mask_data[:, i:i + patch_size, j:j + patch_size]

                patch_transform = transform * Affine.translation(j, i)

                # Image patch save
                img_profile = profile.copy()
                img_profile.update({
                    'height': patch_size,
                    'width': patch_size,
                    'transform': patch_transform
                })

                img_path = os.path.join(output_dir, 'images', f'{patch_id}.tif')
                with rasterio.open(img_path, 'w', **img_profile) as dst:
                    dst.write(img_patch)

                # Mask patch save
                mask_profile_copy = img_profile.copy()
                mask_profile_copy.update({
                    'count': 1,
                    'dtype': mask_patch.dtype
                })

                mask_path_out = os.path.join(output_dir, 'masks', f'{patch_id}.tif')
                with rasterio.open(mask_path_out, 'w', **mask_profile_copy) as dst:
                    dst.write(mask_patch)

                patch_count += 1

        print(f" {patch_count} patches saved with correct geolocation alignment.")

# Example usage
raster_path = ""
mask_path = ""
output_dir = ""
extract_aligned_patches(raster_path, mask_path, output_dir, patch_size=128)
