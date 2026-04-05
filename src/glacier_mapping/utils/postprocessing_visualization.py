import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology

from glacier_mapping.main import get_test_output

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 12


def print_metrics(metrics, region_name):
    print(f"\nResults for {region_name}")
    print(
        f"Test Loss: {metrics['loss']:.4f}, IoU: {metrics['iou']:.4f}, "
        f"Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, F1 Score: {metrics['f1']:.4f}, "
        f"Kappa: {metrics['kappa']:.4f}"
    )


def print_confusion_matrix(cm):
    cm = cm.astype(int)
    print("Confusion Matrix:")
    print("       Pred 0   Pred 1")
    print(f"True 0  {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"True 1  {cm[1][0]:5d}   {cm[1][1]:5d}")


def stitch_canvas(raster_arr, gt_arr, pred_arr, rows, cols):
    height, width = 128, 128
    raster_canvas = np.zeros((rows * height, cols * width), dtype=np.float32)
    gt_canvas = np.zeros_like(raster_canvas)
    pred_canvas = np.zeros_like(raster_canvas)

    for i in range(len(raster_arr)):
        row = i // cols
        col = i % cols

        raster_canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = raster_arr[i, 0]
        gt_canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = gt_arr[i, 0]
        pred_canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = pred_arr[i, 0]

    return raster_canvas, gt_canvas, pred_canvas


def add_patchwise_contours(masks, color, ax, rows, cols):
    height, width = 128, 128

    for i in range(len(masks)):
        row = i // cols
        col = i % cols

        mask = masks[i, 0]
        contours = measure.find_contours(mask, 0.5)

        for contour in contours:
            ax.plot(
                contour[:, 1] + col * width,
                contour[:, 0] + row * height,
                linewidth=0.5,
                color=color,
                alpha=0.7,
            )


def postprocess_mask(binary_mask):
    cleaned = morphology.remove_small_objects(binary_mask.astype(bool), min_size=188).astype(np.uint8)
    cleaned = morphology.remove_small_holes(cleaned.astype(bool), area_threshold=300).astype(np.uint8)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3)).astype(np.uint8)
    return cleaned


def compute_post_iou(cleaned_pred, gt_mask):
    intersection = np.logical_and(cleaned_pred, gt_mask).sum()
    union = np.logical_or(cleaned_pred, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    test_output = get_test_output()

    regions = [
        {"name": "Himachal", "rows": 80, "cols": 83},
        {"name": "Himachal Ladakh", "rows": 48, "cols": 35},
        {"name": "Sikkim", "rows": 33, "cols": 36},
        {"name": "Kashmir", "rows": 38, "cols": 39},
        {"name": "Uttrakhand", "rows": 22, "cols": 33},
    ]

    for region in regions:
        name, rows, cols = region["name"], region["rows"], region["cols"]
        metrics, pred, gt, rt, cm = test_output[name]

        flattened_rt = torch.cat(rt, dim=0).cpu().numpy()
        flattened_gt = torch.cat(gt, dim=0).cpu().numpy()
        flattened_pred = torch.cat(pred, dim=0).cpu().numpy()

        if flattened_gt.ndim == 3:
            flattened_gt = np.expand_dims(flattened_gt, axis=1)
        if flattened_pred.ndim == 3:
            flattened_pred = np.expand_dims(flattened_pred, axis=1)

        raster_canvas, gt_canvas, pred_canvas = stitch_canvas(
            flattened_rt, flattened_gt, flattened_pred, rows, cols
        )

        print_metrics(metrics, name)
        print_confusion_matrix(cm)

        diff_canvas = np.abs(gt_canvas - pred_canvas)

        fig, axs = plt.subplots(2, 2, figsize=(16, 16))

        axs[0, 0].imshow(gt_canvas, cmap="gray")
        axs[0, 0].set_title(f"Ground Truth - {name}")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(pred_canvas, cmap="gray")
        axs[0, 1].set_title("Prediction")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(diff_canvas, cmap="Reds")
        axs[1, 0].set_title("Difference")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(raster_canvas, cmap="gray")
        axs[1, 1].set_title("Contours Overlay")
        axs[1, 1].axis("off")

        add_patchwise_contours(flattened_gt, "yellow", axs[1, 1], rows, cols)
        add_patchwise_contours(flattened_pred, "red", axs[1, 1], rows, cols)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_4panel.png", dpi=600)
        plt.close()

        full_pred = (pred_canvas > 0.5).astype(np.uint8)
        full_gt = (gt_canvas > 0.5).astype(np.uint8)

        cleaned = postprocess_mask(full_pred)
        smoothed = gaussian_filter(cleaned.astype(float), sigma=1.0)

        post_iou = compute_post_iou(smoothed, full_gt)
        print(f"Postprocessed IoU: {post_iou:.4f}")

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(raster_canvas, cmap="gray")

        for contour in measure.find_contours(full_gt, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=0.7)

        for contour in measure.find_contours(smoothed, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color="red", linewidth=0.7, alpha=0.7)

        plt.axis("off")
        plt.title(f"Postprocessed - {name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_postprocessed.png", dpi=600)
        plt.close()
