import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
import torch
from scipy.ndimage import gaussian_filter

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12

output_dir = ""
os.makedirs(output_dir, exist_ok=True)

from main import get_test_output
test_output = get_test_output()
test_metrics1, pred1, gt1, rt1, cm1 = test_output["Himachal"]
test_metrics2, pred2, gt2, rt2, cm2 = test_output["Himachal Ladakh"]
test_metrics3, pred3, gt3, rt3, cm3 = test_output["Sikkim"]
test_metrics4, pred4, gt4, rt4, cm4 = test_output["Kashmir"]
test_metrics5, pred5, gt5, rt5, cm5 = test_output["Uttrakhand"]

def print_metrics(metrics, region_name):
    print(f"\nResults for {region_name}")
    print(f"Test Loss: {metrics['loss']:.4f}, IoU: {metrics['iou']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, "
          f"Precision: {metrics['precision']:.4f}, F1 Score: {metrics['f1']:.4f}, "
          f"Kappa: {metrics['kappa']:.4f}")

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

        raster_canvas[row*height:(row+1)*height, col*width:(col+1)*width] = raster_arr[i, 0, :, :]
        gt_canvas[row*height:(row+1)*height, col*width:(col+1)*width] = gt_arr[i, 0, :, :]
        pred_canvas[row*height:(row+1)*height, col*width:(col+1)*width] = pred_arr[i, 0, :, :]
        
    return raster_canvas, gt_canvas, pred_canvas

def add_patchwise_contours(masks, color, ax, rows, cols):
    height, width = 128, 128
    for i in range(len(masks)):
        row = i // cols
        col = i % cols
        mask = masks[i, 0, :, :]
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1] + col * width,
                    contour[:, 0] + row * height,
                    linewidth=0.5, color=color, alpha = 0.7)

def postprocess_mask(binary_mask):
    cleaned = morphology.remove_small_objects(binary_mask.astype(bool), min_size=188).astype(np.uint8)
    cleaned = morphology.remove_small_holes(cleaned.astype(bool), area_threshold=300).astype(np.uint8)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3)).astype(np.uint8)
    return cleaned

def compute_post_iou(cleaned_pred, gt_mask):
    intersection = np.logical_and(cleaned_pred, gt_mask).sum()
    union = np.logical_or(cleaned_pred, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

regions = [
    {"name": "Himachal", "rows": 80, "cols": 83, "metrics": test_metrics1, "pred": pred1, "gt": gt1, "rt": rt1, "cm": cm1},
    {"name": "Himlad", "rows": 48, "cols": 35, "metrics": test_metrics2, "pred": pred2, "gt": gt2, "rt": rt2, "cm": cm2},
    {"name": "Sikkim", "rows": 33, "cols": 36, "metrics": test_metrics3, "pred": pred3, "gt": gt3, "rt": rt3, "cm": cm3},
    {"name": "Kashmir", "rows": 38, "cols": 39, "metrics": test_metrics4, "pred": pred4, "gt": gt4, "rt": rt4, "cm": cm4},
    {"name": "Uttrakhand", "rows": 22, "cols": 33, "metrics": test_metrics5, "pred": pred5, "gt": gt5, "rt": rt5, "cm": cm5},
]

for region in regions:
    name, rows, cols = region["name"], region["rows"], region["cols"]
    metrics, cm = region["metrics"], region["cm"]
    
    # Flatten and convert
    flattened_rt = torch.cat(region["rt"], dim=0).cpu().numpy()
    flattened_gt = torch.cat(region["gt"], dim=0).cpu().numpy()
    flattened_pred = torch.cat(region["pred"], dim=0).cpu().numpy()

    if flattened_gt.ndim == 3:
        flattened_gt = np.expand_dims(flattened_gt, axis=1)
    if flattened_pred.ndim == 3:
        flattened_pred = np.expand_dims(flattened_pred, axis=1)

    # Stitch for raster/masks
    raster_canvas, gt_canvas, pred_canvas = stitch_canvas(flattened_rt, flattened_gt, flattened_pred, rows, cols)

    # === Print Raw Metrics ===
    print_metrics(metrics, name)
    print_confusion_matrix(cm)

    # === Plot Patchwise Contour Overlay ===
    diff_canvas = np.abs(gt_canvas - pred_canvas)

    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs[0, 0].imshow(gt_canvas, cmap='gray')
    axs[0, 0].set_title(f"Ground Truth Mask - {name}")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(pred_canvas, cmap='gray')
    axs[0, 1].set_title("Predicted Mask")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(diff_canvas, cmap='Reds')
    axs[1, 0].set_title("Difference Mask")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(raster_canvas, cmap='gray')
    axs[1, 1].set_title("Patchwise Overlay Contours")
    axs[1, 1].axis('off')

    add_patchwise_contours(flattened_gt, 'yellow', axs[1, 1], rows, cols)
    add_patchwise_contours(flattened_pred, 'red', axs[1, 1], rows, cols)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_4panel.png", dpi=600, bbox_inches='tight')
    plt.show()

    # === Global Postprocessed Contours ===
    full_pred_mask = (pred_canvas > 0.5).astype(np.uint8)
    full_gt_mask = (gt_canvas > 0.5).astype(np.uint8)

    
    cleaned_pred = postprocess_mask(full_pred_mask)
    smoothed_pred = gaussian_filter(cleaned_pred.astype(float), sigma=1.0)
    
    post_iou = compute_post_iou(smoothed_pred, full_gt_mask)

    print(f"Postprocessed IoU: {post_iou:.4f}")

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(raster_canvas, cmap='gray')
    
    contours_gt = measure.find_contours(full_gt_mask, 0.5)
    for contour in contours_gt:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.7, color='yellow', label='GT')


    
    contours_post = measure.find_contours(smoothed_pred, 0.5)
    for contour in contours_post:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.7, color='red', alpha = 0.7,label='Postprocessed Pred')

    plt.axis('off')
    plt.title(f"Postprocessed Contours - {name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_postprocessed.png", dpi=600, bbox_inches='tight')
    plt.show()
