import argparse
import json
import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch

from glacier_mapping.data.dataloader import (
    compute_dataset_band_stats,
    create_loro_loaders,
)
from glacier_mapping.losses.DiceBCE import CombinedLoss as DiceBCELoss
from glacier_mapping.losses.DiceIoUSSIM import CombinedLoss as DiceIoUSSIMLoss
from glacier_mapping.losses.FocalDiceTversky import CombinedLoss as FocalDiceTverskyLoss
from glacier_mapping.models.AdapterUNet import AdapterUNet
from glacier_mapping.models.ResUNet import ResUNet
from glacier_mapping.training.evaluate import evaluate
from glacier_mapping.training.train import train_model
from glacier_mapping.utils.benchmark import run_benchmark_report

REGION_KEYS = ["himachal", "himlad", "kashmir", "sikkim", "uttrakhand"]


def find_region_directories(data_dir: str) -> dict[str, tuple[str, str]]:
    """
    Auto-discovers images and masks folders for the 5 Himalayan regions.
    Supports names like 'himachal_128', 'himachal', 'himachal_ladakh_128', etc.
    """
    region_paths = {}
    if not os.path.exists(data_dir):
        print(f"Warning: data_dir '{data_dir}' does not exist yet.")
        return region_paths

    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for r_key in REGION_KEYS:
        matched_dir = None
        for d in subdirs:
            d_lower = d.lower().replace("-", "_").replace(" ", "_")
            is_match = False
            if r_key == "himlad":
                is_match = "ladakh" in d_lower or "himlad" in d_lower
            elif r_key == "himachal":
                is_match = "himachal" in d_lower and "ladakh" not in d_lower
            elif r_key == "uttrakhand":
                is_match = "uttrakhand" in d_lower or "uttarakhand" in d_lower
            elif r_key == "kashmir":
                is_match = "kashmir" in d_lower
            elif r_key == "sikkim":
                is_match = "sikkim" in d_lower

            if is_match:
                matched_dir = d
                break

        if matched_dir:
            img_dir = os.path.join(data_dir, matched_dir, "images")
            mask_dir = os.path.join(data_dir, matched_dir, "masks")
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                region_paths[r_key] = (img_dir, mask_dir)

    return region_paths


def get_loss_function(name: str):
    name = name.lower()
    if name == "focal_dice_tversky":
        return FocalDiceTverskyLoss()
    elif name == "dice_iou_ssim":
        return DiceIoUSSIMLoss()
    elif name == "dice_bce":
        return DiceBCELoss()
    else:
        raise ValueError(f"Unknown loss: {name}. Choices: focal_dice_tversky, dice_iou_ssim, dice_bce")


def get_model(model_name: str, backbone: str = "resnet34", in_channels: int = 18):
    model_name = model_name.lower()
    if model_name == "adapter_unet":
        return AdapterUNet(in_channels=in_channels, backbone=backbone, pretrained=True)
    elif model_name == "resunet":
        return ResUNet(in_channels=in_channels, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choices: adapter_unet, resunet")


def main():
    parser = argparse.ArgumentParser(
        description="Leave-One-Region-Out (LORO) Cross-Validation for Himalayan Glacier Segmentation"
    )
    parser.add_argument("--data_dir", type=str, default="datasets128", help="Path to dataset root folder")
    parser.add_argument(
        "--test_region",
        type=str,
        default="sikkim",
        choices=REGION_KEYS,
        help="Region held out for zero-shot testing (e.g. sikkim)",
    )
    parser.add_argument("--model", type=str, default="adapter_unet", choices=["adapter_unet", "resunet"])
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone for AdapterUNet (resnet18/resnet34)")
    parser.add_argument("--loss", type=str, default="focal_dice_tversky", help="Loss function")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--save_dir", type=str, default="checkpoints/loro_run", help="Output directory for weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--benchmark", action="store_true", help="Run latency & parameters benchmark after training")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing LORO Training on Device: {device} | Test Region Held Out: {args.test_region.upper()}")

    # Step 1: Find regions
    region_paths = find_region_directories(args.data_dir)
    if not region_paths:
        print(f"No region directories found in '{args.data_dir}'. Please verify your dataset path.")
        return

    print(f"Discovered regions: {list(region_paths.keys())}")
    if args.test_region not in region_paths:
        print(f"Error: Held-out test region '{args.test_region}' not found in dataset directories.")
        return

    # Step 2: Compute per-band normalization statistics
    train_image_dirs = [paths[0] for r, paths in region_paths.items() if r != args.test_region]
    stats_cache = os.path.join(args.save_dir, "band_statistics.json")
    print("Computing / loading per-channel z-score statistics...")
    means, stds = compute_dataset_band_stats(train_image_dirs, sample_limit=400, save_path=stats_cache)

    # Step 3: Create zero-leakage LORO dataloaders
    train_loader, val_loader, test_loader = create_loro_loaders(
        region_paths=region_paths,
        test_region=args.test_region,
        batch_size=args.batch_size,
        val_ratio=0.15,
        num_workers=2,
        band_means=means,
        band_stds=stds,
    )
    print(
        f"Data Partition: Train Batches: {len(train_loader)} | "
        f"Val Batches: {len(val_loader)} | Zero-Shot Test Batches: {len(test_loader)}"
    )

    # Step 4: Model & Loss
    model = get_model(args.model, backbone=args.backbone, in_channels=18).to(device)
    criterion = get_loss_function(args.loss)

    # Step 5: Train
    train_losses, val_losses, val_ious, val_f1s = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        device=device,
        use_amp=args.amp,
        checkpoint_dir=args.save_dir,
        resume_path=args.resume,
    )

    # Step 6: Zero-Shot Generalization Test on Held-out Region
    print("\n==========================================")
    print(f"ZERO-SHOT EVALUATION ON HELD-OUT REGION: {args.test_region.upper()}")
    print("==========================================")
    test_metrics, cm = evaluate(model, test_loader, criterion=criterion, device=device)

    for k, v in test_metrics.items():
        print(f"  {k.upper():12s}: {v:.4f}")

    # Save results
    results_file = os.path.join(args.save_dir, f"loro_results_{args.test_region}.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "test_region": args.test_region,
                "model": args.model,
                "backbone": args.backbone,
                "loss": args.loss,
                "metrics": test_metrics,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved evaluation metrics to: {results_file}")

    # Optional: Run Benchmark
    if args.benchmark:
        run_benchmark_report(model, model_name=f"{args.model} ({args.backbone})")


if __name__ == "__main__":
    main()
