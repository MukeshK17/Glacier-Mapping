import argparse
import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import torch

from glacier_mapping.data.dataloader import create_dataloaders, get_test_loader
from glacier_mapping.losses.DiceBCE import CombinedLoss as DiceBCELoss
from glacier_mapping.losses.DiceIoUSSIM import CombinedLoss as DiceIoUSSIMLoss
from glacier_mapping.losses.FocalDiceTversky import CombinedLoss as FocalDiceTverskyLoss
from glacier_mapping.models.AdapterUNet import AdapterUNet
from glacier_mapping.models.ResUNet import ResUNet
from glacier_mapping.training.evaluate import evaluate
from glacier_mapping.training.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


def main():
    parser = argparse.ArgumentParser(description="Glacier Mapping End-to-End Baseline Pipeline")
    parser.add_argument("--data_dir", type=str, default="datasets128", help="Path to datasets root")
    parser.add_argument("--model", type=str, default="adapter_unet", choices=["adapter_unet", "resunet"])
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone for AdapterUNet")
    parser.add_argument("--loss", type=str, default="focal_dice_tversky", help="Loss function")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Paths configuration
    dataset_configs = {
        "himachal": (
            os.path.join(args.data_dir, "himachal_128", "images"),
            os.path.join(args.data_dir, "himachal_128", "masks"),
            1.0,
        ),
        "himlad": (
            os.path.join(args.data_dir, "himachal_ladakh_128", "images"),
            os.path.join(args.data_dir, "himachal_ladakh_128", "masks"),
            0.3,
        ),
        "sikkim": (
            os.path.join(args.data_dir, "sikkim128", "images"),
            os.path.join(args.data_dir, "sikkim128", "masks"),
            0.3,
        ),
        "kashmir": (
            os.path.join(args.data_dir, "kashmir128", "images"),
            os.path.join(args.data_dir, "kashmir128", "masks"),
            0.3,
        ),
        "uttrakhand": (
            os.path.join(args.data_dir, "uttrakhand128", "images"),
            os.path.join(args.data_dir, "uttrakhand128", "masks"),
            0.3,
        ),
    }

    # Verify data existence
    existing_configs = {k: v for k, v in dataset_configs.items() if os.path.exists(v[0]) and os.path.exists(v[1])}
    if not existing_configs:
        print(f"No datasets found in '{args.data_dir}'. Please verify your data directory.")
        return

    train_loader, val_loader = create_dataloaders(existing_configs, batch_size=args.batch_size)

    # Initialize Model
    if args.model == "adapter_unet":
        model = AdapterUNet(in_channels=18, backbone=args.backbone, pretrained=True).to(device)
    else:
        model = ResUNet(in_channels=18, out_channels=1).to(device)

    # Loss
    if args.loss == "focal_dice_tversky":
        criterion = FocalDiceTverskyLoss()
    elif args.loss == "dice_iou_ssim":
        criterion = DiceIoUSSIMLoss()
    else:
        criterion = DiceBCELoss()

    # Train
    print(f"Training {args.model} on {len(existing_configs)} regions with {args.loss}...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=args.epochs,
        patience=10,
        lr=args.lr,
        device=device,
        use_amp=True,
    )

    # Evaluate on individual regions
    print("\n--- Final Evaluation Across Regions ---")
    metrics_list = []
    region_names = []

    for name, (img_dir, mask_dir, *_) in existing_configs.items():
        test_loader = get_test_loader(img_dir, mask_dir, batch_size=args.batch_size)
        metrics, _ = evaluate(model, test_loader, criterion=criterion, device=device)
        metrics_list.append(metrics)
        region_names.append(name.capitalize())

    df = pd.DataFrame(metrics_list, index=region_names)
    print("\nEvaluation Metrics Table:")
    print(df)


if __name__ == "__main__":
    main()
