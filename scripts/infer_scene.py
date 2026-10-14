import argparse
import json
import os
import sys

# Ensure src is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch

from glacier_mapping.models.AdapterUNet import AdapterUNet
from glacier_mapping.models.ResUNet import ResUNet
from glacier_mapping.utils.tiled_inference import predict_full_scene


def main():
    parser = argparse.ArgumentParser(
        description="Sliding Window Inference for Large Multi-Band Satellite Scenes"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input 18-band GeoTIFF raster")
    parser.add_argument("--output", type=str, required=True, help="Path for output segmentation GeoTIFF")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained PyTorch model weights (.pth)")
    parser.add_argument("--model", type=str, default="adapter_unet", choices=["adapter_unet", "resunet"])
    parser.add_argument("--backbone", type=str, default="resnet34", help="Backbone for AdapterUNet")
    parser.add_argument("--tile_size", type=int, default=128, help="Tile size (e.g. 128)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction between adjacent tiles (e.g. 0.5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--stats_json", type=str, default=None, help="Path to band_statistics.json for normalization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} weights from: {args.weights}")

    # Build model
    if args.model == "adapter_unet":
        model = AdapterUNet(in_channels=18, backbone=args.backbone, pretrained=False)
    else:
        model = ResUNet(in_channels=18, out_channels=1)

    state_dict = torch.load(args.weights, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load stats if provided
    band_means, band_stds = None, None
    if args.stats_json and os.path.exists(args.stats_json):
        with open(args.stats_json) as f:
            stats = json.load(f)
            band_means = stats.get("means")
            band_stds = stats.get("stds")
        print("Loaded per-band normalization statistics.")

    print(f"Running seamless sliding-window inference on: {args.input}")
    predict_full_scene(
        model=model,
        input_raster_path=args.input,
        output_raster_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=device,
        threshold=args.threshold,
        band_means=band_means,
        band_stds=band_stds,
    )


if __name__ == "__main__":
    main()
