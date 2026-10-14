import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from glacier_mapping.training.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    epochs: int = 50,
    patience: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: torch.device | None = None,
    use_amp: bool = True,
    checkpoint_dir: str = "checkpoints",
    resume_path: str | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Robust training loop with:
    - Automatic Mixed Precision (AMP FP16) for fast Colab execution
    - Checkpoint saving & seamless resumption
    - Early stopping on Validation IoU
    """
    dev = device or DEVICE
    model = model.to(dev)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    # Mixed precision scaler (enabled only if CUDA is available)
    amp_enabled = use_amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_val_iou = 0.0
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_ious = []
    val_f1s = []

    # Resume from checkpoint if provided
    if resume_path and os.path.isfile(resume_path):
        print(f"Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=dev)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_iou = checkpoint.get("best_val_iou", 0.0)
        print(f"Resumed at epoch {start_epoch} with Best Val IoU: {best_val_iou:.4f}")

    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for x, y in pbar:
            x = x.to(dev)
            y = y.to(dev)

            if y.ndim == 3:
                y = y.unsqueeze(1)
            elif y.ndim == 4 and y.shape[1] != 1:
                y = y[:, :1, :, :]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(x)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                if logits.shape[2:] != y.shape[2:]:
                    logits = F.interpolate(logits, size=y.shape[2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, y.float())

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)

        # Validation phase
        val_metrics, _ = evaluate(model, val_loader, criterion=criterion, device=dev)
        val_loss = val_metrics["loss"]
        val_iou = val_metrics["iou"]
        val_f1 = val_metrics["f1"]

        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch+1:03d} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}"
        )

        scheduler.step(val_iou)

        # Save latest checkpoint for recovery
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
            "best_val_iou": best_val_iou,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint_state, last_checkpoint_path)

        # Check for best validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=dev))
        print(f"Loaded best model weights from {best_model_path}")

    return train_losses, val_losses, val_ious, val_f1s
