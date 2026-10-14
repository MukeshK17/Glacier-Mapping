
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module | None = None,
    device: torch.device | None = None,
    return_predictions: bool = False,
    threshold: float = 0.5,
) -> tuple:
    """
    Evaluates segmentation model on a given dataloader.
    Computes loss, IoU, Accuracy, Precision, Recall, F1, Dice, and Cohen's Kappa.
    """
    dev = device or DEVICE
    model = model.to(dev)
    model.eval()

    if hasattr(model, "module"):
        model = model.module

    eps = 1e-7
    total_loss = 0.0
    total_cm = np.zeros((2, 2), dtype=np.int64)
    all_preds, all_gts, all_rasters = [], [], []

    has_batches = len(loader) > 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)

            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

            if y.ndim == 3:
                y = y.unsqueeze(1)
            elif y.ndim == 4 and y.shape[1] != 1:
                y = y[:, :1, :, :]

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            if criterion is not None:
                loss = criterion(logits, y.float())
                total_loss += loss.item()
            else:
                total_loss += F.binary_cross_entropy(probs, y.float()).item()

            preds_np = (preds > 0.5).view(-1).cpu().numpy().astype(bool)
            y_np = (y > 0.5).view(-1).cpu().numpy().astype(bool)
            batch_cm = confusion_matrix(y_np, preds_np, labels=[0, 1])
            total_cm += batch_cm

            if return_predictions:
                all_preds.append(preds.cpu())
                all_gts.append(y.cpu())
                all_rasters.append(x.cpu())

    tn, fp, fn, tp = total_cm.ravel()

    metrics = {
        "loss": (total_loss / len(loader)) if has_batches else 0.0,
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "accuracy": float((tp + tn + eps) / (tp + tn + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "f1": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
    }

    total_pixels = tp + tn + fp + fn
    if total_pixels > 0:
        po = (tp + tn) / total_pixels
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total_pixels**2)
        metrics["kappa"] = float((po - pe) / (1 - pe + eps))
    else:
        metrics["kappa"] = 0.0

    if return_predictions:
        return metrics, all_preds, all_gts, all_rasters, total_cm
    return metrics, total_cm
