import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)

def evaluate(model, loader, criterion, return_predictions=False, plot_confusion_matrix=False):
    model = model.to(device)
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    eps = 1e-7

    total_loss = 0
    total_cm = np.zeros((2, 2))
    all_preds, all_gts, all_rasters = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

            if y.ndim == 3: # [B, 1, H, W]
                y = y.unsqueeze(1)
            elif y.ndim == 4 and y.shape[1] != 1:
                y = y[:, :1, :, :]

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            loss = criterion(logits, y.float())
            total_loss += loss.item()

            # Remove channel dimension for metric calculation
            preds_np = preds.view(-1).cpu().numpy().astype(bool)
            y_np = y.view(-1).cpu().numpy().astype(bool)
            batch_cm = confusion_matrix(y_np, preds_np, labels=[0, 1])
            total_cm += batch_cm

            if return_predictions:
                all_preds.append(preds.cpu())
                all_gts.append(y.cpu())
                all_rasters.append(x.cpu())

    tn, fp, fn, tp = total_cm.ravel()

    metrics = {
        'loss': total_loss / len(loader),
        'iou': (tp + eps) / (tp + fp + fn + eps),
        'accuracy': (tp + tn + eps) / (tp + tn + fp + fn + eps),
        'precision': (tp + eps) / (tp + fp + eps),
        'recall': (tp + eps) / (tp + fn + eps),
        'f1': (2 * (tp + eps)) / (2 * tp + fp + fn + eps),
        'dice': (2 * tp + eps) / (2 * tp + fp + fn + eps),
        'kappa': ((tp + tn)/ (tp + tn + fp + fn) -
                 (((tp + fp)*(tp + fn)+(fn + tn)*(fp + tn))/(tp + tn + fp + fn)**2))
                 / (1 - (((tp + fp)*(tp + fn)+(fn + tn)*(fp + tn))/(tp + tn + fp + fn)**2) + eps)
    }

    if return_predictions:
        return metrics, all_preds, all_gts, all_rasters, total_cm
    else:
        return metrics, total_cm
