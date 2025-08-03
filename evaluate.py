import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)
print(device) # CUDA based code, will run on GPU if available

def evaluate(model, loader, criterion, return_predictions=False, plot_confusion_matrix=False):
    model.eval()
    if isinstance(model, nn.DataParallel):
        model = model.module

    criterion = criterion() 
    eps = 1e-7

    total_loss = 0
    total_cm = np.zeros((2, 2))  # Binary confusion matrix

    all_preds, all_gts, all_rasters = [], [], []

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            logits = outputs.logits  
            logits = F.interpolate(logits, size=Y.shape[-2:], mode='bilinear', align_corners=False)
    
            # Make Y shape: [B, 1, H, W] to match logits
            if Y.ndim == 3:
                Y = Y.unsqueeze(1)
            elif Y.ndim == 4 and Y.shape[1] != 1:
                Y = Y[:, :1, :, :]  # for multiple channels
    
            # Get predicted probs and binarize
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
    
            loss = criterion(logits, Y.float())
            total_loss += loss.item()
    
            # Remove channel dimension for metric calculation
            preds_np = preds.view(-1).cpu().numpy().astype(bool)
            Y_np = Y.view(-1).cpu().numpy().astype(bool)
            batch_cm = confusion_matrix(Y_np, preds_np, labels=[0, 1])
            total_cm += batch_cm
    
            if return_predictions:
                all_preds.append(preds.cpu())
                all_gts.append(Y.cpu())
                all_rasters.append(X.cpu())

    
    # Only calculate metrics after loop
    tn, fp, fn, tp = total_cm.ravel()

    metrics = {
        'loss': total_loss / len(loader),
        'iou': (tp + eps) / (tp + fp + fn + eps),
        'accuracy': (tp + tn + eps) / (tp + tn + fp + fn + eps),
        'precision': (tp + eps) / (tp + fp + eps),
        'recall': (tp + eps) / (tp + fn + eps),
        'f1': (2 * (tp + eps)) / (2 * tp + fp + fn + eps),
        'dice': (2 * tp + eps) / (2 * tp + fp + fn + eps),
        'kappa': ((tp + tn)/ (tp + tn + fp + fn) - (((tp + fp)*(tp + fn)+(fn + tn)*(fp + tn))/(tp + tn + fp + fn)**2)) / (1 - (((tp + fp)*(tp + fn)+(fn + tn)*(fp + tn))/(tp + tn + fp + fn)**2) + eps)
    }

    if return_predictions:
        return metrics, all_preds, all_gts, all_rasters, total_cm
    else:
        return metrics, total_cm
