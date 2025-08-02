import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1. - eps) 
        bce = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
        
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        eps = 1e-7
        y_pred = y_pred.squeeze(1)
        y_true = y_true.squeeze(1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2. * intersection + eps) / (union + eps)
        return 1 - dice
        
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        eps = 1e-7
        y_pred = y_pred.squeeze(1)
        y_true = y_true.squeeze(1)
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()
        tversky = (TP + eps) / (TP + self.alpha * FP + self.beta * FN + eps)
        return 1 - tversky
        
class CombinedLoss(nn.Module):
    def __init__(self, weights=(1.0, 0.5, 0.5)):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()
        self.w_focal, self.w_dice, self.w_tversky = weights

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)  
        loss = (
            self.w_focal * self.focal(probs, targets.float()) +
            self.w_dice * self.dice(probs, targets.float()) +
            self.w_tversky * self.tversky(probs, targets.float())
        )
        return loss

