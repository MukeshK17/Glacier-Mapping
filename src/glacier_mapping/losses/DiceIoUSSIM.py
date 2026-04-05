import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index as ssim_index

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = target.float()
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        intersection = (pred * target).sum(dim=(1, 2))
        total = (pred + target).sum(dim=(1, 2))
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        return 1 - ssim_index(pred, target, data_range=1.0)

class CombinedLoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0)):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.ssim = SSIMLoss()
        self.w_dice, self.w_iou, self.w_ssim = weights
        
    def forward(self, pred, target):
        if pred.shape[1] > 1:
            pred = pred[:, 1, :, :]
        else:
            pred = pred.squeeze(1)

        if target.ndim == 4:
            target = target.squeeze(1)

        loss = (
            self.w_dice * self.dice(pred, target) +
            self.w_iou * self.iou(pred, target) +
            self.w_ssim * self.ssim(pred, target)
        )

        return loss

  