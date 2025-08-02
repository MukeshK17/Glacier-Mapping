import torch 
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, output, target):
        bce = F.binary_cross_entropy_with_logits(output, target)
        probs = torch.sigmoid(output)
        intersection = (probs * target).sum()
        dice = 1 - ((2. * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth))
        loss = 0.5 * bce + 0.5 * dice
        return loss