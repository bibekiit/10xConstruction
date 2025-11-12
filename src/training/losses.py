"""
Loss functions for segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] tensor with values in [0, 1]
            targets: [B, 1, H, W] tensor with values in [0, 1]
        """
        # Flatten
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        # Return Dice loss (1 - Dice)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal_loss.mean()

