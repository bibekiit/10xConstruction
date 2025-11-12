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

class WeightedBCELoss(nn.Module):
    """Weighted BCE loss for handling class imbalance."""
    
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] tensor with logits (before sigmoid)
            targets: [B, 1, H, W] tensor with values in [0, 1]
        """
        # Calculate class weights
        pos_pixels = targets.sum()
        neg_pixels = targets.numel() - pos_pixels
        
        if pos_pixels > 0:
            weight = neg_pixels / pos_pixels
        else:
            weight = 1.0
        
        # Weighted BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets,
            pos_weight=torch.tensor(weight * self.pos_weight).to(predictions.device)
        )
        return bce

class WeightedCombinedLoss(nn.Module):
    """Weighted combined BCE and Dice loss."""
    
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=10.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] tensor with logits (before sigmoid)
            targets: [B, 1, H, W] tensor with values in [0, 1]
        """
        # Weighted BCE
        pos_pixels = targets.sum()
        neg_pixels = targets.numel() - pos_pixels
        weight = (neg_pixels / pos_pixels) if pos_pixels > 0 else 1.0
        
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets,
            pos_weight=torch.tensor(weight * self.pos_weight).to(predictions.device)
        )
        
        # Dice loss (needs sigmoid predictions)
        pred_sigmoid = torch.sigmoid(predictions)
        dice = self.dice(pred_sigmoid, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice

