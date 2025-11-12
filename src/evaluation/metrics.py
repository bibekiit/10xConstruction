"""
Evaluation metrics for segmentation.
"""
import torch
import numpy as np
from sklearn.metrics import jaccard_score

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    
    Args:
        pred_mask: Predicted binary mask (numpy array, 0 or 1)
        gt_mask: Ground truth binary mask (numpy array, 0 or 1)
    
    Returns:
        IoU score (float)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """
    Calculate Dice coefficient for binary masks.
    
    Args:
        pred_mask: Predicted binary mask (numpy array, 0 or 1)
        gt_mask: Ground truth binary mask (numpy array, 0 or 1)
    
    Returns:
        Dice score (float)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0
    
    dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum())
    return dice

def calculate_pixel_accuracy(pred_mask, gt_mask):
    """
    Calculate pixel accuracy.
    
    Args:
        pred_mask: Predicted binary mask (numpy array, 0 or 1)
        gt_mask: Ground truth binary mask (numpy array, 0 or 1)
    
    Returns:
        Pixel accuracy (float)
    """
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    return correct / total if total > 0 else 0.0

def evaluate_batch(predictions, targets, threshold=0.5):
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: Tensor [B, 1, H, W] with values in [0, 1]
        targets: Tensor [B, 1, H, W] with values in [0, 1]
        threshold: Threshold for binarization
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy and binarize
    pred_np = (predictions.cpu().numpy() > threshold).astype(np.uint8)
    target_np = (targets.cpu().numpy() > 0.5).astype(np.uint8)
    
    batch_size = pred_np.shape[0]
    
    ious = []
    dices = []
    pixel_accs = []
    
    for i in range(batch_size):
        pred = pred_np[i, 0]
        target = target_np[i, 0]
        
        iou = calculate_iou(pred, target)
        dice = calculate_dice(pred, target)
        pixel_acc = calculate_pixel_accuracy(pred, target)
        
        ious.append(iou)
        dices.append(dice)
        pixel_accs.append(pixel_acc)
    
    return {
        'iou': np.mean(ious),
        'dice': np.mean(dices),
        'pixel_accuracy': np.mean(pixel_accs),
        'iou_std': np.std(ious),
        'dice_std': np.std(dices)
    }

def calculate_miou(predictions, targets, threshold=0.5):
    """Calculate mean IoU."""
    metrics = evaluate_batch(predictions, targets, threshold)
    return metrics['iou']

def calculate_mean_dice(predictions, targets, threshold=0.5):
    """Calculate mean Dice coefficient."""
    metrics = evaluate_batch(predictions, targets, threshold)
    return metrics['dice']

