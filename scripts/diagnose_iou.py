#!/usr/bin/env python3
"""
Diagnostic script to investigate IoU/Dice issues and find optimal threshold.
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import TextConditionedSegmentationDataset, CombinedDataset
from src.models.clipseg_model import SimpleCLIPSegModel
from src.evaluation.metrics import evaluate_batch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def diagnose_model(model, val_loader, device='cpu'):
    """Diagnose IoU issues and suggest fixes"""
    
    model.eval()
    all_preds = []
    all_masks = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']
            
            preds = model(images, prompts=prompts)
            # Ensure sigmoid is applied
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    
    # Statistics
    print("\n" + "="*60)
    print("DIAGNOSIS RESULTS")
    print("="*60)
    print(f"Prediction range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
    print(f"Prediction mean: {all_preds.mean():.4f}")
    print(f"Prediction std: {all_preds.std():.4f}")
    
    gt_pos_pixels = (all_masks > 0.5).sum().item()
    gt_total_pixels = all_masks.numel()
    gt_pos_ratio = gt_pos_pixels / gt_total_pixels * 100
    print(f"GT positive pixels: {gt_pos_pixels:,} / {gt_total_pixels:,} ({gt_pos_ratio:.2f}%)")
    
    # Test different thresholds
    print("\n" + "="*60)
    print("THRESHOLD TESTING")
    print("="*60)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_iou = 0
    best_thresh = 0.5
    best_dice = 0
    
    for thresh in thresholds:
        pred_binary = (all_preds > thresh).float()
        pos_pixels = (pred_binary > 0.5).sum().item()
        pos_ratio = pos_pixels / pred_binary.numel() * 100
        
        # Calculate metrics
        metrics = evaluate_batch(all_preds, all_masks, threshold=thresh)
        
        print(f"Threshold {thresh:.1f}: "
              f"IoU={metrics['iou']:.4f}, "
              f"Dice={metrics['dice']:.4f}, "
              f"PixelAcc={metrics['pixel_accuracy']:.4f}, "
              f"PosPixels={pos_ratio:.2f}%")
        
        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            best_thresh = thresh
            best_dice = metrics['dice']
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"Best threshold: {best_thresh:.1f}")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    
    if best_iou < 0.1:
        print("\n⚠️ IoU still very low. Consider:")
        print("1. Using weighted/focal loss")
        print("2. Lowering learning rate")
        print("3. Adding data augmentation")
        print("4. Checking mask format")
        print("5. Retraining with class weights")
    elif best_iou < 0.3:
        print("\n✓ Basic segmentation working. Can improve with:")
        print("1. Better loss function (weighted/focal)")
        print("2. More training epochs")
        print("3. Data augmentation")
    else:
        print("\n✓ Good segmentation quality achieved!")
    
    return best_thresh, best_iou, best_dice

def main():
    parser = argparse.ArgumentParser(description='Diagnose IoU/Dice issues')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/training_both_datasets/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load dataset
    print("Loading validation dataset...")
    val_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    data_dir = Path(args.data_dir)
    val_dirs = []
    
    # Check for taping area dataset
    taping_val = data_dir / "taping_area" / "val"
    if taping_val.exists():
        val_dirs.append(taping_val)
    
    # Check for cracks dataset
    cracks_val = data_dir / "cracks" / "val"
    if cracks_val.exists():
        val_dirs.append(cracks_val)
    
    if not val_dirs:
        raise ValueError(f"No validation data found in {args.data_dir}")
    
    # Create datasets
    val_datasets = [TextConditionedSegmentationDataset(d, transform=val_transform) 
                    for d in val_dirs]
    
    if len(val_datasets) > 1:
        val_dataset = CombinedDataset(val_datasets)
    else:
        val_dataset = val_datasets[0]
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = SimpleCLIPSegModel()
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Filter out projection layers that might not exist yet
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if k in model_state_dict and model_state_dict[k].shape == v.shape}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Diagnose
    best_thresh, best_iou, best_dice = diagnose_model(model, val_loader, device)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Optimal threshold: {best_thresh:.1f}")
    print(f"mIoU: {best_iou:.4f}")
    print(f"Dice: {best_dice:.4f}")
    
    return best_thresh, best_iou, best_dice

if __name__ == '__main__':
    main()

