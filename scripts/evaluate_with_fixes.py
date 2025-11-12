#!/usr/bin/env python3
"""
Evaluate model with fixes applied (weighted loss, optimal threshold, etc.)
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import TextConditionedSegmentationDataset, CombinedDataset
from src.models.clipseg_model import SimpleCLIPSegModel
from src.evaluation.metrics import evaluate_batch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def evaluate_model(model, val_loader, threshold=0.5, device='cpu'):
    """Evaluate model with given threshold"""
    
    model.eval()
    all_preds = []
    all_masks = []
    
    print(f"Evaluating with threshold={threshold:.2f}...")
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
    
    # Calculate metrics
    metrics = evaluate_batch(all_preds, all_masks, threshold=threshold)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with fixes')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/training_both_datasets/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarization')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    
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
    
    # Evaluate with base threshold (0.5)
    print("\n" + "="*60)
    print("EVALUATION WITH BASE THRESHOLD (0.5)")
    print("="*60)
    base_metrics = evaluate_model(model, val_loader, threshold=0.5, device=device)
    
    print(f"mIoU: {base_metrics['iou']:.4f}")
    print(f"Dice: {base_metrics['dice']:.4f}")
    print(f"Pixel Accuracy: {base_metrics['pixel_accuracy']:.4f}")
    
    # Evaluate with optimal threshold
    print("\n" + "="*60)
    print(f"EVALUATION WITH OPTIMAL THRESHOLD ({args.threshold:.2f})")
    print("="*60)
    optimal_metrics = evaluate_model(model, val_loader, threshold=args.threshold, device=device)
    
    print(f"mIoU: {optimal_metrics['iou']:.4f}")
    print(f"Dice: {optimal_metrics['dice']:.4f}")
    print(f"Pixel Accuracy: {optimal_metrics['pixel_accuracy']:.4f}")
    
    # Save results
    results = {
        'base_threshold_0.5': {
            'threshold': 0.5,
            'iou': float(base_metrics['iou']),
            'dice': float(base_metrics['dice']),
            'pixel_accuracy': float(base_metrics['pixel_accuracy'])
        },
        'optimal_threshold': {
            'threshold': args.threshold,
            'iou': float(optimal_metrics['iou']),
            'dice': float(optimal_metrics['dice']),
            'pixel_accuracy': float(optimal_metrics['pixel_accuracy'])
        },
        'improvement': {
            'iou_improvement': float(optimal_metrics['iou'] - base_metrics['iou']),
            'dice_improvement': float(optimal_metrics['dice'] - base_metrics['dice']),
            'pixel_acc_change': float(optimal_metrics['pixel_accuracy'] - base_metrics['pixel_accuracy'])
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    return results

if __name__ == '__main__':
    main()

