"""
Training script for text-conditioned segmentation model.
"""
import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import TextConditionedSegmentationDataset, CombinedDataset
from src.models.clipseg_model import CLIPSegSegmentationModel, SimpleCLIPSegModel
from src.training.losses import CombinedLoss, DiceLoss
from src.evaluation.metrics import calculate_miou, calculate_mean_dice, evaluate_batch
from src.training.trainer import EarlyStopping, TrainingTracker, evaluate_per_prompt, save_checkpoint, load_checkpoint

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_data_loaders(data_dir, batch_size=8, num_workers=4, image_size=512):
    """Create data loaders for training and validation."""
    from albumentations import (
        Compose, Normalize, Resize, HorizontalFlip, VerticalFlip,
        RandomBrightnessContrast, ShiftScaleRotate
    )
    from albumentations.pytorch import ToTensorV2
    
    # Training transforms
    train_transform = Compose([
        Resize(image_size, image_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation transforms
    val_transform = Compose([
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load datasets
    train_dirs = []
    val_dirs = []
    
    # Check for taping area dataset
    taping_train = data_dir / "taping_area" / "train"
    taping_val = data_dir / "taping_area" / "val"
    if taping_train.exists():
        train_dirs.append(taping_train)
    if taping_val.exists():
        val_dirs.append(taping_val)
    
    # Check for cracks dataset
    cracks_train = data_dir / "cracks" / "train"
    cracks_val = data_dir / "cracks" / "val"
    if cracks_train.exists():
        train_dirs.append(cracks_train)
    if cracks_val.exists():
        val_dirs.append(cracks_val)
    
    if not train_dirs:
        raise ValueError(f"No training data found in {data_dir}")
    
    # Create datasets
    train_datasets = [TextConditionedSegmentationDataset(d, transform=train_transform) 
                      for d in train_dirs]
    val_datasets = [TextConditionedSegmentationDataset(d, transform=val_transform) 
                    for d in val_dirs]
    
    if len(train_datasets) > 1:
        train_dataset = CombinedDataset(train_datasets)
    else:
        train_dataset = train_datasets[0]
    
    if len(val_datasets) > 1:
        val_dataset = CombinedDataset(val_datasets)
    else:
        val_dataset = val_datasets[0] if val_datasets else train_dataset
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, 
                scaler=None, gradient_clip=0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        prompts = batch['prompt']  # List of prompt strings
        
        # Forward pass
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images, prompts=prompts)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(images, prompts=prompts)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def validate(model, val_loader, criterion, device, epoch, writer):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']  # List of prompt strings
            
            # Forward pass
            outputs = model(images, prompts=prompts)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Store for metrics
            all_predictions.append(outputs)
            all_targets.append(masks)
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = evaluate_batch(all_predictions, all_targets)
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Val/Loss', avg_loss, epoch)
            writer.add_scalar('Val/IoU', metrics['iou'], epoch)
            writer.add_scalar('Val/Dice', metrics['dice'], epoch)
            writer.add_scalar('Val/PixelAccuracy', metrics['pixel_accuracy'], epoch)
        
        return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser(description='Train text-conditioned segmentation model')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for training')
    parser.add_argument('--model_name', type=str, default='CIDAS/clipseg-rd64-refined',
                        help='CLIPSeg model name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--eval_per_prompt', action='store_true',
                        help='Evaluate metrics per prompt')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir / "logs")
    
    # Create data loaders
    data_dir = Path(args.data_dir)
    train_loader, val_loader = create_data_loaders(
        data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Create model
    print("Creating model...")
    model = SimpleCLIPSegModel(model_name=args.model_name)
    model = model.to(device)
    
    # Create loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision training
    scaler = None
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")
    
    # Training tracker and early stopping
    tracker = TrainingTracker(output_dir)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max',
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, device)
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    best_val_iou = 0.0
    best_val_dice = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Device: {device}")
    if args.mixed_precision:
        print("Mixed precision: Enabled")
    if args.gradient_clip > 0:
        print(f"Gradient clipping: {args.gradient_clip}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, 
                                scaler=scaler, gradient_clip=args.gradient_clip)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Per-prompt evaluation (optional, can be slow)
        if args.eval_per_prompt and epoch % 5 == 0:  # Every 5 epochs
            print("\nEvaluating per prompt...")
            prompt_metrics = evaluate_per_prompt(model, val_loader, device)
            for prompt, metrics in prompt_metrics.items():
                print(f"  {prompt}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
                writer.add_scalar(f'PerPrompt/{prompt}/IoU', metrics['iou'], epoch)
                writer.add_scalar(f'PerPrompt/{prompt}/Dice', metrics['dice'], epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track metrics
        epoch_metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_metrics['iou'],
            'val_dice': val_metrics['dice'],
            'val_pixel_accuracy': val_metrics['pixel_accuracy']
        }
        tracker.log_epoch(epoch, epoch_metrics)
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}")
        print(f"  Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        is_best = val_metrics['iou'] > best_val_iou
        if is_best:
            best_val_iou = val_metrics['iou']
            best_val_dice = val_metrics['dice']
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, epoch_metrics, checkpoint_dir, is_best=is_best)
        if is_best:
            print(f"  ✓ Saved best model (IoU: {best_val_iou:.4f}, Dice: {best_val_dice:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['iou']):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")
    
    # Save training summary using tracker
    summary = tracker.save_summary()
    summary['training_args'] = vars(args)
    summary['best_val_iou'] = best_val_iou
    summary['best_val_dice'] = best_val_dice
    
    with open(output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best Val IoU: {best_val_iou:.4f} (epoch {tracker.get_best_epoch('val_iou')})")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Total epochs trained: {epoch}")
    print(f"Summary saved to: {output_dir / 'training_summary.json'}")
    print(f"{'='*60}")
    
    writer.close()

if __name__ == "__main__":
    main()

