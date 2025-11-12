"""
Enhanced training utilities for Phase 3.
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"  Early stopping: Improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  Early stopping: No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  Early stopping triggered!")
        
        return self.early_stop
    
    def _is_better(self, current, best):
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

class TrainingTracker:
    """Track training metrics and experiments."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        
    def log_epoch(self, epoch, metrics):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
            # Track best metrics
            if key not in self.best_metrics:
                self.best_metrics[key] = {'value': value, 'epoch': epoch}
            else:
                if 'iou' in key.lower() or 'dice' in key.lower():
                    # Higher is better
                    if value > self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': epoch}
                elif 'loss' in key.lower():
                    # Lower is better
                    if value < self.best_metrics[key]['value']:
                        self.best_metrics[key] = {'value': value, 'epoch': epoch}
    
    def save_summary(self):
        """Save training summary."""
        summary = {
            'best_metrics': self.best_metrics,
            'final_metrics': {k: v[-1] if v else None 
                            for k, v in self.metrics_history.items()},
            'total_epochs': len(self.metrics_history.get('train_loss', []))
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_best_epoch(self, metric_name='val_iou'):
        """Get epoch with best metric."""
        if metric_name in self.best_metrics:
            return self.best_metrics[metric_name]['epoch']
        return None

def evaluate_per_prompt(model, val_loader, device, prompt_groups=None):
    """
    Evaluate model performance per prompt.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        prompt_groups: Optional dict mapping prompt groups to prompt strings
    
    Returns:
        Dictionary with metrics per prompt
    """
    model.eval()
    prompt_metrics = defaultdict(lambda: {'predictions': [], 'targets': []})
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating per prompt"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']
            
            outputs = model(images, prompts=prompts)
            
            # Group by prompt
            for i, prompt in enumerate(prompts):
                prompt_metrics[prompt]['predictions'].append(outputs[i:i+1])
                prompt_metrics[prompt]['targets'].append(masks[i:i+1])
    
    # Calculate metrics per prompt
    from src.evaluation.metrics import evaluate_batch
    
    results = {}
    for prompt, data in prompt_metrics.items():
        if data['predictions']:
            preds = torch.cat(data['predictions'], dim=0)
            targets = torch.cat(data['targets'], dim=0)
            metrics = evaluate_batch(preds, targets)
            results[prompt] = metrics
    
    return results

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    metrics = checkpoint.get('metrics', {})
    
    return start_epoch, metrics

def log_model_parameters(writer, model, epoch):
    """Log model parameters to TensorBoard."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)
            writer.add_histogram(f'parameters/{name}', param.data, epoch)

