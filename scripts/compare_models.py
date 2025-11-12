"""
Compare different model checkpoints and generate comparison report.
"""
import json
import argparse
from pathlib import Path
import pandas as pd

def load_training_summary(summary_path):
    """Load training summary from JSON file."""
    with open(summary_path, 'r') as f:
        return json.load(f)

def compare_models(experiment_dirs, output_file="model_comparison.csv"):
    """
    Compare multiple model training results.
    
    Args:
        experiment_dirs: List of experiment output directories
        output_file: Path to save comparison CSV
    """
    results = []
    
    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        summary_path = exp_dir / "training_summary.json"
        
        if not summary_path.exists():
            print(f"Warning: No summary found in {exp_dir}")
            continue
        
        summary = load_training_summary(summary_path)
        
        # Extract key metrics
        result = {
            'experiment': exp_dir.name,
            'best_val_iou': summary.get('best_val_iou', 0),
            'best_val_dice': summary.get('best_val_dice', 0),
            'total_epochs': summary.get('total_epochs', 0),
        }
        
        # Add training args if available
        if 'training_args' in summary:
            args = summary['training_args']
            result.update({
                'batch_size': args.get('batch_size', 'N/A'),
                'learning_rate': args.get('lr', 'N/A'),
                'image_size': args.get('image_size', 'N/A'),
                'mixed_precision': args.get('mixed_precision', False),
            })
        
        # Add best epoch info
        if 'best_metrics' in summary and 'val_iou' in summary['best_metrics']:
            result['best_epoch'] = summary['best_metrics']['val_iou'].get('epoch', 'N/A')
        
        results.append(result)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values('best_val_iou', ascending=False)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nComparison saved to: {output_file}")
    print("\nTop models by IoU:")
    print(df.head(10).to_string(index=False))
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Compare model training results')
    parser.add_argument('experiment_dirs', nargs='+',
                        help='Paths to experiment output directories')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # If directory provided, find all experiments
    if len(args.experiment_dirs) == 1:
        base_dir = Path(args.experiment_dirs[0])
        if base_dir.is_dir():
            # Find all subdirectories with training_summary.json
            experiment_dirs = [d for d in base_dir.iterdir() 
                             if d.is_dir() and (d / "training_summary.json").exists()]
            if experiment_dirs:
                args.experiment_dirs = [str(d) for d in experiment_dirs]
    
    compare_models(args.experiment_dirs, args.output)

if __name__ == "__main__":
    main()

