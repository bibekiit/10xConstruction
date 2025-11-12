"""
Script to run multiple training experiments with different configurations.
Useful for Phase 3: Model Iteration and hyperparameter tuning.
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

def run_experiment(config, experiment_name, base_output_dir="outputs/experiments"):
    """
    Run a single training experiment.
    
    Args:
        config: Dictionary with training configuration
        experiment_name: Name for this experiment
        base_output_dir: Base directory for experiment outputs
    """
    output_dir = Path(base_output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build command
    cmd = ["python", "scripts/train.py"]
    
    # Add arguments from config
    for key, value in config.items():
        if key == 'experiment_name':
            continue
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Override output directory
    cmd.extend(["--output_dir", str(output_dir)])
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run training
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print(f"\n✓ Experiment {experiment_name} completed successfully")
    else:
        print(f"\n✗ Experiment {experiment_name} failed with code {result.returncode}")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Run training experiments')
    parser.add_argument('--config_file', type=str, default=None,
                        help='JSON file with experiment configurations')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run specific experiment by name')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    
    args = parser.parse_args()
    
    # Default experiments
    experiments = {
        'baseline': {
            'batch_size': 8,
            'epochs': 50,
            'lr': 1e-4,
            'image_size': 512,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0
        },
        'baseline_mixed_precision': {
            'batch_size': 8,
            'epochs': 50,
            'lr': 1e-4,
            'image_size': 512,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0,
            'mixed_precision': True
        },
        'higher_lr': {
            'batch_size': 8,
            'epochs': 50,
            'lr': 5e-4,
            'image_size': 512,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0
        },
        'lower_lr': {
            'batch_size': 8,
            'epochs': 50,
            'lr': 5e-5,
            'image_size': 512,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0
        },
        'larger_batch': {
            'batch_size': 16,
            'epochs': 50,
            'lr': 1e-4,
            'image_size': 512,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0
        },
        'smaller_image': {
            'batch_size': 8,
            'epochs': 50,
            'lr': 1e-4,
            'image_size': 256,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0
        },
        'with_per_prompt_eval': {
            'batch_size': 8,
            'epochs': 50,
            'lr': 1e-4,
            'image_size': 512,
            'early_stopping_patience': 10,
            'gradient_clip': 1.0,
            'eval_per_prompt': True
        }
    }
    
    # Load from file if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            experiments = json.load(f)
    
    if args.list:
        print("Available experiments:")
        for name in experiments.keys():
            print(f"  - {name}")
        return
    
    # Run experiments
    if args.experiment:
        # Run single experiment
        if args.experiment not in experiments:
            print(f"Error: Experiment '{args.experiment}' not found")
            print(f"Available: {list(experiments.keys())}")
            return
        run_experiment(experiments[args.experiment], args.experiment)
    else:
        # Run all experiments
        print(f"Running {len(experiments)} experiments...")
        results = {}
        for name, config in experiments.items():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{name}_{timestamp}"
            success = run_experiment(config, exp_name)
            results[name] = success
        
        # Summary
        print(f"\n{'='*60}")
        print("Experiment Summary")
        print(f"{'='*60}")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {name}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

