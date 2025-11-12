"""
Test script to validate the training pipeline with synthetic data.
Creates a minimal test dataset and runs a short training session.
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_synthetic_dataset(output_dir, num_images=10, image_size=(512, 512)):
    """
    Create a minimal synthetic dataset for testing.
    
    Args:
        output_dir: Directory to create test dataset
        num_images: Number of synthetic images to create
        image_size: Size of synthetic images
    """
    output_dir = Path(output_dir)
    
    # Create directory structure
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for split_dir in [train_dir, val_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)
        (split_dir / "prompts").mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images and masks
    print(f"Creating {num_images} synthetic images...")
    
    for i in range(num_images):
        # Create random image
        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Create synthetic mask (random rectangular region)
        mask = np.zeros(image_size, dtype=np.uint8)
        x1, y1 = np.random.randint(0, image_size[0]//2, 2)
        x2, y2 = np.random.randint(image_size[0]//2, image_size[0], 2)
        mask[y1:y2, x1:x2] = 255
        
        # Determine split (80% train, 20% val)
        split_dir = train_dir if i < int(num_images * 0.8) else val_dir
        
        # Save image
        image_id = f"test_image_{i:03d}"
        img_path = split_dir / "images" / f"{image_id}.png"
        img.save(img_path)
        
        # Save mask with prompt
        prompt = "segment crack" if i % 2 == 0 else "segment taping area"
        prompt_safe = prompt.replace(" ", "_")
        mask_filename = f"{image_id}__{prompt_safe}.png"
        mask_path = split_dir / "masks" / mask_filename
        Image.fromarray(mask).save(mask_path)
        
        # Save prompt
        prompt_file = split_dir / "prompts" / f"{image_id}.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)
    
    print(f"✓ Created test dataset in {output_dir}")
    print(f"  Train images: {len(list((train_dir / 'images').glob('*.png')))}")
    print(f"  Val images: {len(list((val_dir / 'images').glob('*.png')))}")

def test_data_loading():
    """Test if data loading works."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        from src.data.dataset import TextConditionedSegmentationDataset
        from torch.utils.data import DataLoader
        
        test_data_dir = Path("data/test_dataset/train")
        if not test_data_dir.exists():
            print("✗ Test dataset not found. Creating...")
            create_synthetic_dataset("data/test_dataset", num_images=10)
        
        # Test dataset loading
        dataset = TextConditionedSegmentationDataset(test_data_dir)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test data loader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        print(f"✓ DataLoader works")
        print(f"  Batch image shape: {batch['image'].shape}")
        print(f"  Batch mask shape: {batch['mask'].shape}")
        print(f"  Prompts: {batch['prompt']}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test if model can be created."""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)
    
    try:
        import torch
        from src.models.clipseg_model import SimpleCLIPSegModel
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model (this will download CLIPSeg if not cached)
        print("Creating model (this may download CLIPSeg model on first run)...")
        model = SimpleCLIPSegModel()
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model created successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 2
        test_image = torch.randn(batch_size, 3, 512, 512).to(device)
        test_prompts = ["segment crack", "segment taping area"]
        
        with torch.no_grad():
            output = model(test_image, prompts=test_prompts)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("Testing Training Step")
    print("="*60)
    
    try:
        import torch
        from src.data.dataset import TextConditionedSegmentationDataset
        from src.models.clipseg_model import SimpleCLIPSegModel
        from src.training.losses import CombinedLoss
        from torch.utils.data import DataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test dataset
        test_data_dir = Path("data/test_dataset/train")
        if not test_data_dir.exists():
            create_synthetic_dataset("data/test_dataset", num_images=10)
        
        # Create data loader
        dataset = TextConditionedSegmentationDataset(test_data_dir)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create model
        model = SimpleCLIPSegModel()
        model = model.to(device)
        model.train()
        
        # Create loss and optimizer
        criterion = CombinedLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Get a batch
        batch = next(iter(loader))
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        prompts = batch['prompt']
        
        # Training step
        optimizer.zero_grad()
        outputs = model(images, prompts=prompts)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Gradients computed: {any(p.grad is not None for p in model.parameters())}")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_training():
    """Run a mini training session (1-2 epochs)."""
    print("\n" + "="*60)
    print("Testing Mini Training Session")
    print("="*60)
    
    try:
        # Create test dataset if needed
        test_data_dir = Path("data/test_dataset")
        if not (test_data_dir / "train").exists():
            create_synthetic_dataset("data/test_dataset", num_images=20)
        
        # Import training components
        import torch
        from src.data.dataset import TextConditionedSegmentationDataset, CombinedDataset
        from src.models.clipseg_model import SimpleCLIPSegModel
        from src.training.losses import CombinedLoss
        from src.evaluation.metrics import evaluate_batch
        from torch.utils.data import DataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create datasets
        train_dataset = TextConditionedSegmentationDataset(test_data_dir / "train")
        val_dataset = TextConditionedSegmentationDataset(test_data_dir / "val")
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Create model
        print("Creating model...")
        model = SimpleCLIPSegModel()
        model = model.to(device)
        
        # Create loss and optimizer
        criterion = CombinedLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train for 2 epochs
        num_epochs = 2
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                prompts = batch['prompt']
                
                optimizer.zero_grad()
                outputs = model(images, prompts=prompts)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    prompts = batch['prompt']
                    
                    outputs = model(images, prompts=prompts)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    all_preds.append(outputs)
                    all_targets.append(masks)
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate metrics
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metrics = evaluate_batch(all_preds, all_targets)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val IoU: {metrics['iou']:.4f}")
            print(f"  Val Dice: {metrics['dice']:.4f}")
        
        print(f"\n✓ Mini training completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Mini training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Training Pipeline Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Data Loading
    results['data_loading'] = test_data_loading()
    
    # Test 2: Model Creation
    results['model_creation'] = test_model_creation()
    
    # Test 3: Training Step
    if results['data_loading'] and results['model_creation']:
        results['training_step'] = test_training_step()
    else:
        print("\n⚠ Skipping training step test (prerequisites failed)")
        results['training_step'] = False
    
    # Test 4: Mini Training
    if all([results['data_loading'], results['model_creation'], results['training_step']]):
        results['mini_training'] = test_mini_training()
    else:
        print("\n⚠ Skipping mini training test (prerequisites failed)")
        results['mini_training'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("✓ All tests passed! Training pipeline is ready.")
    else:
        print("✗ Some tests failed. Please check errors above.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

