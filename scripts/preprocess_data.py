"""
Preprocess datasets: convert annotations to binary masks, create splits, and map prompts.
"""
import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def convert_yolo_to_mask(yolo_file, img_width, img_height, class_id=0, process_all_classes=False):
    """
    Convert YOLO format annotation to binary mask.
    
    YOLO format can be:
    1. Bounding box: class_id x_center y_center width height (normalized)
    2. Segmentation: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon coordinates)
    
    Args:
        yolo_file: Path to YOLO annotation file
        img_width: Image width in pixels
        img_height: Image height in pixels
        class_id: Class ID to process (if process_all_classes=False)
        process_all_classes: If True, process all classes; if False, only process class_id
    
    Returns:
        Binary mask (numpy array, uint8, values 0 or 255)
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not os.path.exists(yolo_file):
        return mask
    
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        current_class_id = int(parts[0])
        
        # Filter by class if not processing all classes
        if not process_all_classes and current_class_id != class_id:
            continue
        
        # Check if this is segmentation format (polygon) or bbox format
        # Segmentation: class_id x1 y1 x2 y2 x3 y3 ... (even number of coordinates)
        # Bbox: class_id x_center y_center width height (exactly 5 parts)
        
        if len(parts) > 5:
            # Likely segmentation format (polygon)
            try:
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 == 0 and len(coords) >= 6:  # At least 3 points
                    # Convert normalized polygon to pixel coordinates
                    polygon_points = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * img_width)
                        y = int(coords[i+1] * img_height)
                        polygon_points.append([x, y])
                    
                    # Draw filled polygon
                    if len(polygon_points) >= 3:
                        polygon_points = np.array(polygon_points, dtype=np.int32)
                        cv2.fillPoly(mask, [polygon_points], 255)
                else:
                    # Fall back to bbox if polygon parsing fails
                    raise ValueError("Invalid polygon format")
            except (ValueError, IndexError):
                # Fall back to bbox format
                if len(parts) >= 5:
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = max(0, int(x_center - width / 2))
                    y1 = max(0, int(y_center - height / 2))
                    x2 = min(img_width, int(x_center + width / 2))
                    y2 = min(img_height, int(y_center + height / 2))
                    
                    mask[y1:y2, x1:x2] = 255
        else:
            # Bounding box format: class_id x_center y_center width height
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # Convert to bounding box coordinates with bounds checking
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_width, int(x_center + width / 2))
            y2 = min(img_height, int(y_center + height / 2))
            
            # Draw filled rectangle
            # Note: This creates rectangular masks from bboxes
            # If segmentation data is available, it will be used instead
            mask[y1:y2, x1:x2] = 255
    
    return mask

def process_dataset(dataset_dir, output_dir, dataset_name, prompts, split_ratios=(0.7, 0.15, 0.15), class_id=0):
    """
    Process a dataset: convert annotations to masks and create splits.
    
    Args:
        dataset_dir: Path to raw dataset directory
        output_dir: Path to output processed data
        dataset_name: Name of the dataset
        prompts: List of prompts for this dataset
        split_ratios: (train, val, test) ratios
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}...")
    print(f"{'='*60}")
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)
        (split_dir / "prompts").mkdir(parents=True, exist_ok=True)
    
    # Find images and annotations
    # YOLO format typically has: train/images, train/labels, etc.
    images = []
    annotations = []
    
    # Look for images in common YOLO structure
    # Also handle Roboflow structure which may have version folders
    possible_paths = [
        dataset_dir / "train" / "images",
        dataset_dir / "images",
        dataset_dir,
    ]
    
    # Check for Roboflow version folders (e.g., dataset/v1/train/images)
    for version_dir in dataset_dir.glob("v*"):
        if version_dir.is_dir():
            possible_paths.insert(0, version_dir / "train" / "images")
            possible_paths.insert(0, version_dir / "images")
    
    img_dir = None
    label_dir = None
    
    for path in possible_paths:
        if path.exists():
            img_dir = path
            # Find corresponding labels directory
            label_candidates = [
                path.parent / "labels",
                path.parent / "train" / "labels",
                dataset_dir / "train" / "labels",
                dataset_dir / "labels",
            ]
            for lbl_path in label_candidates:
                if lbl_path.exists():
                    label_dir = lbl_path
                    break
            if label_dir:
                break
    
    if not img_dir or not label_dir:
        print(f"Warning: Could not find standard YOLO structure in {dataset_dir}")
        print("Looking for any image files...")
        # Fallback: find all images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(dataset_dir.rglob(ext)))
    else:
        print(f"Found images in: {img_dir}")
        print(f"Found labels in: {label_dir}")
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + \
                 list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.PNG"))
    
    if not images:
        print(f"✗ No images found in {dataset_dir}")
        return
    
    print(f"Found {len(images)} images")
    
    # Process each image
    processed_data = []
    
    for img_path in tqdm(images, desc="Processing images"):
        # Load image to get dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Find corresponding annotation file
        img_stem = img_path.stem
        label_path = None
        
        if label_dir:
            # Try different possible label file names
            for ext in ['.txt', '.TXT']:
                candidate = label_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    label_path = candidate
                    break
        
        # Create mask
        if label_path and label_path.exists():
            # Process all classes and combine into single binary mask
            # (for segmentation, we typically want all objects as foreground)
            mask = convert_yolo_to_mask(label_path, img_width, img_height, 
                                      class_id=0, process_all_classes=True)
            
            # Validate mask
            if mask.sum() == 0:
                print(f"Warning: Empty mask for {img_path.name} (no valid annotations)")
        else:
            # Create empty mask if no annotation found
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            print(f"Warning: No annotation found for {img_path.name}")
        
        # Store image info
        processed_data.append({
            'image_path': img_path,
            'mask': mask,
            'image_id': img_stem,
            'width': img_width,
            'height': img_height
        })
    
    # Shuffle and split data
    random.shuffle(processed_data)
    n_total = len(processed_data)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    
    train_data = processed_data[:n_train]
    val_data = processed_data[n_train:n_train+n_val]
    test_data = processed_data[n_train+n_val:]
    
    print(f"\nSplit: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Save processed data
    for split_name, split_data, split_output_dir in [
        ("train", train_data, train_dir),
        ("val", val_data, val_dir),
        ("test", test_data, test_dir)
    ]:
        print(f"\nSaving {split_name} split...")
        for item in tqdm(split_data, desc=f"  {split_name}"):
            img_path = item['image_path']
            mask = item['mask']
            image_id = item['image_id']
            
            # Copy image
            dest_img = split_output_dir / "images" / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Save mask for each prompt
            for prompt in prompts:
                # Create prompt-safe filename
                prompt_safe = prompt.replace(" ", "_").replace("/", "_")
                mask_filename = f"{image_id}__{prompt_safe}.png"
                mask_path = split_output_dir / "masks" / mask_filename
                
                # Save mask as PNG with values {0, 255}
                Image.fromarray(mask).save(mask_path)
                
                # Save prompt mapping
                prompt_file = split_output_dir / "prompts" / f"{image_id}.txt"
                with open(prompt_file, 'w') as f:
                    f.write(prompt)
    
    # Create metadata file
    metadata = {
        'dataset_name': dataset_name,
        'prompts': prompts,
        'total_images': n_total,
        'train_count': len(train_data),
        'val_count': len(val_data),
        'test_count': len(test_data),
        'split_ratios': split_ratios
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary statistics
    total_masks = sum([len(list((split_dir / "masks").glob("*.png"))) 
                       for split_dir in [train_dir, val_dir, test_dir]])
    non_empty_masks = 0
    
    print(f"\n✓ Processed {dataset_name}")
    print(f"  Total images: {n_total}")
    print(f"  Total mask files: {total_masks}")
    print(f"  Prompts: {', '.join(prompts)}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"\n  Note: Masks are generated from YOLO annotations.")
    print(f"  If annotations are bounding boxes, masks will be rectangular.")
    print(f"  If annotations include polygon data, precise segmentation will be used.")

def main():
    """Main preprocessing function."""
    set_seed(42)  # Set seed for reproducibility
    
    base_dir = Path(__file__).parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    processed_data_dir = base_dir / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset 1: Drywall-Join-Detect (Taping Area)
    dataset1_dir = raw_data_dir / "drywall-join-detect"
    if dataset1_dir.exists():
        output1_dir = processed_data_dir / "taping_area"
        prompts1 = ["segment taping area", "segment joint/tape", "segment drywall seam"]
        process_dataset(dataset1_dir, output1_dir, "Drywall-Join-Detect", prompts1)
    else:
        print(f"Warning: Dataset 1 not found at {dataset1_dir}")
        print("Please download it first using: python scripts/download_datasets.py")
    
    # Dataset 2: Cracks
    dataset2_dir = raw_data_dir / "cracks"
    if dataset2_dir.exists():
        output2_dir = processed_data_dir / "cracks"
        prompts2 = ["segment crack", "segment wall crack"]
        process_dataset(dataset2_dir, output2_dir, "Cracks", prompts2)
    else:
        print(f"Warning: Dataset 2 not found at {dataset2_dir}")
        print("Please download it first using: python scripts/download_datasets.py")
    
    print("\n" + "="*60)
    print("Data preprocessing completed!")
    print("="*60)

if __name__ == "__main__":
    main()

