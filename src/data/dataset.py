"""
Dataset classes for text-conditioned segmentation.
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class TextConditionedSegmentationDataset(Dataset):
    """
    Dataset for text-conditioned segmentation.
    Loads images, masks, and text prompts.
    """
    
    def __init__(self, data_dir, transform=None, target_transform=None):
        """
        Args:
            data_dir: Path to dataset directory (should contain images/, masks/, prompts/ subdirs)
            transform: Transform to apply to images
            target_transform: Transform to apply to masks
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Find all images
        images_dir = self.data_dir / "images"
        self.image_paths = []
        self.mask_paths = []
        self.prompts = []
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + \
                     list(images_dir.glob("*.png")) + \
                     list(images_dir.glob("*.JPG")) + \
                     list(images_dir.glob("*.PNG"))
        
        masks_dir = self.data_dir / "masks"
        prompts_dir = self.data_dir / "prompts"
        
        for img_path in image_files:
            image_id = img_path.stem
            
            # Find corresponding mask (may have multiple masks per image for different prompts)
            # Format: {image_id}__{prompt}.png
            mask_files = list(masks_dir.glob(f"{image_id}__*.png"))
            
            if mask_files:
                # If multiple masks exist, create separate entries
                for mask_path in mask_files:
                    # Extract prompt from filename
                    prompt_part = mask_path.stem.replace(f"{image_id}__", "")
                    prompt_part = prompt_part.replace("_", " ")
                    
                    # Try to get prompt from prompts directory
                    prompt_file = prompts_dir / f"{image_id}.txt"
                    if prompt_file.exists():
                        with open(prompt_file, 'r') as f:
                            prompt = f.read().strip()
                    else:
                        # Fallback to filename-based prompt
                        prompt = prompt_part
                    
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.prompts.append(prompt)
            else:
                # If no mask found, still add image with empty prompt
                self.image_paths.append(img_path)
                self.mask_paths.append(None)
                self.prompts.append("")
        
        print(f"Loaded {len(self.image_paths)} image-mask pairs from {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_paths[idx]
        if mask_path and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            # Convert to binary: 0 or 255 -> 0 or 1
            mask = np.array(mask)
            mask = (mask > 127).astype(np.float32)
        else:
            # Create empty mask if not found
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        
        # Get prompt
        prompt = self.prompts[idx]
        
        # Apply transforms
        if self.transform:
            # For albumentations, we need to convert PIL to numpy
            image_np = np.array(image)
            transformed = self.transform(image=image_np, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Ensure mask has channel dimension [H, W] -> [1, H, W]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            # Default: convert to tensor
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            mask = torch.from_numpy(mask)
            # Ensure mask has channel dimension [H, W] -> [1, H, W]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'prompt': prompt,
            'image_id': img_path.stem
        }

class CombinedDataset(Dataset):
    """
    Combines multiple datasets (e.g., taping area + cracks).
    """
    
    def __init__(self, datasets):
        """
        Args:
            datasets: List of TextConditionedSegmentationDataset instances
        """
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
    
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx + 1)
        if dataset_idx > 0:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = idx
        
        return self.datasets[dataset_idx][sample_idx]

