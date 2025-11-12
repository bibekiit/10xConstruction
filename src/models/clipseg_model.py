"""
CLIPSeg-based model for text-conditioned segmentation.
"""
import torch
import torch.nn as nn
from transformers import CLIPSegProcessor, CLIPSegModel
from PIL import Image
import numpy as np

class CLIPSegSegmentationModel(nn.Module):
    """
    Text-conditioned segmentation model based on CLIPSeg.
    """
    
    def __init__(self, model_name="CIDAS/clipseg-rd64-refined", freeze_clip=False):
        """
        Args:
            model_name: HuggingFace model name for CLIPSeg
            freeze_clip: Whether to freeze CLIP encoder weights
        """
        super().__init__()
        
        # Load CLIPSeg model
        self.model = CLIPSegModel.from_pretrained(model_name)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        
        # Freeze CLIP if requested
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Optional: Add additional decoder layers for refinement
        # CLIPSeg already has a decoder, but we can add more layers
        self.refinement = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        """
        Forward pass.
        
        Args:
            pixel_values: Preprocessed images (from processor)
            input_ids: Tokenized text prompts
            attention_mask: Attention mask for text
        """
        # Get CLIPSeg outputs
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # CLIPSeg returns logits of shape [batch_size, num_patches, hidden_size]
        # We need to reshape to spatial dimensions
        logits = outputs.logits
        
        # Reshape logits to spatial dimensions
        # CLIPSeg uses patch-based approach, need to reshape
        batch_size = logits.shape[0]
        # Assuming square patches, calculate spatial dims
        num_patches = logits.shape[1]
        spatial_size = int(np.sqrt(num_patches))
        
        if spatial_size * spatial_size == num_patches:
            # Reshape to spatial
            logits = logits.view(batch_size, spatial_size, spatial_size, -1)
            # Take mean over hidden dimension or use a projection
            logits = logits.mean(dim=-1)  # [B, H, W]
            logits = logits.unsqueeze(1)  # [B, 1, H, W]
        else:
            # Fallback: use first hidden dimension
            logits = logits[:, :, 0].view(batch_size, spatial_size, spatial_size)
            logits = logits.unsqueeze(1)
        
        # Apply refinement
        refined = self.refinement(logits)
        
        return refined
    
    def predict(self, image, prompt, device='cuda'):
        """
        Predict segmentation mask for a single image and prompt.
        
        Args:
            image: PIL Image or numpy array
            prompt: Text prompt string
            device: Device to run inference on
        
        Returns:
            Binary mask as numpy array (0 or 1)
        """
        self.eval()
        
        # Process inputs
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
            mask = outputs[0, 0].cpu().numpy()  # Get first (and only) mask
        
        # Threshold to binary
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        return mask

class SimpleCLIPSegModel(nn.Module):
    """
    Simplified version that works directly with image tensors and text prompts.
    More flexible for training with proper text conditioning.
    """
    
    def __init__(self, model_name="CIDAS/clipseg-rd64-refined"):
        super().__init__()
        self.clip_model = CLIPSegModel.from_pretrained(model_name)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        
        # Get the hidden size from the model config
        # CLIPSeg uses reduce_dim for the decoder output dimension
        hidden_size = self.clip_model.config.reduce_dim
        
        # Decoder head to convert CLIPSeg output to segmentation mask
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images, prompts=None, input_ids=None, attention_mask=None):
        """
        Forward pass.
        
        Args:
            images: Image tensors [B, C, H, W] (normalized, already preprocessed)
            prompts: List of text prompt strings (optional, for training)
            input_ids: Pre-tokenized text input_ids [B, seq_len]
            attention_mask: Attention mask for text [B, seq_len]
        """
        batch_size = images.shape[0]
        original_size = images.shape[2:]
        
        # If prompts are provided, tokenize them
        if prompts is not None and input_ids is None:
            # Tokenize prompts
            text_inputs = self.processor.tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=77  # CLIP max length
            )
            input_ids = text_inputs['input_ids'].to(images.device)
            attention_mask = text_inputs['attention_mask'].to(images.device)
        
        # Resize images to CLIPSeg input size (typically 352x352 for refined model)
        target_size = 352
        images_resized = nn.functional.interpolate(
            images, size=(target_size, target_size), mode='bilinear', align_corners=False
        )
        
        # Get CLIPSeg outputs
        # CLIPSeg expects pixel_values in specific format, so we need to ensure proper preprocessing
        # For training, we assume images are already normalized to [-1, 1] or [0, 1]
        # CLIPSeg expects [0, 1] range, so if images are in [-1, 1], convert them
        if images_resized.min() < 0:
            # Assume ImageNet normalization, convert back to [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            images_resized = images_resized * std + mean
            images_resized = torch.clamp(images_resized, 0, 1)
        
        # Forward through CLIPSeg
        if input_ids is not None:
            outputs = self.clip_model(
                pixel_values=images_resized,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            # If no text provided, use a default prompt
            default_prompt = "a photo"
            text_inputs = self.processor.tokenizer(
                [default_prompt] * batch_size,
                padding=True,
                return_tensors="pt",
                truncation=True
            )
            input_ids = text_inputs['input_ids'].to(images.device)
            attention_mask = text_inputs['attention_mask'].to(images.device)
            outputs = self.clip_model(
                pixel_values=images_resized,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # CLIPSeg returns image_embeds and text_embeds
        # We need to use the decoder to get segmentation logits
        # For now, use vision model outputs and process them
        # Get vision features from the output
        if hasattr(outputs, 'vision_model_output'):
            vision_features = outputs.vision_model_output.last_hidden_state
        elif hasattr(outputs, 'image_embeds'):
            # Use image embeddings directly
            vision_features = outputs.image_embeds
        else:
            # Fallback: use vision model directly
            vision_outputs = self.clip_model.vision_model(pixel_values=images_resized)
            vision_features = vision_outputs.last_hidden_state
        
        # Get text embeddings
        if hasattr(outputs, 'text_model_output'):
            text_features = outputs.text_model_output.last_hidden_state
        elif hasattr(outputs, 'text_embeds'):
            text_features = outputs.text_embeds.unsqueeze(1)  # Add sequence dimension
        else:
            # Fallback: use text model directly
            text_outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state
        
        # Combine vision and text features
        # Vision features: [B, num_patches, vision_hidden_size (768)]
        # Text features: [B, seq_len, text_hidden_size (512)] -> take mean or [CLS] token
        text_embed = text_features.mean(dim=1)  # [B, text_hidden_size]
        
        # Project both to same dimension before combining
        hidden_size = self.clip_model.config.reduce_dim  # 64 for refined model
        
        # Create projection layers if they don't exist
        if not hasattr(self, 'vision_proj'):
            vision_hidden = vision_features.shape[-1]
            self.vision_proj = nn.Linear(vision_hidden, hidden_size)
        if not hasattr(self, 'text_proj'):
            text_hidden = text_embed.shape[-1]
            self.text_proj = nn.Linear(text_hidden, hidden_size)
        
        # Project features
        vision_proj = self.vision_proj(vision_features)  # [B, num_patches, hidden_size]
        text_proj = self.text_proj(text_embed)  # [B, hidden_size]
        
        # Expand text embedding to match vision patches
        num_patches = vision_proj.shape[1]
        text_proj_expanded = text_proj.unsqueeze(1).expand(-1, num_patches, -1)  # [B, num_patches, hidden_size]
        
        # Combine features (element-wise multiplication for attention-like mechanism)
        combined_features = vision_proj * text_proj_expanded  # [B, num_patches, hidden_size]
        
        logits = combined_features
        
        # Reshape to spatial dimensions
        # CLIPSeg uses patch-based approach
        # For refined model, typically 22x22 patches for 352x352 input
        # But actual patch count may vary
        num_patches = logits.shape[1]
        hidden_dim = logits.shape[2]
        
        # Calculate spatial dimensions (CLIPSeg vision model output)
        # For 352x352 input with patch size 16, we get 22x22 = 484 patches
        # But actual output may have different number
        patch_size = int(np.sqrt(num_patches))
        
        if patch_size * patch_size == num_patches:
            # Perfect square: reshape to [B, H, W, hidden_size]
            logits = logits.view(batch_size, patch_size, patch_size, hidden_dim)
            # Permute to [B, hidden_size, H, W]
            logits = logits.permute(0, 3, 1, 2)
        else:
            # Not a perfect square: use approximate dimensions
            # Find dimensions that work
            h = int(np.sqrt(num_patches))
            w = num_patches // h
            
            # Only reshape if dimensions match exactly
            if h * w == num_patches:
                logits = logits.view(batch_size, h, w, hidden_dim)
                logits = logits.permute(0, 3, 1, 2)
            else:
                # Fallback: use 1D to 2D interpolation
                # Reshape to [B, hidden_size, num_patches]
                logits = logits.permute(0, 2, 1)  # [B, hidden_dim, num_patches]
                # Reshape to approximate 2D
                h = int(np.sqrt(num_patches))
                w = (num_patches + h - 1) // h
                # Pad or crop to fit
                if h * w > num_patches:
                    # Pad
                    padding = h * w - num_patches
                    logits = nn.functional.pad(logits, (0, padding), mode='constant', value=0)
                logits = logits.view(batch_size, hidden_dim, h, w)
        
        # Decode to segmentation mask
        # The decoder will upsample, so we don't need to upsample to full size first
        # Just ensure we have reasonable spatial dimensions
        mask = self.decoder(logits)
        
        # Final resize to original image size if needed
        if mask.shape[2:] != original_size:
            mask = nn.functional.interpolate(
                mask, size=original_size, mode='bilinear', align_corners=False
            )
        
        return mask

