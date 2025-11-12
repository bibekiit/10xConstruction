# Prompted Segmentation for Drywall QA - Project Plan

## Project Overview

**Goal**: Train/fine-tune a text-conditioned segmentation model that can produce binary masks for:
- "segment crack" (Dataset 2: Cracks)
- "segment taping area" (Dataset 1: Drywall-Join-Detect)

## Datasets

### Dataset 1: Drywall-Join-Detect (Taping Area)
- **Source**: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect
- **Prompts**: "segment taping area", "segment joint/tape", "segment drywall seam"

### Dataset 2: Cracks
- **Source**: https://universe.roboflow.com/10xConstruction/cracks-3ii36
- **Prompts**: "segment crack", "segment wall crack"

## Implementation Plan

### Phase 1: Environment Setup & Data Preparation (Days 1-2)

#### 1.1 Environment Setup
- [ ] Create Python virtual environment
- [ ] Install dependencies:
  - PyTorch / TensorFlow
  - Segmentation model libraries (SAM, CLIPSeg, or similar)
  - Data processing libraries (albumentations, opencv-python)
  - Evaluation metrics (segmentation-metrics)
  - Visualization tools (matplotlib, seaborn)
  - Roboflow SDK for dataset download

#### 1.2 Data Acquisition
- [ ] Download Dataset 1 (Drywall-Join-Detect) from Roboflow
- [ ] Download Dataset 2 (Cracks) from Roboflow
- [ ] Verify dataset structure and annotations
- [ ] Check image formats, resolutions, and annotation types

#### 1.3 Data Preprocessing
- [ ] Convert annotations to binary masks (if needed)
- [ ] Standardize image sizes (resize/crop strategy)
- [ ] Create train/val/test splits (e.g., 70/15/15 or 80/10/10)
- [ ] Create prompt-to-dataset mapping:
  - Dataset 1 → ["segment taping area", "segment joint/tape", "segment drywall seam"]
  - Dataset 2 → ["segment crack", "segment wall crack"]
- [ ] Augment data (rotations, flips, brightness, contrast) if needed
- [ ] Create data loaders with prompt integration

### Phase 2: Model Selection & Baseline (Days 3-5)

#### 2.1 Model Research & Selection
- [ ] Research text-conditioned segmentation models:
  - **CLIPSeg** (CLIP + segmentation head) - Recommended starting point
  - **SAM (Segment Anything Model)** with text prompts
  - **GroupViT** for semantic segmentation
  - **Custom architecture**: CLIP encoder + U-Net/DeepLab decoder
- [ ] Select 2-3 models to experiment with
- [ ] Document model architectures and capabilities

#### 2.2 Baseline Implementation
- [ ] Implement baseline model (start with CLIPSeg or similar)
- [ ] Create model wrapper for training/inference
- [ ] Implement loss functions:
  - Binary Cross-Entropy Loss
  - Dice Loss (for better segmentation)
  - Combined BCE + Dice Loss
- [ ] Set up training loop with validation
- [ ] Implement early stopping and model checkpointing

### Phase 3: Training & Fine-tuning (Days 6-10)

#### 3.1 Training Configuration
- [ ] Set random seeds for reproducibility
- [ ] Configure hyperparameters:
  - Learning rate (start with 1e-4 to 1e-5)
  - Batch size (based on GPU memory)
  - Number of epochs
  - Optimizer (Adam/AdamW)
  - Learning rate scheduler
- [ ] Set up logging (TensorBoard or WandB)

#### 3.2 Training Process
- [ ] Train on Dataset 1 (taping area)
- [ ] Train on Dataset 2 (cracks)
- [ ] Experiment with:
  - Multi-task learning (both datasets together)
  - Transfer learning (pre-train on one, fine-tune on other)
  - Prompt engineering (test different prompt variations)
- [ ] Monitor training metrics (loss, mIoU, Dice)
- [ ] Save best models based on validation performance

#### 3.3 Model Iteration
- [ ] Try different model architectures if baseline underperforms
- [ ] Experiment with different prompt formats
- [ ] Adjust data augmentation strategies
- [ ] Fine-tune hyperparameters based on validation results

### Phase 4: Evaluation & Metrics (Days 11-12)

#### 4.1 Metric Implementation
- [ ] Implement mIoU (mean Intersection over Union) calculation
- [ ] Implement Dice coefficient calculation
- [ ] Calculate metrics for:
  - Each prompt separately
  - Overall performance
  - Per-dataset performance
- [ ] Create confusion matrices if needed

#### 4.2 Testing
- [ ] Evaluate on test set for both prompts
- [ ] Test consistency across varied scenes
- [ ] Document failure cases
- [ ] Measure inference time per image
- [ ] Calculate model size (parameters, disk size)

### Phase 5: Prediction & Output Generation (Day 13)

#### 5.1 Prediction Pipeline
- [ ] Create inference script
- [ ] Generate predictions for test images
- [ ] Ensure output format:
  - PNG format
  - Single-channel
  - Same spatial size as source image
  - Values {0, 255} (binary mask)
- [ ] Implement filename convention: `{image_id}__{prompt}.png`
  - Example: `123__segment_crack.png`, `456__segment_taping_area.png`

#### 5.2 Visualization
- [ ] Create visualization script
- [ ] Generate 3-4 visual examples showing:
  - Original image
  - Ground truth mask
  - Predicted mask
  - Overlay visualization
- [ ] Save visualization images for report

### Phase 6: Documentation & Reporting (Days 14-15)

#### 6.1 README Creation
- [ ] Create comprehensive README.md with:
  - Project description
  - Installation instructions
  - Dataset download instructions
  - Training instructions
  - Inference instructions
  - Model architecture details
  - Random seeds used (for reproducibility)
  - Dependencies list

#### 6.2 Report Generation
- [ ] Create detailed report including:
  - **Approach**: Model selection rationale, training strategy
  - **Models Tried**: List all models/architectures experimented with
  - **Goal Summary**: Brief recap of objectives
  - **Data Split Counts**: Train/val/test counts for each dataset
  - **Metrics Table**: 
    - mIoU for each prompt
    - Dice coefficient for each prompt
    - Overall metrics
  - **Visual Examples**: 3-4 examples (Original | GT | Prediction)
  - **Failure Notes**: Brief analysis of failure cases
  - **Runtime & Footprint**:
    - Training time
    - Average inference time per image
    - Model size (MB/GB)
  - **Tables**: Performance metrics in tabular format
  - **Visuals**: Charts, graphs, example images

## Technical Considerations

### Model Architecture Options

1. **CLIPSeg** (Recommended)
   - Pre-trained CLIP for text-image understanding
   - Lightweight segmentation decoder
   - Good for text-conditioned tasks

2. **SAM + CLIP**
   - Use CLIP to generate text embeddings
   - Use SAM for segmentation with text guidance
   - More complex but potentially more accurate

3. **Custom Architecture**
   - CLIP text encoder
   - CLIP image encoder
   - Cross-attention mechanism
   - U-Net or DeepLab decoder

### Evaluation Metrics

- **mIoU (mean Intersection over Union)**: Primary metric for correctness
- **Dice Coefficient**: Secondary metric, good for binary segmentation
- **Pixel Accuracy**: Additional metric for reference

### Key Requirements Checklist

- [x] Text-conditioned segmentation
- [ ] Support for "segment crack" prompt
- [ ] Support for "segment taping area" prompt
- [ ] Binary mask output (PNG, single-channel, {0,255})
- [ ] Correct filename format: `{image_id}__{prompt}.png`
- [ ] mIoU and Dice metrics on both prompts
- [ ] Consistent performance across varied scenes
- [ ] Comprehensive README with seeds
- [ ] Detailed report with tables and visuals

## Timeline Summary

- **Days 1-2**: Setup & Data Preparation
- **Days 3-5**: Model Selection & Baseline
- **Days 6-10**: Training & Fine-tuning
- **Days 11-12**: Evaluation & Metrics
- **Day 13**: Prediction & Output Generation
- **Days 14-15**: Documentation & Reporting

**Total Estimated Time**: 15 days

## Success Criteria

1. **Correctness (50 pts)**: Achieve good mIoU & Dice scores on both prompts
2. **Consistency (30 pts)**: Stable performance across varied scenes
3. **Presentation (20 pts)**: Clear README, documented seeds, comprehensive report with tables and visuals

## Next Steps

1. Start with Phase 1: Set up environment and download datasets
2. Begin with CLIPSeg as the baseline model
3. Iterate based on validation performance
4. Document everything for the final report

