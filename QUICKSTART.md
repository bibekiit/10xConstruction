# Quick Start Guide

## Phase 1 & 2 Implementation Summary

This guide covers the setup and initial implementation for Phases 1 and 2 of the Drywall QA Segmentation project.

## What's Been Implemented

### Phase 1: Environment Setup & Data Preparation ✅

1. **Project Structure**: Complete directory structure created
2. **Dependencies**: `requirements.txt` with all necessary packages
3. **Data Download Script**: `scripts/download_datasets.py` for Roboflow integration
4. **Data Preprocessing**: `scripts/preprocess_data.py` for:
   - Converting YOLO annotations to binary masks
   - Creating train/val/test splits
   - Mapping prompts to datasets
   - Generating mask files in correct format

### Phase 2: Model Selection & Baseline ✅

1. **Model Architecture**: CLIPSeg-based text-conditioned segmentation model
   - `CLIPSegSegmentationModel`: Full CLIPSeg implementation
   - `SimpleCLIPSegModel`: Training-optimized version with proper text conditioning
2. **Data Loaders**: Custom PyTorch datasets with text prompt support
3. **Loss Functions**: Combined BCE + Dice loss for segmentation
4. **Evaluation Metrics**: IoU, Dice coefficient, pixel accuracy
5. **Training Script**: Complete training pipeline with:
   - TensorBoard logging
   - Model checkpointing
   - Validation metrics
   - Learning rate scheduling

## Setup Instructions

### 1. Environment Setup

```bash
# Option 1: Use setup script
./setup.sh

# Option 2: Manual setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get Roboflow API Key

1. Sign up at https://roboflow.com/
2. Get your API key from https://app.roboflow.com/
3. Set it as environment variable:
   ```bash
   export ROBOFLOW_API_KEY='your_api_key_here'
   ```

### 3. Download Datasets

```bash
python scripts/download_datasets.py
```

This will download:
- **Dataset 1**: Drywall-Join-Detect (taping areas)
- **Dataset 2**: Cracks dataset

**Note**: If automatic download fails, you can manually download from:
- Dataset 1: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect
- Dataset 2: https://universe.roboflow.com/10xConstruction/cracks-3ii36

### 4. Preprocess Data

```bash
python scripts/preprocess_data.py
```

This will:
- Convert annotations to binary masks (PNG, single-channel, {0, 255})
- Create train/val/test splits (70/15/15)
- Map prompts to datasets:
  - Taping area: "segment taping area", "segment joint/tape", "segment drywall seam"
  - Cracks: "segment crack", "segment wall crack"
- Generate mask files with format: `{image_id}__{prompt}.png`

### 5. Start Training

```bash
python scripts/train.py \
    --data_dir data/processed \
    --output_dir outputs \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 512
```

Or use default settings:
```bash
python scripts/train.py
```

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw downloaded datasets
│   ├── processed/        # Processed datasets with masks
│   └── splits/           # Train/val/test splits
├── src/
│   ├── data/             # Data loaders
│   ├── models/           # Model architectures (CLIPSeg)
│   ├── training/         # Loss functions
│   └── evaluation/       # Metrics (IoU, Dice)
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── train.py
├── configs/
│   └── baseline.yaml     # Training configuration
├── outputs/
│   ├── checkpoints/      # Model checkpoints
│   ├── predictions/      # Prediction masks
│   └── visualizations/   # Visualization images
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

## Model Architecture

### CLIPSeg-Based Model

- **Base**: CLIPSeg (CIDAS/clipseg-rd64-refined)
- **Text Conditioning**: CLIP text encoder processes prompts
- **Image Encoding**: CLIP vision encoder processes images
- **Decoder**: Custom CNN decoder upsamples to original image size
- **Output**: Binary segmentation mask (sigmoid activation)

### Key Features

- Text-conditioned: Accepts natural language prompts
- End-to-end trainable: Fine-tunes CLIPSeg for drywall QA
- Flexible prompts: Supports multiple prompt variations per dataset

## Training Configuration

- **Random Seed**: 42 (for reproducibility)
- **Image Size**: 512x512 (configurable)
- **Batch Size**: 8 (adjust based on GPU memory)
- **Learning Rate**: 1e-4
- **Loss**: Combined BCE + Dice (50/50)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau

## Next Steps (Phases 3-6)

1. **Phase 3**: Train and fine-tune models
2. **Phase 4**: Evaluate on test set, calculate metrics
3. **Phase 5**: Generate predictions with correct filename format
4. **Phase 6**: Create comprehensive report with visuals

## Troubleshooting

### Dataset Download Issues

If Roboflow download fails:
1. Check API key is set: `echo $ROBOFLOW_API_KEY`
2. Manually download datasets and place in `data/raw/`
3. Ensure datasets follow YOLO format structure

### CUDA Out of Memory

- Reduce batch size: `--batch_size 4`
- Reduce image size: `--image_size 256`
- Use gradient accumulation (modify training script)

### Model Loading Issues

- Ensure transformers library is installed: `pip install transformers`
- Model will auto-download on first use (~500MB)
- Check internet connection for model download

## Notes

- All random seeds are set to 42 for reproducibility
- Model checkpoints saved in `outputs/checkpoints/`
- TensorBoard logs in `outputs/logs/` (view with `tensorboard --logdir outputs/logs`)
- Best model saved as `best_model.pth` based on validation IoU

