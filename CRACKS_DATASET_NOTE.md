# Cracks Dataset Status

## Current Situation

The cracks dataset project exists at:
- **Workspace**: `test-eswkr`
- **Project**: `cracks-3ii36-aocvl`
- **URL**: https://app.roboflow.com/test-eswkr/cracks-3ii36-aocvl/

However, **no versions have been exported yet** in Roboflow.

## To Download the Cracks Dataset

### Option 1: Export Version in Roboflow (Recommended)

1. Go to: https://app.roboflow.com/test-eswkr/cracks-3ii36-aocvl/
2. Click on "Generate" or "Export" 
3. Select "YOLOv8" format
4. Export version 1
5. Then run: `python3 scripts/download_datasets.py`

### Option 2: Manual Download

1. Go to: https://app.roboflow.com/test-eswkr/cracks-3ii36-aocvl/
2. Click "Download" → Select "YOLOv8" format
3. Extract to: `data/raw/cracks/`
4. Then run: `python3 scripts/preprocess_data.py`

## Current Training Status

We can proceed with training on **just the taping area dataset** for now:
- ✅ 175 training images (525 mask files with 3 prompts each)
- ✅ 37 validation images
- ✅ 38 test images

The model will learn to segment taping areas. Once the cracks dataset is available, we can:
1. Download and preprocess it
2. Continue training with both datasets (multi-task learning)
3. Or fine-tune the model on cracks

## Next Steps

**Option A: Train on Taping Area Only (Now)**
```bash
python3 scripts/train.py \
    --data_dir data/processed \
    --output_dir outputs/training_taping_area \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4
```

**Option B: Wait for Cracks Dataset**
- Export version in Roboflow
- Download: `python3 scripts/download_datasets.py`
- Preprocess: `python3 scripts/preprocess_data.py`
- Train on both datasets

