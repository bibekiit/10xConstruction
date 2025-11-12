# Dataset Setup Guide for Real Training

## Option 1: Automatic Download (Recommended)

### Step 1: Get Roboflow API Key

1. Sign up/Login at https://roboflow.com/
2. Go to https://app.roboflow.com/
3. Navigate to your account settings
4. Copy your API key

### Step 2: Set API Key

```bash
export ROBOFLOW_API_KEY='your_api_key_here'
```

### Step 3: Download Datasets

```bash
python3 scripts/download_datasets.py
```

This will download:
- **Dataset 1**: Drywall-Join-Detect (taping areas)
- **Dataset 2**: Cracks dataset

## Option 2: Manual Download

If you don't have a Roboflow API key, you can download datasets manually:

### Dataset 1: Drywall-Join-Detect
1. Visit: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect
2. Click "Download" → Select "YOLOv8" format
3. Extract to: `data/raw/drywall-join-detect/`

### Dataset 2: Cracks
1. Visit: https://universe.roboflow.com/10xConstruction/cracks-3ii36
2. Click "Download" → Select "YOLOv8" format  
3. Extract to: `data/raw/cracks/`

### Expected Directory Structure

After manual download, your `data/raw/` should look like:

```
data/raw/
├── drywall-join-detect/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/ (or val/)
│   │   ├── images/
│   │   └── labels/
│   └── test/ (optional)
│       ├── images/
│       └── labels/
└── cracks/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/ (or val/)
    │   ├── images/
    │   └── labels/
    └── test/ (optional)
        ├── images/
        └── labels/
```

## Step 4: Preprocess Data

Once datasets are in `data/raw/`, run:

```bash
python3 scripts/preprocess_data.py
```

This will:
- Convert YOLO annotations to binary masks
- Create train/val/test splits (70/15/15)
- Map prompts to datasets
- Generate mask files: `{image_id}__{prompt}.png`

## Step 5: Verify Processed Data

Check that processed data exists:

```bash
ls -la data/processed/
```

You should see:
- `taping_area/` (train, val, test)
- `cracks/` (train, val, test)

## Step 6: Start Training

```bash
python3 scripts/train.py \
    --data_dir data/processed \
    --output_dir outputs/training_run_1 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
```

## Quick Setup Script

You can also use the automated setup script:

```bash
./scripts/setup_training.sh
```

This script will:
1. Check for API key
2. Download datasets (if API key is set)
3. Preprocess data
4. Verify everything is ready

## Troubleshooting

### No datasets found
- Make sure datasets are in `data/raw/` with correct structure
- Check that YOLO format was selected during download

### Preprocessing fails
- Verify YOLO annotation files (.txt) exist in labels/ directories
- Check that image and label filenames match (except extension)

### Training fails
- Ensure processed data exists in `data/processed/`
- Check that train/val splits have images
- Verify prompts are correctly mapped

## Next Steps After Setup

Once datasets are ready:
1. Review `DATA_PROCESSING_REVIEW.md` for data format details
2. Start with baseline training: `python3 scripts/train.py`
3. Monitor training with TensorBoard: `tensorboard --logdir outputs/logs`
4. Experiment with different configs using `scripts/run_experiments.py`

