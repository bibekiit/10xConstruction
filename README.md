# Prompted Segmentation for Drywall QA

Text-conditioned segmentation model for detecting cracks and taping areas in drywall images.

## Project Overview

This project implements a text-conditioned segmentation model that can produce binary masks for:
- **Cracks**: "segment crack", "segment wall crack"
- **Taping Areas**: "segment taping area", "segment joint/tape", "segment drywall seam"

## Datasets

- **Dataset 1 (Drywall-Join-Detect)**: [Roboflow Link](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- **Dataset 2 (Cracks)**: [Roboflow Link](https://universe.roboflow.com/10xConstruction/cracks-3ii36)

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Run the data preparation script:

```bash
python scripts/download_datasets.py
```

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw downloaded datasets
│   ├── processed/        # Processed datasets with masks
│   └── splits/           # Train/val/test splits
├── models/               # Model definitions
├── scripts/              # Utility scripts
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── train.py
├── src/                  # Source code
│   ├── data/             # Data loaders
│   ├── models/           # Model architectures
│   ├── training/         # Training utilities
│   └── evaluation/       # Evaluation metrics
├── outputs/              # Model outputs
│   ├── checkpoints/      # Model checkpoints
│   ├── predictions/      # Prediction masks
│   └── visualizations/   # Visualization images
├── requirements.txt
└── README.md
```

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup and usage instructions.

### Setup

```bash
# 1. Setup environment
./setup.sh
source venv/bin/activate

# 2. Set Roboflow API key
export ROBOFLOW_API_KEY='your_api_key_here'

# 3. Download datasets
python scripts/download_datasets.py

# 4. Preprocess data
python scripts/preprocess_data.py

# 5. Train model
python scripts/train.py
```

## Usage

### Training

```bash
python scripts/train.py \
    --data_dir data/processed \
    --output_dir outputs \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4
```

### Inference

(To be implemented in Phase 5)

```bash
python scripts/inference.py \
    --model_path outputs/checkpoints/best_model.pth \
    --image_path path/to/image.jpg \
    --prompt "segment crack"
```

## Random Seeds

For reproducibility, the following seeds are used:
- **Random Seed**: 42
- **PyTorch Seed**: 42
- **NumPy Seed**: 42
- **CUDA Seed**: 42 (if using GPU)

## License

[Add license information]

