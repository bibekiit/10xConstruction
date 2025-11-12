#!/bin/bash

# Setup script for real dataset training
# This script helps download datasets and prepare for training

set -e

echo "=========================================="
echo "Setting up Real Dataset Training"
echo "=========================================="

# Check for Roboflow API key
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo ""
    echo "⚠ WARNING: ROBOFLOW_API_KEY not set"
    echo ""
    echo "To download datasets automatically:"
    echo "1. Get your API key from: https://app.roboflow.com/"
    echo "2. Set it: export ROBOFLOW_API_KEY='your_key_here'"
    echo ""
    echo "Alternatively, you can download datasets manually:"
    echo "  - Dataset 1: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect"
    echo "  - Dataset 2: https://universe.roboflow.com/10xConstruction/cracks-3ii36"
    echo ""
    read -p "Do you want to continue with manual download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set ROBOFLOW_API_KEY or download datasets manually."
        exit 1
    fi
fi

# Step 1: Download datasets
echo ""
echo "Step 1: Downloading datasets..."
echo "-----------------------------------"
if [ ! -z "$ROBOFLOW_API_KEY" ]; then
    python3 scripts/download_datasets.py
else
    echo "⚠ Skipping automatic download (no API key)"
    echo "Please download datasets manually to data/raw/"
fi

# Step 2: Preprocess data
echo ""
echo "Step 2: Preprocessing data..."
echo "-----------------------------------"
if [ -d "data/raw/drywall-join-detect" ] || [ -d "data/raw/cracks" ]; then
    python3 scripts/preprocess_data.py
else
    echo "⚠ No datasets found in data/raw/"
    echo "Please download datasets first"
    exit 1
fi

# Step 3: Verify processed data
echo ""
echo "Step 3: Verifying processed data..."
echo "-----------------------------------"
if [ -d "data/processed/taping_area/train" ] || [ -d "data/processed/cracks/train" ]; then
    echo "✓ Processed data found"
    echo ""
    echo "Dataset summary:"
    if [ -d "data/processed/taping_area/train" ]; then
        train_count=$(find data/processed/taping_area/train/images -type f | wc -l)
        echo "  Taping area - Train: $train_count images"
    fi
    if [ -d "data/processed/cracks/train" ]; then
        train_count=$(find data/processed/cracks/train/images -type f | wc -l)
        echo "  Cracks - Train: $train_count images"
    fi
else
    echo "✗ No processed data found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete! Ready for training."
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python3 scripts/train.py --data_dir data/processed --epochs 50"
echo ""

