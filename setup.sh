#!/bin/bash

# Setup script for Drywall QA Segmentation Project

echo "=========================================="
echo "Setting up Drywall QA Segmentation Project"
echo "=========================================="

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Set your Roboflow API key: export ROBOFLOW_API_KEY='your_key_here'"
echo "3. Download datasets: python scripts/download_datasets.py"
echo "4. Preprocess data: python scripts/preprocess_data.py"
echo "5. Start training: python scripts/train.py"
echo ""

