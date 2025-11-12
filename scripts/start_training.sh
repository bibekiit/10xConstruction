#!/bin/bash

# Start training script for real dataset training

set -e

echo "=========================================="
echo "Starting Training on Real Datasets"
echo "=========================================="

# Check if processed data exists
if [ ! -d "data/processed/taping_area/train" ] && [ ! -d "data/processed/cracks/train" ]; then
    echo "Error: No processed data found!"
    echo "Please run: python3 scripts/preprocess_data.py"
    exit 1
fi

# Set API key if not set
if [ -z "$ROBOFLOW_API_KEY" ]; then
    export ROBOFLOW_API_KEY='FQQRU5Cbf1JLgKSTaVke'
fi

# Training configuration
DATA_DIR="data/processed"
OUTPUT_DIR="outputs/training_$(date +%Y%m%d_%H%M%S)"
EPOCHS=50
BATCH_SIZE=4  # Reduced for CPU
IMAGE_SIZE=256  # Reduced for faster training on CPU
LR=1e-4

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMAGE_SIZE"
echo "  Learning rate: $LR"
echo ""

# Start training
python3 scripts/train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --image_size $IMAGE_SIZE \
    --num_workers 0 \
    --early_stopping_patience 10 \
    --gradient_clip 1.0

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "View logs: tensorboard --logdir $OUTPUT_DIR/logs"
echo ""

