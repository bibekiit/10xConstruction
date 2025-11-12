# Training Report: Prompted Segmentation for Drywall QA

## Executive Summary

This report documents the training and evaluation of a text-conditioned segmentation model for drywall quality assurance. The model is designed to segment specific regions in drywall images based on text prompts, specifically targeting "cracks" and "taping areas" (drywall seams/joints).

**Training Date**: November 12, 2025  
**Model**: CLIPSeg-based text-conditioned segmentation  
**Status**: Training completed (2 epochs), model checkpointed

---

## 1. Goal Summary

The primary objective was to train/fine-tune a text-conditioned segmentation model capable of producing binary masks for:

1. **"segment crack"** - Detecting cracks in drywall images
2. **"segment taping area"** - Detecting drywall joints/seams/taping areas

The model should:
- Accept text prompts as input
- Generate binary segmentation masks (PNG format, single-channel, values {0, 255})
- Output filenames in format: `{image_id}__{prompt}.png`
- Achieve good mIoU and Dice coefficient scores on both prompts
- Maintain consistent performance across varied scenes

---

## 2. Approach

### 2.1 Model Selection Rationale

**Selected Model: CLIPSeg (CIDAS/clipseg-rd64-refined)**

**Rationale:**
- CLIPSeg combines CLIP's powerful text-image understanding with a lightweight segmentation decoder
- Pre-trained on large-scale vision-language data, providing strong feature representations
- Designed specifically for text-conditioned segmentation tasks
- Efficient architecture suitable for fine-tuning on domain-specific data
- Good balance between performance and computational requirements

**Architecture Details:**
- **Base Model**: CLIPSeg (CIDAS/clipseg-rd64-refined)
- **Vision Encoder**: CLIP vision model (768-dimensional embeddings)
- **Text Encoder**: CLIP text model (512-dimensional embeddings)
- **Projection Layers**: 
  - Vision projection: 768 → 64 dimensions
  - Text projection: 512 → 64 dimensions
- **Feature Fusion**: Element-wise multiplication of vision and text embeddings
- **Decoder**: CNN decoder with upsampling (64 → 256 → 128 → 64 → 1)
- **Output**: Binary segmentation mask [B, 1, H, W]
- **Total Parameters**: 150,137,346

### 2.2 Training Strategy

**Multi-Task Learning Approach:**
- Combined both datasets (cracks and taping area) for training
- Single model trained to handle multiple prompts simultaneously
- Prompts mapped to datasets:
  - Cracks dataset → ["segment crack", "segment wall crack"]
  - Taping area dataset → ["segment taping area", "segment joint/tape", "segment drywall seam"]

**Loss Function:**
- Combined Binary Cross-Entropy (BCE) + Dice Loss
- BCE provides pixel-level supervision
- Dice loss helps with class imbalance and improves segmentation quality

**Training Configuration:**
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 2 (CPU memory constraints)
- **Image Size**: 256×256 (reduced for faster CPU training)
- **Gradient Clipping**: 1.0 (prevents gradient explosion)
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Enabled (patience based on validation IoU)
- **Device**: CPU (CUDA not available)

---

## 3. Models Tried

### 3.1 CLIPSeg (Selected)

**Status**: ✅ Selected and implemented

**Implementation Details:**
- Custom decoder added to CLIPSeg base model
- Projection layers to align vision and text embeddings
- Fine-tuned on combined drywall datasets

**Results**: See metrics section below

### 3.2 Alternative Models Considered

1. **SAM + CLIP**
   - **Status**: Not implemented
   - **Reason**: More complex architecture, CLIPSeg sufficient for initial implementation

2. **Custom CLIP + U-Net**
   - **Status**: Not implemented
   - **Reason**: CLIPSeg provides pre-trained components, faster to implement

---

## 4. Datasets

### 4.1 Dataset 1: Drywall-Join-Detect (Taping Area)

**Source**: https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

**Prompts**:
- "segment taping area"
- "segment joint/tape"
- "segment drywall seam"

**Data Split**:
- **Total Images**: 250
- **Train**: 175 images (525 mask files - 3 prompts per image)
- **Validation**: 37 images (111 mask files)
- **Test**: 38 images (114 mask files)

**Split Ratio**: 70% / 15% / 15%

### 4.2 Dataset 2: Cracks

**Source**: https://universe.roboflow.com/10xConstruction/cracks-3ii36

**Prompts**:
- "segment crack"
- "segment wall crack"

**Data Split**:
- **Total Images**: 201
- **Train**: 140 images (280 mask files - 2 prompts per image)
- **Validation**: 30 images (60 mask files)
- **Test**: 31 images (62 mask files)

**Split Ratio**: 70% / 15% / 15%

### 4.3 Combined Dataset Statistics

**Training Set**:
- Total image-mask pairs: 805 (525 from taping area + 280 from cracks)
- Total training batches: 403 (with batch size 2)

**Validation Set**:
- Total image-mask pairs: 171 (111 from taping area + 60 from cracks)
- Total validation batches: 86 (with batch size 2)

**Test Set**:
- Total image-mask pairs: 176 (114 from taping area + 62 from cracks)

---

## 5. Training Process

### 5.1 Training Execution

**Training Run**: `training_both_datasets`

**Epochs Completed**: 2 (out of planned 10)

**Training Time**:
- Epoch 1: ~1 hour 33 minutes
- Epoch 2: ~1 hour 8 minutes
- Total: ~2 hours 41 minutes (for 2 epochs)

**Training Progress**:
- Training loss decreased from ~0.6 to ~0.5-0.6 range
- Validation loss: 0.5784 (Epoch 2)
- Model showed learning progress with improving validation metrics

### 5.2 Training Metrics

**Epoch 2 Results** (Best model checkpoint):
- **Train Loss**: 0.5963
- **Val Loss**: 0.5784
- **Val IoU**: 0.1109
- **Val Dice**: 0.1830
- **Val Pixel Accuracy**: 0.8728
- **Learning Rate**: 0.0001

**Observations**:
- Training and validation losses are decreasing, indicating model is learning
- IoU and Dice scores are low but improving (0.0 → 0.1109 IoU)
- Pixel accuracy is high (87.28%), suggesting model is learning background/foreground distinction
- Early training stage - model needs more epochs for better segmentation quality

---

## 6. Evaluation Metrics

### 6.1 Overall Metrics (Validation Set)

| Metric | Value | Notes |
|--------|-------|-------|
| **mIoU** | 0.1109 | Mean Intersection over Union |
| **Dice Coefficient** | 0.1830 | F1 score for binary segmentation |
| **Pixel Accuracy** | 0.8728 | Overall pixel classification accuracy |
| **Validation Loss** | 0.5784 | Combined BCE + Dice loss |

### 6.2 Per-Prompt Metrics

*Note: Per-prompt evaluation was not completed in this training run. Future evaluation should include:*

- **"segment crack"** - mIoU, Dice
- **"segment wall crack"** - mIoU, Dice
- **"segment taping area"** - mIoU, Dice
- **"segment joint/tape"** - mIoU, Dice
- **"segment drywall seam"** - mIoU, Dice

### 6.3 Test Set Evaluation

*Test set evaluation pending - requires running inference on test set with trained model*

---

## 7. Model Performance Analysis

### 7.1 Strengths

1. **Training Stability**: Model trains stably with decreasing loss
2. **High Pixel Accuracy**: 87.28% pixel accuracy indicates good background/foreground separation
3. **Learning Progress**: Metrics improving from epoch 1 to epoch 2
4. **Multi-Prompt Capability**: Single model handles multiple prompts

### 7.2 Limitations

1. **Low IoU/Dice Scores**: Current scores (IoU: 0.11, Dice: 0.18) indicate segmentation quality needs improvement
2. **Early Training Stage**: Only 2 epochs completed - model needs more training
3. **CPU Training**: Training on CPU is slow - GPU would significantly speed up training
4. **Small Batch Size**: Batch size of 2 limits gradient estimation quality

### 7.3 Failure Cases (Expected)

Based on low IoU/Dice scores, the model likely struggles with:
- **Fine details**: Thin cracks or narrow taping areas
- **Edge cases**: Unusual lighting conditions, angles, or wall textures
- **Ambiguous regions**: Areas where cracks and seams are similar
- **Small objects**: Very small cracks or taping areas

---

## 8. Runtime & Footprint

### 8.1 Training Time

- **Per Epoch**: ~1-1.5 hours (CPU)
- **Total Training Time**: ~2 hours 41 minutes (2 epochs)
- **Estimated Full Training** (10 epochs): ~10-15 hours on CPU
- **GPU Estimate**: Would reduce to ~1-2 hours for 10 epochs

### 8.2 Inference Time

*Not measured in this training run. Estimated:*
- **Per Image**: ~0.5-1 second (CPU)
- **Batch Inference**: ~0.2-0.5 seconds per image (with batch processing)

### 8.3 Model Size

- **Parameters**: 150,137,346
- **Estimated Size**: ~600 MB (FP32) / ~300 MB (FP16)
- **Checkpoint Size**: ~600 MB (includes optimizer state)

---

## 9. Technical Implementation Details

### 9.1 Data Preprocessing

- Images resized to 256×256 for training
- Masks converted to binary format (0/255)
- Prompts loaded from text files
- Data augmentation: Standard transforms (normalization, etc.)

### 9.2 Model Architecture Modifications

1. **Projection Layers**: Added to align vision (768-dim) and text (512-dim) embeddings to 64 dimensions
2. **Decoder**: Custom CNN decoder with upsampling layers
3. **Output Layer**: Sigmoid activation for binary segmentation

### 9.3 Training Infrastructure

- **Framework**: PyTorch
- **Logging**: TensorBoard
- **Checkpointing**: Best model saved based on validation IoU
- **Early Stopping**: Configured but not triggered (only 2 epochs)

---

## 10. Visual Examples

*Note: Visual examples should be generated after test set evaluation. Placeholder structure:*

### Example 1: Taping Area Segmentation
- Original Image
- Ground Truth Mask
- Predicted Mask
- Overlay Visualization

### Example 2: Crack Segmentation
- Original Image
- Ground Truth Mask
- Predicted Mask
- Overlay Visualization

### Example 3: Multi-Prompt Comparison
- Same image with different prompts
- Comparison of segmentation results

### Example 4: Failure Case Analysis
- Example where model fails
- Analysis of failure reason

---

## 11. Recommendations

### 11.1 Immediate Next Steps

1. **Continue Training**: Train for more epochs (10-20) to improve segmentation quality
2. **Test Set Evaluation**: Run inference on test set and calculate final metrics
3. **Per-Prompt Analysis**: Evaluate performance for each prompt separately
4. **Visual Examples**: Generate and analyze visual examples from test set

### 11.2 Model Improvements

1. **GPU Training**: Use GPU for faster training and larger batch sizes
2. **Larger Batch Size**: Increase to 8-16 with GPU for better gradient estimates
3. **Learning Rate Tuning**: Experiment with different learning rates (1e-5 to 5e-4)
4. **Data Augmentation**: Add more aggressive augmentation (rotations, flips, color jitter)
5. **Longer Training**: Train for 20-50 epochs with early stopping

### 11.3 Architecture Improvements

1. **Attention Mechanisms**: Add cross-attention between vision and text features
2. **Multi-Scale Features**: Use feature pyramid network for better detail capture
3. **Post-Processing**: Add CRF or morphological operations for mask refinement

### 11.4 Evaluation Improvements

1. **Per-Prompt Metrics**: Implement detailed per-prompt evaluation
2. **Confusion Matrices**: Analyze false positives and false negatives
3. **Failure Case Collection**: Systematically collect and analyze failure cases

---

## 12. Conclusion

The training pipeline has been successfully implemented and tested. The CLIPSeg-based model shows learning progress with decreasing loss and improving metrics. However, the model is in early training stages (only 2 epochs completed) and requires more training to achieve satisfactory segmentation quality.

**Key Achievements:**
- ✅ Training pipeline fully functional
- ✅ Model architecture implemented and verified
- ✅ Multi-dataset training working
- ✅ Metrics tracking implemented
- ✅ Checkpointing system working

**Areas for Improvement:**
- ⚠️ Low IoU/Dice scores (needs more training)
- ⚠️ Limited training epochs completed
- ⚠️ CPU training is slow
- ⚠️ Per-prompt evaluation pending

**Next Steps:**
1. Continue training for more epochs
2. Evaluate on test set
3. Generate visual examples
4. Analyze per-prompt performance
5. Optimize hyperparameters

---

## 13. Appendix

### 13.1 Training Command

```bash
python scripts/train.py \
    --data_dir data/processed \
    --output_dir outputs/training_both_datasets \
    --epochs 10 \
    --batch_size 2 \
    --lr 1e-4 \
    --image_size 256 \
    --num_workers 0 \
    --gradient_clip 1.0
```

### 13.2 Model Checkpoints

- **Best Model**: `outputs/training_both_datasets/checkpoints/best_model.pth`
- **Epoch 1**: `outputs/training_both_datasets/checkpoints/checkpoint_epoch_1.pth`
- **Epoch 2**: `outputs/training_both_datasets/checkpoints/checkpoint_epoch_2.pth`

### 13.3 Training Logs

- **TensorBoard Logs**: `outputs/training_both_datasets/logs/`
- **Training Log**: `training_both_datasets.log`

### 13.4 Dependencies

- PyTorch
- transformers (for CLIPSeg)
- albumentations
- torchvision
- numpy
- scikit-learn
- tensorboard

### 13.5 Random Seeds

*Note: Random seeds should be documented for reproducibility. Check training script for seed values.*

---

**Report Generated**: November 12, 2025  
**Model Version**: CLIPSeg-based v1.0  
**Training Status**: In Progress (2/10 epochs completed)

