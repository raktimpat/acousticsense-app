# UrbanSound8K Audio Classification with Audio Spectrogram Transformer (AST)

Deep learning model for urban sound classification using pre-trained Audio Spectrogram Transformer (AST) with 10-fold cross-validation on the UrbanSound8K dataset.

## Overview

This project implements audio classification on the UrbanSound8K dataset using a pre-trained Audio Spectrogram Transformer from MIT. The model classifies 10 urban sound categories: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music.

## Model Architecture

- **Base Model**: MIT/ast-finetuned-audioset-10-10-0.4593 (pre-trained on AudioSet)
- **Transfer Learning**: Fine-tuned classifier head for 10 UrbanSound8K classes
- **Input**: Pre-computed mel spectrograms (128 mel bins × 1024 time frames)
- **Method**: 10-fold cross-validation using predefined UrbanSound8K splits

## Key Design Decisions

### 1. Pre-computed Spectrograms
**Why**: Training speed and consistency
- Spectrograms are computed once during preprocessing
- Saves ~40% training time compared to computing on-the-fly
- Ensures identical feature extraction across all experiments
- Allows rapid experimentation with different models

### 2. Fixed Input Size (1024 × 128)
**Why**: AST architecture requirement
- The pre-trained AST model has fixed positional embeddings for 1024 time frames
- All audio clips are padded or truncated to exactly 1024 frames (~10 seconds at 10ms hop)
- UrbanSound8K clips (≤4 seconds) are zero-padded to 1024 frames
- This is standard practice for AST models

### 3. No Feature Extractor Usage
**Why**: Working with pre-computed spectrograms
- HuggingFace's `ASTFeatureExtractor` expects raw audio waveforms
- We bypass it and apply AudioSet normalization (mean=-4.27, std=4.57) manually
- Spectrograms are fed directly in the shape AST expects after patch embedding

### 4. Respecting Predefined Folds
**Why**: Data leakage prevention and reproducibility
- UrbanSound8K contains related clips from the same recordings
- Predefined folds ensure related clips stay together (no data leakage)
- Using the official 10-fold split makes results comparable to published research
- **Never** reshuffle or create custom splits

### 5. Data Augmentation
**Why**: Improve generalization with limited data
- SpecAugment-style time and frequency masking applied during training
- Operates on spectrograms before normalization
- Only applied to training data (not validation)
- Helps model learn robust features

## Project Structure

```
.
├── train.py                    # Main training script
├── data/
│   ├── raw/                    # Original UrbanSound8K audio files
│   └── processed/
│       ├── manifest.csv        # File paths, labels, and fold assignments
│       └── spectrograms/       # Pre-computed .npy spectrogram files
├── results/                    # Training outputs per fold
├── best_model/                 # Best model across all folds
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── confusion_matrix.npy
│   └── class_metrics.csv
└── logs/                       # TensorBoard logs
```

## Data Format

### manifest.csv
```csv
spectrogram_path,classID,fold
data/processed/spectrograms/fold1/12345.npy,0,1
data/processed/spectrograms/fold1/67890.npy,3,1
...
```

### Spectrogram Files
- Format: NumPy `.npy` files
- Shape: `[128, time]` or `[time, 128]` (automatically detected)
- Values: Mel spectrogram magnitudes (not log scale, not normalized)

## Training Process

### 10-Fold Cross-Validation Flow

```
For each fold k ∈ {1, 2, ..., 10}:
    1. Train on folds ≠ k (9 folds)
    2. Validate on fold k (1 fold)
    3. Track metrics: accuracy, precision, recall, F1
    4. Save best model (highest F1 score)

Final: Average metrics across all 10 folds
```

### Training Configuration
- **Epochs**: 10 per fold
- **Batch Size**: 8
- **Learning Rate**: 5e-5 with 10% warmup
- **Optimizer**: AdamW with weight decay 0.01
- **Mixed Precision**: FP16 (if GPU available)
- **Best Model Selection**: Highest macro F1 score

## Results Format

### Console Output (Per Fold)
```
Fold 1 Results:
  Accuracy:  0.8234
  Precision: 0.8156
  Recall:    0.8089
  F1 Score:  0.8122

Detailed Classification Report:
                    precision    recall  f1-score   support
  air_conditioner      0.8500    0.8095    0.8293        84
  car_horn             0.7800    0.8421    0.8099        95
  ...
```

### Cross-Validation Summary
```
Fold Accuracies: ['0.8234', '0.7891', '0.8156', ...]
Average Accuracy: 0.8045 ± 0.0234
Average F1 Score: 0.7987 ± 0.0198
```

### Saved Artifacts
- `results/cv_results.csv` - All fold metrics
- `best_model/` - Best performing model
- `best_model/confusion_matrix.npy` - Confusion matrix
- `best_model/class_metrics.csv` - Per-class performance

## Installation

```bash
pip install torch torchvision torchaudio
pip install transformers datasets evaluate
pip install scikit-learn pandas numpy
```

## Usage

### 1. Prepare Data
Ensure your data follows the required format with pre-computed spectrograms and manifest.csv.

### 2. Train Model
```bash
python train.py
```

The script will:
- Validate all spectrogram files exist
- Run 10-fold cross-validation
- Save the best model (highest F1)
- Generate detailed metrics and reports

### 3. Monitor Training
```bash
tensorboard --logdir ./logs
```

## Technical Implementation Details

### Why Manual Normalization?
The AST feature extractor expects raw audio and computes spectrograms internally. Since we have pre-computed spectrograms:
1. We extract normalization stats (mean, std) from the feature extractor
2. Apply them manually: `(spec - mean) / (std * 2)`
3. Feed normalized spectrograms directly to the model

### Shape Transformations
```python
Loaded spectrogram:      [128, time] or [time, 128]
After detection:         [128, time]  (frequency × time)
After augmentation:      [128, time]
After normalization:     [128, time]
After transpose:         [time, 128]  (AST format)
After padding/truncate:  [1024, 128]  (fixed size)
Batched:                 [batch, 1024, 128]
```

### Memory Optimization
- Lazy loading: Spectrograms loaded one at a time
- Checkpoint cleanup: Only keeps 2 best checkpoints per fold
- GPU cache clearing: Clears CUDA cache between folds
- Fold cleanup: Removes intermediate fold results after processing

## Reproducibility

- **Random Seeds**: Set for Python, NumPy, and PyTorch (seed=42)
- **Deterministic Training**: Fixed seed in TrainingArguments
- **No Data Shuffling**: Uses predefined UrbanSound8K folds
- **Fixed Splits**: Same 10-fold CV as published research

## Common Issues & Solutions

### Issue: RuntimeError about tensor size mismatch
**Cause**: Positional embedding size mismatch
**Solution**: All spectrograms must be exactly 1024 time frames (handled by collator)

### Issue: Out of memory errors
**Cause**: Large batch size or long spectrograms
**Solution**: Reduce batch size or ensure max_length=1024 is enforced

### Issue: Results not comparable to papers
**Cause**: Custom splits or different fold count
**Solution**: Always use the 10 predefined folds, never reshuffle

## Performance Expectations

UrbanSound8K is a challenging dataset with high intra-class variability. Typical results:
- **Baseline models**: 70-75% accuracy
- **State-of-the-art (2024)**: 85-90% accuracy
- **This implementation**: Results depend on your preprocessing and training duration

Note: Fold 10 typically gives higher scores than Fold 1 due to data characteristics.

## Citation

If you use this code, please cite the UrbanSound8K dataset:

```bibtex
@inproceedings{Salamon:UrbanSound:ACMMM:14,
  author = {Salamon, J. and Jacoby, C. and Bello, J. P.},
  title = {A Dataset and Taxonomy for Urban Sound Research},
  booktitle = {22nd ACM International Conference on Multimedia},
  year = {2014},
  pages = {1041--1044},
  address = {Orlando, FL, USA}
}
```

And the Audio Spectrogram Transformer:

```bibtex
@article{gong2021ast,
  title={AST: Audio Spectrogram Transformer},
  author={Gong, Yuan and Chung, Yu-An and Glass, James},
  journal={arXiv preprint arXiv:2104.01778},
  year={2021}
}
```

## License

This code is provided for research and educational purposes. Please respect the licenses of the UrbanSound8K dataset and the pre-trained AST model.