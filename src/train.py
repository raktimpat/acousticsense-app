import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from transformers import ASTConfig, ASTForAudioClassification, Trainer, TrainingArguments
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import random

# --- Set Random Seeds for Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Configuration ---
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
MANIFEST_PATH = 'data/processed/manifest.csv'
OUTPUT_DIR = './results'
BEST_MODEL_DIR = './best_model'
NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_FOLDS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# AST normalization stats (from AudioSet pretraining)
AUDIOSET_MEAN = -4.2677393
AUDIOSET_STD = 4.5689974

# --- Class Labels ---
CLASS_LABELS = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", 
    "drilling", "engine_idling", "gun_shot", "jackhammer", 
    "siren", "street_music"
]
id2label = {i: label for i, label in enumerate(CLASS_LABELS)}
label2id = {label: i for i, label in enumerate(CLASS_LABELS)}

# --- Custom Dataset with Lazy Loading ---
class SpectrogramDataset(TorchDataset):
    """
    Custom dataset that loads PRE-COMPUTED spectrograms on-the-fly.
    AST internally expects audio waveforms, but we can pass spectrograms 
    directly by matching the expected format after feature extraction.
    """
    def __init__(self, dataframe, mean=AUDIOSET_MEAN, std=AUDIOSET_STD, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def normalize_spectrogram(self, spectrogram):
        """Apply AudioSet normalization."""
        normalized = (spectrogram - self.mean) / (self.std * 2)
        return normalized
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load spectrogram lazily
        spectrogram = np.load(row['spectrogram_path'])
        
        # Handle different possible shapes from preprocessing
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.squeeze(0)
        
        # Ensure shape is [n_mels, time] where n_mels should be 128
        if spectrogram.shape[0] != 128 and spectrogram.shape[1] == 128:
            spectrogram = spectrogram.T
        elif spectrogram.shape[0] != 128 and spectrogram.shape[1] != 128:
            raise ValueError(f"Expected one dimension to be 128 (n_mels), got shape {spectrogram.shape}")
        
        # Now spectrogram is [128, time]
        
        # Apply augmentation if provided (operates on [n_mels, time])
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Normalize the spectrogram
        spectrogram = self.normalize_spectrogram(spectrogram)
        
        # AST expects [time, frequency] after internal processing
        # Transpose from [128, time] to [time, 128]
        spectrogram = spectrogram.T
        
        # Convert to tensor - final shape: [time, 128]
        input_values = torch.tensor(spectrogram, dtype=torch.float32)
        
        # Create the inputs dictionary
        inputs = {
            'input_values': input_values,
            'labels': torch.tensor(row['classID'], dtype=torch.long)
        }
        
        return inputs

# --- Data Collator for Batching ---
class SpectrogramCollator:
    """
    Custom collator to handle batching of spectrograms with padding/truncation.
    AST expects FIXED input shape: [batch_size, 1024, 128]
    All spectrograms must be exactly 1024 time frames.
    """
    def __init__(self, target_length=1024):
        self.target_length = target_length  # Fixed length required by AST
    
    def __call__(self, features):
        padded_inputs = []
        labels = []
        
        for f in features:
            input_val = f['input_values']  # Shape: [time, 128]
            current_time = input_val.shape[0]
            
            # Always resize to exactly target_length
            if current_time > self.target_length:
                # Truncate
                input_val = input_val[:self.target_length, :]
            elif current_time < self.target_length:
                # Pad
                pad_amount = self.target_length - current_time
                padding = torch.zeros((pad_amount, input_val.shape[1]), dtype=input_val.dtype)
                input_val = torch.cat([input_val, padding], dim=0)
            # If equal, use as-is
            
            padded_inputs.append(input_val)
            labels.append(f['labels'])
        
        # Stack all - creates [batch, 1024, 128]
        input_values = torch.stack(padded_inputs, dim=0)
        labels_tensor = torch.stack(labels, dim=0)
        
        # Verify shape
        assert input_values.shape[1] == self.target_length, f"Expected time dim {self.target_length}, got {input_values.shape[1]}"
        assert input_values.shape[2] == 128, f"Expected 128 mel bins, got {input_values.shape[2]}"
        
        return {
            'input_values': input_values,
            'labels': labels_tensor
        }

# --- Simple Data Augmentation ---
class SpectrogramAugmentation:
    """SpecAugment-style augmentation: time and frequency masking"""
    def __init__(self, time_mask_param=20, freq_mask_param=20, probability=0.5):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.probability = probability
    
    def __call__(self, spectrogram):
        if random.random() > self.probability:
            return spectrogram
        
        spec = spectrogram.copy()
        
        # Time masking
        if random.random() > 0.5 and spec.shape[1] > self.time_mask_param:
            t = random.randint(1, self.time_mask_param)
            t0 = random.randint(0, spec.shape[1] - t)
            spec[:, t0:t0+t] = spec.mean()
        
        # Frequency masking
        if random.random() > 0.5 and spec.shape[0] > self.freq_mask_param:
            f = random.randint(1, self.freq_mask_param)
            f0 = random.randint(0, spec.shape[0] - f)
            spec[f0:f0+f, :] = spec.mean()
        
        return spec

# --- Load Data ---
print("Loading manifest...")
manifest_df = pd.read_csv(MANIFEST_PATH)

# Validate spectrogram files
print("Validating spectrogram paths...")
missing_files = [path for path in manifest_df['spectrogram_path'] if not os.path.exists(path)]

if missing_files:
    print(f"WARNING: {len(missing_files)} spectrogram files not found!")
    print(f"First few missing: {missing_files[:5]}")
    raise FileNotFoundError("Some spectrogram files are missing")

print(f"Found {len(manifest_df)} spectrograms")

# --- Custom Metrics ---
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    
    return {
        'accuracy': accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'precision': precision.compute(predictions=predictions, references=labels, average='macro')['precision'],
        'recall': recall.compute(predictions=predictions, references=labels, average='macro')['recall'],
        'f1': f1.compute(predictions=predictions, references=labels, average='macro')['f1']
    }

# --- Main Training Loop ---
fold_accuracies = []
fold_f1_scores = []
best_fold_f1 = 0.0

augmentation = SpectrogramAugmentation()
data_collator = SpectrogramCollator(target_length=1024)

for k in range(1, NUM_FOLDS + 1):
    print(f"\n{'='*60}")
    print(f"Starting Training for Fold {k}/{NUM_FOLDS}")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Split data
    train_df = manifest_df[manifest_df['fold'] != k]
    eval_df = manifest_df[manifest_df['fold'] == k]
    
    print(f"Train samples: {len(train_df)}, Eval samples: {len(eval_df)}")
    
    # Create datasets
    train_dataset = SpectrogramDataset(train_df, transform=augmentation)
    eval_dataset = SpectrogramDataset(eval_df, transform=None)
    
    # Debug: Check shapes
    if k == 1:
        print(f"\nChecking dataset format...")
        sample = train_dataset[0]
        print(f"  Sample input_values shape: {sample['input_values'].shape}")
        print(f"  Expected: [time, 128]")
        
        test_batch = data_collator([train_dataset[0], train_dataset[1]])
        print(f"  Batched input_values shape: {test_batch['input_values'].shape}")
        print(f"  Expected: [2, time, 128]\n")
    
    # Load Model - use from_pretrained but reinitialize classifier
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f'{OUTPUT_DIR}/fold_{k}',
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,
        logging_dir=f'./logs/fold_{k}',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        seed=42,
        remove_unused_columns=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate
    print("Evaluating on validation fold...")
    eval_results = trainer.evaluate()
    
    current_accuracy = eval_results['eval_accuracy']
    current_f1 = eval_results['eval_f1']
    
    fold_accuracies.append(current_accuracy)
    fold_f1_scores.append(current_f1)
    
    print(f"\nFold {k} Results:")
    print(f"  Accuracy:  {current_accuracy:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  F1 Score:  {current_f1:.4f}")
    
    # Detailed classification report
    predictions = trainer.predict(eval_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        true_labels, 
        pred_labels, 
        target_names=CLASS_LABELS,
        digits=4
    ))
    
    # Save best model
    if current_f1 > best_fold_f1:
        best_fold_f1 = current_f1
        print(f"\nNew best model found in fold {k} (F1: {current_f1:.4f})!")
        print(f"Saving to {BEST_MODEL_DIR}")
        trainer.save_model(BEST_MODEL_DIR)
        
        # Save confusion matrix and metrics
        cm = confusion_matrix(true_labels, pred_labels)
        np.save(f'{BEST_MODEL_DIR}/confusion_matrix.npy', cm)
        
        class_report = classification_report(
            true_labels, 
            pred_labels, 
            target_names=CLASS_LABELS,
            output_dict=True
        )
        pd.DataFrame(class_report).transpose().to_csv(
            f'{BEST_MODEL_DIR}/class_metrics.csv'
        )
    
    # Cleanup
    if k < NUM_FOLDS:
        import shutil
        checkpoint_dir = f'{OUTPUT_DIR}/fold_{k}'
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

# --- Final Summary ---
print("\n" + "="*60)
print("Cross-Validation Complete")
print("="*60)

mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
mean_f1 = np.mean(fold_f1_scores)
std_f1 = np.std(fold_f1_scores)

print(f"\nFold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
print(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

print(f"\nFold F1 Scores: {[f'{f1:.4f}' for f1 in fold_f1_scores]}")
print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

print(f"\nBest model saved to: {BEST_MODEL_DIR}")
print(f"Best F1 Score: {best_fold_f1:.4f}")

# Save final results
results_summary = pd.DataFrame({
    'fold': range(1, NUM_FOLDS + 1),
    'accuracy': fold_accuracies,
    'f1_score': fold_f1_scores
})
results_summary.to_csv(f'{OUTPUT_DIR}/cv_results.csv', index=False)
print(f"\nResults saved to: {OUTPUT_DIR}/cv_results.csv")