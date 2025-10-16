import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def audio_to_melspectrogram(audio_path):
    """Loads an audio file and converts it to a log Mel spectrogram."""
    y, sr = librosa.load(audio_path, sr=None)
    # Ensure consistent sampling rate if necessary, e.g., 16000 for AST
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def process_dataset():
    """
    Processes the entire UrbanSound8K dataset by converting audio to spectrograms
    and creating a manifest file for training.
    """
    print("Starting dataset preprocessing...")

    # Define paths
    metadata_path = 'data/raw/UrbanSound8K/metadata/UrbanSound8K.csv'
    audio_base_path = 'data/raw/UrbanSound8K/audio'
    output_dir = 'data/processed/spectrograms'
    manifest_path = 'data/processed/manifest.csv'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    processed_records = []

    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Processing audio files"):
        filename = row['slice_file_name']
        fold = row['fold']
        class_id = row['classID']

        # Construct paths
        audio_path = os.path.join(audio_base_path, f'fold{fold}', filename)
        spectrogram_fold_dir = os.path.join(output_dir, f'fold{fold}')
        os.makedirs(spectrogram_fold_dir, exist_ok=True)
        spectrogram_filename = os.path.splitext(filename)[0] + '.npy'
        spectrogram_path = os.path.join(spectrogram_fold_dir, spectrogram_filename)

        # Process and save
        if not os.path.exists(spectrogram_path):
            log_mel_spec = audio_to_melspectrogram(audio_path)
            np.save(spectrogram_path, log_mel_spec)

        processed_records.append({
            'spectrogram_path': spectrogram_path,
            'fold': fold,
            'classID': class_id
        })

    # Create and save manifest DataFrame
    manifest_df = pd.DataFrame(processed_records)
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Preprocessing complete. Manifest saved to {manifest_path}")

if __name__ == '__main__':
    process_dataset()
