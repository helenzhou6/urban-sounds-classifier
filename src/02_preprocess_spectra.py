from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm
import pickle


# --- LOADING THE DATASET AND NORMALISE AND PAD SPECTOGRAM ---
def normalise_and_pad_spectrogram(spectrogram, min_value=-80, max_len=1501):
    """
    Normalise the spectrogram to a range of [0, 1].
    """
    spectrogram = np.array(spectrogram, dtype=np.float32)
    pad_width = max_len - spectrogram.shape[1]
    if pad_width > 0:
        spectrogram = np.pad(
            spectrogram,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=min_value
        )
    return (spectrogram + 80) / (80)

def preprocess_dataset(data):
    """
    Preprocess all spectrograms once to avoid processing during training.
    """
    print("Preprocessing spectrograms...")
    processed_data = []
    
    for item in tqdm(data, desc="Preprocessing spectrograms"):
        # print(item)
        spectrogram = normalise_and_pad_spectrogram(item['mel_spectrogram'])
        processed_item = {
            'spectrogram': torch.tensor(spectrogram, dtype=torch.float32),
            'classID': torch.tensor(item['classID'], dtype=torch.long),
            'fold': item['fold']
        }
        processed_data.append(processed_item)
    
    print(f"Preprocessed {len(processed_data)} spectrograms")
    return processed_data

data = load_dataset("EthanGLEdwards/urbansounds_melspectrograms")

processed_dataset = preprocess_dataset(data["train"])

with open("data/processed_urbansound_melspectrograms.pkl", "wb") as file:
    pickle.dump(processed_dataset, file)
print("Dataset preprocessing complete. Processed data saved to 'data/processed_urbansound_melspectrograms.pkl'.")