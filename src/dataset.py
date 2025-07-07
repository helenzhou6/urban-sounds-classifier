from datasets import Dataset as  HFDataset, DatasetDict
from datasets import load_dataset
import numpy as np
import pickle
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import login
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

hf_api_token = os.getenv("HF_API_TOKEN")

NUM_FREQ_BINS = 128

train_us_dataset = load_dataset("danavery/urbansound8K", split="train")

def create_mel_spec_from_audio(data, show_spectrogram=False):
    """
    Create a mel spectrogram from audio data.
    """
    audio_data = data['audio']
    waveform = audio_data['array']
    sampling_rate = audio_data['sampling_rate']

    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sampling_rate,
        n_mels=NUM_FREQ_BINS
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot and show one spectrogram image
    if show_spectrogram:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_mel_spectrogram,
            sr=sampling_rate,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Mel Spectrogram')
        plt.tight_layout()
        plt.show()
        plt.close()

    return log_mel_spectrogram

us_with_spectrogram = []
for sample_row in tqdm(train_us_dataset, desc="Processing samples"):
    data = {
        **{k: sample_row[k] for k in sample_row.keys() if k != "audio"},
        "mel_spectrogram": create_mel_spec_from_audio(sample_row),
    }
    us_with_spectrogram.append(data)

if not os.path.exists("data"):
    os.makedirs("data")

with open("data/urbansound_processed.pkl", "wb") as file:
    pickle.dump(us_with_spectrogram, file)

if not hf_api_token:
    raise ValueError("Please set the HF_API_TOKEN environment variable.")

login(token=hf_api_token)

hf_train = HFDataset.from_list(us_with_spectrogram)

dataset_dict = DatasetDict({
    "train": hf_train,
})

dataset_dict.push_to_hub("EthanGLEdwards/urbansounds_melspectrograms")

