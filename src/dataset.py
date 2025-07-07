from datasets import load_dataset
import numpy as np
import pickle
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

NUM_SAMPLES = 100
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

example = train_us_dataset[0]
create_mel_spec_from_audio(example, False)


# samples = []
# for sample_row in train_us_dataset.select(range(NUM_SAMPLES)):
#     data = {
#         "audio": np.array(sample_row["audio"]["array"]),
#         "target_label": sample_row["class"]
#     }
#     samples.append(data)

# if not os.path.exists("data"):
#     os.makedirs("data")

# with open("data/urbansound_samples.pkl.gz", "wb") as file:
#     pickle.dump(samples, file)

