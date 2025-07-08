import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from utils import get_device, init_wandb, clear_gpu_cache, print_gpu_memory
from models.spectrogram_1D_conv_transformer import SpectrogramTransformer
import pickle

# test the model by loading and forward passing a dummy input, inspecting shape at the end
def test_model_dummy(model, input_shape=(128, 1501)):
    # Create a dummy input tensor with the specified shape
    dummy_input = torch.randn(1, *input_shape)  # Batch size of 1
    print(f"Dummy input shape: {dummy_input.shape}")

    # Forward pass through the model
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


def test_model_spectrogram(model, dataset, input_shape=(128, 1501)):
    # load a spectrogram from   the dataset
    if isinstance(dataset, str):
        # If dataset is a string, load it from the file
        with open(dataset, "rb") as file:
            dataset = pickle.load(file)

    spectrogram = dataset[0]['spectrogram']  # Get the first spectrogram
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    print(f"Dummy spectrogram shape: {spectrogram.shape}")

    # Forward pass through the model
    output = model(spectrogram)
    print(f"Output shape: {output.shape}")


model = SpectrogramTransformer()
test_model_spectrogram(model, "data/processed_urbansound_melspectrograms.pkl")