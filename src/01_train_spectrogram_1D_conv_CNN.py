import torch
from models.spectrogram_1D_conv_CNN import SpectrogramCNN1D

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

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


class AudioSpectrogramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        label = item['classID']
        fold = item['fold']
        spectrogram = normalise_and_pad_spectrogram(item['mel_spectrogram'])
        return spectrogram, label, fold


# -- TRAINING AND EVALUATION FUNCTIONS ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels, fold in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, fold in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

NUM_FOLDS = 10
NUM_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

def run_k_fold_training(dataset, num_folds=NUM_FOLDS, num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE, device='cuda', fold_values=None):
    fold_accuracies = []

    for test_fold in tqdm(range(1, 2), desc="K-Fold"):
        print(f"Fold {test_fold}/{num_folds}")

        train_indices = np.where(fold_values != test_fold)[0].tolist()
        test_indices = np.where(fold_values == test_fold)[0].tolist()

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer
        model = SpectrogramCNN1D(num_classes=num_classes, input_dims=[128, 1501]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in tqdm(range(epochs), desc=f"Training Fold {test_fold}", leave=False):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        Uncomment and implement evaluate if needed
        test_acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy fold {test_fold}: {test_acc:.4f}")
        fold_accuracies.append(test_acc)

    avg_acc = sum(fold_accuracies) / num_folds
    print(f"Average accuracy across {num_folds} folds: {avg_acc:.4f}")
    return fold_accuracies, avg_acc

def main():
    # Initialize the model
    data = load_dataset("EthanGLEdwards/urbansounds_melspectrograms")
    fold_values = np.array(data["train"]["fold"])

    dataset = AudioSpectrogramDataset(data['train'])
    # Assuming dataset is already preprocessed and contains mel spectrograms

    run_k_fold_training(dataset, num_folds=NUM_FOLDS, num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE, device='cuda' if torch.cuda.is_available() else 'cpu', fold_values=fold_values)
    

if __name__ == "__main__":
    main()
    # train()  # Uncomment to run the training function when implemented