import torch
from models.spectrogram_1D_conv_CNN import SpectrogramCNN1D

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from utils import get_device, init_wandb



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
    print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    
    for inputs, labels, fold in tqdm(dataloader, desc="Training", leave=False):
        print(f"[DEBUG] Original inputs device: {inputs.device}, labels device: {labels.device}")
        inputs, labels = inputs.to(device), labels.to(device)
        print(f"[DEBUG] After .to(device) - inputs device: {inputs.device}, labels device: {labels.device}")

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
    print(f"[DEBUG] Evaluation - Model device: {next(model.parameters()).device}")
    
    with torch.no_grad():
        for inputs, labels, fold in loader:
            print(f"[DEBUG] Evaluation - Original inputs device: {inputs.device}, labels device: {labels.device}")
            inputs, labels = inputs.to(device), labels.to(device)
            print(f"[DEBUG] Evaluation - After .to(device) - inputs device: {inputs.device}, labels device: {labels.device}")
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

NUM_FOLDS = 10
NUM_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 32 * 8
LR = 0.001

def run_k_fold_training(dataset, num_folds=NUM_FOLDS, num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE, device='cuda', fold_values=None, wandb=None):
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
        print(f"[DEBUG] Fold {test_fold} - Model initialized on device: {next(model.parameters()).device}")
        print(f"[DEBUG] Fold {test_fold} - Target device: {device}")
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in tqdm(range(epochs), desc=f"Training Fold {test_fold}", leave=False):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "fold": test_fold
                })

        # Uncomment and implement evaluate if needed
        test_acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy fold {test_fold}: {test_acc:.4f}")
        if wandb:
            wandb.log({
                "test_accuracy": test_acc,
                "fold": test_fold
            })
        fold_accuracies.append(test_acc)

    avg_acc = sum(fold_accuracies) / num_folds
    print(f"Average accuracy across {num_folds} folds: {avg_acc:.4f}")
    if wandb:
        wandb.log({
            "average_accuracy": avg_acc,
            "num_folds": num_folds
        })  
    return fold_accuracies, avg_acc

def main():
    # Initialize the model
    data = load_dataset("EthanGLEdwards/urbansounds_melspectrograms")
    fold_values = np.array(data["train"]["fold"])

    dataset = AudioSpectrogramDataset(data['train'])
    # Assuming dataset is already preprocessed and contains mel spectrograms
    device=get_device()
    print(device)
    wandb = init_wandb(config={
        "num_folds": NUM_FOLDS,
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": device
    })

    run_k_fold_training(dataset, num_folds=NUM_FOLDS, num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device, fold_values=fold_values, wandb=wandb)
    

if __name__ == "__main__":
    main()
    # train()  # Uncomment to run the training function when implemented