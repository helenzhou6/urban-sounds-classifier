import torch
from models.spectrogram_1D_conv_CNN import SpectrogramCNN1D
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from utils import get_device, init_wandb, clear_gpu_cache, print_gpu_memory



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


class AudioSpectrogramDataset(Dataset):
    def __init__(self, data, preprocess=True):
        if preprocess:
            self.data = preprocess_dataset(data)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'spectrogram' in item:
            # Data is already preprocessed
            return item['spectrogram'], item['classID'], item['fold']
        else:
            # Legacy path - process on the fly
            label = item['classID']
            fold = item['fold']
            spectrogram = normalise_and_pad_spectrogram(item['mel_spectrogram'])
            
            # Convert to tensors for efficient GPU transfer
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            
            return spectrogram, label, fold


# -- TRAINING AND EVALUATION FUNCTIONS ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    # Print model device only once per epoch
    # print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    
    for batch_idx, (inputs, labels, fold) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Only print debug info for first 2 batches to avoid performance hit
        # if batch_idx < 2:
        #     print(f"[DEBUG] Batch {batch_idx} - Original inputs device: {inputs.device}, labels device: {labels.device}")
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # if batch_idx < 2:
        #     print(f"[DEBUG] Batch {batch_idx} - After .to(device) - inputs device: {inputs.device}, labels device: {labels.device}")

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
    # print(f"[DEBUG] Evaluation - Model device: {next(model.parameters()).device}")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, fold) in enumerate(loader):
            # Only print debug info for first batch in evaluation
            # if batch_idx == 0:
            #     print(f"[DEBUG] Evaluation - Original inputs device: {inputs.device}, labels device: {labels.device}")
            
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # if batch_idx == 0:
            #     print(f"[DEBUG] Evaluation - After .to(device) - inputs device: {inputs.device}, labels device: {labels.device}")
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

NUM_FOLDS = 10
NUM_CLASSES = 10
EPOCHS = 30
BATCH_SIZE = 32 * 16
LR = 0.0001

def run_k_fold_training(dataset, num_folds=NUM_FOLDS, num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE, device='cuda', fold_values=None, wandb=None):
    fold_accuracies = []

    for test_fold in tqdm(range(1, num_folds + 1), desc="K-Fold"):
        print(f"Fold {test_fold}/{num_folds}")

        train_indices = np.where(fold_values != test_fold)[0].tolist()
        test_indices = np.where(fold_values == test_fold)[0].tolist()

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        # Use multiple workers and pin memory for better GPU utilization
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True)

        # Initialize model, optimizer
        model = SpectrogramCNN1D(num_classes=num_classes, input_dims=[128, 1501]).to(device)
        # print(f"[DEBUG] Fold {test_fold} - Model initialized on device: {next(model.parameters()).device}")
        # print(f"[DEBUG] Fold {test_fold} - Target device: {device}")
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in tqdm(range(epochs), desc=f"Training Fold {test_fold}", leave=False):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Monitor GPU memory usage
            if device == 'cuda':
                print_gpu_memory()
            
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

    # Preprocess dataset once for better performance
    # Use only a small subset (e.g., 1000 items) for quick testing
    # subset_data = data['train'].select(range(2))
    # print(subset_data)
    dataset = AudioSpectrogramDataset(data['train'], preprocess=True)
    # dataset = AudioSpectrogramDataset(subset_data, preprocess=True)

    
    # Clear GPU cache before training
    device = get_device()
    clear_gpu_cache()
    print_gpu_memory()
    
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