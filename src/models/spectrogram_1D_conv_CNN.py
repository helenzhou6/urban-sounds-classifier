import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN1D(nn.Module):
    def __init__(self, num_classes=10, input_dims=[128, 1501]):
        super(SpectrogramCNN1D, self).__init__()

        # input channels = 128 (frequency bins)
        # out_channels increase to learn more features
        self.conv1 = nn.Conv1d(input_dims[0], 16, kernel_size=3, stride=1, padding=1)  # output: [B, 16, 1501]
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)                   # output: [B, 16, 750]

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)             # output: [B, 32, 750]
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)                   # output: [B, 32, 375]

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)             # output: [B, 64, 375]
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)                   # output: [B, 64, 187]

        self.flattened_size = 64 * int(input_dims[1] // 8)  # channels * length after final pooling, rounded down

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [B, 128, 1501]
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)

        x = x.view(-1, self.flattened_size)    # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x