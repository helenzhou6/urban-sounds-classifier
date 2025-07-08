import torch

class SpectrogramTransformer(torch.nn.Module):
    def __init__(self):
        super(SpectrogramTransformer, self).__init__()
        
        # Transformer layer
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.fc_latent = torch.nn.Linear(128, 256)
        
        # Fully connected layer
        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, x):
        # add positional encoding
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, 128, 1501))
        # x shape: [B, 128, 1501]
        # Permute to [1501, B, 128] for transformer
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        # Permute back to [B, 128, 1501]
        x = x.permute(1, 2, 0)
        # Global average pooling
        x = torch.mean(x, dim=2) # [B, 128]

        # Pass through a linear layer to get a latent representation
        x = self.fc_latent(x)  # [B, 256]
        # Fully connected layer
        x = self.classifier(x)
        return x
