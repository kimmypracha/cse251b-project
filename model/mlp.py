from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, input_features, output_features, num_layers=2):
        super(MLP, self).__init__()
        
        # Define the layers
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(input_features, 1024))
        layers.append(nn.ReLU())
        hidden_size = 1024
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size//2))
            layers.append(nn.ReLU())
            hidden_size //= 2
        layers.append(nn.Linear(hidden_size, output_features))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, data):
        x = data.x
        # x = x[:, :, :, :6] # (batch, 50, 50, 6)
        x = x.reshape(-1, 50 * 50 * 6)
        x = self.mlp(x)
        return x.view(-1, 60, 2)