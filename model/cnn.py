from torch import nn
import torch
class CNN(nn.Module):
    def __init__(self, input_channels=6, output_channels=2):
        super(CNN, self).__init__()
        
        # Define the layers
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, output_channels * 60)
        )

    def forward(self, data):
        x = data.x
        x = x.reshape(-1, 50, 50, 6) # (batch, 50, 50, 6)
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        return x.view(-1, 60, 2)