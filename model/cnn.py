from torch import nn
import torch
class CNN(nn.Module):
    def __init__(self, 
                 input_channels=6, 
                 output_channels=2,
                 num_conv_blocks=3,
                 num_fc_blocks=2,
                 max_pooling=True):
        super(CNN, self).__init__()
        
        # Define the layers
        layers = []
        layers.append(nn.Conv2d(input_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        n_filters = 64
        w, h = 50, 50
        for _ in range(num_conv_blocks - 2):
            layers.append(nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            n_filters *= 2
            if max_pooling:
                layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
                w, h = w, h // 2

        layers.append(nn.Conv2d(n_filters, output_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        # Flatten the output
        layers.append(nn.Flatten())

        # Fully connected layers
        hidden_size = w * h * output_channels
        for _ in range(num_fc_blocks):
            layers.append(nn.Linear(hidden_size, hidden_size // 2))
            layers.append(nn.ReLU())
            hidden_size //= 2
        layers.append(nn.Linear(hidden_size, 60 * 2))
        self.model = nn.Sequential(*layers)


    def forward(self, data):
        x = data.x
        x = x.reshape(-1, 50, 50, 6) # (batch, 50, 50, 6)
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        return x.view(-1, 60, 2)