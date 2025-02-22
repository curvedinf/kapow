import os
import torch
import torch.nn as nn

HIDDEN_SIZE = 1024

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class OptimizedNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=16, dropout=0.1):
        super(OptimizedNN, self).__init__()
        self.input_layer = nn.Linear(input_size, HIDDEN_SIZE)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.LayerNorm(HIDDEN_SIZE),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, x):
        # x should be of shape (input_size,) so we add a batch dimension if necessary:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.input_layer(x)
        # Use residual connection at every layer
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # add the residual connection
        x = self.output_layer(x)
        return x.squeeze(0)  # remove the batch dimension if not needed


def load_or_initialize(model_file, input_size, device):
    if os.path.isfile(model_file):
        model = torch.load(model_file, map_location=device, weights_only=False)
    else:
        output_size = input_size
        model = OptimizedNN(input_size, output_size).to(device)
    return model


def save_model(model, model_file):
    torch.save(model, model_file)