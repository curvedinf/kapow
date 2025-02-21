import os
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_or_initialize(model_file, input_size, device):
    if os.path.isfile(model_file):
        with torch.serialization.safe_globals([SimpleNN, nn.Linear, nn.ReLU]):
            model = torch.load(model_file, map_location=device)
    else:
        output_size = input_size  # Use input size as output size for flexibility
        model = SimpleNN(input_size, output_size).to(device)
    return model

def save_model(model, model_file):
    torch.save(model, model_file)