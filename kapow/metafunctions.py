import json
import os
import torch
import torch.nn as nn
from litellm import littellm

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def nn_metafunction(
        *args,
        _kapow_function_name,
        _kapow_optimizer_function,
        _kapow_function_arg_names,
        _kapow_function_signature,
        _kapow_optimizer_signature,
        _kapow_function_output_signature,
        _kapow_optimizer_output_signature,
        **kwargs,
):
    # Convert function arguments to JSON
    arg_json = json.dumps({'args': args, 'kwargs': kwargs})
    print(f"Arg JSON: {arg_json}")

    # Generate embedding from littellm
    embedding = littellm.embed(arg_json, model='voyage-code-3')
    embedding_vector = embedding['vector']  # Assuming this is the correct attribute
    input_size = len(embedding_vector)
    
    # Check for existing NN model or initialize a new one
    model_file = f"{_kapow_function_name}_nn.pth"
    if os.path.isfile(model_file):
        nn_model = torch.load(model_file)
    else:
        output_size = input_size  # As per spec
        nn_model = SimpleNN(input_size, output_size)
        torch.save(nn_model, model_file)
    
    # Convert embedding to tensor and run it through the neural network
    input_tensor = torch.tensor(embedding_vector, dtype=torch.float32)
    nn_output = nn_model(input_tensor)
    print(f"NN Output: {nn_output}")

    # Produce default output based on the function output signature
    temp_output = tuple(typ() for typ in _kapow_function_output_signature.values())
    print(f"Temp output: {temp_output}")
    output = temp_output if len(temp_output) > 1 else temp_output[0]
    
    optimized_output = _kapow_optimizer_function(*args, **kwargs)
    
    # Note: We're keeping the function's output the same as before for now
    return output