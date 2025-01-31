import json
import os
import time
import torch
import torch.nn as nn
from litellm import embedding

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

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
    arg_json = json.dumps(
        {
            'function_name': _kapow_function_name,
            'time': time.time_ns(),
            'args': args,
            'kwargs': kwargs,
            'arg_names': _kapow_function_arg_names,
        }
    )
    print(f"Arg JSON: {arg_json}")
    
    # Generate embedding from litellm
    embedding_vector = embedding(
        input=[f"JSON representation of a function call:\n\n{arg_json}"],
        model='voyage/voyage-code-3'
    )['data'][0]['embedding']
    input_size = len(embedding_vector)
    print(f"Embedding vector length: {input_size}")  # Debug statement
    print(f"Embedding vector sample: {embedding_vector[:10]}")  # Show first 10 values
    
    # Check for existing NN model or initialize a new one
    model_file = f"{_kapow_function_name}.tnn"
    # Allowlist the SimpleNN class
    if os.path.isfile(model_file):
        with torch.serialization.safe_globals([SimpleNN]):
            nn_model = torch.load(model_file, weights_only=False)
    else:
        output_size = input_size  # Start with input size as output size for flexibility
        nn_model = SimpleNN(input_size, output_size)
        torch.save(nn_model, model_file)
        
    # Convert embedding to tensor and run it through the neural network
    input_tensor = torch.tensor(embedding_vector, dtype=torch.float16)
    print(f"Input tensor shape: {input_tensor.shape}")  # Debug statement
    print(f"Input tensor sample values: {input_tensor[:10]}")  # Show first 10 values
    nn_output = nn_model(input_tensor)
    print(f"NN Output shape: {nn_output.shape}")  # Debug statement
    print(f"NN Output sample values: {nn_output[:10]}")  # Show first 10 values
    
    # Produce default output based on the function output signature
    temp_output = tuple(typ() for typ in _kapow_function_output_signature.values())
    output = temp_output if len(temp_output) > 1 else temp_output[0]
    
    # Encode output_from_nn to JSON and generate embedding
    nn_output_json = json.dumps({
        'output_from_nn': nn_output.tolist()
    })
    output_embedding_vector = embedding(
        input=[f"JSON representation of NN output:\n\n{nn_output_json}"],
        model='voyage/voyage-code-3'
    )['data'][0]['embedding']
    
    # Training using only input_tensor and output_embedding_tensor
    output_embedding_tensor = torch.tensor(output_embedding_vector, dtype=torch.float16)
    print(f"Output embedding tensor shape: {output_embedding_tensor.shape}")  # Debug statement

    # Define a simple loss function and an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    # Training step
    optimizer.zero_grad()
    output_from_nn = nn_model(input_tensor)
    print(f"Output from NN for training shape: {output_from_nn.shape}")  # Debug statement
    loss = criterion(output_from_nn, output_embedding_tensor)
    print(f"Loss: {loss.item()}")  # Debug statement
    loss.backward()
    optimizer.step()
    torch.save(nn_model, model_file)

    return output