import json
import os
import time
import torch
import torch.nn as nn

from kapow.embedding import embed


# Define a simple feedforward neural network
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

nn_model = None

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
    global nn_model

    # Check if CUDA is available
    device = torch.device("cuda") #"cuda" if torch.cuda.is_available() else "cpu")

    print(f"Function name: {_kapow_function_name}")
    print(f"Function signature: {_kapow_function_arg_names}")
    print(f"kwargs: {kwargs}")
    print(f"args: {args}")
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
    embedding_vector = embed(f"JSON representation of a function call:\n\n{arg_json}")
    input_size = len(embedding_vector)
    print(f"Embedding vector length: {input_size}")  # Debug statement
    print(f"Embedding vector sample: {embedding_vector[:10]}")  # Show first 10 values

    # Check for existing NN model or initialize a new one
    model_file = f"{_kapow_function_name}.tnn"

    if nn_model is None:
        if os.path.isfile(model_file):
            # Load the model and move it to the device
            with torch.serialization.safe_globals([SimpleNN, nn.Linear, nn.ReLU]):
                nn_model = torch.load(model_file, map_location=device)
        else:
            output_size = input_size  # Start with input size as output size for flexibility
            nn_model = SimpleNN(input_size, output_size).to(device)

    # Convert embedding to tensor and move it to the device
    input_tensor = torch.tensor(embedding_vector, dtype=torch.float32).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")  # Debug statement
    print(f"Input tensor sample values: {input_tensor[:10]}")  # Show first 10 values
    nn_output = nn_model(input_tensor)
    print(f"NN Output shape: {nn_output.shape}")  # Debug statement
    print(f"NN Output sample values: {nn_output[:10]}")  # Show first 10 values

    # Produce default output based on the function output signature
    temp_output = tuple(typ() for typ in _kapow_function_output_signature.values())
    output = temp_output if len(temp_output) > 1 else temp_output[0]

    # Encode output_from_nn to JSON and generate embedding
    print("pre json.dumps")
    nn_output_json = json.dumps({
        'output_from_nn': nn_output.tolist()
    })
    print("embedding")
    output_embedding_vector = embed(f"JSON representation of NN output:\n\n{nn_output_json}")

    # Training using only input_tensor and output_embedding_tensor
    print("Pre torch.tensor")
    output_embedding_tensor = torch.tensor(output_embedding_vector, dtype=torch.float32).to(device)
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
