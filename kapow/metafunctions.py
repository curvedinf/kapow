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

    # Generate embedding from littellm
    embedding_vector = embedding(
        input=[f"JSON representation of a function call:\n\n{arg_json}"],
        model='voyage/voyage-code-3'
    )['data'][0]['embedding']
    input_size = len(embedding_vector)

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
    input_tensor = torch.tensor(embedding_vector, dtype=torch.float32)
    nn_output = nn_model(input_tensor)
    print(f"NN Output: {embedding_vector}")
    print(f"NN Output: {nn_output}")

    # Produce default output based on the function output signature
    temp_output = tuple(typ() for typ in _kapow_function_output_signature.values())
    output = temp_output if len(temp_output) > 1 else temp_output[0]

    # Get optimizer's output and convert to JSON
    optimized_output = _kapow_optimizer_function(*args, **kwargs)
    optimized_output_json = json.dumps({
        'optimizer_output': optimized_output
    })
    print(f"Optimized Output JSON: {optimized_output_json}")

    # Generate embedding vector for optimizer output
    optimizer_embedding_vector = embedding(
        input=[f"JSON representation of optimizer function output:\n\n{optimized_output_json}"],
        model='voyage/voyage-code-3'
    )['data'][0]['embedding']
    
    # Combine embedding vectors
    combined_embedding_vector = embedding_vector + optimizer_embedding_vector
    
    # Train the NN on combined input/output embeddings
    combined_input_tensor = torch.tensor(combined_embedding_vector, dtype=torch.float32)
    target_tensor = input_tensor  # Use initial input as target for initial style training
    
    # Define a simple loss function and an optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    # Training step
    optimizer.zero_grad()
    output_from_nn = nn_model(combined_input_tensor)
    loss = criterion(output_from_nn, target_tensor)
    loss.backward()
    optimizer.step()

    torch.save(nn_model, model_file)

    return output