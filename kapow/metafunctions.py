import torch
from kapow.model import load_or_initialize, save_model
from kapow.training import train_step
from kapow.processing import serialize_function_call, get_embedding_from_text, process_nn_output

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
    device = torch.device("cuda")  # Using CUDA as per original logic
    print(f"Function name: {_kapow_function_name}")
    print(f"Function signature: {_kapow_function_arg_names}")
    print(f"kwargs: {kwargs}")
    print(f"args: {args}")

    # Convert function arguments to JSON
    arg_json = serialize_function_call(
        _kapow_function_name, args, kwargs, _kapow_function_arg_names
    )
    print(f"Arg JSON: {arg_json}")

    # Generate embedding from litellm
    embedding_vector = get_embedding_from_text(
        f"JSON representation of a function call:\n\n{arg_json}"
    )
    input_size = len(embedding_vector)
    print(f"Embedding vector length: {input_size}")  # Debug statement
    print(f"Embedding vector sample: {embedding_vector[:10]}")  # Show first 10 values

    # Check for existing NN model or initialize a new one
    model_file = f"{_kapow_function_name}.tnn"
    if nn_model is None:
        nn_model = load_or_initialize(model_file, input_size, device)

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

    print("pre json.dumps")
    # Encode output_from_nn to JSON and generate embedding
    output_embedding_vector = process_nn_output(nn_output)
    print("embedding")
    output_embedding_tensor = torch.tensor(output_embedding_vector, dtype=torch.float32).to(device)
    print(f"Output embedding tensor shape: {output_embedding_tensor.shape}")  # Debug statement

    # Training step with input_tensor and output_embedding_tensor
    loss = train_step(nn_model, input_tensor, output_embedding_tensor)
    print(f"Loss: {loss.item()}")  # Debug statement

    save_model(nn_model, model_file)
    return output