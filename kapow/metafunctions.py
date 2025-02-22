from json import loads, dumps
import torch
from kapow.decoder import decode
from kapow.embedding import embed, get_first_token_embedding
from kapow.model import load_or_initialize, save_model
from kapow.qwen_model import tokenizer
from kapow.training import train_step
from kapow.processing import serialize_function_call, process_nn_output, initialize_signature
from kapow.math import distance

nn_models = {}

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
    global nn_models
    device = torch.device("cuda")
    print("Metafunction called==========================================")
    print(f"Function name: {_kapow_function_name}")
    print(f"Function signature: {_kapow_function_arg_names}")
    print(f"kwargs: {kwargs}")
    print(f"args: {args}")

    # Convert function arguments to JSON
    arg_json = serialize_function_call(
        _kapow_function_name, args, kwargs, _kapow_function_arg_names
    )
    print(f"Arg JSON: {arg_json}")

    system_prompt = "You are a helpful assistant that only outputs raw JSON."

    prompt = (
        f"The JSON representation of a function call:\n\n{arg_json}\n\n"
        f"What is the correct output of the function call in raw JSON?"
        f"Example:\n[5.23]\n\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Prompt text: {text}")

    # Generate embedding from litellm
    prompt_embedding_vector = embed(prompt)
    input_size = len(prompt_embedding_vector)
    print(f"Embedding vector length: {input_size}")
    print(f"Embedding vector sample: {prompt_embedding_vector[:10]}")

    # Check for existing NN model or initialize a new one
    model_file = f"{_kapow_function_name}.tnn"
    if model_file not in nn_models:
        nn_models[model_file] = load_or_initialize(model_file, input_size, device)
    nn_model = nn_models[model_file]

    # Convert embedding to tensor and move it to the device
    input_tensor = torch.tensor(prompt_embedding_vector, dtype=torch.float32).to(device)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor sample values: {input_tensor[:10]}")
    nn_output = nn_model(input_tensor)
    print(f"NN Output shape: {nn_output.shape}")
    print(f"NN Output sample values: {nn_output[:10]}")

    # Encode output_from_nn to JSON and generate embedding
    output_embedding_vector = process_nn_output(nn_output)
    output_embedding_tensor = torch.tensor(output_embedding_vector, dtype=torch.float32).to(device)
    print(f"Output embedding tensor shape: {output_embedding_tensor.shape}")

    # Run the decoder on the output embedding vector
    decoded_text = decode(output_embedding_vector)
    print(f"Decoder output (raw text): {decoded_text}")
    try:
        # Attempt to parse the decoded text as JSON:
        output_data = loads(decoded_text)
    except Exception as e:
        print("Error decoding output JSON, returning default output signature:", e)
        output_data = initialize_signature(_kapow_function_output_signature)

    # If the output data is a list or tuple with a single element, unwrap it
    if any([isinstance(output_data, list), isinstance(output_data, tuple)]) and len(output_data) == 1:
        output_data = output_data[0]

    # If the output data is a dict with a single key, unwrap it
    if isinstance(output_data, dict) and len(output_data) == 1:
        output_data = list(output_data.values())[0]

    # Call the optimizer function on the NN output and the metafunction call args/kwargs.
    optimizer_output = _kapow_optimizer_function(output_data, args, kwargs)

    # Wrap the optimizer output in a list if it's not already a collection
    if not any([isinstance(optimizer_output, list), isinstance(optimizer_output, tuple), isinstance(optimizer_output, dict)]):
        optimizer_output = [optimizer_output]

    # Encode the optimizer output directly to json in the form [output_1, output_2]
    optimizer_output_json = dumps(optimizer_output)
    print(f"Optimizer output JSON: {optimizer_output_json}")

    target_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": optimizer_output_json}
    ]

    # Create an embedding vector of the json and convert it to a tensor.
    target_embedding_vector = get_first_token_embedding(target_messages)
    target_embedding_tensor = torch.tensor(target_embedding_vector, dtype=torch.float32).to(device)
    print(f"Target embedding tensor shape: {target_embedding_tensor.shape}")

    # Print the geometric distance between the target embedding vector and the output embedding vector
    vector_distance = distance(target_embedding_vector, output_embedding_vector)
    print(f"Vector distance: {vector_distance}")

    # Training step with input_tensor and target_embedding_tensor
    loss = train_step(nn_model, input_tensor, target_embedding_tensor)
    print(f"Loss: {loss.item()}")  # Debug statement
    save_model(nn_model, model_file)
    print(f"Output data: {output_data}")
    return output_data